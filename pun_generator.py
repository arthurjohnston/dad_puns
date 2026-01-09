#!/usr/bin/env python3
"""
Pun Generator - Finds pun opportunities by matching words to idioms.

Takes a single word, compares its pronunciation to words in common idioms,
and suggests puns where a word can be swapped with a similar-sounding word.
"""

import argparse
import os
import nltk
from nltk.corpus import cmudict
from pathlib import Path
from typing import Optional

from conceptnet_loader import load_conceptnet, get_related_words

# Common words to skip when matching (too short/generic to make good puns)
STOPWORDS = {
    'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'by', 'is', 'it',
    'be', 'as', 'or', 'if', 'so', 'no', 'up', 'we', 'he', 'me', 'my',
    'do', 'go', 'us', 'am',
}

# ConceptNet relations to skip (too generic/noisy)
SKIP_RELATIONS = {
    'HasContext', 'RelatedTo', 'AtLocation', 'HasA', 'HasPrerequisite',
}


def ensure_cmudict():
    """Download cmudict if not already present."""
    try:
        cmudict.dict()
    except LookupError:
        print("Downloading CMU Pronouncing Dictionary...")
        nltk.download('cmudict', quiet=True)


def load_idioms(idioms_file: str) -> list[str]:
    """Load idioms from a text file (one per line)."""
    path = Path(idioms_file)
    if not path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        path = script_dir / idioms_file

    if not path.exists():
        print(f"Warning: Idioms file '{idioms_file}' not found")
        return []

    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]


def get_pronunciation(word: str, pron_dict: dict) -> Optional[list[str]]:
    """
    Get the CMU pronunciation for a word.

    Returns:
        List of phonemes or None if not found
    """
    word_lower = word.lower()
    if word_lower in pron_dict:
        return pron_dict[word_lower][0]
    return None


phone_to_peers = {
    # stops
    "P":  ("K", "T"),
    "T":  ("K", "P"),
    "K":  ("P", "T"),
    "B":  ("D", "G"),

    #voiced stops
    "D":  ("B", "G"),
    "G":  ("B", "D"),

    # affricates
    "CH": (),
    "JH": (),

    # fricatives
    "F":  ("HH", "S", "SH", "TH"),
    "TH": ("F", "HH", "S", "SH"),
    "S":  ("F", "HH", "SH", "TH"),
    "SH": ("F", "HH", "S", "TH"),
    "HH": ("F", "S", "SH", "TH"),

    "V":  ("DH", "Z", "ZH"),
    "DH": ("V", "Z", "ZH"),
    "Z":  ("DH", "V", "ZH"),
    "ZH": ("DH", "V", "Z"),

    # sonorants
    "M":  ("N", "NG"),
    "N":  ("M", "NG"),
    "NG": ("M", "N"),

    "L":  ("R",),
    "R":  ("L",),

    "W":  ("Y",),
    "Y":  ("W",),

    # vowels (CMUdict, stressless phones)
    "IY": ("AE", "EH", "EY", "IH"),
    "IH": ("AE", "EH", "EY", "IY"),
    "EY": ("AE", "EH", "IH", "IY"),
    "EH": ("AE", "EY", "IH", "IY"),
    "AE": ("EH", "EY", "IH", "IY"),

    "AH":  ("AX", "AXR"),
    "AX":  ("AH", "AXR"),
    "AXR": ("AH", "AX"),

    "UW": ("AA", "AO", "OW", "UH"),
    "UH": ("AA", "AO", "OW", "UW"),
    "OW": ("AA", "AO", "UH", "UW"),
    "AO": ("AA", "OW", "UH", "UW"),
    "AA": ("AO", "OW", "UH", "UW"),

    "ER": (),

    "AY": ("AW", "OY"),
    "AW": ("AY", "OY"),
    "OY": ("AW", "AY"),
}

def are_peer_phonemes(p1: str, p2: str) -> bool:
    """Check if two phonemes (without stress markers) are in the same peer group."""
    if p1 == p2:
        return True
    peers = phone_to_peers.get(p1, ())
    return p2 in peers


def phoneme_edit_distance(pron1: list[str], pron2: list[str]) -> float:
    """
    Calculate the Levenshtein edit distance between two pronunciations.
    Stressed vowels (ending in '1') must match - substituting them costs heavily.
    Peer phonemes (similar sounds) cost 0.5 to substitute instead of 1.
    """
    m, n = len(pron1), len(pron2)
    INF = 1000.0  # High cost to prevent stressed vowel changes
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ph1 = pron1[i-1]
            ph2 = pron2[j-1]
            # Strip stress markers for comparison
            p1 = ph1.rstrip('012')
            p2 = ph2.rstrip('012')

            if p1 == p2:
                dp[i][j] = dp[i-1][j-1]
            else:
                # Heavy penalty if either is a primary stressed vowel
                if ph1.endswith('1') or ph2.endswith('1'):
                    sub_cost = INF
                # Reduced cost if phonemes are peers (similar sounds)
                elif are_peer_phonemes(p1, p2):
                    sub_cost = 0.5
                else:
                    sub_cost = 1.0
                dp[i][j] = min(
                    dp[i-1][j] + 1.0,      # deletion
                    dp[i][j-1] + 1.0,      # insertion
                    dp[i-1][j-1] + sub_cost  # substitution
                )

    return dp[m][n]


def get_stressed_vowel(pron: list[str]) -> Optional[str]:
    """Extract the primary stressed vowel (ending in '1') from a pronunciation."""
    for phoneme in pron:
        if phoneme.endswith('1'):
            return phoneme.rstrip('1')
    return None


def find_idiom_puns(
    word: str,
    idioms: list[str],
    pron_dict: dict,
    max_distance: float = 1.0,
    source_word: str | None = None,
    relation: str | None = None
) -> list[tuple[str, str, str, float, str | None, str | None]]:
    """
    Find idioms where a word can be replaced with the input word.

    Args:
        word: The word to find pun matches for
        idioms: List of idiom phrases
        pron_dict: CMU pronunciation dictionary
        max_distance: Maximum edit distance (default 1.0)
        source_word: The original input word (if word is a related word)
        relation: The ConceptNet relation (if word is a related word)

    Returns:
        List of tuples: (original_idiom, punned_idiom, matched_word, distance, source_word, relation)
        Excludes exact matches (distance == 0).
    """
    word_pron = get_pronunciation(word, pron_dict)
    if not word_pron:
        # Only warn for single words (multi-word phrases won't have pronunciations)
        if ' ' not in word:
            print(f"Warning: No pronunciation found for '{word}'")
        return []

    stressed_vowel = get_stressed_vowel(word_pron)

    results = []
    seen_puns = set()

    for idiom in idioms:
        words_in_idiom = idiom.split()

        for i, idiom_word in enumerate(words_in_idiom):
            # Clean punctuation
            clean_word = ''.join(c for c in idiom_word if c.isalpha())
            if not clean_word or clean_word == word.lower():
                continue

            # Skip common stopwords
            if clean_word in STOPWORDS:
                continue

            idiom_word_pron = get_pronunciation(clean_word, pron_dict)
            if not idiom_word_pron:
                continue

            # Stressed vowels must match
            if get_stressed_vowel(idiom_word_pron) != stressed_vowel:
                continue

            distance = phoneme_edit_distance(word_pron, idiom_word_pron)

            if 0 < distance <= max_distance:
                # Create the punned version
                new_words = words_in_idiom.copy()
                new_words[i] = word.upper()
                punned_idiom = ' '.join(new_words)

                # Avoid duplicates
                if punned_idiom not in seen_puns:
                    seen_puns.add(punned_idiom)
                    results.append((idiom, punned_idiom, clean_word, distance, source_word, relation))

    # Sort by distance, then alphabetically
    results.sort(key=lambda x: (x[3], x[0]))
    return results


def format_pronunciation(pron: list[str]) -> str:
    """Format a pronunciation for display."""
    return ' '.join(pron)


def main():
    parser = argparse.ArgumentParser(
        description='Find pun opportunities by matching a word to similar-sounding words in idioms.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "sun"
  %(prog)s "deer" --max-distance 2
  %(prog)s "knight" --show-pronunciation
        """
    )
    parser.add_argument(
        'word',
        help='The word to find pun matches for'
    )
    parser.add_argument(
        '--max-distance', '-m',
        type=int,
        default=1,
        help='Maximum phoneme edit distance (default: 1)'
    )
    parser.add_argument(
        '--idioms-file', '-f',
        default='idioms.txt',
        help='Path to idioms file (default: idioms.txt)'
    )
    parser.add_argument(
        '--show-pronunciation', '-p',
        action='store_true',
        help='Show phonetic pronunciations'
    )
    parser.add_argument(
        '--show-related', '-d',
        action='store_true',
        help='Show related words from ConceptNet and exit'
    )

    args = parser.parse_args()

    # Handle --show-related option
    if args.show_related:
        print(f"\nRelated words for '{args.word}':\n")
        concept_dict = load_conceptnet()
        entries = concept_dict.get(args.word.lower(), [])
        if not entries:
            print("No related words found.")
            return
        # Group by relation
        by_relation = {}
        for entry in entries:
            if entry.relation in SKIP_RELATIONS:
                continue
            if entry.relation not in by_relation:
                by_relation[entry.relation] = []
            by_relation[entry.relation].append(entry.end)
        for relation, words in sorted(by_relation.items()):
            print(f"  {relation}: {', '.join(words)}")
        return

    print(f"\nFinding idiom puns for '{args.word}'...\n")

    ensure_cmudict()
    pron_dict = cmudict.dict()

    # Show input word pronunciation
    word_pron = get_pronunciation(args.word, pron_dict)
    if word_pron and args.show_pronunciation:
        print(f"'{args.word}' pronunciation: [{format_pronunciation(word_pron)}]\n")

    idioms = load_idioms(args.idioms_file)
    if not idioms:
        print("No idioms loaded. Please check your idioms file.")
        return

    print(f"Loaded {len(idioms)} idioms")

    # Load ConceptNet for related words
    concept_dict = load_conceptnet()

    print(f"Searching for words with edit distance <= {args.max_distance}...\n")

    # Search with the original word
    results = find_idiom_puns(args.word, idioms, pron_dict, args.max_distance)

    # Also search with related words from ConceptNet
    seen_puns = set(r[1] for r in results)  # Track punned idioms we've seen
    entries = concept_dict.get(args.word.lower(), [])
    for entry in entries:
        # Skip noisy/generic relations
        if entry.relation in SKIP_RELATIONS:
            continue
        related_results = find_idiom_puns(
            entry.end,
            idioms,
            pron_dict,
            args.max_distance,
            source_word=args.word,
            relation=entry.relation
        )
        # Only add new puns we haven't seen
        for r in related_results:
            if r[1] not in seen_puns:
                seen_puns.add(r[1])
                results.append(r)

    # Re-sort all results
    results.sort(key=lambda x: (x[3], x[0]))

    if not results:
        print("No pun matches found. Try:")
        print("  - A different word")
        print("  - Increasing --max-distance")
        return

    print(f"{'='*60}")
    print(f"Found {len(results)} potential puns:")
    print(f"{'='*60}\n")

    for original, punned, matched_word, distance, source_word, relation in results:
        print(f"  {original}")
        print(f"  → {punned}")

        # Get the word used for the pun (either original input or related word)
        pun_word = args.word if source_word is None else source_word
        # Figure out what word was actually substituted
        substituted_word = args.word if relation is None else next(
            (w for w in punned.split() if w.isupper()), args.word
        ).lower()

        if relation:
            print(f"    ('{matched_word}' → '{substituted_word}', distance: {distance})")
            print(f"    ({substituted_word}: {relation} of '{args.word}')")
        else:
            print(f"    ('{matched_word}' → '{args.word}', distance: {distance})")

        if args.show_pronunciation:
            matched_pron = get_pronunciation(matched_word, pron_dict)
            sub_pron = get_pronunciation(substituted_word, pron_dict)
            if matched_pron and sub_pron:
                print(f"    [{format_pronunciation(matched_pron)}] → [{format_pronunciation(sub_pron)}]")
        print()


if __name__ == '__main__':
    main()
