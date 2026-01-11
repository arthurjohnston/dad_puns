#!/usr/bin/env python3
"""
Pun Generator - Finds pun opportunities by matching words to idioms.

Takes a single word, compares its pronunciation to words in common idioms,
and suggests puns where a word can be swapped with a similar-sounding word.

TODO: Quality improvements to implement:
  Phonetic:
    1. Syllable count matching - prefer same syllable count
    2. Prefer end-rhymes - matching final sounds makes stronger puns
    3. Weight by position - matches at word start/end matter more
  Word Quality:
    4. Use word frequency - prefer common words over obscure ones
    5. Part-of-speech matching - nouns replacing nouns sounds more natural
    6. Filter short substitutions - single letters like "a" sound awkward
  ConceptNet:
    7. Rank relations by humor value - CapableOf, Desires, Antonym > IsA, HasProperty
    8. Filter multi-word phrases - related words with spaces don't work
    9. Filter obscure related words - use word frequency on ConceptNet results
  Semantic:
    10. Bonus for semantic contrast - puns funnier when meaning shifts dramatically
"""

import argparse
import os
from pathlib import Path
from typing import Optional
from phonemizer.separator import Separator
from phonemizer.backend import EspeakBackend

from conceptnet_loader import load_conceptnet, get_related_words
from word_frequency import word_to_count

# Cache for pronunciations to avoid repeated phonemizer calls
_pron_cache: dict[str, list[str]] = {}

# Persistent phonemizer backend and separator (initialized lazily)
_espeak_backend = None
_phoneme_separator = Separator(phone=' ', word='', syllable='')


def _get_backend():
    """Get or create the persistent espeak backend."""
    global _espeak_backend
    if _espeak_backend is None:
        _espeak_backend = EspeakBackend(
            language='en-us',
            with_stress=True,
        )
    return _espeak_backend

# Common words to skip when matching (too short/generic to make good puns)
STOPWORDS = {
    'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'by', 'is', 'it',
    'be', 'as', 'or', 'if', 'so', 'no', 'up', 'we', 'he', 'me', 'my',
    'do', 'go', 'us', 'am',
}

# ConceptNet relations to skip (too generic/noisy)
SKIP_RELATIONS = {
    'HasContext', 'RelatedTo', 'AtLocation', 'HasA', 'HasPrerequisite', 'DerivedFrom',
    'EtymologicallyRelatedTo', 'IsA', 'DistinctFrom'
}

# IPA-ish English phoneme peer groups (space-free tokens like phonemizer output);
# diphthongs/rhotics are multi-char strings (e.g., "eɪ", "oʊ", "aɪ", "ɝ", "ɚ").
phone_to_peers = {
    # stops
    "p": ("t", "k"),
    "t": ("p", "k"),
    "k": ("p", "t"),

    "b": ("d", "ɡ"),
    "d": ("b", "ɡ"),
    "ɡ": ("b", "d"),

    # affricates
    "tʃ": (),
    "dʒ": (),

    # fricatives
    "f": ("θ", "s", "ʃ", "h"),
    "θ": ("f", "s", "ʃ", "h"),
    "s": ("f", "θ", "ʃ", "h"),
    "ʃ": ("f", "θ", "s", "h"),
    "h": ("f", "θ", "s", "ʃ"),

    "v": ("ð", "z", "ʒ"),
    "ð": ("v", "z", "ʒ"),
    "z": ("v", "ð", "ʒ"),
    "ʒ": ("v", "ð", "z"),

    # sonorants
    "m": ("n", "ŋ"),
    "n": ("m", "ŋ"),
    "ŋ": ("m", "n"),

    "l": ("ɹ",),
    "ɹ": ("l",),

    "w": ("j",),
    "j": ("w",),

    # vowels (rough CMUdict->IPA equivalents, stressless)
    # front-ish set
    "i":  ("ɪ", "eɪ", "ɛ", "æ"),
    "ɪ":  ("i", "eɪ", "ɛ", "æ"),
    "eɪ": ("i", "ɪ", "ɛ", "æ"),
    "ɛ":  ("i", "ɪ", "eɪ", "æ"),
    "æ":  ("i", "ɪ", "eɪ", "ɛ"),

    # central-ish set
    "ʌ": ("ə", "ɚ"),
    "ə": ("ʌ", "ɚ"),
    "ɚ": ("ʌ", "ə"),

    # back-ish set
    "u":  ("ʊ", "oʊ", "ɔ", "ɑ"),
    "ʊ":  ("u", "oʊ", "ɔ", "ɑ"),
    "oʊ": ("u", "ʊ", "ɔ", "ɑ"),
    "ɔ":  ("u", "ʊ", "oʊ", "ɑ"),
    "ɑ":  ("u", "ʊ", "oʊ", "ɔ"),

    # rhotic vowel
    "ɝ": (),

    # diphthongs
    "aɪ": ("aʊ", "ɔɪ"),
    "aʊ": ("aɪ", "ɔɪ"),
    "ɔɪ": ("aɪ", "aʊ"),
}


def are_peer_phonemes(p1: str, p2: str) -> bool:
    """Check if two phonemes (without stress markers) are in the same peer group."""
    if p1 == p2:
        return True
    peers = phone_to_peers.get(p1, ())
    return p2 in peers


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


def get_pronunciation(word: str) -> Optional[list[str]]:
    """
    Get the IPA pronunciation for a word using phonemizer.

    Returns:
        List of phonemes or None if not found
    """
    word_lower = word.lower()

    # Check cache first
    if word_lower in _pron_cache:
        return _pron_cache[word_lower]

    try:
        # Use persistent espeak backend for performance
        backend = _get_backend()
        ipa = backend.phonemize(
            [word_lower],
            separator=_phoneme_separator,
            strip=True,
        )[0]
        if ipa:
            # Split into individual phonemes
            phonemes = ipa.split()
            _pron_cache[word_lower] = phonemes
            return phonemes
    except Exception:
        pass

    _pron_cache[word_lower] = None
    return None


IPA_VOWELS = set('aɑæɐeəɛɜiɪɨoɔœøuʊʉɯʌyʏ')

# Minimum word length for substitutions (filters out awkward single-letter replacements)
MIN_WORD_LENGTH = 3


def get_word_frequency(word: str) -> int:
    """Get the frequency count for a word (0 if not in dictionary)."""
    return word_to_count.get(word.lower(), 0)


def pun_rank(edit_distance: float, word_frequency: int) -> tuple[float, int]:
    """
    Calculate ranking score for a pun result.

    Returns a tuple for sorting where:
    - Lower edit_distance is always better (primary sort)
    - Higher word_frequency is better (secondary sort, negated for ascending sort)

    Args:
        edit_distance: Phoneme edit distance (lower is better)
        word_frequency: Word frequency count (higher is better)

    Returns:
        Tuple (edit_distance, -word_frequency) for use in sorting
    """
    return (edit_distance, -word_frequency)


def count_syllables(pron: list[str]) -> int:
    """Count syllables by counting vowel-containing phonemes."""
    count = 0
    for phoneme in pron:
        stripped = phoneme.lstrip('ˈˌ')
        if any(c in IPA_VOWELS for c in stripped):
            count += 1
    return count


def is_stressed_vowel(phoneme: str) -> bool:
    """Check if a phoneme is a primary stressed vowel (starts with ˈ and contains a vowel)."""
    if not phoneme.startswith('ˈ'):
        return False
    return any(c in IPA_VOWELS for c in phoneme)


def get_vowel(phoneme: str) -> Optional[str]:
    """Extract the vowel from a phoneme, stripping stress markers."""
    stripped = phoneme.lstrip('ˈˌ')
    for c in stripped:
        if c in IPA_VOWELS:
            return c
    return None


def phoneme_edit_distance(pron1: list[str], pron2: list[str]) -> float:
    """
    Calculate the Levenshtein edit distance between two pronunciations.
    Stressed vowels (marked with ˈ) must match - substituting them costs heavily.
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
            p1 = ph1.lstrip('ˈˌ')
            p2 = ph2.lstrip('ˈˌ')

            if p1 == p2:
                dp[i][j] = dp[i-1][j-1]
            else:
                # Heavy penalty if either is a primary stressed vowel
                if is_stressed_vowel(ph1) or is_stressed_vowel(ph2):
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
    """Extract the primary stressed vowel (marked with ˈ) from a pronunciation."""
    for phoneme in pron:
        if is_stressed_vowel(phoneme):
            return get_vowel(phoneme)
    return None


def find_idiom_puns(
    word: str,
    idioms: list[str],
    max_distance: float = 1.0,
    source_word: str | None = None,
    relation: str | None = None
) -> list[tuple[str, str, str, float, str | None, str | None]]:
    """
    Find idioms where a word can be replaced with the input word.

    Args:
        word: The word to find pun matches for
        idioms: List of idiom phrases
        max_distance: Maximum edit distance (default 1.0)
        source_word: The original input word (if word is a related word)
        relation: The ConceptNet relation (if word is a related word)

    Returns:
        List of tuples: (original_idiom, punned_idiom, matched_word, distance, source_word, relation)
        Excludes exact matches (distance == 0).
    """
    word_pron = get_pronunciation(word)
    if not word_pron:
        # Only warn for single words (multi-word phrases won't have pronunciations)
        if ' ' not in word:
            print(f"Warning: No pronunciation found for '{word}'")
        return []

    # Filter out short pun words (#6)
    if len(word) < MIN_WORD_LENGTH:
        return []

    stressed_vowel = get_stressed_vowel(word_pron)
    word_syllables = count_syllables(word_pron)

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

            # Skip short words (#6)
            if len(clean_word) < MIN_WORD_LENGTH:
                continue

            idiom_word_pron = get_pronunciation(clean_word)
            if not idiom_word_pron:
                continue

            # Syllable counts must match (#1)
            if count_syllables(idiom_word_pron) != word_syllables:
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

    # Sort by edit distance (primary), then word frequency (secondary, higher is better)
    def sort_key(result):
        distance = result[3]
        punned_idiom = result[1]
        # Extract the uppercase (substituted) word
        sub_word = next((w for w in punned_idiom.split() if w.isupper()), '').lower()
        return pun_rank(distance, get_word_frequency(sub_word))

    results.sort(key=sort_key)
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

    # Show input word pronunciation
    word_pron = get_pronunciation(args.word)
    if word_pron and args.show_pronunciation:
        print(f"'{args.word}' pronunciation: [{format_pronunciation(word_pron)}]\n")

    idioms = load_idioms(args.idioms_file)
    if not idioms:
        print("No idioms loaded. Please check your idioms file.")
        return

    # Load ConceptNet for related words
    concept_dict = load_conceptnet()

    print(f"Searching for words with edit distance <= {args.max_distance}...\n")

    # Search with the original word
    results = find_idiom_puns(args.word, idioms, args.max_distance)

    # Also search with related words from ConceptNet
    seen_puns = set(r[1] for r in results)  # Track punned idioms we've seen
    entries = concept_dict.get(args.word.lower(), [])
    for entry in entries:
        # Skip noisy/generic relations
        if entry.relation in SKIP_RELATIONS:
            continue
        # Skip obscure words not in word frequency list (#9)
        if entry.end.lower() not in word_to_count:
            continue
        # Skip multi-word phrases (#8)
        if ' ' in entry.end:
            continue
        # Skip related words with same pronunciation (no pun if sounds identical)
        related_pron = get_pronunciation(entry.end)
        if related_pron and word_pron and related_pron == word_pron:
            continue
        related_results = find_idiom_puns(
            entry.end,
            idioms,
            args.max_distance,
            source_word=args.word,
            relation=entry.relation
        )
        # Only add new puns we haven't seen
        for r in related_results:
            if r[1] not in seen_puns:
                seen_puns.add(r[1])
                results.append(r)

    # Re-sort all results by edit distance (primary), word frequency (secondary)
    def sort_key(result):
        distance = result[3]
        punned_idiom = result[1]
        sub_word = next((w for w in punned_idiom.split() if w.isupper()), '').lower()
        return pun_rank(distance, get_word_frequency(sub_word))

    results.sort(key=sort_key)

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
            matched_pron = get_pronunciation(matched_word)
            sub_pron = get_pronunciation(substituted_word)
            if matched_pron and sub_pron:
                print(f"    [{format_pronunciation(matched_pron)}] → [{format_pronunciation(sub_pron)}]")
        print()


if __name__ == '__main__':
    main()
