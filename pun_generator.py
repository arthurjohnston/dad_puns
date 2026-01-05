#!/usr/bin/env python3
"""
Pun Generator - Finds pun opportunities between two concepts.

Takes two words/phrases, looks up related words in WordNet (offline),
compares their pronunciations, and finds pairs with small edit distance
(potential puns).
"""

import argparse
import nltk
from nltk.corpus import cmudict, wordnet
from typing import Optional


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    for resource, name in [('cmudict', 'CMU Pronouncing Dictionary'),
                           ('wordnet', 'WordNet')]:
        try:
            if resource == 'cmudict':
                cmudict.dict()
            else:
                wordnet.synsets('test')
        except LookupError:
            print(f"Downloading {name}...")
            nltk.download(resource, quiet=True)


def get_wordnet_related(concept: str, limit: int = 50) -> list[str]:
    """
    Get words related to the given concept using WordNet (offline).

    Finds synonyms, hypernyms (more general), hyponyms (more specific),
    and other semantically related words.

    Args:
        concept: The word or phrase to look up
        limit: Maximum number of related words to return

    Returns:
        List of related words
    """
    related_words = set()
    related_words.add(concept.lower())

    # Get all synsets for the concept
    synsets = wordnet.synsets(concept.lower())

    for synset in synsets:
        # Add lemma names (synonyms)
        for lemma in synset.lemmas():
            word = lemma.name().replace('_', ' ').lower()
            related_words.add(word)

            # Add antonyms
            for antonym in lemma.antonyms():
                related_words.add(antonym.name().replace('_', ' ').lower())

        # Add hypernyms (more general terms)
        for hypernym in synset.hypernyms():
            for lemma in hypernym.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())

        # Add hyponyms (more specific terms)
        for hyponym in synset.hyponyms():
            for lemma in hyponym.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())

        # Add meronyms (part-of relationships)
        for meronym in synset.part_meronyms() + synset.substance_meronyms():
            for lemma in meronym.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())

        # Add holonyms (whole-of relationships)
        for holonym in synset.part_holonyms() + synset.substance_holonyms():
            for lemma in holonym.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())

        # Add similar-to relationships
        for similar in synset.similar_tos():
            for lemma in similar.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())

        # Add also-see relationships
        for also_see in synset.also_sees():
            for lemma in also_see.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())

    return list(related_words)[:limit]


def get_pronunciation(word: str, pron_dict: dict) -> Optional[list[str]]:
    """
    Get the CMU pronunciation for a word.

    Args:
        word: The word to look up
        pron_dict: The CMU pronouncing dictionary

    Returns:
        List of phonemes or None if not found
    """
    word_lower = word.lower()
    if word_lower in pron_dict:
        # Return first pronunciation (some words have multiple)
        return pron_dict[word_lower][0]
    return None


def phoneme_edit_distance(pron1: list[str], pron2: list[str]) -> int:
    """
    Calculate the Levenshtein edit distance between two pronunciations.

    Args:
        pron1: First pronunciation (list of phonemes)
        pron2: Second pronunciation (list of phonemes)

    Returns:
        Edit distance (number of insertions, deletions, or substitutions)
    """
    m, n = len(pron1), len(pron2)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill in the rest
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Strip stress markers for comparison (e.g., "AH0" -> "AH")
            p1 = pron1[i-1].rstrip('012')
            p2 = pron2[j-1].rstrip('012')

            if p1 == p2:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )

    return dp[m][n]


def find_pun_candidates(
    concept1: str,
    concept2: str,
    max_edit_distance: int = 3,
    min_edit_distance: int = 1,
    related_limit: int = 50
) -> list[tuple[str, str, int, list[str], list[str]]]:
    """
    Find pun candidates between two concepts.

    Args:
        concept1: First concept/word
        concept2: Second concept/word
        max_edit_distance: Maximum pronunciation edit distance to consider
        min_edit_distance: Minimum edit distance (0 would be identical)
        related_limit: How many related words to fetch per concept

    Returns:
        List of tuples: (word1, word2, edit_distance, pronunciation1, pronunciation2)
    """
    ensure_nltk_data()
    pron_dict = cmudict.dict()

    print(f"Looking up words related to '{concept1}'...")
    words1 = get_wordnet_related(concept1, limit=related_limit)
    print(f"  Found {len(words1)} related words")

    print(f"Looking up words related to '{concept2}'...")
    words2 = get_wordnet_related(concept2, limit=related_limit)
    print(f"  Found {len(words2)} related words")

    candidates = []

    # Get pronunciations for all words
    prons1 = {}
    for word in words1:
        # Handle multi-word phrases by checking individual words
        tokens = word.split()
        if len(tokens) == 1:
            pron = get_pronunciation(word, pron_dict)
            if pron:
                prons1[word] = pron

    prons2 = {}
    for word in words2:
        tokens = word.split()
        if len(tokens) == 1:
            pron = get_pronunciation(word, pron_dict)
            if pron:
                prons2[word] = pron

    print(f"Found pronunciations for {len(prons1)} words from concept 1")
    print(f"Found pronunciations for {len(prons2)} words from concept 2")
    print("Comparing pronunciations...")

    # Compare all pairs
    for word1, pron1 in prons1.items():
        for word2, pron2 in prons2.items():
            if word1 == word2:
                continue

            distance = phoneme_edit_distance(pron1, pron2)

            if min_edit_distance <= distance <= max_edit_distance:
                candidates.append((word1, word2, distance, pron1, pron2))

    # Sort by edit distance (closest matches first)
    candidates.sort(key=lambda x: (x[2], x[0], x[1]))

    return candidates


def format_pronunciation(pron: list[str]) -> str:
    """Format a pronunciation for display."""
    return ' '.join(pron)


def main():
    parser = argparse.ArgumentParser(
        description='Generate puns by finding phonetically similar words between two concepts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "cat" "music"
  %(prog)s "programming" "food"
  %(prog)s "ocean" "space"
        """
    )
    parser.add_argument(
        'concept1',
        help='First concept (word or phrase in quotes)'
    )
    parser.add_argument(
        'concept2',
        help='Second concept (word or phrase in quotes)'
    )
    parser.add_argument(
        '--max-distance', '-d',
        type=int,
        default=3,
        help='Maximum phoneme edit distance (default: 3)'
    )
    parser.add_argument(
        '--min-distance', '-m',
        type=int,
        default=1,
        help='Minimum phoneme edit distance (default: 1)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=50,
        help='Number of related words to fetch per concept (default: 50)'
    )
    parser.add_argument(
        '--show-pronunciation', '-p',
        action='store_true',
        help='Show phonetic pronunciations in output'
    )

    args = parser.parse_args()

    print(f"\nFinding puns between '{args.concept1}' and '{args.concept2}'...\n")

    candidates = find_pun_candidates(
        args.concept1,
        args.concept2,
        max_edit_distance=args.max_distance,
        min_edit_distance=args.min_distance,
        related_limit=args.limit
    )

    if not candidates:
        print("\nNo pun candidates found. Try:")
        print("  - Different concepts")
        print("  - Increasing --max-distance")
        print("  - Increasing --limit for more related words")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(candidates)} potential pun pairs:")
    print(f"{'='*60}\n")

    for word1, word2, distance, pron1, pron2 in candidates:
        print(f"  {word1} <-> {word2}  (edit distance: {distance})")
        if args.show_pronunciation:
            print(f"    [{format_pronunciation(pron1)}]")
            print(f"    [{format_pronunciation(pron2)}]")
            print()


if __name__ == '__main__':
    main()
