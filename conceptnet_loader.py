"""
ConceptNet CSV loader.

Loads ConceptNet assertions from CSV and creates a dictionary
mapping start words to their related entries.
"""

import csv
import os
from pathlib import Path
from typing import NamedTuple

# TODO: Filter out relations (e.g., only keep RelatedTo, Synonym, IsA, etc.)

CONCEPTNET_CSV = "conceptnet/conceptnet-assertions-5.7.0.csv"


class ConceptNetEntry(NamedTuple):
    """A single ConceptNet assertion."""
    relation: str
    start: str
    end: str
    weight: float


def extract_word(concept_uri: str) -> str | None:
    """
    Extract the word from a ConceptNet URI.

    E.g., '/c/en/cat/n' -> 'cat'
         '/c/en/hot_dog' -> 'hot dog'
    """
    parts = concept_uri.split('/')
    if len(parts) >= 4:
        word = parts[3]
        return word.replace('_', ' ')
    return None


def extract_relation(relation_uri: str) -> str:
    """
    Extract the relation name from a ConceptNet URI.

    E.g., '/r/RelatedTo' -> 'RelatedTo'
    """
    return relation_uri.split('/')[-1]


def is_english(concept_uri: str) -> bool:
    """Check if a concept URI is English."""
    return concept_uri.startswith('/c/en/')


def load_conceptnet(
    csv_path: str | None = None,
    english_only: bool = True,
    limit: int | None = None
) -> dict[str, list[ConceptNetEntry]]:
    """
    Load ConceptNet CSV and create a dictionary from start words to entries.

    Args:
        csv_path: Path to the ConceptNet CSV file
        english_only: Only load English entries (default True)
        limit: Maximum number of lines to process (for testing)

    Returns:
        Dictionary mapping start words to list of ConceptNetEntry objects
    """
    if csv_path is None:
        # Try relative to script location
        script_dir = Path(__file__).parent
        csv_path = script_dir / CONCEPTNET_CSV

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Warning: ConceptNet CSV not found at '{csv_path}'")
        return {}

    print(f"Loading ConceptNet from {csv_path}...")

    concept_dict: dict[str, list[ConceptNetEntry]] = {}
    count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            if len(row) < 5:
                continue

            _, relation_uri, start_uri, end_uri, metadata = row[:5]

            # Filter to English only
            if english_only:
                if not is_english(start_uri) or not is_english(end_uri):
                    continue

            start_word = extract_word(start_uri)
            end_word = extract_word(end_uri)
            relation = extract_relation(relation_uri)

            if not start_word or not end_word:
                continue

            # Parse weight from metadata
            weight = 1.0
            if '"weight":' in metadata:
                try:
                    import json
                    meta = json.loads(metadata)
                    weight = meta.get('weight', 1.0)
                except json.JSONDecodeError:
                    pass

            entry = ConceptNetEntry(
                relation=relation,
                start=start_word,
                end=end_word,
                weight=weight
            )

            if start_word not in concept_dict:
                concept_dict[start_word] = []
            concept_dict[start_word].append(entry)

            count += 1
            if count % 500000 == 0:
                print(f"  Processed {count:,} entries...")

            if limit and count >= limit:
                break

    print(f"Loaded {count:,} entries for {len(concept_dict):,} unique words")
    return concept_dict


def get_related_words(concept_dict: dict[str, list[ConceptNetEntry]], word: str) -> list[str]:
    """Get all words related to the given word."""
    entries = concept_dict.get(word.lower(), [])
    return list(set(entry.end for entry in entries))


# Global dictionary - populated by load_conceptnet()
CONCEPTNET: dict[str, list[ConceptNetEntry]] = {}


if __name__ == '__main__':
    # Test loading
    concepts = load_conceptnet(limit=100000)
    print(f"\nSample entries for 'cat':")
    for entry in concepts.get('cat', [])[:10]:
        print(f"  {entry.relation}: {entry.start} -> {entry.end} (weight: {entry.weight})")
