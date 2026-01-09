"""
ConceptNet SQLite loader.

Loads ConceptNet assertions from SQLite database and provides
dictionary-like access mapping start words to their related entries.

Run build_conceptnet_db.py first to create the database from CSV.
"""

import sqlite3
from pathlib import Path
from typing import NamedTuple

# TODO: Filter out relations (e.g., only keep RelatedTo, Synonym, IsA, etc.)

CONCEPTNET_DB = "conceptnet.db"


class ConceptNetEntry(NamedTuple):
    """A single ConceptNet assertion."""
    relation: str
    start: str
    end: str
    weight: float


class ConceptNetDict:
    """
    Dictionary-like wrapper around SQLite database.

    Provides dict-style access (concept_dict[word] or concept_dict.get(word))
    but fetches from SQLite on demand for fast startup.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def get(self, word: str, default=None) -> list[ConceptNetEntry]:
        """Get entries for a word, returning default if not found."""
        try:
            return self[word]
        except KeyError:
            return default if default is not None else []

    def __getitem__(self, word: str) -> list[ConceptNetEntry]:
        """Get all entries for a given start word."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT relation, start, end, weight FROM entries WHERE start = ?',
            (word.lower(),)
        )

        rows = cursor.fetchall()
        if not rows:
            raise KeyError(word)

        return [
            ConceptNetEntry(relation=r[0], start=r[1], end=r[2], weight=r[3])
            for r in rows
        ]

    def __contains__(self, word: str) -> bool:
        """Check if word exists in database."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM entries WHERE start = ? LIMIT 1', (word.lower(),))
        return cursor.fetchone() is not None

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


def load_conceptnet(db_path: str | None = None) -> ConceptNetDict:
    """
    Load ConceptNet from SQLite database.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        ConceptNetDict providing dictionary-like access to entries
    """
    if db_path is None:
        script_dir = Path(__file__).parent
        db_path = script_dir / CONCEPTNET_DB

    db_path = Path(db_path)
    if not db_path.exists():
        print(f"Warning: ConceptNet database not found at '{db_path}'")
        print("Run build_conceptnet_db.py first to create the database.")
        return ConceptNetDict(db_path)  # Return empty dict-like object

    print(f"Loaded ConceptNet database from {db_path}")
    return ConceptNetDict(db_path)


def get_related_words(concept_dict: ConceptNetDict, word: str) -> list[str]:
    """Get all words related to the given word."""
    entries = concept_dict.get(word.lower(), [])
    return list(set(entry.end for entry in entries))


if __name__ == '__main__':
    # Test loading
    concepts = load_conceptnet()
    print(f"\nSample entries for 'cat':")
    for entry in concepts.get('cat', [])[:10]:
        print(f"  {entry.relation}: {entry.start} -> {entry.end} (weight: {entry.weight})")
