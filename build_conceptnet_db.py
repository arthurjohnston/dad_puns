#!/usr/bin/env python3
"""
Build SQLite database from ConceptNet CSV.

This script reads the ConceptNet CSV file and creates a SQLite database
for fast lookups by start word.
"""

import csv
import json
import sqlite3
from pathlib import Path

CONCEPTNET_CSV = "conceptnet/conceptnet-assertions-5.7.0.csv"
CONCEPTNET_DB = "conceptnet.db"


def extract_word(concept_uri: str) -> str | None:
    """Extract the word from a ConceptNet URI."""
    parts = concept_uri.split('/')
    if len(parts) >= 4:
        word = parts[3]
        return word.replace('_', ' ')
    return None


def extract_relation(relation_uri: str) -> str:
    """Extract the relation name from a ConceptNet URI."""
    return relation_uri.split('/')[-1]


def is_english(concept_uri: str) -> bool:
    """Check if a concept URI is English."""
    return concept_uri.startswith('/c/en/')


def build_database(csv_path: str | None = None, db_path: str | None = None):
    """
    Build SQLite database from ConceptNet CSV.

    Args:
        csv_path: Path to the ConceptNet CSV file
        db_path: Path to the output SQLite database
    """
    script_dir = Path(__file__).parent

    if csv_path is None:
        csv_path = script_dir / CONCEPTNET_CSV
    else:
        csv_path = Path(csv_path)

    if db_path is None:
        db_path = script_dir / CONCEPTNET_DB
    else:
        db_path = Path(db_path)

    if not csv_path.exists():
        print(f"Error: ConceptNet CSV not found at '{csv_path}'")
        return

    print(f"Building database from {csv_path}...")
    print(f"Output: {db_path}")

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    # Create database and table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start TEXT NOT NULL,
            relation TEXT NOT NULL,
            end TEXT NOT NULL,
            weight REAL DEFAULT 1.0
        )
    ''')

    # Create index on start for fast lookups
    cursor.execute('CREATE INDEX idx_start ON entries (start)')

    count = 0
    batch = []
    batch_size = 10000

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            if len(row) < 5:
                continue

            _, relation_uri, start_uri, end_uri, metadata = row[:5]

            # Filter to English only
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
                    meta = json.loads(metadata)
                    weight = meta.get('weight', 1.0)
                except json.JSONDecodeError:
                    pass

            batch.append((start_word, relation, end_word, weight))

            if len(batch) >= batch_size:
                cursor.executemany(
                    'INSERT INTO entries (start, relation, end, weight) VALUES (?, ?, ?, ?)',
                    batch
                )
                conn.commit()
                count += len(batch)
                print(f"  Processed {count:,} entries...")
                batch = []

    # Insert remaining batch
    if batch:
        cursor.executemany(
            'INSERT INTO entries (start, relation, end, weight) VALUES (?, ?, ?, ?)',
            batch
        )
        conn.commit()
        count += len(batch)

    # Get stats
    cursor.execute('SELECT COUNT(DISTINCT start) FROM entries')
    unique_words = cursor.fetchone()[0]

    conn.close()

    print(f"\nDone! Created database with {count:,} entries for {unique_words:,} unique words")
    print(f"Database size: {db_path.stat().st_size / (1024*1024):.1f} MB")


if __name__ == '__main__':
    build_database()
