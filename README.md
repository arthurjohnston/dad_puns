# Pun Generator

A tool that finds pun opportunities by matching words to similar-sounding words in common idioms.

## How It Works

1. Takes a single word as input (e.g., "cat")
2. Finds related words using ConceptNet (synonyms, antonyms, etc.)
3. Compares pronunciations to words in common idioms using phoneme edit distance
4. Suggests puns where words can be swapped with similar-sounding related words

## Installation

### Prerequisites

- Python 3.10+
- espeak-ng (for phonemizer)

On Ubuntu/Debian:
```bash
sudo apt install espeak-ng
```

On macOS:
```bash
brew install espeak-ng
```

### Python Dependencies

```bash
pip install phonemizer nltk
```

## Setting Up the Database

### 1. Download ConceptNet

Download the ConceptNet assertions file:

```bash
mkdir -p conceptnet
cd conceptnet
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
cd ..
```

### 2. Build the Database

```bash
python build_conceptnet_db.py
```

This creates `conceptnet.db` (~50MB) from the ConceptNet CSV (~3GB).

## Usage

### Basic Usage

```bash
python pun_generator.py <word>
```

### Examples

```bash
# Find cat puns
python pun_generator.py cat

# Find dog puns with higher edit distance threshold
python pun_generator.py dog -m 2

# Show related words from ConceptNet
python pun_generator.py cat -d

# Show pronunciations
python pun_generator.py cat -p
```

### Sample Output

```
$ python pun_generator.py cat

Finding idiom puns for 'cat'...
Searching for words with edit distance <= 1...

============================================================
Found 66 potential puns:
============================================================

  a feather in your cap
  → a feather in your CAT
    ('cap' → 'cat', distance: 0.5)

  hear something straight from the horse's mouth
  → hear something straight from the horse's MOUSE
    ('mouth' → 'mouse', distance: 0.5)
    (mouse: Desires of 'cat')

  let sleeping dogs lie
  → let sleeping DOG lie
    ('dogs' → 'dog', distance: 1.0)
    (dog: Antonym of 'cat')

  once bitten twice shy
  → once KITTEN twice shy
    ('bitten' → 'kitten', distance: 1.0)
    (kitten: EAT of 'cat')
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-m, --max-distance` | Maximum phoneme edit distance (default: 1) |
| `-d, --show-related` | Show related words from ConceptNet |
| `-p, --show-pronunciation` | Show IPA pronunciations |
| `-i, --idioms-file` | Custom idioms file (default: idioms.txt) |

## Data Sources

- **ConceptNet** - Knowledge graph for word relationships
- **Edinburgh Associative Thesaurus (EAT)** - Human word associations
- **idioms.txt** - Common English idioms and phrases

## Files

| File | Description |
|------|-------------|
| `pun_generator.py` | Main pun generation script |
| `build_conceptnet_db.py` | Build SQLite database from ConceptNet |
| `conceptnet_loader.py` | Load and query ConceptNet database |
| `word_frequency.py` | Word frequency data for filtering |
| `idioms.txt` | List of common idioms |
| `eat.json` | Edinburgh Associative Thesaurus data |
