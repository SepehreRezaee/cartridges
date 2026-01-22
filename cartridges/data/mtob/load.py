"""
MTOB Benchmark Data Loading Functions.

Data is loaded from the official MTOB benchmark dataset:
https://github.com/lukemelas/mtob

The _data directory should contain the following files from the official repo:
- test_examples_ke.json  (Kalamang to English test, 50 examples)
- test_examples_ek.json  (English to Kalamang test, 50 examples)
- train_examples.json    (Training examples)
- wordlist.json          (Bilingual wordlist)
- grammar_book.tex       (Original grammar book in LaTeX)
- grammar_book.txt       (Preprocessed plaintext grammar book)
- grammar_book_for_claude_long.txt
- grammar_book_for_claude_medium.txt

To download the data, run:
    curl -L -o mtob-dataset.zip https://github.com/lukemelas/mtob/raw/main/dataset-encrypted-with-password-kalamang.zip
    unzip -P kalamang mtob-dataset.zip -d mtob-data
    mkdir -p _data && cp mtob-data/splits/*.json _data/ && cp mtob-data/resources/* _data/
"""
import json
from pathlib import Path


dataset_root = Path(__file__).resolve().parent / "_data"


def _check_data_exists():
    """Check if the official MTOB data files exist."""
    required_files = [
        "test_examples_ke.json",
        "test_examples_ek.json",
    ]
    missing = [f for f in required_files if not (dataset_root / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"MTOB data files not found: {missing}. "
            f"Please download the official MTOB benchmark data from "
            f"https://github.com/lukemelas/mtob and extract to {dataset_root}/. "
            f"See module docstring for download instructions."
        )


def load_book_long():
    return (dataset_root / "grammar_book_for_claude_long.txt").read_text()


def load_book_medium():
    return (dataset_root / "grammar_book_for_claude_medium.txt").read_text()


def load_book_full():
    return (dataset_root / "grammar_book.txt").read_text()


def load_book_full_tex():
    return (dataset_root / "grammar_book.tex").read_text()


def load_wordlist():
    return json.loads((dataset_root / "wordlist.json").read_text())


def load_test_ek():
    """Load English-to-Kalamang test examples (50 examples from official MTOB benchmark)."""
    file_path = dataset_root / "test_examples_ek.json"
    if not file_path.exists():
        _check_data_exists()
    data = json.loads(file_path.read_text())[1:]  # Skip first element (canary)
    assert len(data) == 50, f"Expected 50 test examples, got {len(data)}"
    return data


def load_test_ke():
    """Load Kalamang-to-English test examples (50 examples from official MTOB benchmark)."""
    file_path = dataset_root / "test_examples_ke.json"
    if not file_path.exists():
        _check_data_exists()  
    data = json.loads(file_path.read_text())[1:]  # Skip first element (canary)
    assert len(data) == 50, f"Expected 50 test examples, got {len(data)}"
    return data


def load_train_examples():
    """Load training examples for parallel sentences."""
    data = json.loads((dataset_root / "train_examples.json").read_text())[1:]
    return data


def wordlist_to_lines(wordlist: dict[str, list[str] | str]) -> list:
    return (
        [
            f"{source}: {','.join(target) if isinstance(target, list) else target}"
            for (source, target) in wordlist.items()
        ]
    )
