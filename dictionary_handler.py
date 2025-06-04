"""Minimal dictionary handling utilities."""

DICTIONARY_WORDS = set()


def load_dictionary(path: str) -> None:
    """Load words from the given file into DICTIONARY_WORDS."""
    global DICTIONARY_WORDS
    with open(path, "r", encoding="utf-8") as f:
        DICTIONARY_WORDS = {w.strip().lower() for w in f if w.strip()}


def is_valid_word(word: str) -> bool:
    """Return True if the word is in the loaded dictionary."""
    return word.lower() in DICTIONARY_WORDS
