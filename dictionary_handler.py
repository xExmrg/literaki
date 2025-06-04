"""Minimal dictionary handling utilities."""

import os

DICTIONARY_WORDS = set()


def load_dictionary(path: str) -> None:
    """Load words from the given file into ``DICTIONARY_WORDS``.

    Parameters
    ----------
    path:
        Path to the dictionary file.

    Raises
    ------
    ValueError
        If ``path`` is not a non-empty string.
    FileNotFoundError
        If ``path`` does not point to an existing file.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    global DICTIONARY_WORDS
    with open(path, "r", encoding="utf-8") as f:
        DICTIONARY_WORDS = {w.strip().lower() for w in f if w.strip()}


def is_valid_word(word: str) -> bool:
    """Return True if the word is in the loaded dictionary."""
    return word.lower() in DICTIONARY_WORDS
