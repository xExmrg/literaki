"""Minimal dictionary handling utilities."""

# Path to the currently loaded dictionary file.  The file is read lazily so the
# entire contents are never stored in memory at once.
_DICTIONARY_PATH = None


def load_dictionary(path: str) -> None:
    """Register the dictionary file to use for lookups.

    The file is not loaded into memory. Instead, its path is stored and the
    contents are streamed when needed.  This avoids constructing a large set
    when working with very big dictionaries.
    """
    global _DICTIONARY_PATH
    _DICTIONARY_PATH = path


def iter_dictionary_words():
    """Yield words from the loaded dictionary one by one."""
    if _DICTIONARY_PATH is None:
        raise RuntimeError("Dictionary path not set. Call load_dictionary first.")
    with open(_DICTIONARY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                yield word


def is_valid_word(word: str) -> bool:
    """Return True if the word is present in the dictionary file."""
    word_l = word.lower()
    return any(word_l == w for w in iter_dictionary_words())
