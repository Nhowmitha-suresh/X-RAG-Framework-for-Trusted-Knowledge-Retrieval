from typing import List


def chunk_text(text: str, chunk_size: int = 80) -> List[str]:
    """Split `text` into word-based chunks of approximately `chunk_size` words.

    Returns a list of chunk strings.
    """
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
