"""Recursive, overlap-aware text chunking.

We split on the most semantic boundary that fits (paragraph -> line ->
sentence -> word), which keeps related sentences together and gives the
embedding model coherent passages. Overlap preserves context that would
otherwise be cut across a chunk boundary.
"""
import re
from typing import List

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on(text: str, sep: str) -> List[str]:
    if sep == "":
        return list(text)
    parts = text.split(sep)
    # Re-attach the separator (except the trailing piece) so we don't lose it.
    return [p + sep for p in parts[:-1]] + parts[-1:]


def _recursive_split(text: str, size: int, separators: List[str]) -> List[str]:
    if len(text) <= size or not separators:
        return [text]
    sep, *rest = separators
    chunks: List[str] = []
    buf = ""
    for piece in _split_on(text, sep):
        if len(piece) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_recursive_split(piece, size, rest))
        elif len(buf) + len(piece) <= size:
            buf += piece
        else:
            if buf:
                chunks.append(buf)
            buf = piece
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into ~``size`` char chunks with ``overlap`` char overlap."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    base = _recursive_split(text, size, _SEPARATORS)
    if overlap <= 0 or len(base) <= 1:
        return [c.strip() for c in base if c.strip()]

    # Add a tail of the previous chunk to the front of the next one.
    overlapped: List[str] = []
    for i, chunk in enumerate(base):
        if i == 0:
            overlapped.append(chunk)
        else:
            tail = base[i - 1][-overlap:]
            overlapped.append((tail + chunk).strip())
    return [c.strip() for c in overlapped if c.strip()]
