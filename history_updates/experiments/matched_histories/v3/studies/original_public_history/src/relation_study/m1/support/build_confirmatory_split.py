"""Stable row and split digests used by development-only construction."""
import hashlib

def row_id(situation: str, text: str, valence: str, vrd: str) -> str:
    """Content-addressed, stable across reloads and dataset re-downloads."""
    h = hashlib.sha256("\x1f".join([situation, text, valence, vrd]).encode("utf-8"))
    return h.hexdigest()[:16]


def manifest_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()

