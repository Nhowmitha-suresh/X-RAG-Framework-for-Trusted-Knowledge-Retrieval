from typing import List, Optional


def get_sources(chunks: List[str], distances: Optional[List[float]] = None):
    """Return a list of source metadata dicts including a short snippet and similarity metric.

    `distances` corresponds to L2 distances for each chunk; smaller is better.
    We expose a simple similarity = 1/(1+distance) when distances are provided.
    """
    sources = []
    for i, c in enumerate(chunks):
        snippet = c[:300].strip()
        entry = {"index": i, "source_text": snippet}
        if distances is not None and len(distances) > i:
            d = float(distances[i])
            sim = 1.0 / (1.0 + d)
            entry["similarity"] = float(sim)
            entry["distance"] = d
        sources.append(entry)

    return sources
