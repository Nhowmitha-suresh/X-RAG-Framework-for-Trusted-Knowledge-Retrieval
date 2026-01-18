from explainability.source_attribution import get_sources
from explainability.confidence_score import calculate_confidence


def explain(chunks, distances):
    """Return explainability artifacts for a set of retrieved chunks.

    - sources: list of dicts with snippet, index, distance and similarity
    - similarities: list of similarity floats
    - confidence: aggregated confidence score
    """
    sources = get_sources(chunks, distances)
    similarities = [s.get("similarity") for s in sources]
    confidence = calculate_confidence(distances)

    return {
        "sources": sources,
        "similarities": similarities,
        "confidence": confidence,
    }
