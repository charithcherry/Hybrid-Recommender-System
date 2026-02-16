"""Filter engine for applying LLM-parsed filters to candidates."""

from typing import List, Dict


def apply_llm_filters(
    candidates: List[Dict],
    filters: Dict,
    top_k: int = 10
) -> List[Dict]:
    """
    Apply LLM-extracted filters to candidate products.

    Args:
        candidates: List of product dictionaries
        filters: Filters extracted from LLM (gender, category, color, price, etc.)
        top_k: Number of results to return

    Returns:
        Filtered and ranked candidates
    """
    filtered = []

    for candidate in candidates:
        metadata = candidate.get('metadata', {})

        # Apply filters
        if filters.get('gender') and metadata.get('gender') != filters['gender']:
            continue

        if filters.get('articleType') and metadata.get('articleType') != filters['articleType']:
            continue

        if filters.get('baseColour') and metadata.get('baseColour') != filters['baseColour']:
            continue

        if filters.get('season') and metadata.get('season') != filters['season']:
            continue

        if filters.get('usage') and metadata.get('usage') != filters['usage']:
            continue

        # Price range filter (if we have price data)
        price_range = filters.get('price_range', {})
        if price_range:
            # TODO: Add price filtering when price data is available
            pass

        filtered.append(candidate)

    # Return top k
    return filtered[:top_k]
