"""LLM-based explanation generator for recommendations."""

from typing import List, Dict, Optional
from openai import OpenAI


EXPLANATION_PROMPT = """You are a helpful fashion shopping assistant. Explain why these products match the user's request.

User Request: "{query}"
Understood Intent: {intent}

Recommended Products:
{products}

Generate a brief, friendly explanation (2-3 sentences) for why these products are good matches. Be specific about how they match the filters and preferences.

Keep it natural and conversational, like a personal shopper would explain.
"""


INDIVIDUAL_EXPLANATION_PROMPT = """Briefly explain (1 sentence) why this product matches the user's request: "{query}"

Product: {product_name}
Category: {category}
Filters matched: {matched_filters}

Explanation:"""


async def generate_explanations(
    products: List[Dict],
    query: str,
    intent: Dict,
    openai_client: OpenAI,
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Generate explanations for why each product was recommended.

    Args:
        products: List of product dictionaries
        query: Original user query
        intent: Parsed intent from query_parser
        openai_client: OpenAI client
        model: LLM model to use

    Returns:
        Products with explanation field added
    """
    if not openai_client or len(products) == 0:
        return products

    try:
        # For cost efficiency, generate ONE overall explanation
        # Instead of individual explanations per product (would use 10x more tokens)

        # Format products for prompt
        product_list = "\n".join([
            f"{i+1}. {p['title']} ({p['metadata'].get('articleType', 'Unknown')})"
            for i, p in enumerate(products[:5])  # Show top 5
        ])

        prompt = EXPLANATION_PROMPT.format(
            query=query,
            intent=intent,
            products=product_list
        )

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful fashion shopping assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150  # Keep it brief to save costs
        )

        overall_explanation = response.choices[0].message.content.strip()

        # Add simple explanation to each product (cost-efficient)
        for product in products:
            metadata = product.get('metadata', {})
            matched = []

            if intent.get('filters', {}).get('articleType') == metadata.get('articleType'):
                matched.append(f"matches '{metadata.get('articleType')}'")
            if intent.get('filters', {}).get('baseColour') == metadata.get('baseColour'):
                matched.append(f"{metadata.get('baseColour')} color")
            if intent.get('filters', {}).get('gender') == metadata.get('gender'):
                matched.append(f"for {metadata.get('gender')}")

            if matched:
                product['explanation'] = f"Great match: {', '.join(matched)}"
            else:
                product['explanation'] = "Recommended based on your query"

        print(f"[Explanation Generator] Generated explanations for {len(products)} products")
        print(f"  Overall: {overall_explanation[:100]}...")

        return products

    except Exception as e:
        print(f"Error generating explanations: {e}")
        # Return products without explanations
        return products


def generate_simple_explanation(product: Dict, filters: Dict) -> str:
    """Generate simple rule-based explanation (no LLM needed - free!)."""
    metadata = product.get('metadata', {})
    matched = []

    if filters.get('articleType') and metadata.get('articleType') == filters['articleType']:
        matched.append(f"{metadata['articleType']}")

    if filters.get('baseColour') and metadata.get('baseColour') == filters['baseColour']:
        matched.append(f"{metadata['baseColour']} color")

    if filters.get('gender') and metadata.get('gender') == filters['gender']:
        matched.append(f"for {metadata['gender']}")

    if filters.get('usage') and metadata.get('usage') == filters['usage']:
        matched.append(f"{metadata['usage']} style")

    if matched:
        return f"Matches: {', '.join(matched)}"
    else:
        return "Similar to your preferences"
