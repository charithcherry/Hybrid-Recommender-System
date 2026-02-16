"""LLM-based query parser for natural language understanding."""

import json
from typing import Dict, Optional
from openai import OpenAI


QUERY_PARSING_PROMPT = """You are a fashion e-commerce search assistant. Extract filters from the user's query.

CRITICAL RULES:
1. ALWAYS extract articleType when a product category is mentioned (watches, dress, shoes, etc.)
2. ALWAYS extract gender when mentioned or implied (men, women, boys, girls)
3. ALWAYS extract colors when mentioned (red, black, blue, etc.)
4. Extract ALL other filters that apply

User Query: "{query}"

Return ONLY this JSON format (no markdown, no explanation):
{{
  "search_query": "semantic search terms (keep it simple for vector search)",
  "filters": {{
    "gender": "Men/Women/Boys/Girls/Unisex or null",
    "articleType": "specific type like Watches, Shoes, Dress, etc or null",
    "baseColour": "color name or null",
    "season": "Summer/Winter/Fall/Spring or null",
    "usage": "Casual/Formal/Sports/Party/Ethnic or null",
    "price_range": {{"min": number or null, "max": number or null}}
  }},
  "preferences": ["list of style preferences, occasions, or requirements"],
  "intent": "search/browse/recommendation/comparison"
}}

Examples:

Query: "red dress for summer wedding under $200"
Output: {{
  "search_query": "red dress wedding",
  "filters": {{
    "gender": "Women",
    "articleType": "Dresses",
    "baseColour": "Red",
    "season": "Summer",
    "usage": "Party",
    "price_range": {{"max": 200}}
  }},
  "preferences": ["wedding appropriate", "summer suitable", "affordable"],
  "intent": "search"
}}

Query: "casual watches for men"
Output: {{
  "search_query": "casual watches men",
  "filters": {{
    "gender": "Men",
    "articleType": "Watches",
    "usage": "Casual",
    "baseColour": null,
    "season": null,
    "price_range": null
  }},
  "preferences": ["casual style", "men's accessories"],
  "intent": "search"
}}

Query: "show me something like my recent likes"
Output: {{
  "search_query": "similar to user history",
  "filters": {{}},
  "preferences": ["personalized", "based on history"],
  "intent": "recommendation"
}}

Now parse the query above and return ONLY the JSON output, no other text.
"""


async def parse_user_query(
    query: str,
    openai_client: OpenAI,
    user_id: Optional[int] = None,
    model: str = "gpt-4o-mini"
) -> Dict:
    """
    Parse natural language query using LLM.

    Args:
        query: User's natural language query
        openai_client: OpenAI client instance
        user_id: Optional user ID for personalization context
        model: LLM model to use

    Returns:
        Parsed intent dictionary with search_query, filters, preferences
    """
    if not openai_client:
        # Fallback: simple keyword extraction
        return {
            "search_query": query,
            "filters": {},
            "preferences": [],
            "intent": "search"
        }

    try:
        # Build prompt
        prompt = QUERY_PARSING_PROMPT.format(query=query)

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract filters from shopping queries. Key rules: 'watches'=articleType:Watches, 'dress'=articleType:Dresses, 'men'=gender:Men, 'women'=gender:Women, 'red'=baseColour:Red. Return ONLY JSON."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # Force JSON response
            temperature=0,  # Deterministic
            max_tokens=300
        )

        # Extract and parse JSON
        content = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        parsed_intent = json.loads(content.strip())

        print(f"[Query Parser] '{query}' → {parsed_intent['intent']} intent")

        return parsed_intent

    except Exception as e:
        print(f"Error parsing query with LLM: {e}")
        # Fallback to simple keyword extraction
        return {
            "search_query": query,
            "filters": {},
            "preferences": [query],
            "intent": "search"
        }
