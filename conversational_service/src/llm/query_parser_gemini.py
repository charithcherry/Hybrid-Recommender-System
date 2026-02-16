"""Gemini-based query parser (FREE tier!)."""

import json
import os
from typing import Dict, Optional
from google import genai


QUERY_PARSING_PROMPT = """Extract filters from this fashion shopping query.

Query: "{query}"

Return a JSON object with these fields:
- search_query: simplified search terms
- filters: {{gender, articleType, baseColour, season, usage}}
- intent: search/browse/recommendation

Examples:

Query: "casual watches for men"
Output: {{"search_query": "casual watches", "filters": {{"gender": "Men", "articleType": "Watches", "usage": "Casual"}}, "intent": "search"}}

Query: "red summer dress"
Output: {{"search_query": "red dress", "filters": {{"articleType": "Dresses", "baseColour": "Red", "season": "Summer"}}, "intent": "search"}}

Query: "black backpack"
Output: {{"search_query": "black backpack", "filters": {{"articleType": "Backpacks", "baseColour": "Black"}}, "intent": "search"}}

Now parse: "{query}"

Return ONLY valid JSON, no markdown, no explanation."""


async def parse_with_gemini(
    query: str,
    api_key: str,
    model: str = "gemini-2.5-flash"
) -> Dict:
    """
    Parse query using Gemini (FREE tier!).

    Args:
        query: User's natural language query
        api_key: Gemini API key
        model: Gemini model to use

    Returns:
        Parsed intent dictionary
    """
    try:
        # Create Gemini client
        client = genai.Client(api_key=api_key)

        # Build prompt
        prompt = QUERY_PARSING_PROMPT.format(query=query)

        # Generate with new API (force JSON output!)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": 300,
                "response_mime_type": "application/json"  # KEY: Force JSON!
            }
        )

        content = response.text.strip()

        print(f"[Gemini] Raw response: {content[:100]}...")

        # Clean up response (remove markdown if present)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        # Parse JSON
        parsed = json.loads(content.strip())

        print(f"[Gemini] Parsed filters: {parsed.get('filters', {})}")

        return parsed

    except Exception as e:
        print(f"[Gemini] Error parsing: {e}")
        # Fallback
        return {
            "search_query": query,
            "filters": {},
            "intent": "search"
        }


async def parse_user_query(
    query: str,
    gemini_api_key: Optional[str] = None,
    openai_client = None,
    user_id: Optional[int] = None,
    provider: str = "gemini"
) -> Dict:
    """
    Parse query using configured LLM provider.

    Args:
        query: User's query
        gemini_api_key: Gemini API key
        openai_client: OpenAI client (fallback)
        user_id: Optional user ID
        provider: "gemini" or "openai"

    Returns:
        Parsed intent dictionary
    """
    # Use Gemini if available
    if provider == "gemini" and gemini_api_key:
        model = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
        return await parse_with_gemini(query, gemini_api_key, model)

    # Fallback to OpenAI
    if openai_client:
        print("[Fallback] Using OpenAI...")
        # Import original OpenAI parser
        from src.llm.query_parser import parse_user_query as parse_openai
        return await parse_openai(query, openai_client, user_id)

    # No LLM available - return basic
    return {
        "search_query": query,
        "filters": {},
        "intent": "search"
    }
