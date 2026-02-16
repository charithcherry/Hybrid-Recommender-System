"""Debug LLM query parsing."""

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Initialize OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Test query
test_query = "casual watches for men"

# Simple direct prompt
prompt = f"""Extract filters from this shopping query and return JSON:

Query: "{test_query}"

Return this exact JSON structure:
{{
  "articleType": "Watches",
  "gender": "Men",
  "usage": "Casual"
}}

Return ONLY the JSON, nothing else."""

print("Testing LLM filter extraction...")
print(f"Query: '{test_query}'")
print("\nCalling OpenAI API...")

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract shopping filters and return JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    content = response.choices[0].message.content
    print(f"\nLLM Response:")
    print(content)

    # Try to parse
    import json
    parsed = json.loads(content)
    print(f"\nParsed JSON:")
    print(json.dumps(parsed, indent=2))

    print("\n[OK] LLM is working and returning JSON!")

except Exception as e:
    print(f"\n[FAIL] Error: {e}")
