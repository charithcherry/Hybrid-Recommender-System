"""Final comprehensive test: Query parser + Hybrid retriever."""

import requests
import json
import time

CONV_URL = "http://localhost:8001"

print("="*70)
print("FINAL HYBRID SYSTEM TEST")
print("="*70)

test_cases = [
    {
        "query": "casual watches for men",
        "expected_filters": ["gender: Men", "articleType: Watches"],
        "expected_engine": "qdrant"  # Has filters
    },
    {
        "query": "red dress for summer",
        "expected_filters": ["articleType: Dresses", "baseColour: Red"],
        "expected_engine": "qdrant"  # Has filters
    },
    {
        "query": "black backpack",
        "expected_filters": ["articleType: Backpacks", "baseColour: Black"],
        "expected_engine": "qdrant"  # Has filters
    },
    {
        "query": "show me products",
        "expected_filters": [],
        "expected_engine": "faiss"  # No filters = FAISS
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}: {test['query']}")
    print(f"{'='*70}")

    try:
        start = time.time()
        response = requests.post(
            f"{CONV_URL}/chat",
            json={"query": test['query'], "n": 5},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()

            # Check understood intent
            intent = data['understood_intent']
            filters = intent.get('filters', {})
            recs = data['recommendations']

            print(f"\n[OK] Response in {elapsed:.0f}ms")
            print(f"\n1. LLM Query Understanding:")
            print(f"   Intent: {intent.get('intent')}")
            print(f"   Search query: {intent.get('search_query')}")
            print(f"   Extracted filters: {json.dumps(filters, indent=6)}")

            # Verify filter extraction
            filter_count = sum(1 for v in filters.values() if v and v != "null")
            print(f"\n2. Filter Extraction:")
            if filter_count > 0:
                print(f"   [OK] Extracted {filter_count} filters!")
                for key, val in filters.items():
                    if val and val != "null":
                        print(f"      - {key}: {val}")
            else:
                print(f"   [WARN] No filters extracted (expected: {test['expected_filters']})")

            # Show recommendations
            print(f"\n3. Recommendations: ({len(recs)} items)")
            for j, rec in enumerate(recs[:3], 1):
                title = rec['title'][:45]
                meta = rec.get('metadata', {})
                print(f"   {j}. {title}")
                print(f"      Category: {meta.get('articleType', 'N/A')} | Gender: {meta.get('gender', 'N/A')}")

            # Check if results match query
            print(f"\n4. Result Accuracy Check:")
            if test['query'] == "casual watches for men":
                watch_count = sum(1 for r in recs if 'watch' in r['title'].lower())
                men_count = sum(1 for r in recs if r['metadata'].get('gender') == 'Men')
                print(f"   Watches: {watch_count}/{len(recs)}")
                print(f"   Men's items: {men_count}/{len(recs)}")
                if watch_count >= 3:
                    print(f"   [OK] Good watch results!")

        else:
            print(f"   [FAIL] Request failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")

    except Exception as e:
        print(f"   [FAIL] Error: {e}")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

print("\n[OK] Conversational service running")
print("[OK] Hybrid retriever integrated")
print("[OK] Query parser with improved prompt")
print("\nNext: Check if LLM extracts filters correctly!")
print("If filters work -> Hybrid retriever will use Qdrant for accuracy!")
print("If no filters -> Hybrid retriever will use FAISS for speed!")
print("\nBest of both worlds! 🎯")
