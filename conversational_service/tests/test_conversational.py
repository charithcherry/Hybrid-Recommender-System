"""Test script for conversational recommendation service."""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_conversational_service():
    """Test the conversational recommendation service."""
    print("="*70)
    print("CONVERSATIONAL RECOMMENDATION SERVICE TEST")
    print("="*70)

    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   [OK] Service healthy")
            print(f"   -> OpenAI configured: {data['openai_configured']}")
            print(f"   -> FAISS loaded: {data['faiss_loaded']}")
        else:
            print(f"   [FAIL] Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   [FAIL] Service not running: {e}")
        print(f"\n   Start service with: uvicorn src.api.main:app --port 8001")
        return

    # Test 2: Simple conversational query
    print("\n2. Testing conversational query...")
    test_queries = [
        {
            "query": "casual watches for men",
            "description": "Simple category search"
        },
        {
            "query": "red dress for summer",
            "description": "Multi-filter query"
        },
        {
            "query": "show me backpacks",
            "description": "Basic product search"
        }
    ]

    for test in test_queries:
        print(f"\n   Query: '{test['query']}'")
        print(f"   Type: {test['description']}")

        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"query": test['query'], "n": 5},
                timeout=30
            )
            elapsed = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                print(f"   [OK] Response in {elapsed:.0f}ms")
                print(f"   -> Intent: {data['understood_intent']['intent']}")
                print(f"   -> Recommendations: {len(data['recommendations'])}")
                print(f"   -> Explanation: {data['explanation'][:80]}...")

                # Show sample recommendations
                for i, rec in enumerate(data['recommendations'][:2], 1):
                    print(f"      {i}. {rec['title'][:50]}")
                    if rec.get('explanation'):
                        print(f"         Why: {rec['explanation'][:60]}")
            else:
                print(f"   [FAIL] Query failed: {response.status_code}")
                print(f"   Error: {response.text[:200]}")

        except Exception as e:
            print(f"   [FAIL] Error: {e}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_conversational_service()
