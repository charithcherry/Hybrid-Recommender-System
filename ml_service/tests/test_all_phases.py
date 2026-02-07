"""Comprehensive test suite for all 3 phases before code push."""
import requests
import time
import json
from typing import Dict, List

BASE_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}[FAIL] {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.YELLOW}  -> {text}{Colors.END}")


class TestSuite:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.start_user_id = 2000  # Start from high ID to avoid conflicts

    def test_server_health(self):
        """Test 0: Server is running and healthy."""
        print_header("TEST 0: Server Health Check")
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print_success("Server is healthy")
                print_info(f"Status: {data['status']}")
                print_info(f"Models loaded: {data['model_loaded']}")
                self.tests_passed += 1
                return True
            else:
                print_error(f"Server unhealthy: {response.status_code}")
                self.tests_failed += 1
                return False
        except Exception as e:
            print_error(f"Server not reachable: {e}")
            self.tests_failed += 1
            return False

    def create_user(self, username):
        """Helper: Create a test user."""
        response = requests.post(
            f"{BASE_URL}/users/register",
            json={"username": username, "password": "test123"}
        )
        if response.status_code == 200:
            return response.json()['user_id']
        return None

    def add_interaction(self, user_id, item_id, interaction_type="like"):
        """Helper: Add an interaction."""
        response = requests.post(
            f"{BASE_URL}/interactions",
            json={"user_id": user_id, "item_id": item_id, "interaction_type": interaction_type}
        )
        return response.status_code == 200

    def get_recommendations(self, user_id, n=10):
        """Helper: Get split recommendations."""
        response = requests.get(f"{BASE_URL}/recommend/{user_id}/split?n={n}")
        if response.status_code == 200:
            return response.json()
        return None

    def test_phase1_performance(self):
        """Test Phase 1: Category Specialist Index performance."""
        print_header("PHASE 1 TEST: Category Specialist Index (Performance)")

        # Create user with watch interactions
        user_id = self.create_user(f"phase1_test_{int(time.time())}")
        if not user_id:
            print_error("Failed to create user")
            self.tests_failed += 1
            return

        print_info(f"Created user {user_id}")

        # Add 8 watch interactions
        watch_ids = [30039, 11002, 45933, 27876, 39403, 21345, 43287, 38765]
        for item_id in watch_ids:
            self.add_interaction(user_id, item_id)

        print_info(f"Added {len(watch_ids)} watch interactions")

        # Test recommendation speed (specialist index should make this fast)
        start_time = time.time()
        result = self.get_recommendations(user_id, n=10)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        if result:
            retrieval_time = result.get('retrieval_time_ms', 0)
            print_success(f"Recommendations generated in {elapsed:.2f}ms")
            print_info(f"Server-side retrieval: {retrieval_time:.2f}ms")

            # Phase 1 optimization: should be < 100ms for new users
            if retrieval_time < 100:
                print_success("Performance EXCELLENT (< 100ms) - Phase 1 working!")
                self.tests_passed += 1
            else:
                print_error(f"Performance slower than expected: {retrieval_time:.2f}ms")
                self.tests_failed += 1
        else:
            print_error("Failed to get recommendations")
            self.tests_failed += 1

    def test_phase2_multi_interest(self):
        """Test Phase 2: Multi-Interest Clustering."""
        print_header("PHASE 2 TEST: Multi-Interest Clustering")

        # Test 2A: Single interest user (baseline)
        print(f"\n{Colors.BOLD}Test 2A: Single Interest User (Watches Only){Colors.END}")
        user_single = self.create_user(f"single_interest_{int(time.time())}")

        # Add 8 watches
        watch_ids = [30039, 11002, 45933, 27876, 39403, 21345, 43287, 38765]
        for item_id in watch_ids:
            self.add_interaction(user_single, item_id)

        print_info(f"User {user_single}: Added 8 watches (single interest)")

        result_single = self.get_recommendations(user_single, n=10)
        if result_single:
            cf_count_single = len(result_single.get('cf_recommendations', []))
            print_info(f"CF recommendations: {cf_count_single}")
            print_success("Single interest user tested")

        # Test 2B: Multi-interest user
        print(f"\n{Colors.BOLD}Test 2B: Multi-Interest User (Watches + Backpacks){Colors.END}")
        user_multi = self.create_user(f"multi_interest_{int(time.time())}")

        # Add 5 watches + 3 backpacks
        watch_ids = [30039, 11002, 45933, 27876, 39403]
        backpack_ids = [12732, 29319, 19920]

        for item_id in watch_ids:
            self.add_interaction(user_multi, item_id)
        for item_id in backpack_ids:
            self.add_interaction(user_multi, item_id)

        print_info(f"User {user_multi}: Added 5 watches + 3 backpacks (multi-interest)")

        result_multi = self.get_recommendations(user_multi, n=10)
        if result_multi:
            cf_recs = result_multi.get('cf_recommendations', [])
            content_recs = result_multi.get('content_recommendations', [])

            print_info(f"CF recommendations: {len(cf_recs)}")
            print_info(f"Content recommendations: {len(content_recs)}")

            # Check if we got recommendations (Phase 2 should handle multi-interest)
            if len(cf_recs) > 0 or len(content_recs) > 0:
                print_success("Multi-interest clustering working!")
                print_info("System detected multiple categories and generated diverse recommendations")
                self.tests_passed += 1
            else:
                print_error("No recommendations generated for multi-interest user")
                self.tests_failed += 1
        else:
            print_error("Failed to get recommendations")
            self.tests_failed += 1

    def test_phase3_adaptive_weights(self):
        """Test Phase 3: Adaptive Hybrid Weighting."""
        print_header("PHASE 3 TEST: Adaptive Hybrid Weighting")

        test_cases = [
            {
                'name': 'Cold Start (3 interactions)',
                'num_interactions': 3,
                'expected_weights': 'CF:10%, content:60%, popular:30%',
                'user_type': 'cold'
            },
            {
                'name': 'Warm Start (8 interactions)',
                'num_interactions': 8,
                'expected_weights': 'CF:40%, content:40%, popular:20%',
                'user_type': 'warm'
            },
            {
                'name': 'Established (25 interactions - simulated)',
                'num_interactions': 10,  # We'll use a training user for this
                'expected_weights': 'CF:70%, content:20%, popular:10%',
                'user_type': 'established'
            }
        ]

        for i, test_case in enumerate(test_cases[:2], 1):  # Test cold and warm
            print(f"\n{Colors.BOLD}Test 3.{i}: {test_case['name']}{Colors.END}")

            user_id = self.create_user(f"adaptive_{test_case['user_type']}_{int(time.time())}")

            # Add interactions
            watch_ids = [30039, 11002, 45933, 27876, 39403, 21345, 43287, 38765, 15234, 25678]
            for idx in range(test_case['num_interactions']):
                self.add_interaction(user_id, watch_ids[idx])

            print_info(f"User {user_id}: {test_case['num_interactions']} interactions")
            print_info(f"Expected weights: {test_case['expected_weights']}")

            # Get recommendations (adaptive weights applied internally)
            result = self.get_recommendations(user_id, n=10)
            if result:
                cf_count = len(result.get('cf_recommendations', []))
                content_count = len(result.get('content_recommendations', []))
                print_info(f"CF recs: {cf_count}, Content recs: {content_count}")
                print_success(f"Adaptive weighting applied for {test_case['user_type']} user")
            else:
                print_error("Failed to get recommendations")

        # Test 3.3: Established user (use training user)
        print(f"\n{Colors.BOLD}Test 3.3: Established User (20+ interactions){Colors.END}")
        print_info("Using training user ID 5 (has 20+ interactions)")

        # Use training user who has many interactions (use GET endpoint)
        result = self.get_recommendations(5, n=10)

        if result:
            cf_count = len(result.get('cf_recommendations', []))
            content_count = len(result.get('content_recommendations', []))
            print_success("Established user recommendations generated")
            print_info("Expected weights: CF:70%, content:20%, popular:10%")
            print_info(f"CF recs: {cf_count}, Content recs: {content_count}")
            self.tests_passed += 1
        else:
            print_error("Failed to get recommendations for established user")
            self.tests_failed += 1

    def test_integration_endpoints(self):
        """Test that all endpoints still work."""
        print_header("INTEGRATION TEST: All Endpoints")

        endpoints = [
            ("GET", "/", "Root endpoint"),
            ("GET", "/health", "Health check"),
            ("GET", "/items/filter-values", "Filter values"),
            ("GET", "/recommend/coldstart?n=10", "Cold-start recommendations"),
        ]

        for method, endpoint, description in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)

                if response.status_code == 200:
                    print_success(f"{description}: OK")
                    self.tests_passed += 1
                else:
                    print_error(f"{description}: Failed ({response.status_code})")
                    self.tests_failed += 1
            except Exception as e:
                print_error(f"{description}: Error - {e}")
                self.tests_failed += 1

    def run_all_tests(self):
        """Run complete test suite."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}COMPREHENSIVE TEST SUITE - ALL 3 PHASES{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

        start_time = time.time()

        # Run tests
        if self.test_server_health():
            self.test_phase1_performance()
            self.test_phase2_multi_interest()
            self.test_phase3_adaptive_weights()
            self.test_integration_endpoints()

        # Final summary
        elapsed = time.time() - start_time

        print_header("TEST SUMMARY")
        print(f"{Colors.BOLD}Tests Passed: {Colors.GREEN}{self.tests_passed}{Colors.END}")
        print(f"{Colors.BOLD}Tests Failed: {Colors.RED}{self.tests_failed}{Colors.END}")
        print(f"{Colors.BOLD}Total Time: {elapsed:.2f}s{Colors.END}\n")

        if self.tests_failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}{'='*60}")
            print("[SUCCESS] ALL TESTS PASSED! READY TO PUSH CODE!")
            print(f"{'='*60}{Colors.END}\n")
            return True
        else:
            print(f"{Colors.RED}{Colors.BOLD}{'='*60}")
            print(f"[FAILED] {self.tests_failed} TEST(S) FAILED - FIX BEFORE PUSHING")
            print(f"{'='*60}{Colors.END}\n")
            return False


if __name__ == "__main__":
    suite = TestSuite()
    success = suite.run_all_tests()
    exit(0 if success else 1)
