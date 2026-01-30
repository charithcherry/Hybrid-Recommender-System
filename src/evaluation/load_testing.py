"""Load testing and performance evaluation for recommender API."""

import sys
from pathlib import Path
import requests
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


class LoadTester:
    """Load testing for recommender system API."""

    def __init__(self, base_url: str = "http://localhost:8000", config=None):
        """Initialize load tester.

        Args:
            base_url: Base URL of the API
            config: Configuration object
        """
        self.base_url = base_url
        self.config = config or get_config()

        self.results = []

    def check_health(self) -> bool:
        """Check if API is healthy.

        Returns:
            True if API is healthy
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def make_request(self, user_id: int, n: int = 10) -> Dict:
        """Make a recommendation request.

        Args:
            user_id: User ID
            n: Number of recommendations

        Returns:
            Dictionary with request results
        """
        start_time = time.time()

        try:
            response = requests.get(
                f"{self.base_url}/recommend/{user_id}",
                params={"n": n},
                timeout=30
            )

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms

            result = {
                'user_id': user_id,
                'status_code': response.status_code,
                'latency_ms': latency,
                'success': response.status_code == 200,
                'timestamp': start_time
            }

            if response.status_code == 200:
                data = response.json()
                result['server_time_ms'] = data.get('retrieval_time_ms', 0)
                result['n_recommendations'] = len(data.get('recommendations', []))

            return result

        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000

            return {
                'user_id': user_id,
                'status_code': 0,
                'latency_ms': latency,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            }

    def sequential_load_test(self, user_ids: List[int], n: int = 10) -> List[Dict]:
        """Run sequential load test.

        Args:
            user_ids: List of user IDs to test
            n: Number of recommendations per request

        Returns:
            List of request results
        """
        print(f"Running sequential load test with {len(user_ids)} requests...")

        results = []
        for user_id in tqdm(user_ids, desc="Sequential requests"):
            result = self.make_request(user_id, n)
            results.append(result)
            time.sleep(0.01)  # Small delay between requests

        return results

    def concurrent_load_test(self, user_ids: List[int],
                            n: int = 10,
                            max_workers: int = 10) -> List[Dict]:
        """Run concurrent load test.

        Args:
            user_ids: List of user IDs to test
            n: Number of recommendations per request
            max_workers: Maximum number of concurrent workers

        Returns:
            List of request results
        """
        print(f"Running concurrent load test with {len(user_ids)} requests "
              f"and {max_workers} workers...")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_user = {
                executor.submit(self.make_request, user_id, n): user_id
                for user_id in user_ids
            }

            # Collect results
            for future in tqdm(as_completed(future_to_user),
                             total=len(user_ids),
                             desc="Concurrent requests"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Request failed: {e}")

        return results

    def ramp_up_load_test(self, user_ids: List[int],
                         n: int = 10,
                         ramp_up_seconds: int = 10,
                         max_workers: int = 20) -> List[Dict]:
        """Run ramp-up load test.

        Gradually increases load over time.

        Args:
            user_ids: List of user IDs to test
            n: Number of recommendations per request
            ramp_up_seconds: Time to ramp up to max load
            max_workers: Maximum number of concurrent workers

        Returns:
            List of request results
        """
        print(f"Running ramp-up load test over {ramp_up_seconds} seconds...")

        results = []
        requests_per_worker = len(user_ids) // max_workers

        start_time = time.time()

        for i in range(max_workers):
            # Calculate delay for ramp-up
            delay = (i / max_workers) * ramp_up_seconds
            time.sleep(delay - (time.time() - start_time))

            # Start batch of requests
            batch_start = i * requests_per_worker
            batch_end = (i + 1) * requests_per_worker if i < max_workers - 1 else len(user_ids)
            batch_users = user_ids[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(self.make_request, uid, n) for uid in batch_users]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Request failed: {e}")

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze load test results.

        Args:
            results: List of request results

        Returns:
            Dictionary with analysis metrics
        """
        df = pd.DataFrame(results)

        # Filter successful requests
        successful = df[df['success'] == True]

        if len(successful) == 0:
            return {
                'error': 'No successful requests',
                'total_requests': len(df),
                'success_rate': 0.0
            }

        analysis = {
            'total_requests': len(df),
            'successful_requests': len(successful),
            'failed_requests': len(df) - len(successful),
            'success_rate': len(successful) / len(df),

            # Latency metrics (end-to-end)
            'latency_mean_ms': successful['latency_ms'].mean(),
            'latency_median_ms': successful['latency_ms'].median(),
            'latency_std_ms': successful['latency_ms'].std(),
            'latency_p50_ms': successful['latency_ms'].quantile(0.50),
            'latency_p95_ms': successful['latency_ms'].quantile(0.95),
            'latency_p99_ms': successful['latency_ms'].quantile(0.99),
            'latency_min_ms': successful['latency_ms'].min(),
            'latency_max_ms': successful['latency_ms'].max(),

            # Throughput
            'duration_seconds': df['timestamp'].max() - df['timestamp'].min(),
        }

        analysis['throughput_rps'] = analysis['total_requests'] / max(analysis['duration_seconds'], 1)

        # Server-side timing (if available)
        if 'server_time_ms' in successful.columns:
            analysis['server_time_mean_ms'] = successful['server_time_ms'].mean()
            analysis['server_time_p95_ms'] = successful['server_time_ms'].quantile(0.95)

        return analysis

    def print_analysis(self, analysis: Dict, title: str = "Load Test Results"):
        """Print analysis results.

        Args:
            analysis: Analysis dictionary
            title: Title for the report
        """
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        if 'error' in analysis:
            print(f"\nError: {analysis['error']}")
            return

        print(f"\nRequest Statistics:")
        print(f"  Total Requests: {analysis['total_requests']}")
        print(f"  Successful: {analysis['successful_requests']}")
        print(f"  Failed: {analysis['failed_requests']}")
        print(f"  Success Rate: {analysis['success_rate']*100:.2f}%")

        print(f"\nLatency (End-to-End):")
        print(f"  Mean: {analysis['latency_mean_ms']:.2f} ms")
        print(f"  Median: {analysis['latency_median_ms']:.2f} ms")
        print(f"  P95: {analysis['latency_p95_ms']:.2f} ms")
        print(f"  P99: {analysis['latency_p99_ms']:.2f} ms")
        print(f"  Min: {analysis['latency_min_ms']:.2f} ms")
        print(f"  Max: {analysis['latency_max_ms']:.2f} ms")

        if 'server_time_mean_ms' in analysis:
            print(f"\nServer-Side Timing:")
            print(f"  Mean: {analysis['server_time_mean_ms']:.2f} ms")
            print(f"  P95: {analysis['server_time_p95_ms']:.2f} ms")

        print(f"\nThroughput:")
        print(f"  Duration: {analysis['duration_seconds']:.2f} seconds")
        print(f"  Throughput: {analysis['throughput_rps']:.2f} requests/second")

        print("=" * 60)

    def plot_results(self, results: List[Dict], output_path: str = "experiments/load_test_results.png"):
        """Plot load test results.

        Args:
            results: List of request results
            output_path: Path to save plot
        """
        df = pd.DataFrame(results)
        successful = df[df['success'] == True]

        if len(successful) == 0:
            print("No successful requests to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Latency distribution
        axes[0, 0].hist(successful['latency_ms'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Latency (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Latency Distribution')
        axes[0, 0].axvline(successful['latency_ms'].median(), color='r',
                          linestyle='--', label='Median')
        axes[0, 0].axvline(successful['latency_ms'].quantile(0.95), color='g',
                          linestyle='--', label='P95')
        axes[0, 0].legend()

        # Latency over time
        successful_sorted = successful.sort_values('timestamp')
        axes[0, 1].plot(range(len(successful_sorted)), successful_sorted['latency_ms'])
        axes[0, 1].set_xlabel('Request Number')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Latency Over Time')

        # Box plot
        axes[1, 0].boxplot(successful['latency_ms'])
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].set_title('Latency Box Plot')

        # CDF
        sorted_latencies = np.sort(successful['latency_ms'])
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        axes[1, 1].plot(sorted_latencies, cdf)
        axes[1, 1].set_xlabel('Latency (ms)')
        axes[1, 1].set_ylabel('CDF')
        axes[1, 1].set_title('Cumulative Distribution Function')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")


def run_comprehensive_load_test():
    """Run comprehensive load testing suite."""
    print("=" * 60)
    print("Comprehensive Load Testing Suite")
    print("=" * 60)

    tester = LoadTester()

    # Check health
    if not tester.check_health():
        print("\nERROR: API is not responding. Please start the server first.")
        print("Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        return

    print("\nAPI is healthy!")

    # Load test data
    try:
        train_df = pd.read_csv("data/processed/interactions_train.csv")
        test_users = train_df['user_id'].unique()[:100]  # Use 100 users for testing
    except:
        print("Using synthetic user IDs")
        test_users = list(range(100))

    # Test 1: Sequential load test
    print("\n" + "-" * 60)
    print("Test 1: Sequential Load Test")
    print("-" * 60)

    seq_results = tester.sequential_load_test(test_users[:20], n=10)
    seq_analysis = tester.analyze_results(seq_results)
    tester.print_analysis(seq_analysis, "Sequential Load Test")

    # Test 2: Concurrent load test
    print("\n" + "-" * 60)
    print("Test 2: Concurrent Load Test")
    print("-" * 60)

    conc_results = tester.concurrent_load_test(test_users[:50], n=10, max_workers=10)
    conc_analysis = tester.analyze_results(conc_results)
    tester.print_analysis(conc_analysis, "Concurrent Load Test (10 workers)")

    # Test 3: High concurrency test
    print("\n" + "-" * 60)
    print("Test 3: High Concurrency Load Test")
    print("-" * 60)

    high_conc_results = tester.concurrent_load_test(test_users, n=10, max_workers=20)
    high_conc_analysis = tester.analyze_results(high_conc_results)
    tester.print_analysis(high_conc_analysis, "High Concurrency Load Test (20 workers)")

    # Plot results
    all_results = seq_results + conc_results + high_conc_results
    tester.plot_results(all_results)

    # Save results
    Path("experiments").mkdir(exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("experiments/load_test_results.csv", index=False)

    print("\n" + "=" * 60)
    print("Load Testing Complete!")
    print("=" * 60)
    print("\nResults saved to:")
    print("  - experiments/load_test_results.csv")
    print("  - experiments/load_test_results.png")


if __name__ == "__main__":
    run_comprehensive_load_test()
