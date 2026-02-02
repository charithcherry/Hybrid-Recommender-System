"""Evaluation metrics for recommender systems."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict


class RecommenderMetrics:
    """Compute evaluation metrics for recommender systems."""

    @staticmethod
    def precision_at_k(recommended_items: List[int],
                      relevant_items: List[int],
                      k: int) -> float:
        """Compute Precision@K.

        Precision@K = (# of recommended items @K that are relevant) / K

        Args:
            recommended_items: List of recommended item IDs (ranked)
            relevant_items: List of relevant (ground truth) item IDs
            k: Cutoff position

        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0

        # Get top-k recommendations
        recommended_at_k = set(recommended_items[:k])
        relevant_set = set(relevant_items)

        # Count relevant items in recommendations
        num_relevant = len(recommended_at_k & relevant_set)

        return num_relevant / k

    @staticmethod
    def recall_at_k(recommended_items: List[int],
                   relevant_items: List[int],
                   k: int) -> float:
        """Compute Recall@K.

        Recall@K = (# of recommended items @K that are relevant) / (total # of relevant items)

        Args:
            recommended_items: List of recommended item IDs (ranked)
            relevant_items: List of relevant (ground truth) item IDs
            k: Cutoff position

        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0

        # Get top-k recommendations
        recommended_at_k = set(recommended_items[:k])
        relevant_set = set(relevant_items)

        # Count relevant items in recommendations
        num_relevant = len(recommended_at_k & relevant_set)

        return num_relevant / len(relevant_items)

    @staticmethod
    def average_precision(recommended_items: List[int],
                         relevant_items: List[int],
                         k: int = None) -> float:
        """Compute Average Precision.

        Args:
            recommended_items: List of recommended item IDs (ranked)
            relevant_items: List of relevant (ground truth) item IDs
            k: Optional cutoff position

        Returns:
            Average Precision score
        """
        if len(relevant_items) == 0:
            return 0.0

        relevant_set = set(relevant_items)

        if k is not None:
            recommended_items = recommended_items[:k]

        score = 0.0
        num_hits = 0.0

        for i, item in enumerate(recommended_items):
            if item in relevant_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if num_hits == 0:
            return 0.0

        return score / min(len(relevant_items), len(recommended_items))

    @staticmethod
    def ndcg_at_k(recommended_items: List[int],
                 relevant_items: List[int],
                 k: int,
                 relevance_scores: Dict[int, float] = None) -> float:
        """Compute Normalized Discounted Cumulative Gain (NDCG@K).

        NDCG@K considers both relevance and position in ranking.

        Args:
            recommended_items: List of recommended item IDs (ranked)
            relevant_items: List of relevant (ground truth) item IDs
            k: Cutoff position
            relevance_scores: Optional dict mapping item_id to relevance score
                             If None, binary relevance is used

        Returns:
            NDCG@K score
        """
        if len(relevant_items) == 0:
            return 0.0

        # Get top-k recommendations
        recommended_at_k = recommended_items[:k]

        # Create relevance mapping
        if relevance_scores is None:
            # Binary relevance: 1 for relevant, 0 for non-relevant
            relevance_scores = {item: 1.0 for item in relevant_items}

        # Compute DCG
        dcg = 0.0
        for i, item in enumerate(recommended_at_k):
            if item in relevance_scores:
                # DCG formula: rel / log2(i + 2)
                dcg += relevance_scores[item] / np.log2(i + 2)

        # Compute IDCG (ideal DCG)
        # Sort relevant items by relevance score
        ideal_items = sorted(
            relevant_items,
            key=lambda x: relevance_scores.get(x, 0.0),
            reverse=True
        )[:k]

        idcg = 0.0
        for i, item in enumerate(ideal_items):
            idcg += relevance_scores.get(item, 0.0) / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mean_reciprocal_rank(recommended_items: List[int],
                            relevant_items: List[int]) -> float:
        """Compute Mean Reciprocal Rank (MRR).

        MRR is the reciprocal of the rank of the first relevant item.

        Args:
            recommended_items: List of recommended item IDs (ranked)
            relevant_items: List of relevant (ground truth) item IDs

        Returns:
            MRR score
        """
        relevant_set = set(relevant_items)

        for i, item in enumerate(recommended_items):
            if item in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def hit_rate_at_k(recommended_items: List[int],
                     relevant_items: List[int],
                     k: int) -> float:
        """Compute Hit Rate@K.

        Hit Rate@K = 1 if any recommended item @K is relevant, else 0.

        Args:
            recommended_items: List of recommended item IDs (ranked)
            relevant_items: List of relevant (ground truth) item IDs
            k: Cutoff position

        Returns:
            Hit rate (0 or 1)
        """
        recommended_at_k = set(recommended_items[:k])
        relevant_set = set(relevant_items)

        return 1.0 if len(recommended_at_k & relevant_set) > 0 else 0.0


class RecommenderEvaluator:
    """Evaluate recommender system on test data."""

    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """Initialize evaluator.

        Args:
            k_values: List of K values for @K metrics
        """
        self.k_values = k_values
        self.metrics = RecommenderMetrics()

    def evaluate_user(self, user_id: int,
                     recommended_items: List[int],
                     test_interactions: pd.DataFrame,
                     relevance_threshold: float = 3.0) -> Dict[str, float]:
        """Evaluate recommendations for a single user.

        Args:
            user_id: User ID
            recommended_items: List of recommended item IDs (ranked)
            test_interactions: Test interactions DataFrame
            relevance_threshold: Minimum rating to consider item relevant

        Returns:
            Dictionary of metric scores
        """
        # Get relevant items for this user
        user_test = test_interactions[test_interactions['user_id'] == user_id]

        if len(user_test) == 0:
            return {}

        # Relevant items are those with rating >= threshold
        relevant_items = user_test[user_test['rating'] >= relevance_threshold]['item_id'].tolist()

        if len(relevant_items) == 0:
            return {}

        # Create relevance scores dictionary
        relevance_scores = dict(zip(user_test['item_id'], user_test['rating']))

        # Compute metrics for each K
        results = {}

        for k in self.k_values:
            results[f'precision@{k}'] = self.metrics.precision_at_k(
                recommended_items, relevant_items, k
            )
            results[f'recall@{k}'] = self.metrics.recall_at_k(
                recommended_items, relevant_items, k
            )
            results[f'ndcg@{k}'] = self.metrics.ndcg_at_k(
                recommended_items, relevant_items, k, relevance_scores
            )
            results[f'hit_rate@{k}'] = self.metrics.hit_rate_at_k(
                recommended_items, relevant_items, k
            )

        # Compute MAP and MRR
        results['map'] = self.metrics.average_precision(
            recommended_items, relevant_items
        )
        results['mrr'] = self.metrics.mean_reciprocal_rank(
            recommended_items, relevant_items
        )

        return results

    def evaluate_model(self, recommender_func,
                      test_interactions: pd.DataFrame,
                      train_interactions: pd.DataFrame = None,
                      n_recommendations: int = 20) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluate recommender model on test set.

        Args:
            recommender_func: Function that takes (user_id, n) and returns recommendations
            test_interactions: Test interactions DataFrame
            train_interactions: Training interactions for filtering (optional)
            n_recommendations: Number of recommendations to generate

        Returns:
            Tuple of (average_metrics, per_user_metrics)
        """
        print("Evaluating model...")

        # Get unique users in test set
        test_users = test_interactions['user_id'].unique()
        print(f"Evaluating {len(test_users)} users")

        # Store per-user metrics
        all_user_metrics = []

        for user_id in test_users:
            # Generate recommendations
            try:
                recommendations = recommender_func(user_id, n_recommendations)

                # Extract item IDs
                if isinstance(recommendations[0], tuple):
                    recommended_items = [item_id for item_id, _ in recommendations]
                else:
                    recommended_items = recommendations

                # Evaluate user
                user_metrics = self.evaluate_user(
                    user_id, recommended_items, test_interactions
                )

                if user_metrics:
                    user_metrics['user_id'] = user_id
                    all_user_metrics.append(user_metrics)

            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue

        # Create DataFrame
        metrics_df = pd.DataFrame(all_user_metrics)

        # Compute average metrics
        avg_metrics = {}
        for col in metrics_df.columns:
            if col != 'user_id':
                avg_metrics[col] = metrics_df[col].mean()
                avg_metrics[f'{col}_std'] = metrics_df[col].std()

        print("\nEvaluation complete!")

        return avg_metrics, metrics_df

    def print_metrics(self, metrics: Dict[str, float], title: str = "Model Metrics"):
        """Print metrics in a formatted table.

        Args:
            metrics: Dictionary of metrics
            title: Title for the table
        """
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        # Group metrics by K
        for k in self.k_values:
            print(f"\n@{k} Metrics:")
            for metric_name in ['precision', 'recall', 'ndcg', 'hit_rate']:
                key = f'{metric_name}@{k}'
                if key in metrics:
                    value = metrics[key]
                    std_key = f'{key}_std'
                    if std_key in metrics:
                        std = metrics[std_key]
                        print(f"  {metric_name.capitalize()}@{k}: {value:.4f} ± {std:.4f}")
                    else:
                        print(f"  {metric_name.capitalize()}@{k}: {value:.4f}")

        # Print MAP and MRR
        print(f"\nOther Metrics:")
        if 'map' in metrics:
            print(f"  MAP: {metrics['map']:.4f}", end="")
            if 'map_std' in metrics:
                print(f" ± {metrics['map_std']:.4f}")
            else:
                print()

        if 'mrr' in metrics:
            print(f"  MRR: {metrics['mrr']:.4f}", end="")
            if 'mrr_std' in metrics:
                print(f" ± {metrics['mrr_std']:.4f}")
            else:
                print()

        print("=" * 60)


def evaluate_recommender():
    """Example evaluation script."""
    print("=" * 60)
    print("Evaluating Recommender System")
    print("=" * 60)

    # Load test data
    test_df = pd.read_csv("data/processed/interactions_test.csv")
    train_df = pd.read_csv("data/processed/interactions_train.csv")

    # Load a model (example: Matrix Factorization)
    from src.models.matrix_factorization import MatrixFactorizationCF

    model = MatrixFactorizationCF.load("models/collaborative_filtering/matrix_factorization.pkl")

    # Create recommender function
    def recommender_func(user_id, n):
        return model.recommend(user_id, n=n, interactions_df=train_df)

    # Evaluate
    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
    avg_metrics, user_metrics = evaluator.evaluate_model(
        recommender_func, test_df, train_df, n_recommendations=20
    )

    # Print results
    evaluator.print_metrics(avg_metrics, title="Matrix Factorization Results")

    # Save results
    user_metrics.to_csv("experiments/evaluation_results.csv", index=False)

    print(f"\nDetailed results saved to experiments/evaluation_results.csv")


if __name__ == "__main__":
    evaluate_recommender()
