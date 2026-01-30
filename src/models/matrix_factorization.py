"""Matrix Factorization for Collaborative Filtering using ALS."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
import pickle
from typing import List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


class MatrixFactorizationCF:
    """Matrix Factorization Collaborative Filtering using ALS."""

    def __init__(self, config=None):
        """Initialize MF model.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()

        # Model parameters
        self.factors = self.config.get('collaborative_filtering.matrix_factorization.factors', 128)
        self.regularization = self.config.get('collaborative_filtering.matrix_factorization.regularization', 0.01)
        self.iterations = self.config.get('collaborative_filtering.matrix_factorization.iterations', 15)
        self.alpha = self.config.get('collaborative_filtering.matrix_factorization.alpha', 1.0)

        # Initialize model
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42
        )

        self.user_factors = None
        self.item_factors = None
        self.user_mapping = None
        self.item_mapping = None

    def prepare_data(self, interactions_df: pd.DataFrame,
                    user_mapping: dict, item_mapping: dict) -> csr_matrix:
        """Prepare interaction data for training.

        Args:
            interactions_df: Interactions DataFrame
            user_mapping: User ID to index mapping
            item_mapping: Item ID to index mapping

        Returns:
            Sparse user-item matrix
        """
        # Map IDs to indices
        user_indices = interactions_df['user_id'].map(user_mapping['to_idx']).values
        item_indices = interactions_df['item_id'].map(item_mapping['to_idx']).values
        ratings = interactions_df['rating'].values

        # Create sparse matrix (item-user for implicit library)
        n_users = len(user_mapping['to_idx'])
        n_items = len(item_mapping['to_idx'])

        # implicit library expects item-user matrix
        interaction_matrix = csr_matrix(
            (ratings, (item_indices, user_indices)),
            shape=(n_items, n_users)
        )

        return interaction_matrix

    def train(self, interactions_df: pd.DataFrame,
             user_mapping: dict, item_mapping: dict):
        """Train the matrix factorization model.

        Args:
            interactions_df: Training interactions
            user_mapping: User ID mappings
            item_mapping: Item ID mappings
        """
        print("Training Matrix Factorization model...")
        print(f"Parameters: factors={self.factors}, reg={self.regularization}, "
              f"iterations={self.iterations}")

        # Store mappings
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping

        # Prepare data
        interaction_matrix = self.prepare_data(interactions_df, user_mapping, item_mapping)

        print(f"Interaction matrix shape: {interaction_matrix.shape}")
        print(f"Sparsity: {1 - interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.4f}")

        # Apply confidence weighting (alpha * rating)
        interaction_matrix = interaction_matrix.multiply(self.alpha)

        # Train model
        self.model.fit(interaction_matrix, show_progress=True)

        # Extract learned factors
        # Note: implicit library trains on item-user matrix, so factors are swapped
        self.user_factors = self.model.item_factors  # Actually user factors (1000 users)
        self.item_factors = self.model.user_factors  # Actually item factors (9043 items)

        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")
        print("Training complete!")

    def recommend(self, user_id: int, n: int = 10,
                 filter_already_liked: bool = True,
                 interactions_df: pd.DataFrame = None) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            filter_already_liked: Whether to filter items user has already interacted with
            interactions_df: User interactions for filtering

        Returns:
            List of (item_id, score) tuples
        """
        # Map user ID to index
        if user_id not in self.user_mapping['to_idx']:
            print(f"User {user_id} not found in training data")
            return []

        user_idx = self.user_mapping['to_idx'][user_id]

        # Get items to filter
        filter_items = set()
        if filter_already_liked and interactions_df is not None:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            filter_items = set(user_interactions['item_id'].values)

        # Get recommendations
        # Note: recommend expects (user_id, item_user_matrix)
        # We'll compute scores manually for more control
        user_vector = self.user_factors[user_idx]
        scores = np.dot(self.item_factors, user_vector)

        # Get top items
        top_indices = np.argsort(scores)[::-1]

        recommendations = []
        for idx in top_indices:
            item_id = self.item_mapping['to_id'][idx]

            if filter_already_liked and item_id in filter_items:
                continue

            recommendations.append((item_id, float(scores[idx])))

            if len(recommendations) >= n:
                break

        return recommendations

    def recommend_similar_items(self, item_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Find similar items based on learned item factors.

        Args:
            item_id: Item ID
            n: Number of similar items to return

        Returns:
            List of (item_id, similarity) tuples
        """
        if item_id not in self.item_mapping['to_idx']:
            print(f"Item {item_id} not found in training data")
            return []

        item_idx = self.item_mapping['to_idx'][item_id]
        item_vector = self.item_factors[item_idx]

        # Compute similarities with all items
        similarities = np.dot(self.item_factors, item_vector)

        # Normalize
        norms = np.linalg.norm(self.item_factors, axis=1)
        similarities = similarities / (norms * np.linalg.norm(item_vector) + 1e-8)

        # Get top similar items (excluding the item itself)
        top_indices = np.argsort(similarities)[::-1][1:n+1]

        similar_items = [
            (self.item_mapping['to_id'][idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return similar_items

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get learned embedding for a user.

        Args:
            user_id: User ID

        Returns:
            User embedding vector
        """
        if user_id not in self.user_mapping['to_idx']:
            return None

        user_idx = self.user_mapping['to_idx'][user_id]
        return self.user_factors[user_idx]

    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get learned embedding for an item.

        Args:
            item_id: Item ID

        Returns:
            Item embedding vector
        """
        if item_id not in self.item_mapping['to_idx']:
            return None

        item_idx = self.item_mapping['to_idx'][item_id]
        return self.item_factors[item_idx]

    def save(self, path: str):
        """Save model to disk.

        Args:
            path: Save path
        """
        model_data = {
            'model': self.model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'config': {
                'factors': self.factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'alpha': self.alpha
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load model from disk.

        Args:
            path: Model path

        Returns:
            Loaded MatrixFactorizationCF instance
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls()
        instance.model = model_data['model']
        instance.user_factors = model_data['user_factors']
        instance.item_factors = model_data['item_factors']
        instance.user_mapping = model_data['user_mapping']
        instance.item_mapping = model_data['item_mapping']

        config = model_data['config']
        instance.factors = config['factors']
        instance.regularization = config['regularization']
        instance.iterations = config['iterations']
        instance.alpha = config['alpha']

        print(f"Model loaded from {path}")

        return instance


def train_matrix_factorization():
    """Train and save matrix factorization model."""
    print("=" * 60)
    print("Training Matrix Factorization Model")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv("data/processed/interactions_train.csv")

    with open("data/processed/user_mapping.pkl", 'rb') as f:
        user_mapping = pickle.load(f)
    with open("data/processed/item_mapping.pkl", 'rb') as f:
        item_mapping = pickle.load(f)

    print(f"Train interactions: {len(train_df)}")
    print(f"Users: {len(user_mapping['to_idx'])}")
    print(f"Items: {len(item_mapping['to_idx'])}")

    # Initialize and train model
    model = MatrixFactorizationCF()
    model.train(train_df, user_mapping, item_mapping)

    # Save model
    model_dir = Path("models/collaborative_filtering")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "matrix_factorization.pkl")

    # Test recommendations
    print("\n" + "=" * 60)
    print("Sample Recommendations")
    print("=" * 60)

    sample_user = train_df['user_id'].iloc[0]
    recommendations = model.recommend(sample_user, n=10, interactions_df=train_df)

    print(f"\nTop 10 recommendations for user {sample_user}:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"{i}. Item {item_id}: {score:.4f}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_matrix_factorization()
