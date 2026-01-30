"""Data preprocessing pipeline for multimodal data."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


class DataPreprocessor:
    """Preprocess multimodal data for recommendation."""

    def __init__(self, config=None):
        """Initialize preprocessor.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text string

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def preprocess_products(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess product data.

        Args:
            products_df: Raw product DataFrame

        Returns:
            Processed product DataFrame
        """
        print("Preprocessing product data...")

        df = products_df.copy()

        # Ensure required columns exist
        if 'title' not in df.columns:
            df['title'] = df['id'].apply(lambda x: f"Product {x}")

        if 'description' not in df.columns:
            df['description'] = df['title']

        # Clean text fields
        df['title_clean'] = df['title'].apply(self.clean_text)
        df['description_clean'] = df['description'].apply(self.clean_text)

        # Create combined text field for embedding
        df['combined_text'] = df['title_clean'] + '. ' + df['description_clean']

        # Handle missing values
        df['combined_text'] = df['combined_text'].fillna('')

        print(f"Processed {len(df)} products")

        return df

    def create_interaction_matrix(self, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        """Create user-item interaction matrix.

        Args:
            interactions_df: Interactions DataFrame

        Returns:
            Tuple of (interaction_matrix, user_mapping, item_mapping)
        """
        print("Creating interaction matrix...")

        # Create user and item mappings
        unique_users = sorted(interactions_df['user_id'].unique())
        unique_items = sorted(interactions_df['item_id'].unique())

        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}

        # Create matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        matrix = np.zeros((n_users, n_items))

        # Fill matrix with ratings
        for _, row in interactions_df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            # Use max rating if multiple interactions
            matrix[user_idx, item_idx] = max(matrix[user_idx, item_idx], row['rating'])

        user_mapping = {'to_idx': user_to_idx, 'to_id': idx_to_user}
        item_mapping = {'to_idx': item_to_idx, 'to_id': idx_to_item}

        print(f"Created matrix of shape {matrix.shape}")
        print(f"Sparsity: {1 - np.count_nonzero(matrix) / matrix.size:.4f}")

        return matrix, user_mapping, item_mapping

    def split_interactions(self, interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions into train/val/test sets.

        Uses temporal split for more realistic evaluation.

        Args:
            interactions_df: Interactions DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("Splitting interactions...")

        # Sort by timestamp
        df = interactions_df.sort_values('timestamp').reset_index(drop=True)

        # Get split sizes
        train_split = self.config.get('data.train_split', 0.8)
        val_split = self.config.get('data.val_split', 0.1)

        # Calculate split indices
        n = len(df)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))

        # Split data
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]

        print(f"Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        print(f"Val: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/n*100:.1f}%)")

        return train_df, val_df, test_df

    def create_evaluation_sets(self, interactions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create evaluation sets for each user.

        For each user, hold out their last N interactions for testing.

        Args:
            interactions_df: Interactions DataFrame

        Returns:
            Dictionary with train/val/test DataFrames
        """
        print("Creating user-based evaluation sets...")

        # Sort by user and timestamp
        df = interactions_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        train_data = []
        val_data = []
        test_data = []

        # For each user, hold out last interactions
        for user_id, user_df in df.groupby('user_id'):
            user_interactions = user_df.to_dict('records')
            n = len(user_interactions)

            if n < 5:  # Skip users with too few interactions
                continue

            # Last 2 interactions for test, 1 for val, rest for train
            n_test = min(2, max(1, n // 5))
            n_val = min(1, max(1, n // 10))

            train_data.extend(user_interactions[:-n_test-n_val])
            val_data.extend(user_interactions[-n_test-n_val:-n_test])
            test_data.extend(user_interactions[-n_test:])

        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        test_df = pd.DataFrame(test_data)

        print(f"Train: {len(train_df)} interactions")
        print(f"Val: {len(val_df)} interactions")
        print(f"Test: {len(test_df)} interactions")

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

    def save_processed_data(self, products_df: pd.DataFrame,
                           interaction_splits: Dict[str, pd.DataFrame]):
        """Save processed data to disk.

        Args:
            products_df: Processed products DataFrame
            interaction_splits: Dictionary with train/val/test splits
        """
        print("\nSaving processed data...")

        # Save products
        products_df.to_csv(self.processed_dir / "products_processed.csv", index=False)

        # Save interaction splits
        for split_name, split_df in interaction_splits.items():
            split_df.to_csv(self.processed_dir / f"interactions_{split_name}.csv", index=False)

        print(f"Saved processed data to {self.processed_dir}")


def main():
    """Main preprocessing pipeline."""
    preprocessor = DataPreprocessor()

    # Load raw data
    print("Loading raw data...")
    products_df = pd.read_csv("data/raw/products.csv")
    interactions_df = pd.read_csv("data/processed/interactions.csv")

    # Preprocess products
    products_df = preprocessor.preprocess_products(products_df)

    # Create interaction splits
    interaction_splits = preprocessor.create_evaluation_sets(interactions_df)

    # Create interaction matrix for training set
    matrix, user_mapping, item_mapping = preprocessor.create_interaction_matrix(
        interaction_splits['train']
    )

    # Save mappings
    import pickle
    with open(preprocessor.processed_dir / "user_mapping.pkl", 'wb') as f:
        pickle.dump(user_mapping, f)
    with open(preprocessor.processed_dir / "item_mapping.pkl", 'wb') as f:
        pickle.dump(item_mapping, f)

    # Save processed data
    preprocessor.save_processed_data(products_df, interaction_splits)

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
