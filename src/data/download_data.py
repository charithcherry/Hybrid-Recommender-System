"""Download and prepare Pinterest-style dataset."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


class DatasetDownloader:
    """Download and prepare multimodal dataset."""

    def __init__(self, config=None):
        """Initialize dataset downloader.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_fashion_dataset(self):
        """Download Fashion Product Images dataset from HuggingFace.

        This dataset contains product images with titles and descriptions.
        """
        print("Downloading fashion product dataset...")

        try:
            # Use a fashion dataset from HuggingFace
            dataset = load_dataset("ashraq/fashion-product-images-small", split="train")

            print(f"Downloaded {len(dataset)} items")

            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)

            # Save raw data
            df.to_csv(self.raw_dir / "products.csv", index=False)
            print(f"Saved product data to {self.raw_dir / 'products.csv'}")

            return df

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Creating synthetic dataset instead...")
            return self._create_synthetic_dataset()

    def _create_synthetic_dataset(self):
        """Create a synthetic dataset for testing.

        Returns:
            DataFrame with synthetic product data
        """
        num_items = self.config.get('data.num_items', 10000)

        print(f"Creating synthetic dataset with {num_items} items...")

        # Create synthetic product data
        categories = ['fashion', 'home', 'art', 'food', 'travel', 'technology']
        styles = ['modern', 'vintage', 'minimalist', 'rustic', 'elegant', 'casual']

        data = {
            'id': range(num_items),
            'title': [f"Product {i}: {np.random.choice(styles)} {np.random.choice(categories)}"
                     for i in range(num_items)],
            'description': [f"This is a {np.random.choice(styles)} {np.random.choice(categories)} item "
                          f"with unique characteristics and high quality."
                          for _ in range(num_items)],
            'category': [np.random.choice(categories) for _ in range(num_items)],
            'style': [np.random.choice(styles) for _ in range(num_items)],
        }

        df = pd.DataFrame(data)

        # Save synthetic data
        df.to_csv(self.raw_dir / "products.csv", index=False)
        print(f"Saved synthetic product data to {self.raw_dir / 'products.csv'}")

        return df

    def generate_interactions(self, products_df):
        """Generate synthetic user-item interactions.

        Args:
            products_df: DataFrame containing product information

        Returns:
            DataFrame with user interactions
        """
        num_users = self.config.get('data.num_users', 1000)
        num_items = len(products_df)
        min_interactions = self.config.get('data.min_interactions', 5)

        print(f"Generating interactions for {num_users} users...")

        interactions = []

        for user_id in tqdm(range(num_users), desc="Generating interactions"):
            # Each user interacts with 5-50 items
            num_user_interactions = np.random.randint(min_interactions, 50)

            # Users tend to interact with items from similar categories
            user_preference = np.random.choice(products_df['category'].unique()
                                             if 'category' in products_df.columns
                                             else range(num_items))

            for _ in range(num_user_interactions):
                # 70% chance of interacting with preferred category
                if 'category' in products_df.columns and np.random.random() < 0.7:
                    item_id = np.random.choice(
                        products_df[products_df['category'] == user_preference]['id'].values
                    )
                else:
                    item_id = np.random.randint(0, num_items)

                # Interaction types: view, like, save, purchase
                interaction_type = np.random.choice(
                    ['view', 'like', 'save', 'purchase'],
                    p=[0.5, 0.3, 0.15, 0.05]
                )

                # Implicit feedback (higher for more engaging interactions)
                rating_map = {'view': 1, 'like': 3, 'save': 4, 'purchase': 5}
                rating = rating_map[interaction_type]

                # Add timestamp
                timestamp = np.random.randint(1, 365 * 24 * 3600)  # Random time in last year

                interactions.append({
                    'user_id': user_id,
                    'item_id': int(item_id),
                    'interaction_type': interaction_type,
                    'rating': rating,
                    'timestamp': timestamp
                })

        interactions_df = pd.DataFrame(interactions)

        # Save interactions
        interactions_df.to_csv(self.processed_dir / "interactions.csv", index=False)
        print(f"Saved {len(interactions_df)} interactions to {self.processed_dir / 'interactions.csv'}")

        # Print statistics
        print("\nInteraction Statistics:")
        print(f"Total users: {interactions_df['user_id'].nunique()}")
        print(f"Total items: {interactions_df['item_id'].nunique()}")
        print(f"Total interactions: {len(interactions_df)}")
        print(f"Avg interactions per user: {len(interactions_df) / interactions_df['user_id'].nunique():.2f}")
        print(f"Sparsity: {1 - len(interactions_df) / (num_users * num_items):.4f}")
        print("\nInteraction type distribution:")
        print(interactions_df['interaction_type'].value_counts())

        return interactions_df

    def prepare_dataset(self):
        """Main method to download and prepare the complete dataset.

        Returns:
            Tuple of (products_df, interactions_df)
        """
        print("=" * 60)
        print("Preparing Multimodal Recommender Dataset")
        print("=" * 60)

        # Download or create products dataset
        products_df = self.download_fashion_dataset()

        # Limit to configured number of items
        max_items = self.config.get('data.num_items', 10000)
        if len(products_df) > max_items:
            products_df = products_df.head(max_items).reset_index(drop=True)
            products_df['id'] = range(len(products_df))

        print(f"\nUsing {len(products_df)} products")

        # Generate interactions
        interactions_df = self.generate_interactions(products_df)

        print("\n" + "=" * 60)
        print("Dataset preparation complete!")
        print("=" * 60)

        return products_df, interactions_df


def main():
    """Main function to download and prepare dataset."""
    downloader = DatasetDownloader()
    products_df, interactions_df = downloader.prepare_dataset()

    print("\nDataset files created:")
    print(f"  - data/raw/products.csv ({len(products_df)} items)")
    print(f"  - data/processed/interactions.csv ({len(interactions_df)} interactions)")


if __name__ == "__main__":
    main()
