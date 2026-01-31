"""Neural Collaborative Filtering model."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from typing import List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


class InteractionDataset(Dataset):
    """Dataset for user-item interactions."""

    def __init__(self, interactions_df: pd.DataFrame,
                 user_mapping: dict, item_mapping: dict,
                 negative_sampling: bool = True,
                 n_negative: int = 4):
        """Initialize dataset.

        Args:
            interactions_df: Interactions DataFrame
            user_mapping: User ID to index mapping
            item_mapping: Item ID to index mapping
            negative_sampling: Whether to add negative samples
            n_negative: Number of negative samples per positive
        """
        self.interactions = interactions_df
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.negative_sampling = negative_sampling
        self.n_negative = n_negative

        # Create positive samples
        self.samples = []
        for _, row in interactions_df.iterrows():
            user_idx = user_mapping['to_idx'][row['user_id']]
            item_idx = item_mapping['to_idx'][row['item_id']]
            self.samples.append((user_idx, item_idx, 1.0))

        # Add negative samples
        if negative_sampling:
            self._add_negative_samples()

    def _add_negative_samples(self):
        """Add negative samples for each user."""
        # Get all items each user has interacted with
        user_items = self.interactions.groupby('user_id')['item_id'].apply(set).to_dict()

        n_items = len(self.item_mapping['to_idx'])

        for user_id, interacted_items in user_items.items():
            user_idx = self.user_mapping['to_idx'][user_id]
            n_samples = len(interacted_items) * self.n_negative

            # Sample negative items
            for _ in range(n_samples):
                # Keep sampling until we find an item the user hasn't interacted with
                item_idx = np.random.randint(0, n_items)
                item_id = self.item_mapping['to_id'][item_idx]

                if item_id not in interacted_items:
                    self.samples.append((user_idx, item_idx, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_idx, item_idx, label = self.samples[idx]
        return torch.tensor(user_idx, dtype=torch.long), \
               torch.tensor(item_idx, dtype=torch.long), \
               torch.tensor(label, dtype=torch.float32)


class NeuralCF(nn.Module):
    """Neural Collaborative Filtering model.

    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP).
    """

    def __init__(self, n_users: int, n_items: int,
                 embedding_dim: int = 64,
                 hidden_layers: List[int] = [128, 64, 32],
                 dropout: float = 0.2):
        """Initialize Neural CF model.

        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension
            hidden_layers: Hidden layer sizes for MLP
            dropout: Dropout rate
        """
        super(NeuralCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # GMF part
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

        # MLP part
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2

        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer
        self.output = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            user_indices: User indices tensor
            item_indices: Item indices tensor

        Returns:
            Predicted scores
        """
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_indices)
        item_emb_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_emb_gmf * item_emb_gmf

        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_indices)
        item_emb_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate GMF and MLP outputs
        concat = torch.cat([gmf_output, mlp_output], dim=-1)

        # Final prediction
        output = self.output(concat)
        output = self.sigmoid(output)

        return output.squeeze()

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get combined user embedding.

        Args:
            user_idx: User index

        Returns:
            User embedding
        """
        user_tensor = torch.tensor([user_idx], dtype=torch.long)

        with torch.no_grad():
            emb_gmf = self.user_embedding_gmf(user_tensor).numpy()
            emb_mlp = self.user_embedding_mlp(user_tensor).numpy()

        # Concatenate embeddings
        return np.concatenate([emb_gmf, emb_mlp], axis=1).squeeze()

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get combined item embedding.

        Args:
            item_idx: Item index

        Returns:
            Item embedding
        """
        item_tensor = torch.tensor([item_idx], dtype=torch.long)

        with torch.no_grad():
            emb_gmf = self.item_embedding_gmf(item_tensor).numpy()
            emb_mlp = self.item_embedding_mlp(item_tensor).numpy()

        # Concatenate embeddings
        return np.concatenate([emb_gmf, emb_mlp], axis=1).squeeze()


class NeuralCFTrainer:
    """Trainer for Neural CF model."""

    def __init__(self, config=None):
        """Initialize trainer.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()

        self.embedding_dim = self.config.get('collaborative_filtering.neural_cf.embedding_dim', 64)
        self.hidden_layers = self.config.get('collaborative_filtering.neural_cf.hidden_layers', [128, 64, 32])
        self.dropout = self.config.get('collaborative_filtering.neural_cf.dropout', 0.2)
        self.learning_rate = self.config.get('collaborative_filtering.neural_cf.learning_rate', 0.001)
        self.batch_size = self.config.get('collaborative_filtering.neural_cf.batch_size', 256)
        self.epochs = self.config.get('collaborative_filtering.neural_cf.epochs', 20)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.user_mapping = None
        self.item_mapping = None

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
             user_mapping: dict, item_mapping: dict):
        """Train the model.

        Args:
            train_df: Training interactions
            val_df: Validation interactions
            user_mapping: User ID mappings
            item_mapping: Item ID mappings
        """
        print("Training Neural Collaborative Filtering model...")
        print(f"Parameters: embedding_dim={self.embedding_dim}, "
              f"hidden_layers={self.hidden_layers}, dropout={self.dropout}")

        self.user_mapping = user_mapping
        self.item_mapping = item_mapping

        n_users = len(user_mapping['to_idx'])
        n_items = len(item_mapping['to_idx'])

        # Initialize model
        self.model = NeuralCF(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)

        print(f"Model initialized with {n_users} users and {n_items} items")

        # Create datasets
        train_dataset = InteractionDataset(train_df, user_mapping, item_mapping,
                                          negative_sampling=True, n_negative=4)
        val_dataset = InteractionDataset(val_df, user_mapping, item_mapping,
                                        negative_sampling=True, n_negative=4)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=0)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for user_indices, item_indices, labels in tqdm(train_loader,
                                                          desc=f"Epoch {epoch+1}/{self.epochs}"):
                user_indices = user_indices.to(self.device)
                item_indices = item_indices.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(user_indices, item_indices)
                loss = criterion(predictions, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for user_indices, item_indices, labels in val_loader:
                    user_indices = user_indices.to(self.device)
                    item_indices = item_indices.to(self.device)
                    labels = labels.to(self.device)

                    predictions = self.model(user_indices, item_indices)
                    loss = criterion(predictions, labels)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(Path("models/collaborative_filtering") / "neural_cf_best.pt")

        print("Training complete!")

    def recommend(self, user_id: int, n: int = 10,
                 filter_already_liked: bool = True,
                 interactions_df: pd.DataFrame = None) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            filter_already_liked: Whether to filter already interacted items
            interactions_df: User interactions for filtering

        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_mapping['to_idx']:
            print(f"User {user_id} not found")
            return []

        user_idx = self.user_mapping['to_idx'][user_id]

        # Get items to filter
        filter_items = set()
        if filter_already_liked and interactions_df is not None:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            filter_items = set(user_interactions['item_id'].values)

        # Score all items
        self.model.eval()
        n_items = len(self.item_mapping['to_idx'])

        with torch.no_grad():
            user_tensor = torch.tensor([user_idx] * n_items, dtype=torch.long).to(self.device)
            item_tensor = torch.arange(n_items, dtype=torch.long).to(self.device)

            scores = self.model(user_tensor, item_tensor).cpu().numpy()

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

    def save(self, path: Path):
        """Save model to disk.

        Args:
            path: Save path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'config': {
                'n_users': self.model.n_users,
                'n_items': self.model.n_items,
                'embedding_dim': self.embedding_dim,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout
            }
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load model from disk.

        Args:
            path: Model path

        Returns:
            Loaded NeuralCFTrainer instance
        """
        checkpoint = torch.load(path, map_location='cpu')

        instance = cls()
        instance.user_mapping = checkpoint['user_mapping']
        instance.item_mapping = checkpoint['item_mapping']

        config = checkpoint['config']
        instance.model = NeuralCF(
            n_users=config['n_users'],
            n_items=config['n_items'],
            embedding_dim=config['embedding_dim'],
            hidden_layers=config['hidden_layers'],
            dropout=config['dropout']
        )

        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()

        print(f"Model loaded from {path}")

        return instance


def train_neural_cf():
    """Train and save Neural CF model."""
    print("=" * 60)
    print("Training Neural Collaborative Filtering Model")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv("data/processed/interactions_train.csv")
    val_df = pd.read_csv("data/processed/interactions_val.csv")

    with open("data/processed/user_mapping.pkl", 'rb') as f:
        user_mapping = pickle.load(f)
    with open("data/processed/item_mapping.pkl", 'rb') as f:
        item_mapping = pickle.load(f)

    print(f"Train interactions: {len(train_df)}")
    print(f"Val interactions: {len(val_df)}")

    # Initialize and train
    trainer = NeuralCFTrainer()
    trainer.train(train_df, val_df, user_mapping, item_mapping)

    # Test recommendations
    print("\n" + "=" * 60)
    print("Sample Recommendations")
    print("=" * 60)

    sample_user = train_df['user_id'].iloc[0]
    recommendations = trainer.recommend(sample_user, n=10, interactions_df=train_df)

    print(f"\nTop 10 recommendations for user {sample_user}:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"{i}. Item {item_id}: {score:.4f}")


if __name__ == "__main__":
    train_neural_cf()
