"""Re-ranking model for precise ranking of candidate items."""

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
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


class RankingDataset(Dataset):
    """Dataset for learning-to-rank."""

    def __init__(self, interactions_df: pd.DataFrame,
                 user_mapping: dict, item_mapping: dict,
                 user_features: np.ndarray = None,
                 item_features: np.ndarray = None):
        """Initialize ranking dataset.

        Args:
            interactions_df: Interactions DataFrame
            user_mapping: User ID to index mapping
            item_mapping: Item ID to index mapping
            user_features: User feature matrix
            item_features: Item feature matrix
        """
        self.interactions = interactions_df
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.user_features = user_features
        self.item_features = item_features

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]

        user_idx = self.user_mapping['to_idx'][row['user_id']]
        item_idx = self.item_mapping['to_idx'][row['item_id']]

        # Get features
        user_feat = self.user_features[user_idx] if self.user_features is not None else np.array([user_idx])
        item_feat = self.item_features[item_idx] if self.item_features is not None else np.array([item_idx])

        # Get label (rating)
        label = row['rating']

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(user_feat, dtype=torch.float32),
            torch.tensor(item_feat, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


class NeuralRanker(nn.Module):
    """Neural ranking model with multimodal features."""

    def __init__(self, n_users: int, n_items: int,
                 user_feature_dim: int, item_feature_dim: int,
                 embedding_dim: int = 64,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        """Initialize neural ranker.

        Args:
            n_users: Number of users
            n_items: Number of items
            user_feature_dim: User feature dimension
            item_feature_dim: Item feature dimension
            embedding_dim: Embedding dimension for user/item IDs
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super(NeuralRanker, self).__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Feature transformation layers
        self.user_feature_transform = nn.Linear(user_feature_dim, embedding_dim)
        self.item_feature_transform = nn.Linear(item_feature_dim, embedding_dim)

        # Combined feature size
        combined_dim = embedding_dim * 4  # user_emb + item_emb + user_feat + item_feat

        # Deep neural network
        layers = []
        input_dim = combined_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.dnn = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor,
               user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            user_indices: User indices
            item_indices: Item indices
            user_features: User features
            item_features: Item features

        Returns:
            Ranking scores
        """
        # Get embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        # Transform features
        user_feat = self.user_feature_transform(user_features)
        item_feat = self.item_feature_transform(item_features)

        # Concatenate all features
        combined = torch.cat([user_emb, item_emb, user_feat, item_feat], dim=-1)

        # Pass through DNN
        hidden = self.dnn(combined)

        # Get score
        score = self.output(hidden)

        return score.squeeze()


class ReRanker:
    """Re-ranking model trainer and predictor."""

    def __init__(self, config=None):
        """Initialize re-ranker.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()

        self.hidden_dims = self.config.get('reranking.hidden_dims', [256, 128, 64])
        self.dropout = self.config.get('reranking.dropout', 0.3)
        self.learning_rate = self.config.get('reranking.learning_rate', 0.0005)
        self.batch_size = self.config.get('reranking.batch_size', 128)
        self.epochs = self.config.get('reranking.epochs', 15)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.user_mapping = None
        self.item_mapping = None
        self.user_features = None
        self.item_features = None

    def prepare_features(self, cf_model, item_embeddings: np.ndarray):
        """Prepare user and item features.

        Args:
            cf_model: Collaborative filtering model
            item_embeddings: Content-based item embeddings
        """
        print("Preparing features...")

        # Get CF embeddings
        n_users = len(self.user_mapping['to_idx'])
        n_items = len(self.item_mapping['to_idx'])

        # User features from CF
        self.user_features = np.zeros((n_users, cf_model.factors))
        for user_id, user_idx in self.user_mapping['to_idx'].items():
            cf_emb = cf_model.get_user_embedding(user_id)
            if cf_emb is not None:
                self.user_features[user_idx] = cf_emb

        # Item features: concatenate CF embeddings and content embeddings
        cf_item_features = np.zeros((n_items, cf_model.factors))
        filtered_content_features = np.zeros((n_items, item_embeddings.shape[1]))

        for item_id, item_idx in self.item_mapping['to_idx'].items():
            # CF embedding
            cf_emb = cf_model.get_item_embedding(item_id)
            if cf_emb is not None:
                cf_item_features[item_idx] = cf_emb

            # Content embedding (filter from full embeddings)
            if item_id < len(item_embeddings):
                filtered_content_features[item_idx] = item_embeddings[item_id]

        # Combine CF and content features
        self.item_features = np.concatenate([cf_item_features, filtered_content_features], axis=1)

        print(f"User features shape: {self.user_features.shape}")
        print(f"Item features shape: {self.item_features.shape}")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
             user_mapping: dict, item_mapping: dict,
             cf_model, item_embeddings: np.ndarray):
        """Train the re-ranking model.

        Args:
            train_df: Training interactions
            val_df: Validation interactions
            user_mapping: User ID mappings
            item_mapping: Item ID mappings
            cf_model: Trained collaborative filtering model
            item_embeddings: Item content embeddings
        """
        print("Training Re-Ranking Model...")

        self.user_mapping = user_mapping
        self.item_mapping = item_mapping

        # Prepare features
        self.prepare_features(cf_model, item_embeddings)

        n_users = len(user_mapping['to_idx'])
        n_items = len(item_mapping['to_idx'])

        # Initialize model
        self.model = NeuralRanker(
            n_users=n_users,
            n_items=n_items,
            user_feature_dim=self.user_features.shape[1],
            item_feature_dim=self.item_features.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        print(f"Model initialized with {n_users} users and {n_items} items")

        # Filter val_df to only include users and items from training
        valid_users = set(user_mapping['to_idx'].keys())
        valid_items = set(item_mapping['to_idx'].keys())

        val_df_filtered = val_df[
            val_df['user_id'].isin(valid_users) & val_df['item_id'].isin(valid_items)
        ].copy()

        print(f"Filtered validation: {len(val_df)} -> {len(val_df_filtered)} interactions")

        # Create datasets
        train_dataset = RankingDataset(
            train_df, user_mapping, item_mapping,
            self.user_features, self.item_features
        )
        val_dataset = RankingDataset(
            val_df_filtered, user_mapping, item_mapping,
            self.user_features, self.item_features
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=0)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                user_idx, item_idx, user_feat, item_feat, labels = batch

                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                user_feat = user_feat.to(self.device)
                item_feat = item_feat.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(user_idx, item_idx, user_feat, item_feat)
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
                for batch in val_loader:
                    user_idx, item_idx, user_feat, item_feat, labels = batch

                    user_idx = user_idx.to(self.device)
                    item_idx = item_idx.to(self.device)
                    user_feat = user_feat.to(self.device)
                    item_feat = item_feat.to(self.device)
                    labels = labels.to(self.device)

                    predictions = self.model(user_idx, item_idx, user_feat, item_feat)
                    loss = criterion(predictions, labels)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(Path("models/reranking") / "reranker_best.pt")

        print("Training complete!")

    def rerank(self, user_id: int, candidate_items: List[int]) -> List[Tuple[int, float]]:
        """Re-rank candidate items for a user.

        Args:
            user_id: User ID
            candidate_items: List of candidate item IDs

        Returns:
            List of (item_id, score) tuples sorted by score
        """
        if user_id not in self.user_mapping['to_idx']:
            return [(item_id, 0.0) for item_id in candidate_items]

        user_idx = self.user_mapping['to_idx'][user_id]

        # Filter valid items
        valid_items = [
            item_id for item_id in candidate_items
            if item_id in self.item_mapping['to_idx']
        ]

        if len(valid_items) == 0:
            return []

        # Prepare batch
        item_indices = [self.item_mapping['to_idx'][item_id] for item_id in valid_items]

        user_indices = torch.tensor([user_idx] * len(item_indices), dtype=torch.long).to(self.device)
        item_indices_tensor = torch.tensor(item_indices, dtype=torch.long).to(self.device)

        user_feats = torch.tensor(self.user_features[user_idx], dtype=torch.float32).unsqueeze(0).repeat(len(item_indices), 1).to(self.device)
        item_feats = torch.tensor(self.item_features[item_indices], dtype=torch.float32).to(self.device)

        # Predict scores
        self.model.eval()
        with torch.no_grad():
            scores = self.model(user_indices, item_indices_tensor, user_feats, item_feats)
            scores = scores.cpu().numpy()

        # Create ranked list
        ranked_items = list(zip(valid_items, scores))
        ranked_items.sort(key=lambda x: x[1], reverse=True)

        return [(item_id, float(score)) for item_id, score in ranked_items]

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
            'user_features': self.user_features,
            'item_features': self.item_features,
            'config': {
                'hidden_dims': self.hidden_dims,
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
            Loaded ReRanker instance
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        instance = cls()
        instance.user_mapping = checkpoint['user_mapping']
        instance.item_mapping = checkpoint['item_mapping']
        instance.user_features = checkpoint['user_features']
        instance.item_features = checkpoint['item_features']

        config = checkpoint['config']
        instance.hidden_dims = config['hidden_dims']
        instance.dropout = config['dropout']

        n_users = len(instance.user_mapping['to_idx'])
        n_items = len(instance.item_mapping['to_idx'])

        instance.model = NeuralRanker(
            n_users=n_users,
            n_items=n_items,
            user_feature_dim=instance.user_features.shape[1],
            item_feature_dim=instance.item_features.shape[1],
            hidden_dims=instance.hidden_dims,
            dropout=instance.dropout
        )

        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()

        print(f"Model loaded from {path}")

        return instance


def train_reranker():
    """Train and save re-ranker model."""
    print("=" * 60)
    print("Training Re-Ranker Model")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv("data/processed/interactions_train.csv")
    val_df = pd.read_csv("data/processed/interactions_val.csv")

    with open("data/processed/user_mapping.pkl", 'rb') as f:
        user_mapping = pickle.load(f)
    with open("data/processed/item_mapping.pkl", 'rb') as f:
        item_mapping = pickle.load(f)

    # Load CF model and embeddings
    from src.models.matrix_factorization import MatrixFactorizationCF

    cf_model = MatrixFactorizationCF.load("models/collaborative_filtering/matrix_factorization.pkl")
    item_embeddings = np.load("data/embeddings/product_text_embeddings.npy")

    # Initialize and train
    reranker = ReRanker()
    reranker.train(train_df, val_df, user_mapping, item_mapping, cf_model, item_embeddings)

    print("\n" + "=" * 60)
    print("Re-Ranker Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_reranker()
