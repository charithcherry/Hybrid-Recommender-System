"""Candidate Generation for two-stage recommendation pipeline."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import pickle
import faiss
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


# Adaptive weighting tiers based on user interaction count
# Format: (min_interactions, cf_weight, content_weight, popular_weight)
ADAPTIVE_WEIGHT_TIERS = [
    (20, 0.70, 0.20, 0.10),  # Established users: trust CF
    (5,  0.40, 0.40, 0.20),  # Warm start: balanced approach
    (0,  0.10, 0.60, 0.30),  # Cold start: trust content + popular
]


class CandidateGenerator:
    """Hybrid candidate generation combining CF and content-based retrieval."""

    def __init__(self, config=None):
        """Initialize candidate generator.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()

        self.num_candidates = self.config.get('candidate_generation.num_candidates', 100)
        self.similarity_metric = self.config.get('candidate_generation.similarity_metric', 'cosine')

        # Models
        self.cf_model = None
        self.item_embeddings = None  # Filtered embeddings (only training items)
        self.full_item_embeddings = None  # ALL 44K product embeddings
        self.user_mapping = None
        self.item_mapping = None

        # Vector index for fast similarity search
        self.vector_index = None
        self.index_type = self.config.get('candidate_generation.vector_index', 'faiss')

    def load_cf_model(self, model_path: str):
        """Load collaborative filtering model.

        Args:
            model_path: Path to CF model
        """
        from src.models.matrix_factorization import MatrixFactorizationCF

        print(f"Loading CF model from {model_path}")
        self.cf_model = MatrixFactorizationCF.load(model_path)

    def load_embeddings(self, embeddings_path: str):
        """Load item embeddings.

        Args:
            embeddings_path: Path to embeddings file
        """
        print(f"Loading embeddings from {embeddings_path}")
        full_embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings with shape {full_embeddings.shape}")

        # Keep FULL embeddings for querying new items
        self.full_item_embeddings = full_embeddings.copy()
        self.item_embeddings = full_embeddings  # Will be filtered later in build_vector_index

    def build_vector_index(self):
        """Build FAISS index for fast similarity search."""
        if self.item_embeddings is None:
            raise ValueError("Item embeddings not loaded")

        if self.item_mapping is None:
            raise ValueError("Item mapping not loaded")

        print("Building FAISS index...")

        # Filter embeddings to only include items in the interaction data
        n_items = len(self.item_mapping['to_idx'])
        filtered_embeddings = np.zeros((n_items, self.item_embeddings.shape[1]))

        for item_id, item_idx in self.item_mapping['to_idx'].items():
            if item_id < len(self.item_embeddings):
                filtered_embeddings[item_idx] = self.item_embeddings[item_id]
            else:
                # Use zero vector for missing items
                filtered_embeddings[item_idx] = np.zeros(self.item_embeddings.shape[1])

        # Update item embeddings to filtered version
        self.item_embeddings = filtered_embeddings

        embedding_dim = self.item_embeddings.shape[1]

        # Normalize embeddings for cosine similarity
        if self.similarity_metric == 'cosine':
            norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
            normalized_embeddings = self.item_embeddings / (norms + 1e-8)

            # Use inner product index for normalized vectors (equivalent to cosine)
            self.vector_index = faiss.IndexFlatIP(embedding_dim)
        else:
            # Use L2 distance
            self.vector_index = faiss.IndexFlatL2(embedding_dim)
            normalized_embeddings = self.item_embeddings

        # Add vectors to index
        self.vector_index.add(normalized_embeddings.astype('float32'))

        print(f"Built FAISS index with {self.vector_index.ntotal} vectors")

    def generate_cf_candidates(self, user_id: int,
                               n: int = 100,
                               interactions_df: pd.DataFrame = None) -> List[Tuple[int, float, str]]:
        """Generate candidates using collaborative filtering.

        Args:
            user_id: User ID
            n: Number of candidates
            interactions_df: User interactions for filtering

        Returns:
            List of (item_id, score, source) tuples
        """
        if self.cf_model is None:
            return []

        recommendations = self.cf_model.recommend(
            user_id, n=n,
            filter_already_liked=True,
            interactions_df=interactions_df
        )

        return [(item_id, score, 'cf') for item_id, score in recommendations]

    def generate_content_candidates(self, user_id: int,
                                   n: int = 100,
                                   interactions_df: pd.DataFrame = None) -> List[Tuple[int, float, str]]:
        """Generate candidates using content-based filtering.

        Uses user's interaction history to find similar items.

        Args:
            user_id: User ID
            n: Number of candidates
            interactions_df: User interactions

        Returns:
            List of (item_id, score, source) tuples
        """
        if self.vector_index is None or interactions_df is None:
            return []

        # Get user's recent interactions
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]

        if len(user_interactions) == 0:
            return []

        # Get user's interaction history sorted by rating/timestamp
        user_items = user_interactions.sort_values('rating', ascending=False)['item_id'].tolist()

        # Use top K items to represent user's interests
        top_k = min(5, len(user_items))
        representative_items = user_items[:top_k]

        print(f"Representative items for query: {representative_items[:10]}")

        # IMPROVED STRATEGY: Query each liked item separately, then combine
        # This prevents mixing different categories (watches + backpacks)
        all_recommendations = {}  # {item_id: max_score}
        seen_items = set(user_items)

        for query_item_id in representative_items[:10]:  # Query top 10 liked items
            # Get embedding for this item
            query_emb = None

            if query_item_id in self.item_mapping['to_idx']:
                # Item in index
                item_idx = self.item_mapping['to_idx'][query_item_id]
                query_emb = self.item_embeddings[item_idx]
            elif self.full_item_embeddings is not None and query_item_id < len(self.full_item_embeddings):
                # Item has embedding but not in index - use full embedding
                query_emb = self.full_item_embeddings[query_item_id]
                print(f"Using full embedding for item {query_item_id}")

            if query_emb is None:
                continue

            # Normalize
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

            # Query FAISS for similar items
            k = min(20, self.vector_index.ntotal)
            distances, indices = self.vector_index.search(
                query_emb.reshape(1, -1).astype('float32'), k
            )

            # Add results to recommendations dict (keep highest score)
            for idx, score in zip(indices[0], distances[0]):
                item_id = self.item_mapping['to_id'][int(idx)]

                # Skip already seen items
                if item_id in seen_items:
                    continue

                # Keep highest score if item already found
                if item_id not in all_recommendations or score > all_recommendations[item_id]:
                    all_recommendations[item_id] = float(score)

        # Convert to list and sort by score
        recommendations = [(item_id, score, 'content') for item_id, score in sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)[:n]]

        print(f"Generated {len(recommendations)} recommendations from {len(representative_items)} query items")

        return recommendations

    def generate_popular_candidates(self, interactions_df: pd.DataFrame,
                                   user_id: int = None,
                                   n: int = 50) -> List[Tuple[int, float, str]]:
        """Generate popular item candidates as fallback.

        Args:
            interactions_df: All interactions
            user_id: User ID (for filtering)
            n: Number of candidates

        Returns:
            List of (item_id, score, source) tuples
        """
        # Get user's interactions for filtering
        seen_items = set()
        if user_id is not None:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            seen_items = set(user_interactions['item_id'].values)

        # Calculate item popularity
        item_counts = interactions_df['item_id'].value_counts()

        recommendations = []
        for item_id, count in item_counts.items():
            if item_id in seen_items:
                continue

            # Normalize score
            score = float(count) / item_counts.max()
            recommendations.append((item_id, score, 'popular'))

            if len(recommendations) >= n:
                break

        return recommendations

    def get_adaptive_weights(self, interaction_count: int) -> Tuple[float, float, float]:
        """Get adaptive weights based on user's interaction count.

        Args:
            interaction_count: Number of interactions (rating >= 3) the user has

        Returns:
            Tuple of (cf_weight, content_weight, popular_weight)
        """
        for min_count, cf_w, content_w, pop_w in ADAPTIVE_WEIGHT_TIERS:
            if interaction_count >= min_count:
                return (cf_w, content_w, pop_w)

        # Default to coldstart weights if no tier matches
        return ADAPTIVE_WEIGHT_TIERS[-1][1:]

    def generate_hybrid_candidates(self, user_id: int,
                                   interactions_df: pd.DataFrame,
                                   n: int = 100,
                                   adaptive_weights: bool = True,
                                   cf_weight: float = None,
                                   content_weight: float = None,
                                   popular_weight: float = None) -> List[Tuple[int, float, Dict]]:
        """Generate hybrid candidates combining multiple strategies.

        Args:
            user_id: User ID
            interactions_df: User interactions
            n: Total number of candidates
            adaptive_weights: If True, automatically determine weights based on user profile
            cf_weight: Weight for CF candidates (manual override)
            content_weight: Weight for content-based candidates (manual override)
            popular_weight: Weight for popularity-based candidates (manual override)

        Returns:
            List of (item_id, score, metadata) tuples
        """
        # Determine weights (adaptive or manual)
        if adaptive_weights and all(w is None for w in [cf_weight, content_weight, popular_weight]):
            # Count user interactions (liked items)
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            interaction_count = len(user_interactions[user_interactions['rating'] >= 3])

            # Get adaptive weights based on interaction count
            cf_weight, content_weight, popular_weight = self.get_adaptive_weights(interaction_count)

            print(f"[Adaptive Weights] User {user_id}: {interaction_count} interactions → "
                  f"CF:{cf_weight:.2f}, content:{content_weight:.2f}, popular:{popular_weight:.2f}")
        else:
            # Use provided weights or defaults
            cf_weight = cf_weight if cf_weight is not None else 0.5
            content_weight = content_weight if content_weight is not None else 0.3
            popular_weight = popular_weight if popular_weight is not None else 0.2

        # Number of candidates from each source
        n_cf = int(n * cf_weight)
        n_content = int(n * content_weight)
        n_popular = int(n * popular_weight)

        # Generate candidates from each source
        cf_candidates = self.generate_cf_candidates(user_id, n=n_cf, interactions_df=interactions_df)
        content_candidates = self.generate_content_candidates(user_id, n=n_content, interactions_df=interactions_df)
        popular_candidates = self.generate_popular_candidates(interactions_df, user_id=user_id, n=n_popular)

        # Combine and deduplicate
        all_candidates = defaultdict(lambda: {'score': 0.0, 'sources': []})

        # Add CF candidates
        for item_id, score, source in cf_candidates:
            all_candidates[item_id]['score'] += score * cf_weight
            all_candidates[item_id]['sources'].append(source)

        # Add content candidates
        for item_id, score, source in content_candidates:
            all_candidates[item_id]['score'] += score * content_weight
            all_candidates[item_id]['sources'].append(source)

        # Add popular candidates
        for item_id, score, source in popular_candidates:
            all_candidates[item_id]['score'] += score * popular_weight
            all_candidates[item_id]['sources'].append(source)

        # Convert to list and sort by score
        candidates = [
            (item_id, metadata['score'], metadata)
            for item_id, metadata in all_candidates.items()
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return candidates[:n]

    def save(self, path: str):
        """Save candidate generator state.

        Args:
            path: Save path
        """
        state = {
            'num_candidates': self.num_candidates,
            'similarity_metric': self.similarity_metric,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'item_embeddings': self.item_embeddings,
            'full_item_embeddings': getattr(self, 'full_item_embeddings', None),
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        # Save FAISS index
        if self.vector_index is not None:
            index_path = Path(path).parent / f"{Path(path).stem}_faiss.index"
            faiss.write_index(self.vector_index, str(index_path))

        print(f"Candidate generator saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load candidate generator from disk.

        Args:
            path: Load path

        Returns:
            Loaded CandidateGenerator instance
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        instance = cls()
        instance.num_candidates = state['num_candidates']
        instance.similarity_metric = state['similarity_metric']
        instance.user_mapping = state['user_mapping']
        instance.item_mapping = state['item_mapping']
        instance.item_embeddings = state.get('item_embeddings', None)
        instance.full_item_embeddings = state.get('full_item_embeddings', None)

        # Load FAISS index
        index_path = Path(path).parent / f"{Path(path).stem}_faiss.index"
        if index_path.exists():
            instance.vector_index = faiss.read_index(str(index_path))

        print(f"Candidate generator loaded from {path}")

        return instance


def build_candidate_generator():
    """Build and save candidate generator."""
    print("=" * 60)
    print("Building Candidate Generator")
    print("=" * 60)

    generator = CandidateGenerator()

    # Load CF model
    cf_model_path = "models/collaborative_filtering/matrix_factorization.pkl"
    if Path(cf_model_path).exists():
        generator.load_cf_model(cf_model_path)

    # Load embeddings
    embeddings_path = "data/embeddings/product_text_embeddings.npy"
    if Path(embeddings_path).exists():
        generator.load_embeddings(embeddings_path)

        # Load mappings
        with open("data/processed/item_mapping.pkl", 'rb') as f:
            generator.item_mapping = pickle.load(f)
        with open("data/processed/user_mapping.pkl", 'rb') as f:
            generator.user_mapping = pickle.load(f)

        # Build vector index
        generator.build_vector_index()

    # Save generator
    save_path = Path("models/candidate_generation")
    save_path.mkdir(parents=True, exist_ok=True)
    generator.save(save_path / "candidate_generator.pkl")

    # Test candidate generation
    print("\n" + "=" * 60)
    print("Testing Candidate Generation")
    print("=" * 60)

    train_df = pd.read_csv("data/processed/interactions_train.csv")
    sample_user = train_df['user_id'].iloc[0]

    candidates = generator.generate_hybrid_candidates(
        sample_user, train_df, n=20
    )

    print(f"\nTop 20 candidates for user {sample_user}:")
    for i, (item_id, score, metadata) in enumerate(candidates, 1):
        sources = ','.join(metadata['sources'])
        print(f"{i}. Item {item_id}: {score:.4f} (sources: {sources})")

    print("\n" + "=" * 60)
    print("Candidate Generator Built Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    build_candidate_generator()
