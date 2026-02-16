"""FAISS-based retriever for conversational service."""

import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Optional
from pathlib import Path


class FAISSRetriever:
    """Wrapper for FAISS vector search with text query support."""

    def __init__(self, embeddings_path: str, products_path: str):
        """
        Initialize FAISS retriever.

        Args:
            embeddings_path: Path to product CLIP embeddings (.npy)
            products_path: Path to products CSV
        """
        print(f"Loading FAISS retriever...")

        # Load embeddings
        self.embeddings = np.load(embeddings_path)
        self.num_items = len(self.embeddings)
        print(f"  Loaded {self.num_items} product embeddings")

        # Load products
        self.products_df = pd.read_csv(products_path)
        print(f"  Loaded {len(self.products_df)} products")

        # Build FAISS index
        self.index = self._build_index(self.embeddings)
        print(f"  Built FAISS index")

        # Initialize CLIP encoder for query embedding
        from src.retrieval.clip_encoder import CLIPTextEncoder
        self.clip_encoder = CLIPTextEncoder()
        print(f"  CLIP encoder ready")

    def _build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings."""
        # Normalize embeddings for cosine similarity
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Create inner product index (equivalent to cosine with normalized vectors)
        dimension = normalized.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(normalized.astype('float32'))

        return index

    def search(
        self,
        query_text: str,
        filters: Optional[Dict] = None,
        n: int = 10,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Search for products using text query or embedding.

        Args:
            query_text: Text query (will be encoded with CLIP)
            filters: Optional filters (gender, category, color, price, etc.)
            n: Number of results
            query_embedding: Optional pre-computed query embedding

        Returns:
            List of product dictionaries with scores
        """
        # Encode query text with CLIP
        if query_embedding is None and query_text:
            print(f"[FAISS] Encoding query: '{query_text}'")
            query_embedding = self.clip_encoder.encode(query_text)

        # Search FAISS index
        if query_embedding is not None:
            # Get more candidates for filtering
            k = min(n * 10, self.num_items)
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k
            )

            # Convert to product list
            candidates = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(self.products_df):
                    continue

                product = self.products_df.iloc[idx]
                candidates.append({
                    'item_id': int(product['id']),
                    'title': product['productDisplayName'],
                    'description': product['productDisplayName'],
                    'score': float(score),
                    'metadata': {
                        'gender': product.get('gender'),
                        'articleType': product.get('articleType'),
                        'baseColour': product.get('baseColour'),
                        'season': product.get('season'),
                        'usage': product.get('usage')
                    }
                })
        else:
            # No query embedding - return filtered products
            candidates = []
            for idx, row in self.products_df.head(n * 10).iterrows():
                candidates.append({
                    'item_id': int(row['id']),
                    'title': row['productDisplayName'],
                    'description': row['productDisplayName'],
                    'score': 1.0,
                    'metadata': {
                        'gender': row.get('gender'),
                        'articleType': row.get('articleType'),
                        'baseColour': row.get('baseColour'),
                        'season': row.get('season'),
                        'usage': row.get('usage')
                    }
                })

        # Apply filters
        if filters:
            filtered = []
            for product in candidates:
                metadata = product['metadata']

                # Check each filter
                if filters.get('gender') and metadata.get('gender') != filters['gender']:
                    continue
                if filters.get('articleType') and metadata.get('articleType') != filters['articleType']:
                    continue
                if filters.get('baseColour') and metadata.get('baseColour') != filters['baseColour']:
                    continue
                if filters.get('season') and metadata.get('season') != filters['season']:
                    continue
                if filters.get('usage') and metadata.get('usage') != filters['usage']:
                    continue

                filtered.append(product)

            candidates = filtered

        # Return top n
        return candidates[:n]

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        filters: Optional[Dict] = None,
        n: int = 10
    ) -> List[Dict]:
        """
        Search using pre-computed query embedding.

        Args:
            query_embedding: Query vector (512-dim)
            filters: Optional metadata filters
            n: Number of results

        Returns:
            List of product dictionaries with scores
        """
        # Normalize query
        query_emb = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Search FAISS (get more for filtering)
        k = min(n * 10, self.num_items)
        distances, indices = self.index.search(
            query_emb.reshape(1, -1).astype('float32'),
            k
        )

        # Get products
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.products_df):
                continue

            product = self.products_df.iloc[idx]

            # Apply filters if provided
            if filters:
                if filters.get('gender') and product['gender'] != filters['gender']:
                    continue
                if filters.get('articleType') and product['articleType'] != filters['articleType']:
                    continue
                # Add more filter logic as needed

            results.append({
                'item_id': int(product['id']),
                'title': product['productDisplayName'],
                'description': product['productDisplayName'],
                'score': float(score),
                'metadata': {
                    'gender': product.get('gender'),
                    'articleType': product.get('articleType'),
                    'baseColour': product.get('baseColour'),
                    'season': product.get('season'),
                    'usage': product.get('usage')
                }
            })

            if len(results) >= n:
                break

        return results
