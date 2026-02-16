"""Qdrant client for vector database operations - No Docker required!"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path


class QdrantVectorDB:
    """Qdrant vector database client (local storage, no Docker needed!)."""

    def __init__(self, storage_path: str = "./qdrant_data"):
        """
        Initialize Qdrant client with local storage.

        Args:
            storage_path: Path to store Qdrant data (persistent)
        """
        print(f"Initializing Qdrant (no Docker required)...")

        # Create client with local storage
        self.client = QdrantClient(path=storage_path)
        self.collection_name = "products"
        self.vector_size = 512  # CLIP embedding dimension

        print(f"[OK] Qdrant initialized at {storage_path}")

    def create_collection(self, recreate: bool = False):
        """
        Create products collection.

        Args:
            recreate: If True, delete existing collection first
        """
        print("Creating Qdrant collection...")

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists:
                if recreate:
                    self.client.delete_collection(self.collection_name)
                    print(f"  Deleted existing collection: {self.collection_name}")
                else:
                    print(f"  Collection '{self.collection_name}' already exists")
                    return

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE  # Same as FAISS
                )
            )

            print(f"[OK] Collection '{self.collection_name}' created")

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def index_products(
        self,
        products_df: pd.DataFrame,
        embeddings: np.ndarray,
        batch_size: int = 100
    ):
        """
        Index products into Qdrant.

        Args:
            products_df: Products DataFrame
            embeddings: Product embeddings (44K x 512)
            batch_size: Batch size for indexing
        """
        print(f"Indexing {len(products_df)} products into Qdrant...")

        points = []
        indexed_count = 0

        for idx, row in products_df.iterrows():
            # Get embedding
            if idx >= len(embeddings):
                continue

            vector = embeddings[idx].tolist()

            # Create point with payload (metadata)
            point = PointStruct(
                id=int(row['id']),  # Use product ID as point ID
                vector=vector,
                payload={
                    "item_id": int(row['id']),
                    "title": str(row['productDisplayName']),
                    "description": str(row['productDisplayName']),
                    # Filters (Qdrant indexes these automatically!)
                    "gender": str(row.get('gender', '')),
                    "masterCategory": str(row.get('masterCategory', '')),
                    "subCategory": str(row.get('subCategory', '')),
                    "articleType": str(row.get('articleType', '')),
                    "baseColour": str(row.get('baseColour', '')),
                    "season": str(row.get('season', '')),
                    "usage": str(row.get('usage', '')),
                    "year": int(row.get('year', 2020)),
                    "interaction_count": 0
                }
            )

            points.append(point)
            indexed_count += 1

            # Upload in batches
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"  Indexed {indexed_count}/{len(products_df)}...")
                points = []

        # Upload remaining
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        print(f"[OK] Successfully indexed {indexed_count} products")

    def search(
        self,
        query_vector: np.ndarray,
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search products using vector similarity and filters.

        Args:
            query_vector: Query embedding (512-dim)
            filters: Optional metadata filters
            limit: Number of results

        Returns:
            List of product dictionaries
        """
        # Build Qdrant filter
        qdrant_filter = self._build_filter(filters) if filters else None

        # Search (Qdrant v1.7+ API)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        ).points

        # Convert to standard format
        products = []
        for result in results:
            payload = result.payload
            products.append({
                "item_id": payload["item_id"],
                "title": payload["title"],
                "description": payload["description"],
                "score": float(result.score),  # Qdrant returns similarity score directly
                "metadata": {
                    "gender": payload.get("gender"),
                    "articleType": payload.get("articleType"),
                    "baseColour": payload.get("baseColour"),
                    "season": payload.get("season"),
                    "usage": payload.get("usage"),
                    "interaction_count": payload.get("interaction_count", 0)
                }
            })

        return products

    def _build_filter(self, filters: Dict) -> Optional[Filter]:
        """Build Qdrant filter from dictionary."""
        conditions = []

        # Text filters
        if filters.get('gender'):
            conditions.append(
                FieldCondition(
                    key="gender",
                    match=MatchValue(value=filters['gender'])
                )
            )

        if filters.get('articleType'):
            conditions.append(
                FieldCondition(
                    key="articleType",
                    match=MatchValue(value=filters['articleType'])
                )
            )

        if filters.get('baseColour'):
            conditions.append(
                FieldCondition(
                    key="baseColour",
                    match=MatchValue(value=filters['baseColour'])
                )
            )

        if filters.get('season'):
            conditions.append(
                FieldCondition(
                    key="season",
                    match=MatchValue(value=filters['season'])
                )
            )

        if filters.get('usage'):
            conditions.append(
                FieldCondition(
                    key="usage",
                    match=MatchValue(value=filters['usage'])
                )
            )

        # Numeric range filters
        price_range = filters.get('price_range', {})
        if price_range.get('min') or price_range.get('max'):
            conditions.append(
                FieldCondition(
                    key="price",
                    range=Range(
                        gte=price_range.get('min'),
                        lte=price_range.get('max')
                    )
                )
            )

        # Return filter or None
        if not conditions:
            return None

        return Filter(must=conditions)

    def count_products(self) -> int:
        """Get total number of products."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except:
            return 0

    def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            collections = self.client.get_collections()
            return True
        except:
            return False

    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": info.name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status
        }
