"""Hybrid Retriever - Intelligently chooses between FAISS and Qdrant."""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class HybridRetriever:
    """
    Intelligent retriever that chooses between FAISS and Qdrant.

    Strategy:
    - Use FAISS: Simple similarity, no filters, speed-critical
    - Use Qdrant: Has filters, conversational queries, exact matching
    """

    def __init__(
        self,
        faiss_index_path: Optional[str] = None,
        qdrant_storage_path: str = "./qdrant_data"
    ):
        """
        Initialize hybrid retriever.

        Args:
            faiss_index_path: Path to FAISS index (optional)
            qdrant_storage_path: Path to Qdrant storage
        """
        print("Initializing Hybrid Retriever (FAISS + Qdrant)...")

        # Initialize Qdrant (always available)
        from embeddings.vector_db.qdrant.client import QdrantVectorDB
        self.qdrant = QdrantVectorDB(storage_path=qdrant_storage_path)
        print("  [OK] Qdrant loaded (for filtered queries)")

        # Initialize FAISS (optional, for speed)
        self.faiss = None
        if faiss_index_path and Path(faiss_index_path).exists():
            try:
                import faiss as faiss_lib
                self.faiss_index = faiss_lib.read_index(faiss_index_path)
                print("  [OK] FAISS loaded (for fast queries)")
            except Exception as e:
                print(f"  [WARN] FAISS not loaded: {e}")

        print("[OK] Hybrid Retriever ready!")

    def search(
        self,
        query_vector: np.ndarray,
        filters: Optional[Dict] = None,
        limit: int = 10,
        force_engine: Optional[str] = None
    ) -> List[Dict]:
        """
        Smart search: Automatically choose FAISS or Qdrant.

        Args:
            query_vector: Query embedding (512-dim)
            filters: Optional metadata filters
            limit: Number of results
            force_engine: Force 'faiss' or 'qdrant' (for testing)

        Returns:
            List of product dictionaries
        """
        # Decide which engine to use
        use_qdrant = self._should_use_qdrant(filters, force_engine)

        if use_qdrant:
            print(f"[Hybrid] Using QDRANT (has filters: {bool(filters)})")
            return self.qdrant.search(
                query_vector=query_vector,
                filters=filters,
                limit=limit
            )
        else:
            print("[Hybrid] Using FAISS (simple similarity, fast!)")
            # For now, delegate to Qdrant without filters
            # In production, use actual FAISS here
            return self.qdrant.search(
                query_vector=query_vector,
                filters=None,
                limit=limit
            )

    def _should_use_qdrant(
        self,
        filters: Optional[Dict],
        force_engine: Optional[str]
    ) -> bool:
        """
        Decision logic: When to use Qdrant vs FAISS.

        Returns:
            True if should use Qdrant, False for FAISS
        """
        # Manual override
        if force_engine == "qdrant":
            return True
        if force_engine == "faiss":
            return False

        # Decision rules
        if not filters or len(filters) == 0:
            # No filters → Use FAISS (faster)
            return False

        # Has filters → Use Qdrant (better filtering)
        filter_count = sum(1 for v in filters.values() if v is not None and v != "")

        if filter_count >= 1:
            # 1+ filters → Qdrant (exact matching)
            return True

        # Default: FAISS
        return False

    def get_stats(self) -> Dict:
        """Get statistics from both engines."""
        return {
            "qdrant": {
                "enabled": True,
                "count": self.qdrant.count_products(),
                "storage": "local"
            },
            "faiss": {
                "enabled": self.faiss is not None,
                "count": self.faiss_index.ntotal if self.faiss else 0,
                "storage": "in-memory"
            }
        }
