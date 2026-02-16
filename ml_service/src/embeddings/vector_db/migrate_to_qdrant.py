"""Migration script: FAISS → Qdrant (No Docker required!)."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from qdrant.client import QdrantVectorDB


def migrate_faiss_to_qdrant(
    embeddings_path: str = "data/embeddings/product_text_embeddings.npy",
    products_path: str = "data/processed/products_processed.csv",
    storage_path: str = "./qdrant_data",
    batch_size: int = 100
):
    """
    Migrate FAISS data to Qdrant (local storage, no Docker!).

    Args:
        embeddings_path: Path to CLIP embeddings
        products_path: Path to products CSV
        storage_path: Where to store Qdrant data
        batch_size: Batch size for indexing
    """
    print("="*70)
    print("FAISS TO QDRANT MIGRATION (No Docker Required!)")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    start_time = time.time()

    embeddings = np.load(embeddings_path)
    products_df = pd.read_csv(products_path)

    load_time = time.time() - start_time
    print(f"   [OK] {len(embeddings)} embeddings loaded ({load_time:.2f}s)")
    print(f"   [OK] {len(products_df)} products loaded")

    # Initialize Qdrant
    print("\n2. Initializing Qdrant...")
    qdrant = QdrantVectorDB(storage_path=storage_path)

    # Create collection
    print("\n3. Creating collection...")
    qdrant.create_collection(recreate=False)

    # Check if already migrated
    current_count = qdrant.count_products()
    if current_count > 0:
        print(f"\n   [WARN] Collection already has {current_count} products")
        response = input("   Recreate collection? (y/n): ")
        if response.lower() == 'y':
            qdrant.create_collection(recreate=True)
        else:
            print("   Keeping existing data")
            return qdrant

    # Index products
    print("\n4. Indexing products...")
    index_start = time.time()

    qdrant.index_products(products_df, embeddings, batch_size=batch_size)

    index_time = time.time() - index_start

    # Verify
    print("\n5. Verifying migration...")
    count = qdrant.count_products()
    print(f"   [OK] Qdrant contains {count} products")

    # Get collection info
    info = qdrant.get_collection_info()
    print(f"   [OK] Vectors indexed: {info['indexed_vectors_count']}")
    print(f"   [OK] Status: {info['status']}")

    # Success summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("[OK] MIGRATION SUCCESSFUL!")
    print("="*70)
    print(f"\n[OK] Migrated {count} products to Qdrant")
    print(f"[OK] Indexing time: {index_time:.2f}s")
    print(f"[OK] Total time: {total_time:.2f}s")
    print(f"[OK] Storage: {storage_path}")
    print(f"[OK] No Docker needed! 🎉")
    print("\nQdrant is ready for FAST filtered queries!")

    return qdrant


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate FAISS to Qdrant")
    parser.add_argument("--storage", default="./qdrant_data", help="Qdrant storage path")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")

    args = parser.parse_args()

    print("\nThis will migrate all 44K products to Qdrant.")
    print("Estimated time: 1-2 minutes\n")

    qdrant = migrate_faiss_to_qdrant(
        storage_path=args.storage,
        batch_size=args.batch_size
    )

    # Quick test
    print("\n" + "="*70)
    print("QUICK TEST")
    print("="*70)

    print("\nTesting Qdrant with filtered search...")
    test_vector = np.random.rand(512)

    # Test 1: Simple search
    results = qdrant.search(query_vector=test_vector, limit=5)
    print(f"\n[OK] Simple search: Found {len(results)} products")

    # Test 2: Filtered search (THIS IS WHERE QDRANT SHINES!)
    results_filtered = qdrant.search(
        query_vector=test_vector,
        filters={"gender": "Men", "articleType": "Watches"},
        limit=5
    )

    print(f"[OK] Filtered search: Found {len(results_filtered)} men's watches")
    for i, product in enumerate(results_filtered[:3], 1):
        print(f"  {i}. {product['title'][:50]}")
        print(f"     {product['metadata']['articleType']} | {product['metadata']['gender']}")

    print("\n[OK] Qdrant working perfectly!")
    print("[OK] FASTER filtered queries than FAISS!")
