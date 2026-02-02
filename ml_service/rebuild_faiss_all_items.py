"""
Rebuild FAISS index with ALL 44K products instead of just 9K training items.
This enables content-based recommendations for all items, not just training items.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import faiss

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from src.models.candidate_generation import CandidateGenerator

def rebuild_faiss_with_all_items():
    """Rebuild FAISS index with all 44K products."""

    print("="*80)
    print("REBUILDING FAISS INDEX WITH ALL 44K PRODUCTS")
    print("="*80)

    # Load the existing candidate generator
    print("\n1. Loading existing candidate generator...")
    cg_path = Path("models/candidate_generation/candidate_generator.pkl")
    generator = CandidateGenerator.load(str(cg_path))
    print("   [OK] Loaded candidate generator")

    # Load full CLIP embeddings (44K items)
    print("\n2. Loading full CLIP embeddings...")
    embeddings_path = Path("data/embeddings/product_text_embeddings.npy")
    full_embeddings = np.load(embeddings_path)
    print(f"   [OK] Loaded embeddings: {full_embeddings.shape}")

    # Load products to create item mapping
    print("\n3. Creating item mapping for all products...")
    products_df = pd.read_csv("data/processed/products_processed.csv")
    print(f"   [OK] Loaded {len(products_df)} products")

    # Create new item mapping for ALL items
    new_item_mapping = {
        'to_idx': {},  # {item_id: index}
        'to_id': {}    # {index: item_id}
    }

    for idx, row in products_df.iterrows():
        item_id = int(row['id'])
        new_item_mapping['to_idx'][item_id] = idx
        new_item_mapping['to_id'][idx] = item_id

    print(f"   [OK] Created mapping for {len(new_item_mapping['to_idx'])} items")

    # Update generator with new mapping and embeddings
    print("\n4. Updating candidate generator...")
    generator.item_mapping = new_item_mapping
    generator.item_embeddings = full_embeddings
    generator.full_item_embeddings = full_embeddings.copy()
    print(f"   [OK] Updated item mapping and embeddings")

    # Build new FAISS index
    print("\n5. Building FAISS index (this may take a few minutes)...")
    embedding_dim = full_embeddings.shape[1]

    # Normalize embeddings for cosine similarity
    print("   - Normalizing embeddings...")
    norms = np.linalg.norm(full_embeddings, axis=1, keepdims=True)
    normalized_embeddings = full_embeddings / (norms + 1e-8)

    # Create FAISS index
    print("   - Creating FAISS index...")
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product = cosine similarity for normalized vectors
    index.add(normalized_embeddings.astype('float32'))

    generator.vector_index = index
    print(f"   [OK] Built FAISS index with {index.ntotal} items")

    # Verify index
    print("\n6. Verifying FAISS index...")
    test_query = normalized_embeddings[0].reshape(1, -1).astype('float32')
    distances, indices = index.search(test_query, 5)
    print(f"   [OK] Test query successful - found {len(indices[0])} similar items")

    # Save updated generator
    print("\n7. Saving updated candidate generator...")
    backup_path = Path("models/candidate_generation/candidate_generator_backup_9k.pkl")
    if cg_path.exists() and not backup_path.exists():
        print(f"   - Creating backup at {backup_path}")
        import shutil
        shutil.copy(cg_path, backup_path)

    generator.save(str(cg_path))
    print(f"   [OK] Saved to {cg_path}")

    # Print statistics
    print("\n" + "="*80)
    print("REBUILD COMPLETE!")
    print("="*80)
    print(f"FAISS Index Size: {index.ntotal:,} items")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"Total Watches Available: 2,542")
    print(f"Watches in Index: {len(products_df[products_df['articleType'] == 'Watches'])}")
    print(f"\nBackup of old index saved to: {backup_path}")
    print("\nRestart the backend server to use the new index!")
    print("="*80)

if __name__ == "__main__":
    try:
        rebuild_faiss_with_all_items()
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
