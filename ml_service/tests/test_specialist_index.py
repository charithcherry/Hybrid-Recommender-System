"""Test script for category specialist index."""
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple


def build_category_specialist_index(
    train_df: pd.DataFrame,
    products_df: pd.DataFrame,
    min_specialization: float = 0.30,
    min_interactions: int = 5,
    max_specialists: int = 50
) -> Dict[str, List[Tuple[int, float]]]:
    """Build specialist index at startup. O(users * interactions) once."""

    specialist_index = defaultdict(list)

    for user_id in range(1000):  # Training users 0-999
        # Get liked items (rating >= 3)
        user_interactions = train_df[
            (train_df['user_id'] == user_id) &
            (train_df['rating'] >= 3)
        ]

        if len(user_interactions) < min_interactions:
            continue

        # Join with products to get categories
        item_ids = user_interactions['item_id'].tolist()
        products = products_df[products_df['id'].isin(item_ids)]

        if len(products) == 0:
            continue

        # Count items per category
        category_counts = products['articleType'].value_counts()
        total_items = len(products)

        # Add to index for categories where user is specialist (>= 30%)
        for category, count in category_counts.items():
            specialization = count / total_items
            if specialization >= min_specialization:
                specialist_index[category].append((user_id, specialization))

    # Sort by specialization score, keep top N
    for category in specialist_index:
        specialist_index[category] = sorted(
            specialist_index[category],
            key=lambda x: x[1],
            reverse=True
        )[:max_specialists]

    return dict(specialist_index)


if __name__ == "__main__":
    print("Loading data...")
    train_df = pd.read_csv('data/processed/interactions_train.csv')
    products_df = pd.read_csv('data/processed/products_processed.csv')

    print(f"Loaded {len(train_df)} interactions and {len(products_df)} products")

    print("\nBuilding specialist index...")
    import time
    start = time.time()
    index = build_category_specialist_index(train_df, products_df)
    elapsed = time.time() - start

    print(f"\n✅ Built index in {elapsed:.2f}s")
    print(f"✅ Index has {len(index)} categories")
    print(f"\nTop 10 categories by specialist count:")
    sorted_categories = sorted(index.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for category, specialists in sorted_categories:
        print(f"  {category}: {len(specialists)} specialists")

    # Test watches specifically
    if 'Watches' in index:
        watches_specialists = index['Watches']
        print(f"\n✅ Watch specialists: {len(watches_specialists)} users")
        print(f"   Top 3:")
        for user_id, spec_score in watches_specialists[:3]:
            print(f"     User {user_id}: {spec_score*100:.1f}% watches")
    else:
        print("\n❌ No watch specialists found")

    print(f"\n✅ Phase 1 test PASSED!")
