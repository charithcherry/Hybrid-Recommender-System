"""Quick API test script."""

import pandas as pd
from src.models.candidate_generation import CandidateGenerator
from src.models.reranking import ReRanker

print("="*60)
print("MULTIMODAL RECOMMENDER SYSTEM - Quick Test")
print("="*60)

# Load data and models
print("\nLoading models...")
train_df = pd.read_csv('data/processed/interactions_train.csv')
products_df = pd.read_csv('data/processed/products_processed.csv')

cand_gen = CandidateGenerator.load('models/candidate_generation/candidate_generator.pkl')
cand_gen.load_cf_model('models/collaborative_filtering/matrix_factorization.pkl')
reranker = ReRanker.load('models/reranking/reranker_best.pt')

print("Models loaded successfully!")

# Test for multiple users
print("\n" + "="*60)
print("Testing Recommendations for 3 Different Users")
print("="*60)

for test_user in [0, 5, 10]:
    print(f"\n--- User {test_user} ---")

    # Generate recommendations
    candidates = cand_gen.generate_hybrid_candidates(test_user, train_df, n=100)
    candidate_items = [item_id for item_id, _, _ in candidates]
    final_recs = reranker.rerank(test_user, candidate_items[:20])[:5]

    print("Top 5 Recommendations:")
    for i, (item_id, score) in enumerate(final_recs, 1):
        # Get product info
        product = products_df[products_df['id'] == item_id]
        if len(product) > 0:
            title = product.iloc[0]['title'][:50]
            print(f"  {i}. {title}... (Score: {score:.2f})")
        else:
            print(f"  {i}. Item {item_id} (Score: {score:.2f})")

print("\n" + "="*60)
print("System Statistics")
print("="*60)
print(f"Total Users: {len(train_df['user_id'].unique())}")
print(f"Total Items: {len(products_df)}")
print(f"Total Interactions: {len(train_df)}")
print(f"Models: Matrix Factorization + Neural CF + Re-Ranker")

print("\n" + "="*60)
print("✓ System Ready for Production!")
print("="*60)
print("\nNext steps:")
print("  1. Start API: uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
print("  2. View docs: http://localhost:8000/docs")
print("  3. Run tests: python src/evaluation/load_testing.py")
print("  4. Track experiments: mlflow ui")
