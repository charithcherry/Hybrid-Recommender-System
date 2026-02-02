"""Master script to run the complete multimodal recommender system pipeline."""

import sys
import argparse
from pathlib import Path


def run_data_preparation():
    """Step 1: Download and prepare dataset."""
    print("\n" + "="*60)
    print("STEP 1: Data Preparation")
    print("="*60)

    from src.data.download_data import main as download_main
    download_main()

    from src.data.preprocessing import main as preprocess_main
    preprocess_main()


def run_embedding_extraction():
    """Step 2: Extract CLIP embeddings."""
    print("\n" + "="*60)
    print("STEP 2: Embedding Extraction")
    print("="*60)

    from src.embeddings.clip_embeddings import extract_product_embeddings
    extract_product_embeddings()


def run_model_training():
    """Step 3: Train all models."""
    print("\n" + "="*60)
    print("STEP 3: Model Training")
    print("="*60)

    # Train Matrix Factorization
    print("\n--- Training Matrix Factorization ---")
    from src.models.matrix_factorization import train_matrix_factorization
    train_matrix_factorization()

    # Train Neural CF
    print("\n--- Training Neural CF ---")
    try:
        from src.models.neural_cf import train_neural_cf
        train_neural_cf()
    except Exception as e:
        print(f"Neural CF training failed (optional): {e}")

    # Build Candidate Generator
    print("\n--- Building Candidate Generator ---")
    from src.models.candidate_generation import build_candidate_generator
    build_candidate_generator()

    # Train Re-ranker
    print("\n--- Training Re-ranker ---")
    try:
        from src.models.reranking import train_reranker
        train_reranker()
    except Exception as e:
        print(f"Re-ranker training failed (optional): {e}")


def run_evaluation():
    """Step 4: Evaluate models with MLflow tracking."""
    print("\n" + "="*60)
    print("STEP 4: Model Evaluation")
    print("="*60)

    from src.evaluation.mlflow_tracking import main as mlflow_main
    mlflow_main()


def run_visualization():
    """Step 5: Generate visualizations."""
    print("\n" + "="*60)
    print("STEP 5: Visualization")
    print("="*60)

    from src.evaluation.visualization import visualize_evaluation_results
    visualize_evaluation_results()


def run_complete_pipeline():
    """Run the complete pipeline."""
    print("\n" + "="*60)
    print("MULTIMODAL RECOMMENDER SYSTEM - COMPLETE PIPELINE")
    print("="*60)

    try:
        # Step 1: Data Preparation
        run_data_preparation()

        # Step 2: Embedding Extraction
        run_embedding_extraction()

        # Step 3: Model Training
        run_model_training()

        # Step 4: Evaluation
        run_evaluation()

        # Step 5: Visualization
        run_visualization()

        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. View MLflow results: mlflow ui")
        print("     Then navigate to http://localhost:5000")
        print("\n  2. Start API server: uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        print("     Then navigate to http://localhost:8000/docs")
        print("\n  3. Run load tests: python src/evaluation/load_testing.py")
        print("\n  4. View visualizations in: experiments/visualizations/")

    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        raise


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Multimodal Recommender System Pipeline"
    )

    parser.add_argument(
        '--step',
        type=str,
        choices=['data', 'embeddings', 'train', 'evaluate', 'visualize', 'all'],
        default='all',
        help='Which step to run (default: all)'
    )

    args = parser.parse_args()

    # Run specified step
    if args.step == 'data':
        run_data_preparation()
    elif args.step == 'embeddings':
        run_embedding_extraction()
    elif args.step == 'train':
        run_model_training()
    elif args.step == 'evaluate':
        run_evaluation()
    elif args.step == 'visualize':
        run_visualization()
    elif args.step == 'all':
        run_complete_pipeline()


if __name__ == "__main__":
    main()
