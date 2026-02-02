"""MLflow experiment tracking integration."""

import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Dict, Any
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


class MLflowTracker:
    """MLflow experiment tracker for recommender system."""

    def __init__(self, config=None):
        """Initialize MLflow tracker.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()

        # Setup MLflow
        self.experiment_name = self.config.get('mlflow.experiment_name', 'multimodal_recommender')
        self.tracking_uri = self.config.get('mlflow.tracking_uri', 'mlruns')

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

        mlflow.set_experiment(self.experiment_name)

        self.active_run = None

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Additional tags for the run
        """
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags)
        return self.active_run

    def end_run(self):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"Error logging param {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for key, value in metrics.items():
            try:
                # Skip non-numeric metrics
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                print(f"Error logging metric {key}: {e}")

    def log_artifact(self, file_path: str):
        """Log artifact to MLflow.

        Args:
            file_path: Path to artifact file
        """
        try:
            mlflow.log_artifact(file_path)
        except Exception as e:
            print(f"Error logging artifact {file_path}: {e}")

    def log_model(self, model, model_name: str, framework: str = 'pytorch'):
        """Log model to MLflow.

        Args:
            model: Model object
            model_name: Name for the model
            framework: Model framework ('pytorch', 'sklearn', etc.)
        """
        try:
            if framework == 'pytorch':
                mlflow.pytorch.log_model(model, model_name)
            elif framework == 'sklearn':
                mlflow.sklearn.log_model(model, model_name)
            else:
                print(f"Unsupported framework: {framework}")
        except Exception as e:
            print(f"Error logging model: {e}")

    def log_config(self, config: Dict[str, Any]):
        """Log configuration as JSON artifact.

        Args:
            config: Configuration dictionary
        """
        try:
            config_path = "mlruns/temp_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.log_artifact(config_path)
        except Exception as e:
            print(f"Error logging config: {e}")


class ExperimentRunner:
    """Run and track experiments for recommender models."""

    def __init__(self, config=None):
        """Initialize experiment runner.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.tracker = MLflowTracker(config)

    def run_cf_experiment(self, model_type: str = 'matrix_factorization'):
        """Run collaborative filtering experiment.

        Args:
            model_type: Type of CF model ('matrix_factorization' or 'neural_cf')
        """
        import pandas as pd
        import pickle

        print(f"\n{'='*60}")
        print(f"Running {model_type.upper()} Experiment")
        print(f"{'='*60}\n")

        # Start MLflow run
        self.tracker.start_run(run_name=f"{model_type}_experiment")

        try:
            # Load data
            train_df = pd.read_csv("data/processed/interactions_train.csv")
            val_df = pd.read_csv("data/processed/interactions_val.csv")
            test_df = pd.read_csv("data/processed/interactions_test.csv")

            with open("data/processed/user_mapping.pkl", 'rb') as f:
                user_mapping = pickle.load(f)
            with open("data/processed/item_mapping.pkl", 'rb') as f:
                item_mapping = pickle.load(f)

            # Log data statistics
            self.tracker.log_params({
                'n_train_interactions': len(train_df),
                'n_val_interactions': len(val_df),
                'n_test_interactions': len(test_df),
                'n_users': len(user_mapping['to_idx']),
                'n_items': len(item_mapping['to_idx'])
            })

            # Train model
            if model_type == 'matrix_factorization':
                from src.models.matrix_factorization import MatrixFactorizationCF

                model = MatrixFactorizationCF(self.config)

                # Log model parameters
                self.tracker.log_params({
                    'model_type': 'matrix_factorization',
                    'factors': model.factors,
                    'regularization': model.regularization,
                    'iterations': model.iterations,
                    'alpha': model.alpha
                })

                model.train(train_df, user_mapping, item_mapping)

            elif model_type == 'neural_cf':
                from src.models.neural_cf import NeuralCFTrainer

                model = NeuralCFTrainer(self.config)

                # Log model parameters
                self.tracker.log_params({
                    'model_type': 'neural_cf',
                    'embedding_dim': model.embedding_dim,
                    'hidden_layers': str(model.hidden_layers),
                    'dropout': model.dropout,
                    'learning_rate': model.learning_rate,
                    'batch_size': model.batch_size,
                    'epochs': model.epochs
                })

                model.train(train_df, val_df, user_mapping, item_mapping)

            # Evaluate model
            from src.evaluation.metrics import RecommenderEvaluator

            evaluator = RecommenderEvaluator(k_values=[5, 10, 20])

            def recommender_func(user_id, n):
                return model.recommend(user_id, n=n, interactions_df=train_df)

            avg_metrics, user_metrics = evaluator.evaluate_model(
                recommender_func, test_df, train_df, n_recommendations=20
            )

            # Log metrics
            self.tracker.log_metrics(avg_metrics)

            # Print results
            evaluator.print_metrics(avg_metrics, title=f"{model_type.upper()} Results")

            # Save results
            results_path = f"experiments/{model_type}_results.csv"
            Path("experiments").mkdir(exist_ok=True)
            user_metrics.to_csv(results_path, index=False)
            self.tracker.log_artifact(results_path)

            print(f"\nExperiment complete! Results saved to MLflow.")

        except Exception as e:
            print(f"Error during experiment: {e}")
            raise

        finally:
            self.tracker.end_run()

    def run_pipeline_experiment(self):
        """Run complete two-stage pipeline experiment."""
        import pandas as pd
        import pickle
        import numpy as np

        print(f"\n{'='*60}")
        print("Running Two-Stage Pipeline Experiment")
        print(f"{'='*60}\n")

        # Start MLflow run
        self.tracker.start_run(run_name="two_stage_pipeline")

        try:
            # Load data
            test_df = pd.read_csv("data/processed/interactions_test.csv")
            train_df = pd.read_csv("data/processed/interactions_train.csv")

            # Load models
            from src.models.candidate_generation import CandidateGenerator
            from src.models.reranking import ReRanker

            print("Loading models...")
            candidate_gen = CandidateGenerator.load("models/candidate_generation/candidate_generator.pkl")
            candidate_gen.load_cf_model("models/collaborative_filtering/matrix_factorization.pkl")

            reranker_path = Path("models/reranking/reranker_best.pt")
            if reranker_path.exists():
                reranker = ReRanker.load(reranker_path)
            else:
                print("Re-ranker not found, using candidate generation only")
                reranker = None

            # Log parameters
            self.tracker.log_params({
                'pipeline': 'two_stage',
                'num_candidates': candidate_gen.num_candidates,
                'use_reranker': reranker is not None
            })

            # Create recommender function
            def pipeline_recommender(user_id, n):
                # Stage 1: Candidate generation
                candidates = candidate_gen.generate_hybrid_candidates(
                    user_id, train_df, n=100
                )
                candidate_items = [item_id for item_id, _, _ in candidates]

                # Stage 2: Re-ranking
                if reranker:
                    ranked = reranker.rerank(user_id, candidate_items)
                    return ranked[:n]
                else:
                    return candidates[:n]

            # Evaluate
            from src.evaluation.metrics import RecommenderEvaluator

            evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
            avg_metrics, user_metrics = evaluator.evaluate_model(
                pipeline_recommender, test_df, train_df, n_recommendations=20
            )

            # Log metrics
            self.tracker.log_metrics(avg_metrics)

            # Print results
            evaluator.print_metrics(avg_metrics, title="Two-Stage Pipeline Results")

            # Save results
            results_path = "experiments/pipeline_results.csv"
            user_metrics.to_csv(results_path, index=False)
            self.tracker.log_artifact(results_path)

            print(f"\nPipeline experiment complete!")

        except Exception as e:
            print(f"Error during pipeline experiment: {e}")
            raise

        finally:
            self.tracker.end_run()


def main():
    """Run all experiments."""
    runner = ExperimentRunner()

    # Run CF experiments
    print("\n" + "="*60)
    print("Starting Experiment Suite")
    print("="*60)

    try:
        runner.run_cf_experiment('matrix_factorization')
    except Exception as e:
        print(f"Matrix Factorization experiment failed: {e}")

    try:
        runner.run_cf_experiment('neural_cf')
    except Exception as e:
        print(f"Neural CF experiment failed: {e}")

    try:
        runner.run_pipeline_experiment()
    except Exception as e:
        print(f"Pipeline experiment failed: {e}")

    print("\n" + "="*60)
    print("All Experiments Complete!")
    print("="*60)
    print("\nView results with: mlflow ui")
    print("Then navigate to http://localhost:5000")


if __name__ == "__main__":
    main()
