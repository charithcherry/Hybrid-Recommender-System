"""Visualization dashboard for ranking quality analysis."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class RankingVisualizer:
    """Visualize and analyze recommendation ranking quality."""

    def __init__(self, output_dir: str = "experiments/visualizations"):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_metrics_comparison(self, metrics_data: Dict[str, Dict],
                               output_file: str = "metrics_comparison.png"):
        """Plot comparison of metrics across models.

        Args:
            metrics_data: Dictionary mapping model names to metrics
            output_file: Output filename
        """
        # Prepare data
        models = list(metrics_data.keys())
        metric_names = ['precision@10', 'recall@10', 'ndcg@10', 'map', 'mrr']

        data = {metric: [metrics_data[model].get(metric, 0) for model in models]
                for metric in metric_names}

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metric_names):
            ax = axes[i]
            values = data[metric]

            bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=10)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Remove extra subplot
        fig.delaxes(axes[-1])

        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {output_path}")
        plt.close()

    def plot_metrics_at_k(self, user_metrics_df: pd.DataFrame,
                         output_file: str = "metrics_at_k.png"):
        """Plot metrics at different K values.

        Args:
            user_metrics_df: DataFrame with per-user metrics
            output_file: Output filename
        """
        k_values = [5, 10, 20]
        metrics = ['precision', 'recall', 'ndcg']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, metric in enumerate(metrics):
            ax = axes[i]

            mean_values = []
            std_values = []

            for k in k_values:
                col_name = f'{metric}@{k}'
                if col_name in user_metrics_df.columns:
                    mean_values.append(user_metrics_df[col_name].mean())
                    std_values.append(user_metrics_df[col_name].std())
                else:
                    mean_values.append(0)
                    std_values.append(0)

            ax.errorbar(k_values, mean_values, yerr=std_values,
                       marker='o', capsize=5, capthick=2, linewidth=2)
            ax.set_xlabel('K')
            ax.set_ylabel(f'{metric.capitalize()}@K')
            ax.set_title(f'{metric.capitalize()}@K vs K')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics@K plot to {output_path}")
        plt.close()

    def plot_user_distribution(self, user_metrics_df: pd.DataFrame,
                              metric: str = 'ndcg@10',
                              output_file: str = "user_metric_distribution.png"):
        """Plot distribution of metrics across users.

        Args:
            user_metrics_df: DataFrame with per-user metrics
            metric: Metric to plot
            output_file: Output filename
        """
        if metric not in user_metrics_df.columns:
            print(f"Metric {metric} not found in data")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        axes[0].hist(user_metrics_df[metric], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(user_metrics_df[metric].mean(), color='r',
                       linestyle='--', linewidth=2, label='Mean')
        axes[0].axvline(user_metrics_df[metric].median(), color='g',
                       linestyle='--', linewidth=2, label='Median')
        axes[0].set_xlabel(metric.upper())
        axes[0].set_ylabel('Number of Users')
        axes[0].set_title(f'Distribution of {metric.upper()} Across Users')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(user_metrics_df[metric], vert=True)
        axes[1].set_ylabel(metric.upper())
        axes[1].set_title(f'{metric.upper()} Box Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved user distribution plot to {output_path}")
        plt.close()

    def plot_ranking_quality_heatmap(self, user_metrics_df: pd.DataFrame,
                                    output_file: str = "ranking_quality_heatmap.png"):
        """Plot heatmap of ranking quality metrics.

        Args:
            user_metrics_df: DataFrame with per-user metrics
            output_file: Output filename
        """
        # Select metrics to include
        metric_cols = [col for col in user_metrics_df.columns
                      if '@' in col or col in ['map', 'mrr']]

        if not metric_cols:
            print("No metrics found for heatmap")
            return

        # Compute correlation matrix
        corr_matrix = user_metrics_df[metric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8})
        plt.title('Correlation Between Ranking Metrics')
        plt.tight_layout()

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
        plt.close()

    def create_interactive_dashboard(self, metrics_data: Dict[str, Dict],
                                    user_metrics_df: pd.DataFrame,
                                    output_file: str = "interactive_dashboard.html"):
        """Create interactive Plotly dashboard.

        Args:
            metrics_data: Dictionary mapping model names to metrics
            user_metrics_df: DataFrame with per-user metrics
            output_file: Output filename
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Metrics Comparison', 'Precision@K vs Recall@K',
                          'NDCG@K Distribution', 'User Metric Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                  [{"type": "box"}, {"type": "histogram"}]]
        )

        # 1. Metrics comparison
        models = list(metrics_data.keys())
        metrics = ['precision@10', 'recall@10', 'ndcg@10']

        for metric in metrics:
            values = [metrics_data[model].get(metric, 0) for model in models]
            fig.add_trace(
                go.Bar(name=metric, x=models, y=values),
                row=1, col=1
            )

        # 2. Precision-Recall scatter
        if 'precision@10' in user_metrics_df.columns and 'recall@10' in user_metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=user_metrics_df['recall@10'],
                    y=user_metrics_df['precision@10'],
                    mode='markers',
                    marker=dict(size=5, opacity=0.6),
                    name='Users'
                ),
                row=1, col=2
            )

        # 3. NDCG box plots
        if 'ndcg@10' in user_metrics_df.columns:
            fig.add_trace(
                go.Box(y=user_metrics_df['ndcg@10'], name='NDCG@10'),
                row=2, col=1
            )

        # 4. Metric histogram
        if 'ndcg@10' in user_metrics_df.columns:
            fig.add_trace(
                go.Histogram(x=user_metrics_df['ndcg@10'], name='NDCG@10'),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Ranking Quality Dashboard",
            showlegend=True,
            height=800
        )

        # Save
        output_path = self.output_dir / output_file
        fig.write_html(str(output_path))
        print(f"Saved interactive dashboard to {output_path}")

    def plot_learning_curves(self, training_history: Dict[str, List[float]],
                            output_file: str = "learning_curves.png"):
        """Plot model training learning curves.

        Args:
            training_history: Dictionary with 'train_loss' and 'val_loss' lists
            output_file: Output filename
        """
        if 'train_loss' not in training_history or 'val_loss' not in training_history:
            print("Training history must contain 'train_loss' and 'val_loss'")
            return

        epochs = range(1, len(training_history['train_loss']) + 1)

        plt.figure(figsize=(12, 6))

        plt.plot(epochs, training_history['train_loss'],
                marker='o', label='Training Loss', linewidth=2)
        plt.plot(epochs, training_history['val_loss'],
                marker='s', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {output_path}")
        plt.close()

    def generate_full_report(self, metrics_data: Dict[str, Dict],
                           user_metrics_df: pd.DataFrame):
        """Generate complete visualization report.

        Args:
            metrics_data: Dictionary mapping model names to metrics
            user_metrics_df: DataFrame with per-user metrics
        """
        print("\n" + "=" * 60)
        print("Generating Visualization Report")
        print("=" * 60)

        # 1. Metrics comparison
        print("\n1. Creating metrics comparison plot...")
        self.plot_metrics_comparison(metrics_data)

        # 2. Metrics@K
        print("2. Creating metrics@K plot...")
        self.plot_metrics_at_k(user_metrics_df)

        # 3. User distribution
        print("3. Creating user distribution plots...")
        for metric in ['precision@10', 'recall@10', 'ndcg@10']:
            if metric in user_metrics_df.columns:
                self.plot_user_distribution(
                    user_metrics_df, metric,
                    f"user_{metric.replace('@', '_at_')}_distribution.png"
                )

        # 4. Correlation heatmap
        print("4. Creating correlation heatmap...")
        self.plot_ranking_quality_heatmap(user_metrics_df)

        # 5. Interactive dashboard
        print("5. Creating interactive dashboard...")
        self.create_interactive_dashboard(metrics_data, user_metrics_df)

        print("\n" + "=" * 60)
        print(f"Visualization report complete!")
        print(f"All visualizations saved to: {self.output_dir}")
        print("=" * 60)


def visualize_evaluation_results():
    """Load and visualize evaluation results."""
    print("Loading evaluation results...")

    # Load evaluation results
    try:
        # Load results from different models
        metrics_data = {}
        user_metrics_df = None

        # Try to load Matrix Factorization results
        mf_path = Path("experiments/matrix_factorization_results.csv")
        if mf_path.exists():
            mf_df = pd.read_csv(mf_path)
            user_metrics_df = mf_df
            metrics_data['Matrix Factorization'] = {
                col: mf_df[col].mean()
                for col in mf_df.columns if col != 'user_id'
            }

        # Try to load Neural CF results
        ncf_path = Path("experiments/neural_cf_results.csv")
        if ncf_path.exists():
            ncf_df = pd.read_csv(ncf_path)
            metrics_data['Neural CF'] = {
                col: ncf_df[col].mean()
                for col in ncf_df.columns if col != 'user_id'
            }

        # Try to load Pipeline results
        pipeline_path = Path("experiments/pipeline_results.csv")
        if pipeline_path.exists():
            pipeline_df = pd.read_csv(pipeline_path)
            user_metrics_df = pipeline_df  # Use pipeline for detailed analysis
            metrics_data['Two-Stage Pipeline'] = {
                col: pipeline_df[col].mean()
                for col in pipeline_df.columns if col != 'user_id'
            }

        if not metrics_data:
            print("No evaluation results found. Please run experiments first.")
            return

        if user_metrics_df is None:
            print("No detailed user metrics found.")
            return

        # Create visualizations
        visualizer = RankingVisualizer()
        visualizer.generate_full_report(metrics_data, user_metrics_df)

    except Exception as e:
        print(f"Error loading results: {e}")
        raise


if __name__ == "__main__":
    visualize_evaluation_results()
