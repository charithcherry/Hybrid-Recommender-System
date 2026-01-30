"""Verify that the multimodal recommender system is set up correctly."""

import sys
from pathlib import Path
from typing import List, Tuple


class SetupVerifier:
    """Verify project setup and dependencies."""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def check_dependency(self, module_name: str, package_name: str = None) -> bool:
        """Check if a Python package is installed.

        Args:
            module_name: Name of the module to import
            package_name: Display name (if different from module)

        Returns:
            True if package is installed
        """
        package_name = package_name or module_name
        try:
            __import__(module_name)
            print(f"[OK] {package_name} is installed")
            self.checks_passed += 1
            return True
        except ImportError:
            print(f"[FAIL] {package_name} is NOT installed")
            self.checks_failed += 1
            return False

    def check_directory(self, dir_path: str) -> bool:
        """Check if a directory exists.

        Args:
            dir_path: Path to directory

        Returns:
            True if directory exists
        """
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"[OK] Directory exists: {dir_path}")
            self.checks_passed += 1
            return True
        else:
            print(f"[FAIL] Directory missing: {dir_path}")
            self.checks_failed += 1
            return False

    def check_file(self, file_path: str, optional: bool = False) -> bool:
        """Check if a file exists.

        Args:
            file_path: Path to file
            optional: Whether the file is optional

        Returns:
            True if file exists or is optional
        """
        path = Path(file_path)
        if path.exists() and path.is_file():
            print(f"[OK] File exists: {file_path}")
            self.checks_passed += 1
            return True
        else:
            if optional:
                print(f"[WARN] Optional file missing: {file_path}")
                self.warnings += 1
                return True
            else:
                print(f"[FAIL] File missing: {file_path}")
                self.checks_failed += 1
                return False

    def verify_dependencies(self):
        """Verify all required Python packages."""
        print("\n" + "=" * 60)
        print("Checking Python Dependencies")
        print("=" * 60 + "\n")

        dependencies = [
            ('torch', 'PyTorch'),
            ('torchvision', 'torchvision'),
            ('transformers', 'Transformers'),
            ('PIL', 'Pillow'),
            ('fastapi', 'FastAPI'),
            ('uvicorn', 'Uvicorn'),
            ('pydantic', 'Pydantic'),
            ('mlflow', 'MLflow'),
            ('pandas', 'pandas'),
            ('numpy', 'numpy'),
            ('sklearn', 'scikit-learn'),
            ('scipy', 'scipy'),
            ('implicit', 'implicit'),
            ('faiss', 'faiss-cpu'),
            ('matplotlib', 'matplotlib'),
            ('seaborn', 'seaborn'),
            ('plotly', 'plotly'),
            ('tqdm', 'tqdm'),
            ('requests', 'requests'),
            ('yaml', 'PyYAML'),
        ]

        for module, name in dependencies:
            self.check_dependency(module, name)

    def verify_structure(self):
        """Verify project directory structure."""
        print("\n" + "=" * 60)
        print("Checking Project Structure")
        print("=" * 60 + "\n")

        directories = [
            'data/raw',
            'data/processed',
            'data/embeddings',
            'models/collaborative_filtering',
            'models/candidate_generation',
            'models/reranking',
            'src/data',
            'src/embeddings',
            'src/models',
            'src/evaluation',
            'src/api',
            'config',
            'experiments',
            'mlruns',
        ]

        for directory in directories:
            self.check_directory(directory)

    def verify_config(self):
        """Verify configuration files."""
        print("\n" + "=" * 60)
        print("Checking Configuration Files")
        print("=" * 60 + "\n")

        files = [
            'config/config.yaml',
            'config/config_loader.py',
            'requirements.txt',
            'README.md',
            'GETTING_STARTED.md',
            'run_pipeline.py',
        ]

        for file in files:
            self.check_file(file)

    def verify_data(self):
        """Verify data files (optional checks)."""
        print("\n" + "=" * 60)
        print("Checking Data Files (Optional)")
        print("=" * 60 + "\n")

        optional_files = [
            'data/raw/products.csv',
            'data/processed/interactions_train.csv',
            'data/processed/interactions_val.csv',
            'data/processed/interactions_test.csv',
            'data/processed/products_processed.csv',
            'data/processed/user_mapping.pkl',
            'data/processed/item_mapping.pkl',
            'data/embeddings/product_text_embeddings.npy',
        ]

        for file in optional_files:
            self.check_file(file, optional=True)

    def verify_models(self):
        """Verify trained models (optional checks)."""
        print("\n" + "=" * 60)
        print("Checking Trained Models (Optional)")
        print("=" * 60 + "\n")

        optional_models = [
            'models/collaborative_filtering/matrix_factorization.pkl',
            'models/candidate_generation/candidate_generator.pkl',
            'models/reranking/reranker_best.pt',
        ]

        for model in optional_models:
            self.check_file(model, optional=True)

    def verify_torch_cuda(self):
        """Check CUDA availability."""
        print("\n" + "=" * 60)
        print("Checking CUDA Support")
        print("=" * 60 + "\n")

        try:
            import torch
            if torch.cuda.is_available():
                print(f"[OK] CUDA is available")
                print(f"  CUDA version: {torch.version.cuda}")
                print(f"  GPU device: {torch.cuda.get_device_name(0)}")
                self.checks_passed += 1
            else:
                print(f"[WARN] CUDA is not available (will use CPU)")
                print(f"  This is fine but training will be slower")
                self.warnings += 1
        except Exception as e:
            print(f"[WARN] Error checking CUDA: {e}")
            self.warnings += 1

    def print_summary(self):
        """Print verification summary."""
        print("\n" + "=" * 60)
        print("Verification Summary")
        print("=" * 60 + "\n")

        print(f"[OK] Checks passed: {self.checks_passed}")
        if self.warnings > 0:
            print(f"[WARN] Warnings: {self.warnings}")
        if self.checks_failed > 0:
            print(f"[FAIL] Checks failed: {self.checks_failed}")

        print("\n" + "-" * 60)

        if self.checks_failed == 0:
            print("[OK] Setup verification PASSED!")
            print("\nYou can now run the pipeline:")
            print("  python run_pipeline.py --step all")
        else:
            print("[FAIL] Setup verification FAILED!")
            print("\nPlease fix the errors above before proceeding.")
            print("\nTo install missing dependencies:")
            print("  pip install -r requirements.txt")

        if self.warnings > 0:
            print("\n[WARN] Note: Warnings indicate optional components that are missing.")
            print("   The system will still work, but some features may be limited.")

        print("\n" + "=" * 60)

    def run_all_checks(self):
        """Run all verification checks."""
        print("\n" + "=" * 60)
        print("MULTIMODAL RECOMMENDER SYSTEM - Setup Verification")
        print("=" * 60)

        self.verify_dependencies()
        self.verify_structure()
        self.verify_config()
        self.verify_torch_cuda()
        self.verify_data()
        self.verify_models()
        self.print_summary()


def main():
    """Main verification function."""
    verifier = SetupVerifier()
    verifier.run_all_checks()


if __name__ == "__main__":
    main()
