"""CLIP embeddings for multimodal recommendation."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
from typing import List, Union
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


class CLIPEmbeddingExtractor:
    """Extract CLIP embeddings for text and images."""

    def __init__(self, config=None):
        """Initialize CLIP model.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.device = torch.device(
            self.config.get('embeddings.device', 'cuda')
            if torch.cuda.is_available()
            else 'cpu'
        )

        print(f"Using device: {self.device}")

        # Load CLIP model
        model_name = self.config.get('embeddings.model_name', 'openai/clip-vit-base-patch32')
        print(f"Loading CLIP model: {model_name}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()

        self.embedding_dim = self.config.get('embeddings.embedding_dim', 512)
        self.batch_size = self.config.get('embeddings.batch_size', 32)

        print(f"CLIP model loaded successfully")

    def extract_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings for text descriptions.

        Args:
            texts: List of text strings

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting text embeddings"):
                batch_texts = texts[i:i + self.batch_size]

                # Process texts
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)

                # Get embeddings
                text_features = self.model.get_text_features(**inputs)

                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                embeddings.append(text_features.cpu().numpy())

        embeddings = np.vstack(embeddings)
        return embeddings

    def extract_image_embeddings(self, images: List[Union[Image.Image, str]]) -> np.ndarray:
        """Extract embeddings for images.

        Args:
            images: List of PIL Images or image paths

        Returns:
            Array of shape (n_images, embedding_dim)
        """
        embeddings = []

        # Load images if paths are provided
        loaded_images = []
        for img in tqdm(images, desc="Loading images"):
            if isinstance(img, str):
                try:
                    loaded_images.append(Image.open(img).convert('RGB'))
                except Exception as e:
                    print(f"Error loading image {img}: {e}")
                    # Create blank image as fallback
                    loaded_images.append(Image.new('RGB', (224, 224), color='white'))
            else:
                loaded_images.append(img)

        with torch.no_grad():
            for i in tqdm(range(0, len(loaded_images), self.batch_size), desc="Extracting image embeddings"):
                batch_images = loaded_images[i:i + self.batch_size]

                # Process images
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                # Get embeddings
                image_features = self.model.get_image_features(**inputs)

                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                embeddings.append(image_features.cpu().numpy())

        embeddings = np.vstack(embeddings)
        return embeddings

    def extract_multimodal_embeddings(self, texts: List[str],
                                     images: List[Union[Image.Image, str]] = None,
                                     fusion: str = 'concat') -> np.ndarray:
        """Extract combined multimodal embeddings.

        Args:
            texts: List of text descriptions
            images: List of images (optional)
            fusion: Fusion strategy ('concat', 'mean', 'max')

        Returns:
            Array of combined embeddings
        """
        print(f"Extracting multimodal embeddings with fusion={fusion}")

        # Extract text embeddings
        text_embeddings = self.extract_text_embeddings(texts)

        if images is None:
            return text_embeddings

        # Extract image embeddings
        image_embeddings = self.extract_image_embeddings(images)

        # Fuse embeddings
        if fusion == 'concat':
            multimodal_embeddings = np.concatenate([text_embeddings, image_embeddings], axis=1)
        elif fusion == 'mean':
            multimodal_embeddings = (text_embeddings + image_embeddings) / 2
        elif fusion == 'max':
            multimodal_embeddings = np.maximum(text_embeddings, image_embeddings)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion}")

        return multimodal_embeddings

    def compute_similarity(self, query_embedding: np.ndarray,
                          item_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and items.

        Args:
            query_embedding: Query embedding of shape (embedding_dim,) or (1, embedding_dim)
            item_embeddings: Item embeddings of shape (n_items, embedding_dim)

        Returns:
            Similarity scores of shape (n_items,)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        items_norm = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(query_norm, items_norm.T).squeeze()

        return similarities


class EmbeddingCache:
    """Cache for pre-computed embeddings."""

    def __init__(self, cache_dir: Path = None):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = cache_dir or Path("data/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, embeddings: np.ndarray, name: str):
        """Save embeddings to cache.

        Args:
            embeddings: Embedding array
            name: Cache name
        """
        cache_path = self.cache_dir / f"{name}.npy"
        np.save(cache_path, embeddings)
        print(f"Saved embeddings to {cache_path}")

    def load(self, name: str) -> np.ndarray:
        """Load embeddings from cache.

        Args:
            name: Cache name

        Returns:
            Embedding array or None if not found
        """
        cache_path = self.cache_dir / f"{name}.npy"
        if cache_path.exists():
            print(f"Loading embeddings from {cache_path}")
            return np.load(cache_path)
        return None

    def exists(self, name: str) -> bool:
        """Check if cache exists.

        Args:
            name: Cache name

        Returns:
            True if cache exists
        """
        cache_path = self.cache_dir / f"{name}.npy"
        return cache_path.exists()


def extract_product_embeddings():
    """Extract and cache embeddings for all products."""
    print("=" * 60)
    print("Extracting Product Embeddings")
    print("=" * 60)

    # Load processed products
    products_df = pd.read_csv("data/processed/products_processed.csv")
    print(f"Loaded {len(products_df)} products")

    # Initialize extractor and cache
    extractor = CLIPEmbeddingExtractor()
    cache = EmbeddingCache()

    # Extract text embeddings
    if not cache.exists("product_text_embeddings"):
        texts = products_df['combined_text'].tolist()
        text_embeddings = extractor.extract_text_embeddings(texts)
        cache.save(text_embeddings, "product_text_embeddings")
    else:
        print("Text embeddings already cached")
        text_embeddings = cache.load("product_text_embeddings")

    print(f"\nText embeddings shape: {text_embeddings.shape}")

    # For synthetic data, we'll use text embeddings only
    # In a real scenario with images, you would also extract image embeddings

    print("\n" + "=" * 60)
    print("Embedding extraction complete!")
    print("=" * 60)

    return text_embeddings


if __name__ == "__main__":
    extract_product_embeddings()
