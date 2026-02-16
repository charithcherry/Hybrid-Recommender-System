"""CLIP text encoder for query embeddings."""

import numpy as np
from typing import List
import torch


class CLIPTextEncoder:
    """Lightweight CLIP text encoder for query embeddings."""

    def __init__(self):
        """Initialize CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel

            print("Loading CLIP model for text encoding...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
            print("CLIP model loaded successfully")

        except Exception as e:
            print(f"Error loading CLIP: {e}")
            self.model = None
            self.processor = None

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text query into 512-dim embedding.

        Args:
            text: Query text

        Returns:
            512-dim numpy array
        """
        if not self.model or not self.processor:
            # Return random embedding as fallback
            return np.random.rand(512).astype('float32')

        try:
            with torch.no_grad():
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )

                # Get text features
                text_features = self.model.get_text_features(**inputs)

                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                return text_features.cpu().numpy()[0]

        except Exception as e:
            print(f"Error encoding text: {e}")
            return np.random.rand(512).astype('float32')

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts at once."""
        if not self.model or not self.processor:
            return np.random.rand(len(texts), 512).astype('float32')

        try:
            with torch.no_grad():
                inputs = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )

                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                return text_features.cpu().numpy()

        except Exception as e:
            print(f"Error encoding batch: {e}")
            return np.random.rand(len(texts), 512).astype('float32')
