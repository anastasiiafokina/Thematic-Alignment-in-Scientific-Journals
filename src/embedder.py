"""
Transforms text into vector embeddings using Sentence-BERT.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Encodes text into dense vector embeddings using a pretrained
    Sentence-BERT model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model name for sentence-transformers.
                        'all-MiniLM-L6-v2' is fast and good quality.
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encodes a list of strings into embeddings.

        Args:
            texts: list of strings to encode
            batch_size: number of texts to encode at once

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_from_df(self, df: pd.DataFrame, text_col: str = "abstract") -> np.ndarray:
        """
        Encodes a text column from a DataFrame.

        Args:
            df: DataFrame containing the text column
            text_col: name of the column to encode

        Returns:
            numpy array of embeddings
        """
        texts = df[text_col].fillna("").tolist()
        return self.encode(texts)

    def save_embeddings(self, embeddings: np.ndarray, path: str = "data/embeddings.npy") -> None:
        """
        Saves embeddings to disk as a numpy file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        print(f"Embeddings saved to {path}")

    def load_embeddings(self, path: str = "data/embeddings.npy") -> np.ndarray:
        """
        Loads embeddings from disk.
        """
        embeddings = np.load(path)
        print(f"Embeddings loaded from {path} — shape: {embeddings.shape}")
        return embeddings