"""
Computes thematic alignment scores between paper abstracts
and the journal's Aims & Scope using cosine similarity.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class AlignmentScorer:
    """
    Computes cosine similarity between each paper embedding
    and the Aims & Scope embedding.
    """

    def __init__(self, scope_embedding: np.ndarray):
        """
        Args:
            scope_embedding: 1D numpy array representing the
                             encoded Aims & Scope text.
        """
        self.scope_embedding = scope_embedding.reshape(1, -1)

    def compute_scores(self, paper_embeddings: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity between each paper and the scope.

        Args:
            paper_embeddings: 2D array of shape (n_papers, embedding_dim)

        Returns:
            1D array of similarity scores in range [-1, 1]
        """
        scores = cosine_similarity(paper_embeddings, self.scope_embedding)
        return scores.flatten()

    def add_scores_to_df(self, df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
        """
        Adds alignment scores to the DataFrame.

        Args:
            df: DataFrame with paper metadata
            scores: 1D array of alignment scores

        Returns:
            DataFrame with new 'alignment_score' column
        """
        df = df.copy()
        df["alignment_score"] = scores
        return df

    def get_outliers(self, df: pd.DataFrame, n: int = 5) -> tuple:
        """
        Returns the top-n and bottom-n papers by alignment score.

        Args:
            df: DataFrame with 'alignment_score' column
            n: number of papers to return for each group

        Returns:
            tuple of (top_n_df, bottom_n_df)
        """
        sorted_df = df.sort_values("alignment_score", ascending=False)
        top = sorted_df.head(n)
        bottom = sorted_df.tail(n)
        return top, bottom
    def get_topic_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean, std and count of alignment scores per topic.

        Args:
            df: DataFrame with 'topic' and 'alignment_score' columns

        Returns:
            DataFrame with topic statistics
        """
        stats = (
            df[df["topic"] != -1]
            .groupby("topic")["alignment_score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={
                "mean": "avg_alignment",
                "std": "std_alignment",
                "count": "n_papers"
            })
            .sort_values("topic")
        )
        return stats