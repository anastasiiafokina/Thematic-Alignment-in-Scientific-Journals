"""
Applies BERTopic to discover latent topics in the paper abstracts
and tracks how topic distributions evolve over time.
"""

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN


class TopicModeler:
    """
    Discovers topics in a corpus of abstracts using BERTopic.
    """

    def __init__(self, n_topics: int = 10, language: str = "english"):
        """
        Args:
            n_topics: number of topics to extract (approximate)
            language: language of the corpus
        """
        hdbscan_model = HDBSCAN(
            min_cluster_size=3,
            min_samples=1,
            prediction_data=True
        )

        vectorizer = CountVectorizer(
            stop_words=language,
            min_df=2,
            ngram_range=(1, 2)
        )

        self.model = BERTopic(
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics=n_topics,
            calculate_probabilities=True,
            verbose=True,
        )
        self.topics = None
        self.probs = None

    def fit(self, abstracts: list[str], embeddings=None):
        """
        Fits BERTopic on the abstracts.

        Args:
            abstracts: list of abstract strings
            embeddings: optional precomputed embeddings (speeds up fitting)

        Returns:
            tuple of (topics, probabilities)
        """
        self.topics, self.probs = self.model.fit_transform(
            abstracts,
            embeddings=embeddings
        )
        return self.topics, self.probs

    def get_topic_info(self) -> pd.DataFrame:
        """
        Returns a DataFrame with topic IDs, sizes, and top keywords.
        """
        return self.model.get_topic_info()

    def add_topics_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds topic assignments to the DataFrame.
        """
        df = df.copy()
        df["topic"] = self.topics
        return df

    def get_topics_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes how topic frequencies evolve year by year.
        """
        abstracts = df["abstract"].fillna("").tolist()
        timestamps = df["year"].tolist()

        topics_over_time = self.model.topics_over_time(
            abstracts,
            timestamps,
            nr_bins=10,
        )
        return topics_over_time