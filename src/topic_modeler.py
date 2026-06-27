"""
Applies BERTopic to discover latent topics in the paper abstracts
and tracks how topic distributions evolve over time.
"""

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


class TopicModeler:
    """
    Discovers topics in a corpus of abstracts using BERTopic.
    """

    def __init__(self, n_topics: int = 20, language: str = "english"):
        """
        Args:
            n_topics: number of topics to extract (approximate)
            language: language of the corpus
        """
        vectorizer = CountVectorizer(
            stop_words=language,
            min_df=2,
            ngram_range=(1, 2)
        )

        self.model = BERTopic(
            language=language,
            nr_topics=n_topics,
            vectorizer_model=vectorizer,
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

        Args:
            df: DataFrame with paper metadata

        Returns:
            DataFrame with new 'topic' column
        """
        df = df.copy()
        df["topic"] = self.topics
        return df

    def get_topics_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes how topic frequencies evolve year by year.

        Args:
            df: DataFrame with 'topic' and 'year' columns

        Returns:
            DataFrame with topic distribution per year
        """
        abstracts = df["abstract"].fillna("").tolist()
        timestamps = df["year"].tolist()

        topics_over_time = self.model.topics_over_time(
            abstracts,
            timestamps,
            nr_bins=10,
        )
        return topics_over_time