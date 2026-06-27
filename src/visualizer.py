"""
Generates all plots for the thematic alignment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """
    Creates all visualizations for the alignment analysis.
    """

    def __init__(self, output_dir: str = "figures"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def plot_score_distribution(self, df: pd.DataFrame) -> None:
        """
        Plots the distribution of alignment scores across all papers.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(
            df["alignment_score"],
            bins=40,
            kde=True,
            color="steelblue",
            ax=ax
        )

        ax.axvline(
            df["alignment_score"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {df['alignment_score'].mean():.3f}"
        )

        ax.set_title("Distribution of Thematic Alignment Scores", fontsize=14)
        ax.set_xlabel("Cosine Similarity with Aims & Scope")
        ax.set_ylabel("Number of Papers")
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/score_distribution.png", dpi=150)
        plt.show()
        print("Saved: score_distribution.png")

    def plot_drift_over_time(self, df: pd.DataFrame) -> None:
        """
        Plots mean alignment score per year to detect thematic drift.
        """
        yearly = (
            df.groupby("year")["alignment_score"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            yearly["year"],
            yearly["mean"],
            marker="o",
            color="steelblue",
            linewidth=2,
            label="Mean alignment score"
        )

        ax.fill_between(
            yearly["year"],
            yearly["mean"] - yearly["std"],
            yearly["mean"] + yearly["std"],
            alpha=0.2,
            color="steelblue",
            label="± 1 std"
        )

        ax.set_title("Thematic Alignment Drift Over Time", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/drift_over_time.png", dpi=150)
        plt.show()
        print("Saved: drift_over_time.png")

    def plot_outliers(self, top_df: pd.DataFrame, bottom_df: pd.DataFrame) -> None:
        """
        Plots top and bottom papers by alignment score as horizontal bar chart.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        for ax, data, title, color in zip(
            axes,
            [top_df, bottom_df],
            ["Top 5 Most Aligned Papers", "Bottom 5 Least Aligned Papers"],
            ["steelblue", "tomato"]
        ):
            labels = [t[:50] + "..." if len(t) > 50 else t for t in data["title"]]
            ax.barh(labels, data["alignment_score"], color=color)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Alignment Score")
            ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/outliers.png", dpi=150)
        plt.show()
        print("Saved: outliers.png")

    def plot_topics_over_time(self, topics_over_time: pd.DataFrame) -> None:
        """
        Plots BERTopic topic frequency evolution over time.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        for topic_id in topics_over_time["Topic"].unique():
            if topic_id == -1:
                continue
            subset = topics_over_time[topics_over_time["Topic"] == topic_id]
            ax.plot(
                subset["Timestamp"],
                subset["Frequency"],
                label=f"Topic {topic_id}",
                linewidth=1.5
            )

        ax.set_title("Topic Frequency Over Time (BERTopic)", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/topics_over_time.png", dpi=150)
        plt.show()
        print("Saved: topics_over_time.png")