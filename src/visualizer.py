"""
Generates all plots for the thematic alignment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Shared Color Palette ──────────────────────────────────
MAIN_COLOR = "#2E86AB"
ACCENT_COLOR = "#E84855"
GRAY_COLOR = "#6B6B6B"

TOPIC_COLORS = {
    0: "#2E86AB",
    1: "#F18F01",
    2: "#C73E1D",
    3: "#3B1F2B",
    4: "#44BBA4",
    5: "#E94F37",
    6: "#393E41",
    7: "#A4036F",
    8: "#048A81",
}


class Visualizer:
    """
    Creates all visualizations for the alignment analysis.
    """

    def __init__(self, output_dir: str = "figures"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def plot_score_distribution(self, df: pd.DataFrame, threshold: float = None) -> None:
        """
        Plots the distribution of alignment scores, highlighting outliers if threshold is provided.
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        # Compute shared bins
        bin_edges = np.linspace(df["alignment_score"].min(), df["alignment_score"].max(), 41)

        if threshold is not None:
            outliers = df[df["alignment_score"] < threshold]
            normal = df[df["alignment_score"] >= threshold]

            sns.histplot(
                normal["alignment_score"], bins=bin_edges,
                color=MAIN_COLOR, label=f"Normal papers ({len(normal)})",
                ax=ax, kde=True
            )
            sns.histplot(
                outliers["alignment_score"], bins=bin_edges,
                color=ACCENT_COLOR,
                label=f"Outliers ({len(outliers)}, {len(outliers)/len(df)*100:.1f}%)",
                ax=ax, kde=False
            )
            ax.axvline(
                threshold, color=ACCENT_COLOR, linestyle="--", linewidth=2,
                label=f"Threshold: {threshold:.3f}"
            )
        else:
            sns.histplot(
                df["alignment_score"], bins=bin_edges, kde=True,
                color=MAIN_COLOR, ax=ax
            )

        ax.axvline(
            df["alignment_score"].mean(), color="black", linestyle="--",
            linewidth=2, label=f"Mean: {df['alignment_score'].mean():.3f}"
        )
        ax.axvline(
            df["alignment_score"].median(), color=GRAY_COLOR, linestyle=":",
            linewidth=2, label=f"Median: {df['alignment_score'].median():.3f}"
        )

        ax.set_title("Distribution of Thematic Alignment Scores", fontsize=14)
        ax.set_xlabel("Cosine Similarity with Aims & Scope")
        ax.set_ylabel("Number of Papers")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/score_distribution.png", dpi=150)
        plt.show()
        print("Saved: score_distribution.png")

    def plot_drift_over_time(self, df: pd.DataFrame) -> None:
        """
        Plots mean and median alignment score per year.
        """
        yearly = (
            df.groupby("year")["alignment_score"]
            .agg(["mean", "median", "std"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            yearly["year"], yearly["mean"],
            marker="o", color=MAIN_COLOR,
            linewidth=2, label="Mean alignment score"
        )

        ax.plot(
            yearly["year"], yearly["median"],
            marker="s", color=ACCENT_COLOR,
            linewidth=2, linestyle="--", label="Median alignment score"
        )

        ax.fill_between(
            yearly["year"],
            yearly["mean"] - yearly["std"],
            yearly["mean"] + yearly["std"],
            alpha=0.15, color=MAIN_COLOR, label="± 1 std"
        )

        ax.axhline(
            df["alignment_score"].mean(),
            color=GRAY_COLOR, linestyle=":", alpha=0.7, label="Corpus mean"
        )

        ax.set_title("Thematic Alignment Drift Over Time", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Cosine Similarity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/drift_over_time.png", dpi=150)
        plt.show()
        print("Saved: drift_over_time.png")

    def plot_outliers(self, top_df: pd.DataFrame, bottom_df: pd.DataFrame) -> None:
        """
        Plots top and bottom papers by alignment score.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        for ax, data, title, color in zip(
            axes,
            [top_df, bottom_df],
            ["Top 5 Most Aligned Papers", "Bottom 5 Least Aligned Papers"],
            [MAIN_COLOR, ACCENT_COLOR]
        ):
            labels = [t[:50] + "..." if len(t) > 50 else t for t in data["title"]]
            ax.barh(labels, data["alignment_score"], color=color)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Alignment Score")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/outliers.png", dpi=150)
        plt.show()
        print("Saved: outliers.png")

    def plot_outlier_distribution(self, df: pd.DataFrame, threshold: float) -> None:
        """
        Plots alignment score distribution highlighting outlier papers.
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        outliers = df[df["alignment_score"] < threshold]
        normal = df[df["alignment_score"] >= threshold]

        sns.histplot(
            normal["alignment_score"], bins=35,
            color=MAIN_COLOR, label=f"Normal papers ({len(normal)})",
            ax=ax, kde=False
        )
        sns.histplot(
            outliers["alignment_score"], bins=10,
            color=ACCENT_COLOR,
            label=f"Outliers ({len(outliers)}, {len(outliers)/len(df)*100:.1f}%)",
            ax=ax, kde=False
        )

        ax.axvline(
            threshold, color=ACCENT_COLOR, linestyle="--", linewidth=2,
            label=f"Threshold: {threshold:.3f}"
        )
        ax.axvline(
            df["alignment_score"].mean(), color=GRAY_COLOR, linestyle=":",
            linewidth=2, label=f"Mean: {df['alignment_score'].mean():.3f}"
        )

        ax.set_title("Alignment Score Distribution with Outlier Threshold", fontsize=14)
        ax.set_xlabel("Cosine Similarity with Aims & Scope")
        ax.set_ylabel("Number of Papers")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/outlier_distribution.png", dpi=150)
        plt.show()
        print("Saved: outlier_distribution.png")

    def plot_topics_over_time(self, topics_over_time: pd.DataFrame, topic_names: dict = None) -> None:
        """
        Plots BERTopic topic frequency evolution over time.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        for topic_id in sorted(topics_over_time["Topic"].unique()):
            if topic_id == -1:
                continue
            subset = topics_over_time[topics_over_time["Topic"] == topic_id]
            name = topic_names.get(topic_id, f"Topic {topic_id}") if topic_names else f"Topic {topic_id}"
            ax.plot(
                subset["Timestamp"], subset["Frequency"],
                label=name,
                color=TOPIC_COLORS.get(topic_id, GRAY_COLOR),
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

    def plot_drift_by_topic(self, df: pd.DataFrame, topic_model, topic_names: dict) -> None:
        """
        Plots mean alignment score per year for each topic.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        yearly_mean = df.groupby("year")["alignment_score"].mean()
        ax.plot(
            yearly_mean.index, yearly_mean.values,
            color="black", linewidth=3, marker="o",
            label="Overall mean", zorder=5
        )

        yearly_median = df.groupby("year")["alignment_score"].median()
        ax.plot(
            yearly_median.index, yearly_median.values,
            color="black", linewidth=1.5, marker="o",
            linestyle="--", label="Overall median", zorder=5
        )

        valid_topics = sorted([t for t in df["topic"].unique() if t != -1])

        for topic_id in valid_topics:
            topic_yearly = (
                df[df["topic"] == topic_id]
                .groupby("year")["alignment_score"]
                .mean()
            )
            name = topic_names.get(topic_id, f"Topic {topic_id}")
            ax.plot(
                topic_yearly.index, topic_yearly.values,
                color=TOPIC_COLORS.get(topic_id, GRAY_COLOR),
                linewidth=1.2, marker=".", alpha=0.8, label=name
            )

        ax.axhline(
            df["alignment_score"].mean(),
            color=GRAY_COLOR, linestyle=":", alpha=0.5, label="Corpus mean"
        )

        ax.set_title("Thematic Alignment Drift Over Time by Topic", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/drift_by_topic.png", dpi=150)
        plt.show()
        print("Saved: drift_by_topic.png")