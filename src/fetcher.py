"""
Fetches paper abstracts from Semantic Scholar API
for a given journal (venue) and time range.
"""

import time
import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class SemanticScholarFetcher:
    """
    Fetches papers from Semantic Scholar API filtered by venue and year range.
    """

    BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    FIELDS = "title,abstract,year,venue"

    def __init__(self, venue: str, year_start: int, year_end: int, api_key: str = None):
        self.venue = venue
        self.year_start = year_start
        self.year_end = year_end
        self.api_key = api_key or os.getenv("S2_API_KEY")

    def fetch(self, max_papers: int = 1000) -> pd.DataFrame:
        """
        Fetches papers using bulk endpoint and returns a cleaned DataFrame.
        """
        papers = []
        token = None

        print(f"Fetching papers from '{self.venue}' ({self.year_start}–{self.year_end})...")

        while len(papers) < max_papers:
            params = {
                "query": "machine learning",
                "venue": self.venue,
                "fields": self.FIELDS,
                "limit": 100,
                "year": f"{self.year_start}-{self.year_end}",
            }

            if token:
                params["token"] = token

            headers = {"x-api-key": self.api_key} if self.api_key else {}

            while True:
                response = requests.get(
                    self.BULK_URL,
                    params=params,
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 429:
                    print("Rate limit hit, waiting 30 seconds...")
                    time.sleep(30)
                    continue

                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    return pd.DataFrame(papers)

                break

            data = response.json()
            batch = data.get("data", [])
            token = data.get("token")

            if not batch:
                print("No more results.")
                break

            for paper in batch:
                abstract = paper.get("abstract") or ""
                year = paper.get("year") or 0

                if len(abstract) > 50:
                    papers.append({
                        "paperId": paper.get("paperId", ""),
                        "title": paper.get("title", ""),
                        "abstract": abstract,
                        "year": year,
                        "venue": paper.get("venue", ""),
                    })

            print(f"  Collected: {len(papers)} papers")
            time.sleep(2)

            if not token:
                print("Reached end of results.")
                break

        df = pd.DataFrame(papers).drop_duplicates(subset="paperId")
        print(f"\nDone. Total papers collected: {len(df)}")
        return df

    def save(self, df: pd.DataFrame, path: str = "data/abstracts.csv") -> None:
        """
        Saves the DataFrame to a CSV file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved to {path}")