"""
Fetches paper abstracts from Semantic Scholar API
for a given journal (venue) and time range.
"""

import time
import requests
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()


class SemanticScholarFetcher:
    """
    Fetches papers from Semantic Scholar API filtered by venue and year range.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,venue,externalIds"

    def __init__(self, venue: str, year_start: int, year_end: int, api_key: str = None):
        self.venue = venue
        self.year_start = year_start
        self.year_end = year_end
        self.api_key = api_key

    def fetch(self, max_papers: int = 1000) -> pd.DataFrame:
        """
        Fetches papers and returns a cleaned DataFrame.
        """
        papers = []
        offset = 0
        limit = 100  # max allowed per request

        print(f"Fetching papers from '{self.venue}' ({self.year_start}–{self.year_end})...")

        while len(papers) < max_papers:
            params = {
                "query": "machine learning",
                "fields": self.FIELDS,
                "limit": limit,
                "offset": offset,
            }

            response = requests.get(self.BASE_URL, params=params, timeout=30)

            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break

            data = response.json()
            batch = data.get("data", [])

            if not batch:
                print("No more results.")
                break

            for paper in batch:
                venue = paper.get("venue", "") or ""
                year = paper.get("year") or 0
                abstract = paper.get("abstract") or ""

                # Filter by venue name and year range
                if (
                    "journal of machine learning research" in venue.lower()
                    and self.year_start <= year <= self.year_end
                    and len(abstract) > 50
                ):
                    papers.append({
                        "paperId": paper.get("paperId", ""),
                        "title": paper.get("title", ""),
                        "abstract": abstract,
                        "year": year,
                        "venue": venue,
                    })

            print(f"  Collected so far: {len(papers)} papers (offset {offset})")
            offset += limit

            # Respect API rate limit
            time.sleep(1)

            # Stop if API has no more data
            if offset >= data.get("total", 0):
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