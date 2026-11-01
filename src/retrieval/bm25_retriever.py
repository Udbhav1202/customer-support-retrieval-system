
import re

import pandas as pd
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into word tokens."""
    return re.findall(r"\b\w+\b", str(text).lower())


class BM25Retriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()

        if self.df["customer_message"].isna().any():
            self.df["customer_message"] = (
                self.df["customer_message"].fillna("")
            )

        tokenized_corpus = [
            tokenize(message)
            for message in self.df["customer_message"]
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_tokens = tokenize(query)

        if not query_tokens or top_k <= 0:
            return pd.DataFrame(
                columns=[
                    "conversation_id",
                    "customer_tweet_id",
                    "support_tweet_id",
                    "customer_message",
                    "support_response",
                    "similarity",
                ]
            )

        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:top_k]

        results = self.df.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]

        return results[
            [
                "conversation_id",
                "customer_tweet_id",
                "support_tweet_id",
                "customer_message",
                "support_response",
                "similarity",
            ]
        ]