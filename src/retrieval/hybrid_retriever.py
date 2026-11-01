
import pandas as pd

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        df: pd.DataFrame,
        rrf_k: int = 60,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.bm25 = BM25Retriever(df)
        self.dense = DenseRetriever(df, model_name=model_name)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        columns = [
            "conversation_id",
            "customer_tweet_id",
            "support_tweet_id",
            "customer_message",
            "support_response",
            "similarity",
        ]

        if not query.strip() or top_k <= 0:
            return pd.DataFrame(columns=columns)

        # Retrieve a larger candidate pool from both systems.
        candidate_k = min(len(self.bm25.df), max(top_k * 5, 20))

        bm25_results = self.bm25.retrieve(query, candidate_k)
        dense_results = self.dense.retrieve(query, candidate_k)

        fused_scores = {}

        # Add each result's reciprocal-rank contribution.
        for results in (bm25_results, dense_results):
            for rank, tweet_id in enumerate(
                results["customer_tweet_id"], start=1
            ):
                key = str(tweet_id)
                fused_scores[key] = fused_scores.get(key, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )

        if not fused_scores:
            return pd.DataFrame(columns=columns)

        # Keep one canonical row per customer case.
        combined = pd.concat(
            [bm25_results, dense_results],
            ignore_index=True,
        ).drop_duplicates(subset=["customer_tweet_id"])

        combined["_key"] = combined["customer_tweet_id"].astype(str)
        combined["similarity"] = combined["_key"].map(fused_scores)
        combined = combined.sort_values(
            "similarity", ascending=False, kind="stable"
        )

        return combined.head(top_k)[columns].reset_index(drop=True)
