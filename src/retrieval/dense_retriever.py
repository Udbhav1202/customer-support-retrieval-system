
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed" / "embedding_cache"


class DenseRetriever:
    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.df["customer_message"] = (
            self.df["customer_message"].fillna("").astype(str)
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Include the model and dataset contents in the cache identity.
        import hashlib

        dataset_hash = hashlib.sha256(
            "\n".join(self.df["customer_message"]).encode("utf-8")
        ).hexdigest()[:16]
        model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
        cache_path = CACHE_DIR / f"embeddings_{model_hash}_{dataset_hash}.npy"

        self.model = SentenceTransformer(model_name)

        if cache_path.exists():
            import numpy as np

            self.document_embeddings = np.load(cache_path)
            print(f"Loaded cached embeddings: {cache_path.name}")
        else:
            import numpy as np

            print("Building document embeddings for the first time...")
            self.document_embeddings = self.model.encode(
                self.df["customer_message"].tolist(),
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            np.save(cache_path, self.document_embeddings)
            print(f"Saved embeddings cache: {cache_path.name}")

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        columns = [
            "conversation_id",
            "customer_tweet_id",
            "support_tweet_id",
            "customer_message",
            "support_response",
            "similarity",
        ]

        if not isinstance(query, str) or not query.strip() or top_k <= 0:
            return pd.DataFrame(columns=columns)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = self.document_embeddings @ query_embedding[0]
        top_indices = scores.argsort()[::-1][:top_k]

        results = self.df.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]

        return results[columns]
