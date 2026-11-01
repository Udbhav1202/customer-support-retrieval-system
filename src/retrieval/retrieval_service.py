
from pathlib import Path

import pandas as pd

from src.retrieval.hybrid_retriever import HybridRetriever


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "verizon_retrieval_cases_clean.csv"


class RetrievalService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Retrieval dataset not found: {DATA_PATH}"
            )

        self.df = pd.read_csv(DATA_PATH)
        self.retriever = HybridRetriever(
            self.df,
            model_name=model_name,
        )


    def retrieve_cases(
        self,
        query: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        if not isinstance(query, str) or not query.strip():
            return self.retriever.retrieve("", top_k=5)

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return self.retriever.retrieve(query, top_k=top_k)
