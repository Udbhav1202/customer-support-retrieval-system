
from pathlib import Path

import pandas as pd

from src.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "verizon_retrieval_cases_clean.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset rows:", len(df))
print("Initializing hybrid retriever...")

retriever = HybridRetriever(df)

queries = [
    "I cannot connect to my wireless network",
    "why is my bill so high",
    "I want to cancel my Verizon service",
]

for query in queries:
    print(f"\nQuery: {query}")
    results = retriever.retrieve(query, top_k=5)

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        print(f"\nRank {rank} | RRF score: {row['similarity']:.5f}")
        print("Customer:", row["customer_message"])
        print("Support: ", row["support_response"])
