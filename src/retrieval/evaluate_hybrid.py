
from pathlib import Path

import pandas as pd

from src.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

DATA_PATH = PROCESSED / "verizon_retrieval_cases_clean.csv"
QUERIES_PATH = PROCESSED / "retrieval_eval_queries_v2.csv"
OUTPUT_PATH = PROCESSED / "hybrid_retrieval_results_v2.csv"

df = pd.read_csv(DATA_PATH)
queries = pd.read_csv(QUERIES_PATH).reset_index(drop=True)
queries["query_id"] = queries.index + 1

retriever = HybridRetriever(df)

all_results = []

for _, query_row in queries.iterrows():
    results = retriever.retrieve(query_row["query"], top_k=5).copy()

    results["query_id"] = query_row["query_id"]
    results["query"] = query_row["query"]
    results["expected_intent"] = query_row["expected_intent"]
    results["rank"] = range(1, len(results) + 1)

    all_results.append(results)

if not all_results:
    raise ValueError("No evaluation queries were found.")

output = pd.concat(all_results, ignore_index=True)
output.to_csv(OUTPUT_PATH, index=False)

print(f"Queries evaluated: {len(queries)}")
print(f"Results generated: {len(output)}")
print(f"Saved to: {OUTPUT_PATH}")
