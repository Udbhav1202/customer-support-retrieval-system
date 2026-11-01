
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

RESULTS_PATH = PROCESSED / "hybrid_retrieval_results.csv"
LABELS_PATH = PROCESSED / "hybrid_relevance_labels_final.csv"
QUERIES_PATH = PROCESSED / "retrieval_eval_queries.csv"

results = pd.read_csv(RESULTS_PATH)
labels = pd.read_csv(LABELS_PATH)
queries = pd.read_csv(QUERIES_PATH)

# Use query IDs to connect results with their manual relevance judgments.
merged = results.merge(
    labels[["query_id", "rank", "relevance"]],
    on=["query_id", "rank"],
    how="left",
    validate="one_to_one",
)

summary = (
    merged.groupby("query_id")
    .agg(
        directly_relevant=("relevance", lambda s: (s == 2).sum()),
        weakly_relevant=("relevance", lambda s: (s == 1).sum()),
        irrelevant=("relevance", lambda s: (s == 0).sum()),
    )
    .reset_index()
)

summary = summary.merge(
    queries.assign(query_id=range(1, len(queries) + 1)),
    on="query_id",
    how="left",
)

failures = summary[summary["directly_relevant"] < 3]

print("Queries with fewer than 3 directly relevant results:")
print(failures.to_string(index=False))
print(f"\nFailure queries: {len(failures)} out of {len(summary)}")
