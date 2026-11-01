
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

LABEL_FILES = {
    "tfidf": PROCESSED / "tfidf_relevance_labels.csv",
    "bm25": PROCESSED / "bm25_relevance_labels_final.csv",
    "dense": PROCESSED / "dense_relevance_labels_final.csv",
    "hybrid": PROCESSED / "hybrid_relevance_labels_final.csv",
}

OUTPUT_PATH = PROCESSED / "master_relevance_judgments.csv"


def normalize_query(series):
    return series.astype(str).str.strip().str.lower()


def normalize_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


frames = []

for system, path in LABEL_FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing label file for {system}: {path}"
        )

    df = pd.read_csv(path)

    required = {"query", "customer_tweet_id", "relevance"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{system} labels are missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["query_key"] = normalize_query(df["query"])
    df["case_key"] = normalize_id(df["customer_tweet_id"])
    df["relevance"] = pd.to_numeric(
        df["relevance"], errors="coerce"
    )

    if df["relevance"].isna().any():
        raise ValueError(
            f"{system} labels contain missing or invalid relevance scores."
        )

    if not df["relevance"].isin([0, 1, 2]).all():
        raise ValueError(
            f"{system} labels must use scores 0, 1, or 2."
        )

    df["source_system"] = system

    frames.append(
        df[
            [
                "query",
                "query_key",
                "customer_tweet_id",
                "case_key",
                "relevance",
                "source_system",
            ]
        ]
    )

all_labels = pd.concat(frames, ignore_index=True)

# A query-case pair must have the same judgment in every system.
conflicts = (
    all_labels.groupby(["query_key", "case_key"])["relevance"]
    .nunique()
)

if (conflicts > 1).any():
    conflict_keys = conflicts[conflicts > 1].index
    examples = all_labels.set_index(
        ["query_key", "case_key"]
    ).loc[conflict_keys].reset_index()

    print("Conflicting relevance judgments found:")
    print(
        examples[
            [
                "query",
                "customer_tweet_id",
                "relevance",
                "source_system",
            ]
        ].head(20).to_string(index=False)
    )

    raise ValueError(
        "Resolve conflicting judgments before creating the master dataset."
    )

master = (
    all_labels.groupby(
        ["query_key", "case_key"], as_index=False
    )
    .agg(
        query=("query", "first"),
        customer_tweet_id=("customer_tweet_id", "first"),
        relevance=("relevance", "first"),
        source_systems=(
            "source_system",
            lambda values: ",".join(sorted(set(values))),
        ),
        judgment_count=("source_system", "nunique"),
    )
)

master = master[
    [
        "query",
        "customer_tweet_id",
        "relevance",
        "source_systems",
        "judgment_count",
    ]
].sort_values(["query", "customer_tweet_id"])

master["relevance"] = master["relevance"].astype(int)
master.to_csv(OUTPUT_PATH, index=False)

print("\nMaster judgment dataset created.")
print("Input rows across all systems:", len(all_labels))
print("Unique query-case judgments:", len(master))
print("Duplicate judgments consolidated:", len(all_labels) - len(master))
print("Missing scores:", master["relevance"].isna().sum())

print("\nRelevance distribution:")
print(master["relevance"].value_counts().sort_index())

print("\nJudgments reused across multiple systems:")
print(master["judgment_count"].value_counts().sort_index())

print("\nSaved to:", OUTPUT_PATH)
