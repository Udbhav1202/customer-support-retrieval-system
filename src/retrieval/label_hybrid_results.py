
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

INPUT_PATH = PROCESSED / "hybrid_retrieval_results_v2.csv"
OUTPUT_PATH = PROCESSED / "hybrid_relevance_labels_v2.csv"


def main():
    results = pd.read_csv(INPUT_PATH)

    required_columns = {
        "query_id",
        "rank",
        "customer_tweet_id",
        "query",
        "expected_intent",
        "customer_message",
        "support_response",
    }
    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


    if OUTPUT_PATH.exists():
        labels = pd.read_csv(OUTPUT_PATH)

        required_label_columns = {
            "query_id",
            "customer_tweet_id",
            "relevance",
        }
        missing_labels = required_label_columns - set(labels.columns)

        if missing_labels:
            raise ValueError(
                f"Existing labels file is missing columns: {missing_labels}"
            )

        # Keep only rows with valid, completed relevance judgments.
        labels["relevance"] = pd.to_numeric(
        labels["relevance"], errors="coerce"
        )
        labels = labels[
                labels["relevance"].isin([0, 1, 2])
            ].copy()

        # Remove unfinished rows from the saved file.
        labels.to_csv(OUTPUT_PATH, index=False)

        completed = set(
            zip(
                labels["query_id"].astype(int),
                labels["customer_tweet_id"].astype(str),
            )
        )
    else:
        labels = pd.DataFrame()
        completed = set()


    print("Relevance scale:")
    print("0 = Irrelevant")
    print("1 = Related, but not useful enough")
    print("2 = Useful historical support case")
    print("q = Save and quit")

    for _, row in results.iterrows():
        key = (
            int(row["query_id"]),
            str(row["customer_tweet_id"]),
        )

        if key in completed:
            continue

        print("\n" + "=" * 70)
        print(f"Query: {row['query']}")
        print(f"Expected intent: {row['expected_intent']}")
        print(f"Rank: {row['rank']}")
        print(f"Customer case ID: {row['customer_tweet_id']}")
        print(f"\nHistorical customer: {row['customer_message']}")
        print(f"Historical support: {row['support_response']}")

        while True:
            rating = input("\nRelevance (0/1/2/q): ").strip().lower()

            if rating in {"0", "1", "2", "q"}:
                break

            print("Enter 0, 1, 2, or q.")

        if rating == "q":
            break

        labeled_row = row.to_dict()
        labeled_row["relevance"] = int(rating)

        labels = pd.concat(
            [labels, pd.DataFrame([labeled_row])],
            ignore_index=True,
        )

        labels.to_csv(OUTPUT_PATH, index=False)
        completed.add(key)

    print(f"\nSaved {len(labels)} labels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
