import pandas as pd

from src.retrieval.tfidf_retriever import TfidfRetriever


DATA_PATH = "data/processed/verizon_retrieval_cases_clean.csv"
QUERY_PATH = "data/processed/retrieval_eval_queries.csv"
OUTPUT_PATH = "data/processed/tfidf_retrieval_results.csv"


print("=" * 80)
print("TF-IDF RETRIEVAL EVALUATION")
print("=" * 80)


# Load retrieval dataset
df = pd.read_csv(DATA_PATH)

print(f"Retrieval cases: {len(df):,}")
print(f"Conversations: {df['conversation_id'].nunique():,}")


# Build retriever
retriever = TfidfRetriever(df)


# Load evaluation queries
queries = pd.read_csv(QUERY_PATH)

print(f"Evaluation queries: {len(queries)}")


# Run retrieval
results = []

for query_id, row in queries.iterrows():

    query = row["query"]
    expected_intent = row["expected_intent"]

    retrieved = retriever.retrieve(query, top_k=5)

    for rank, (_, result) in enumerate(retrieved.iterrows(), start=1):

        results.append({
            "query_id": query_id,
            "query": query,
            "expected_intent": expected_intent,
            "rank": rank,
            "similarity": result["similarity"],
            "conversation_id": result["conversation_id"],
            "customer_tweet_id": result["customer_tweet_id"],
            "support_tweet_id": result["support_tweet_id"],
            "customer_message": result["customer_message"],
            "support_response": result["support_response"],
        })


results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_PATH, index=False)


print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print(f"Total retrieval results: {len(results_df):,}")
print(f"Saved to: {OUTPUT_PATH}")


# Display every query with its top 5 results
for query_id in queries.index:

    query_rows = results_df[
        results_df["query_id"] == query_id
    ]

    query = query_rows.iloc[0]["query"]
    expected_intent = query_rows.iloc[0]["expected_intent"]

    print("\n" + "-" * 80)
    print(f"QUERY #{query_id + 1}")
    print(f"Query: {query}")
    print(f"Expected intent: {expected_intent}")
    print("-" * 80)

    for _, result in query_rows.iterrows():

        print(
            f"\nRank {int(result['rank'])} "
            f"| Similarity: {result['similarity']:.4f}"
        )

        print(f"Customer: {result['customer_message']}")
        print(f"Support:  {result['support_response']}")