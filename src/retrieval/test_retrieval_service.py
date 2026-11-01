
from src.retrieval.retrieval_service import RetrievalService


def main():
    service = RetrievalService()

    query = "My internet keeps disconnecting"
    results = service.retrieve_cases(query, top_k=5)

    expected_columns = {
        "conversation_id",
        "customer_tweet_id",
        "support_tweet_id",
        "customer_message",
        "support_response",
        "similarity",
    }

    assert expected_columns.issubset(results.columns), (
        f"Missing columns: {expected_columns - set(results.columns)}"
    )
    assert len(results) <= 5
    assert results["customer_tweet_id"].is_unique

    print("Query:", query)
    print("Results returned:", len(results))
    print(results.to_string(index=False))
    print("\nRetrieval service test passed.")


if __name__ == "__main__":
    main()
