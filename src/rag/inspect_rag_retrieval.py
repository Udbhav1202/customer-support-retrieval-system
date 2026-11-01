
from src.retrieval.retrieval_service import RetrievalService


def main():
    service = RetrievalService()

    queries = [
        "My internet keeps disconnecting. What should I do?",
        "Why has my Verizon bill suddenly increased?",
        "I want to cancel my Verizon service. What are my options?",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}\n")

        results = service.retrieve_cases(query, top_k=3)

        for rank, row in enumerate(results.itertuples(index=False), start=1):
            print(f"CASE {rank}")
            print(f"Customer: {row.customer_message}")
            print(f"Support:  {row.support_response}")
            print(f"RRF score: {row.similarity:.5f}")
            print("-" * 60)


if __name__ == "__main__":
    main()
