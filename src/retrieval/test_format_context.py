
from src.retrieval.retrieval_service import RetrievalService
from src.retrieval.format_context import format_retrieved_context


def main():
    service = RetrievalService()

    query = "My internet keeps disconnecting"
    results = service.retrieve_cases(query, top_k=3)

    context = format_retrieved_context(results)

    # Verify the formatted context.
    assert isinstance(context, str)
    assert context.strip()
    assert "Customer message:" in context
    assert "Support response:" in context
    assert "Case 1" in context

    print("Formatted context:\n")
    print(context)

    # Verify empty results are handled.
    empty_context = format_retrieved_context(results.iloc[0:0])
    assert empty_context == "No relevant historical support cases were found."

    print("\nAll context formatting tests passed.")


if __name__ == "__main__":
    main()
