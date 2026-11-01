
from src.retrieval.retrieval_service import RetrievalService


def main():
    service = RetrievalService()

    # Empty queries should return no cases.
    assert service.retrieve_cases("").empty
    assert service.retrieve_cases("   ").empty

    # Valid queries should return the requested number at most.
    assert len(service.retrieve_cases("internet outage", top_k=3)) <= 3

    # Invalid top_k values should fail clearly.
    for value, expected_error in [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ]:
        try:
            service.retrieve_cases("internet outage", top_k=value)
        except expected_error:
            pass
        else:
            raise AssertionError(
                f"Expected {expected_error.__name__} for top_k={value!r}"
            )

    print("All retrieval service edge-case tests passed.")


if __name__ == "__main__":
    main()
