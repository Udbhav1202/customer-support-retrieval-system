
import pandas as pd
import pytest
from unittest.mock import Mock

from src.rag.rag_service import RAGService


def test_prepare_prompt_uses_retrieved_context():
    mock_retrieval = Mock()

    mock_retrieval.retrieve_cases.return_value = pd.DataFrame(
        [
            {
                "customer_message": "My internet keeps cutting out.",
                "support_response": "Are all devices disconnecting?",
                "similarity": 0.0318,
            }
        ]
    )


    service = RAGService(retrieval_service=mock_retrieval)

    query = "My internet keeps disconnecting"
    prompt = service.prepare_prompt(query, top_k=3)

    mock_retrieval.retrieve_cases.assert_called_once_with(
        query,
        top_k=3,
    )

    assert "You are a customer support assistant." in prompt
    assert "<retrieved_historical_cases>" in prompt
    assert "</retrieved_historical_cases>" in prompt
    assert query in prompt
    assert "My internet keeps cutting out." in prompt
    assert "Are all devices disconnecting?" in prompt


def test_prepare_prompt_rejects_empty_query():
    mock_retrieval = Mock()
    service = RAGService(retrieval_service=mock_retrieval)

    with pytest.raises(
        ValueError,
        match="Customer query must not be empty",
    ):
        service.prepare_prompt("   ")

    mock_retrieval.retrieve_cases.assert_not_called()
