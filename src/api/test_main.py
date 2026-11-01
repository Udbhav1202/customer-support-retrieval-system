from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_chat_returns_generated_answer():
    with patch("src.api.main.rag_service") as mock_rag:
        mock_rag.answer_with_sources.return_value = {
            "answer": (
                "Restart your router and check whether the connection drops."
            ),

            "sources": [
                {
                    "rank": 1,
                    "customer_message": "My internet keeps disconnecting.",
                    "support_response": "Try restarting your router.",
                    "rrf_score": 0.0318,
                }
            ],

        }

        response = client.post(
            "/chat",
            json={
                "query": "My internet keeps disconnecting",
                "top_k": 5,
            },
        )

    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == (
        "Restart your router and check whether the connection drops."
    )
    assert len(data["sources"]) == 1
    assert data["sources"][0]["customer_message"] == (
        "My internet keeps disconnecting."
    )
    assert data["sources"][0]["support_response"] == (
        "Try restarting your router."
    )
