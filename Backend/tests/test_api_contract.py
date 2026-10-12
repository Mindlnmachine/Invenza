from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ideas_endpoint_returns_evidence_and_used_patents():
    payload = {
        "topic": "smart irrigation sensors",
        "top_k": 3,
    }

    response = client.post("/ideas", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert isinstance(body["source_patents"], list)
    assert "used_patent_ids" in body
    assert isinstance(body["used_patent_ids"], list)
    assert len(body["used_patent_ids"]) <= len(body["source_patents"])
    assert "evidence" in body
    assert isinstance(body["evidence"], list)


def test_chat_endpoint_accepts_conversation_context():
    payload = {
        "topic": "precision agriculture",
        "conversation": [
            {"role": "user", "content": "Suggest a business model for agriculture sensors"},
            {"role": "assistant", "content": "I can help with a grounded business idea."},
        ],
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert "response" in body
    assert body["response"].strip()
    assert "conversation" in body
    assert isinstance(body["conversation"], list)
    assert body["conversation"][-2]["content"] == payload["topic"]
    assert body["conversation"][-1]["role"] == "assistant"
