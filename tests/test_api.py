"""REST API contract tests."""

import uuid


def unique_session():
    return "api-" + uuid.uuid4().hex[:10]


# ------------------------------------------------------------------ system
def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["database"]["status"] == "ok"
    assert payload["checks"]["nlp_engine"]["intents"] > 0


def test_model_info(client):
    response = client.get("/api/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert "engines" in payload
    assert payload["name"]
    assert "artifacts" in payload


def test_metrics(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "latency_ms" in payload
    assert "p50" in payload["latency_ms"]


def test_catalogues(client):
    assert client.get("/api/languages").json()["priority"] == ["id", "en"]
    assert client.get("/api/intents").json()["count"] > 0
    assert client.get("/api/knowledge-base").json()["entries"] > 0


def test_openapi_and_docs(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/api/chat" in schema["paths"]
    assert "/api/nlp/sentiment" in schema["paths"]


def test_web_client_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ChatGangs" in response.text


# -------------------------------------------------------------------- chat
def test_chat_roundtrip(client):
    session = unique_session()
    response = client.post(
        "/api/chat", json={"message": "Halo, apa kabar?", "session_id": session}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"]
    assert payload["intent"]["name"] == "greeting"
    assert payload["session_id"] == session
    assert response.headers["X-Request-ID"]


def test_chat_conversation_flow_and_deletion(client):
    session = unique_session()
    client.post("/api/chat", json={"message": "apa itu NLP", "session_id": session})
    followup = client.post(
        "/api/chat", json={"message": "jelaskan lebih detail", "session_id": session}
    ).json()
    assert followup["context"]["is_followup"] is True

    deleted = client.delete(f"/api/memory/{session}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    # After deletion the session must be gone, not merely emptied.
    assert client.get(f"/api/memory/{session}").status_code == 404


def test_chat_validation_errors(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422
    too_long = client.post("/api/chat", json={"message": "x" * 5000})
    assert too_long.status_code == 422
    invalid_session = client.post(
        "/api/chat", json={"message": "halo", "session_id": "bad id!"}
    )
    assert invalid_session.status_code == 422


def test_chat_unknown_field_rejected(client):
    response = client.post(
        "/api/chat", json={"message": "halo", "unexpected": "field"}
    )
    assert response.status_code == 422


def test_memory_endpoints(client):
    session = unique_session()
    client.post("/api/chat", json={"message": "simpan pesan ini", "session_id": session})
    listing = client.get("/api/memory").json()
    assert "sessions" in listing
    detail = client.get(f"/api/memory/{session}").json()
    assert detail["messages"]


def test_feedback_endpoint(client):
    response = client.post(
        "/api/feedback",
        json={"session_id": unique_session(), "rating": 1, "comment": "bagus"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"


# --------------------------------------------------------------------- NLP
def test_analyze_endpoint(client):
    response = client.post(
        "/api/nlp/analyze",
        json={"text": "Pelayanannya bagus tapi pengirimannya lambat sekali"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["language"]["code"] == "id"
    assert payload["intent"]["name"] == "complaint"
    assert payload["tokens"]


def test_tokenize_endpoint(client):
    payload = client.post(
        "/api/nlp/tokenize", json={"text": "Halo dunia! Ini tes."}
    ).json()
    assert len(payload["tokens"]) >= 4
    assert payload["stats"]["sentences"] == 2
    assert any("stem" in token for token in payload["tokens"] if token["is_word"])


def test_sentiment_endpoint(client):
    positive = client.post("/api/nlp/sentiment", json={"text": "sangat memuaskan"}).json()
    negative = client.post("/api/nlp/sentiment", json={"text": "buruk sekali"}).json()
    assert positive["label"] == "positive"
    assert negative["label"] == "negative"


def test_intent_endpoint(client):
    payload = client.post("/api/nlp/intent", json={"text": "berapa harganya?"}).json()
    assert payload["name"] == "pricing"
    filtered = client.post(
        "/api/nlp/intent", json={"text": "berapa harganya?", "include_candidates": False}
    ).json()
    assert "candidates" not in filtered


def test_entities_endpoint(client):
    payload = client.post(
        "/api/nlp/entities",
        json={"text": "PT Maju Jaya di Bandung, email budi@mail.com"},
    ).json()
    labels = {entity["label"] for entity in payload["entities"]}
    assert {"ORGANIZATION", "EMAIL"} <= labels
    filtered = client.post(
        "/api/nlp/entities",
        json={"text": "PT Maju Jaya di Bandung, email budi@mail.com", "labels": ["EMAIL"]},
    ).json()
    assert all(entity["label"] == "EMAIL" for entity in filtered["entities"])


def test_language_similarity_classify_qa_endpoints(client):
    assert client.post(
        "/api/nlp/language", json={"text": "Saya ingin bertanya"}
    ).json()["language"] == "id"

    similarity = client.post(
        "/api/nlp/similarity", json={"texts": ["saya suka kopi", "saya suka kopi", "harga mahal"]}
    ).json()
    assert similarity["score"] > 0.9

    classification = client.post(
        "/api/nlp/classify", json={"text": "Tolong jelaskan sistem ini"}
    ).json()
    assert classification["label"]

    qa = client.post("/api/nlp/qa", json={"question": "siapa kamu"}).json()
    assert qa["answered"] is True
    assert qa["matches"]


def test_normalize_stem_stopwords_batch(client):
    normalized = client.post(
        "/api/nlp/normalize", json={"text": "gmn cara pkai API"}
    ).json()
    assert "bagaimana" in normalized["normalized"]

    stem = client.post("/api/nlp/stem", json={"text": "pelayanan makanan"}).json()
    stems = {row["stem"] for row in stem["tokens"]}
    assert {"layan", "makan"} <= stems

    stop = client.post(
        "/api/nlp/stopwords", json={"text": "saya ingin bertanya tentang cuaca"}
    ).json()
    assert "cuaca" in stop["content_tokens"]
    assert "saya" in stop["removed"]

    batch = client.post(
        "/api/nlp/batch",
        json={"texts": ["halo", "saya kecewa"], "tasks": ["sentiment", "intent"]},
    ).json()
    assert batch["count"] == 2
    assert batch["results"][1]["sentiment"]["label"] == "negative"


def test_input_size_limit_enforced(client):
    response = client.post("/api/nlp/analyze", json={"text": "a" * 9000})
    assert response.status_code == 422


def test_batch_size_limit(client):
    response = client.post(
        "/api/nlp/batch", json={"texts": ["halo"] * 100, "tasks": ["sentiment"]}
    )
    assert response.status_code == 422


def test_similarity_requires_two_texts(client):
    assert client.post("/api/nlp/similarity", json={"texts": ["satu"]}).status_code == 422


def test_not_found_session(client):
    assert client.get("/api/memory/does-not-exist-xyz").status_code == 404


def test_metrics_record_real_requests(client):
    client.post("/api/nlp/tokenize", json={"text": "halo dunia"})
    payload = client.get("/api/metrics").json()
    assert payload["requests_recent"] >= 1
    assert payload["by_status"]["2xx"] >= 1
    assert payload["latency_ms"]["p50"] >= 0


def test_security_headers_present(client):
    response = client.get("/api/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "X-Request-ID" in response.headers
