"""End-to-end pipeline tests (analysis + conversation)."""

import uuid

import pytest

from chatgangs.core.pipeline import NLPengine


@pytest.fixture(scope="module")
def engine():
    return NLPengine()


def test_full_analysis_shape(engine):
    result = engine.analyze(
        "Pelayanan PT Maju Jaya bagus tapi pengirimannya lambat. "
        "Hubungi budi@mail.com di Jakarta pada 12/03/2025."
    )
    payload = result.to_dict()
    for key in (
        "language", "normalized", "tokens", "token_stats", "stems", "lemmas",
        "stopwords", "content_terms", "entities", "entity_summary", "intent",
        "sentiment", "classification", "keywords", "qa_matches", "stats",
    ):
        assert key in payload
    assert payload["language"]["code"] == "id"
    assert payload["intent"]["name"] == "complaint"
    assert payload["sentiment"]["label"] in ("negative", "neutral")
    assert any(entity["label"] == "EMAIL" for entity in payload["entities"])
    assert payload["stats"]["total_ms"] >= 0


def test_analyze_rejects_empty_text(engine):
    with pytest.raises(ValueError):
        engine.analyze("   ")


def test_analyze_truncates_long_input(engine):
    result = engine.analyze("halo " * 5000)
    assert len(result.text) <= engine.settings.max_input_chars


def test_chat_greeting_and_context(engine):
    session_id = "pipe-" + uuid.uuid4().hex[:8]
    response = engine.chat("halo", session_id=session_id)
    assert response.intent["name"] == "greeting"
    assert response.language == "id"
    assert response.reply
    assert response.memory["stored"] is True

    followup = engine.chat("jelaskan lagi", session_id=session_id)
    assert followup.context["is_followup"] is True
    assert followup.context["turn_count"] == 2


def test_chat_answers_knowledge_questions(engine):
    session_id = "pipe-" + uuid.uuid4().hex[:8]
    response = engine.chat("bagaimana cara memakai api", session_id=session_id)
    assert "api" in response.generation["template_id"] or "/api/" in response.reply


def test_chat_deletion_request_clears_memory(engine):
    session_id = "pipe-" + uuid.uuid4().hex[:8]
    engine.chat("halo saya budi", session_id=session_id)
    response = engine.chat("hapus riwayat percakapan saya", session_id=session_id)
    assert response.intent["name"] == "privacy"
    assert "hapus" in response.reply.lower() or "selesai" in response.reply.lower()
    assert engine.memory.get_messages(session_id) == []


def test_followup_detail_request_expands_the_answer(engine):
    """A detail follow-up must deepen the previous answer, not repeat it."""
    session_id = "pipe-" + uuid.uuid4().hex[:8]
    first = engine.chat("halo, apa saja kemampuanmu?", session_id=session_id)
    assert first.intent["name"] == "capabilities"

    detailed = engine.chat("jelaskan lebih detail", session_id=session_id)
    assert detailed.context["is_followup"] is True
    assert len(detailed.reply) > len(first.reply)
    assert detailed.reply.strip() != first.reply.strip()


def test_privacy_answer_renders_without_placeholder_errors(engine):
    """Templates that mention {session_id} must not crash str.format."""
    session_id = "pipe-" + uuid.uuid4().hex[:8]
    response = engine.chat("bagaimana privasi data saya?", session_id=session_id)
    assert response.intent["name"] == "privacy"
    assert "/api/memory/" in response.reply
    # the literal path placeholder is shown to the user, not crashed on
    assert "session_id}" in response.reply


def test_chat_language_adaptation(engine):
    session_id = "pipe-" + uuid.uuid4().hex[:8]
    response = engine.chat("hello, what can you do?", session_id=session_id)
    assert response.language == "en"
    assert response.intent["name"] == "capabilities"
    assert any(word in response.reply.lower() for word in ("tokenise", "tokenize", "sentiment"))


def test_chat_is_not_repetitive(engine):
    session_id = "pipe-" + uuid.uuid4().hex[:8]
    replies = {engine.chat("halo", session_id=session_id).reply for _ in range(6)}
    assert len(replies) > 1


def test_tokenize_and_module_helpers(engine):
    payload = engine.tokenize("Halo dunia!")
    assert payload["stats"]["words"] == 2
    assert engine.analyze_sentiment("sangat bagus", "id")["label"] == "positive"
    assert engine.detect_intent("berapa harganya", "id")["name"] == "pricing"
    entities = engine.extract_entities("Hubungi budi@mail.com", "id")
    assert entities["entities"][0]["label"] == "EMAIL"
    assert engine.answer_question("siapa kamu", "id")["answered"] is True


def test_similarity_helper(engine):
    payload = engine.similarity(["saya suka kopi", "saya suka kopi", "harga mobil mahal"])
    assert payload["score"] == pytest.approx(1.0, abs=1e-6)


def test_model_info_is_honest(engine):
    info = engine.model_info()
    assert info["training"]["intent_model_trained"] in (True, False)
    assert "engines" in info
    assert "languages" in info


def test_classification_helper(engine):
    payload = engine.classify("Saya sangat kecewa, produknya rusak", "id")
    assert payload["label"] == "keluhan"
