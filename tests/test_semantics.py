"""Semantic similarity, classification and QA tests."""

import pytest

from chatgangs.core.semantics import (
    SemanticIndex,
    TFIDFVectorizer,
    cosine,
    get_knowledge_base,
    get_semantic_index,
    get_text_classifier,
    tokenize_for_index,
)


def test_identical_texts_are_maximally_similar():
    result = get_semantic_index().similarity(["saya suka kucing", "saya suka kucing"])
    assert result.matrix[0][1] == pytest.approx(1.0, abs=1e-6)


def test_related_texts_score_higher_than_unrelated():
    result = get_semantic_index().similarity(
        ["harga produk ini murah", "produk ini harganya murah", "cuaca hari ini cerah"]
    )
    assert result.matrix[0][1] > result.matrix[0][2]


def test_morphological_variants_match_after_stemming():
    tokens_a = tokenize_for_index("harganya berapa", "id")
    tokens_b = tokenize_for_index("harga produk ini", "id")
    assert set(tokens_a) & set(tokens_b)


def test_rank_returns_sorted_scores():
    ranked = get_semantic_index().rank(
        "berapa harga", ["berapa harga produk", "cara install", "halo apa kabar"]
    )
    assert ranked[0][0] == "berapa harga produk"
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_vectorizer_requires_fit():
    with pytest.raises(RuntimeError):
        TFIDFVectorizer().transform([["a"]])


def test_cosine_edge_cases():
    assert cosine([], []) == 0.0
    assert cosine([1, 0], [0, 1]) == 0.0


def test_knowledge_base_retrieval():
    kb = get_knowledge_base()
    assert kb.entries
    match = kb.best("siapa kamu", "id")
    assert match is not None
    assert match.entry["id"] == "kb.identity"


def test_knowledge_base_exact_variants():
    kb = get_knowledge_base()
    for question, expected in [
        ("halo", "kb.greeting"),
        ("terima kasih", "kb.thanks"),
        ("what can you do", "kb.capabilities"),
    ]:
        match = kb.best(question, "id")
        assert match is not None and match.entry["id"] == expected


def test_knowledge_base_returns_none_for_unknown_topic():
    assert get_knowledge_base().best("resep rendang padang asli", "id", min_score=0.4) is None


def test_knowledge_base_topics():
    topics = get_knowledge_base().topics()
    assert "api" in topics and "privacy" in topics


def test_classifier_taxonomy():
    classifier = get_text_classifier()
    assert classifier.classify("Tolong jelaskan cara kerjanya", language="id").label == "perintah"
    assert classifier.classify("Saya sangat kecewa produknya rusak", language="id").label == "keluhan"


def test_classifier_labels_present():
    classifier = get_text_classifier()
    labels = classifier.labels()
    assert "pertanyaan" in labels
    # the trained topic model is reported separately, never merged into the
    # speech-act label space
    assert all(label not in labels for label in classifier.topic_labels())


def test_classifier_task_switch():
    classifier = get_text_classifier()
    auto = classifier.classify("Tolong jelaskan cara kerjanya")
    assert auto.speech_act == "perintah"
    assert auto.label == "perintah"
    forced = classifier.classify("Tolong jelaskan cara kerjanya", task="speech_act")
    assert forced.label == "perintah" and forced.topic is None
    if classifier.model_available:
        topic = classifier.classify("film horor itu pecahkan rekor", task="topic")
        assert topic.label in classifier.topic_labels()
        assert topic.speech_act == "lainnya"
