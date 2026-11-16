"""Intent detection tests."""

import pytest

from chatgangs.core.intent import IntentDetector, get_intent_detector


@pytest.mark.parametrize(
    "text,expected",
    [
        ("halo", "greeting"),
        ("Hai bot, apa kabar?", "greeting"),
        ("selamat tinggal", "farewell"),
        ("terima kasih banyak", "gratitude"),
        ("siapa kamu?", "identity"),
        ("apa saja kemampuanmu", "capabilities"),
        ("cara pakai API nya gimana?", "api_help"),
        ("berapa harganya?", "pricing"),
        ("hapus riwayat percakapan saya", "privacy"),
        ("bagaimana melatih model sendiri", "training_help"),
        ("terjemahkan kalimat ini ke bahasa inggris", "translate"),
        ("saya kecewa, pelayanannya lambat sekali", "complaint"),
        ("tolong", "help"),
    ],
)
def test_common_intents(text, expected):
    result = get_intent_detector().detect(text, "id")
    assert result.name == expected, result.to_dict()


def test_social_frame_does_not_mask_request():
    result = get_intent_detector().detect("hello, what can you do?", "en")
    assert result.name == "capabilities"


def test_typo_tolerance():
    detector = get_intent_detector()
    result = detector.detect("bagaimna cara pakai api", "id")
    assert result.name == "api_help"


def test_low_confidence_falls_back_to_unknown():
    result = get_intent_detector().detect("zxqv blorf nardis", "id")
    assert result.name in ("unknown", "question", "small_talk")


def test_catalogue_loaded():
    detector = get_intent_detector()
    names = {intent["name"] for intent in detector.intents}
    assert "greeting" in names
    assert "complaint" in names
    assert len(detector.training_examples()) > 100


def test_result_contains_candidates_and_method():
    payload = get_intent_detector().detect("berapa harga langganan", "id").to_dict()
    assert payload["method"] in ("rule", "model", "rule+model", "fuzzy", "low-confidence")
    assert payload["candidates"]
    assert 0.0 <= payload["confidence"] <= 1.0


def test_descriptions_are_available():
    detector = get_intent_detector()
    assert detector.description("greeting")
    assert detector.response_hint("greeting") == "greeting"


def test_cache_returns_equivalent_result():
    detector = IntentDetector()
    first = detector.detect("halo", "id")
    second = detector.detect("halo", "id")
    assert first.name == second.name
    assert first.confidence == second.confidence
