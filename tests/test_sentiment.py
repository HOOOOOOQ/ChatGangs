"""Sentiment analysis tests."""

import pytest

from chatgangs.core.sentiment import SentimentAnalyzer, get_sentiment_analyzer


@pytest.mark.parametrize(
    "text,language,expected",
    [
        ("Saya sangat puas dengan layanan ini, mantap!", "id", "positive"),
        ("Pelayanan buruk sekali, saya kecewa", "id", "negative"),
        ("Biasa saja", "id", "neutral"),
        ("I love this product, it works great", "en", "positive"),
        ("This is terrible, worst experience ever", "en", "negative"),
        ("Pengiriman tidak tepat waktu dan barang rusak", "id", "negative"),
        ("Produknya bagus tapi pengirimannya lambat banget", "id", "negative"),
    ],
)
def test_labels(text, language, expected):
    result = get_sentiment_analyzer().analyze(text, language)
    assert result.label == expected, result.to_dict()


def test_score_range_and_confidence():
    result = get_sentiment_analyzer().analyze("sangat bagus sekali", "id")
    assert -1.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_negation_flips_polarity():
    analyzer = get_sentiment_analyzer()
    positive = analyzer.analyze("produk ini bagus", "id")
    negated = analyzer.analyze("produk ini tidak bagus", "id")
    assert positive.score > negated.score


def test_amplifier_increases_intensity():
    analyzer = get_sentiment_analyzer()
    plain = analyzer.analyze("bagus", "id")
    amplified = analyzer.analyze("sangat bagus sekali", "id")
    assert amplified.score >= plain.score


def test_evidence_is_explainable():
    result = get_sentiment_analyzer().analyze("pelayanannya buruk", "id", explain=True)
    terms = [item["term"] for item in result.evidence]
    assert "buruk" in terms


def test_emoji_carries_sentiment():
    result = get_sentiment_analyzer().analyze("mantap 😍", "id")
    assert result.label == "positive"


def test_empty_text_is_neutral():
    result = get_sentiment_analyzer().analyze("", "id")
    assert result.label == "neutral"
    assert result.confidence == 0.0


def test_short_greeting_is_not_scored_as_sentiment():
    result = get_sentiment_analyzer().analyze("halo", "id")
    assert result.label == "neutral"


def test_model_info_reports_sources():
    info = get_sentiment_analyzer().model_info()
    assert "available" in info
    assert info["lexicon"]["positive_terms"] > 0
    assert info["labels"] == ["negative", "neutral", "positive"]


def test_per_sentence_breakdown():
    result = get_sentiment_analyzer().analyze(
        "Makanannya enak. Pelayanannya buruk sekali.", "id"
    )
    assert len(result.sentences) == 2
    assert result.sentences[0]["label"] == "positive"
    assert result.sentences[1]["label"] == "negative"


def test_language_fallback_does_not_crash():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("これはテストです", "ja")
    assert result.label in ("positive", "negative", "neutral")
