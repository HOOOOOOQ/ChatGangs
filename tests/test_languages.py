"""Language detection tests."""

import pytest

from chatgangs.core.languages import (
    SUPPORTED_LANGUAGES,
    detect_language,
    is_supported,
    language_name,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Saya ingin bertanya tentang layanan kalian hari ini", "id"),
        ("I would like to ask about your service today", "en"),
        ("Terima kasih kerana sudi membantu, apa khabar awak", "ms"),
        ("こんにちは、今日はいい天気ですね", "ja"),
        ("안녕하세요 오늘 날씨가 좋네요", "ko"),
        ("Me gustaría preguntar sobre el servicio de hoy", "es"),
        ("Ich möchte heute nach dem Service fragen", "de"),
        ("Je voudrais demander des informations sur le service", "fr"),
    ],
)
def test_detection(text, expected):
    assert detect_language(text).language == expected


def test_empty_text_uses_default():
    result = detect_language("")
    assert result.language == "id"
    assert result.confidence == 0.0


def test_confidence_and_method_reported():
    result = detect_language("Pelayanannya bagus tapi pengirimannya lambat sekali")
    assert 0.0 <= result.confidence <= 1.0
    assert result.method
    payload = result.to_dict()
    assert payload["supported"] is True
    assert payload["name"] == "Indonesian"


def test_supported_language_helpers():
    assert is_supported("id")
    assert not is_supported("xx")
    assert language_name("ja") == "Japanese"
    assert set(SUPPORTED_LANGUAGES) == {"id", "en", "ms", "ja", "ko", "es", "de", "fr"}


def test_indonesian_and_english_are_priority():
    from chatgangs.core.languages import PRIORITY_LANGUAGES

    assert PRIORITY_LANGUAGES == ("id", "en")
