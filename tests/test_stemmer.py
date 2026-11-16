"""Stemming and lemmatization tests."""

import pytest

from chatgangs.core.stemmer import (
    Stemmer,
    get_stemmer,
    stem_english,
    stem_indonesian,
)


@pytest.mark.parametrize(
    "word,expected",
    [
        ("makanan", "makan"),
        ("pelajaran", "ajar"),
        ("kesehatan", "sehat"),
        ("keuangan", "uang"),
        ("pembayaran", "bayar"),
        ("pertanyaan", "tanya"),
        ("menulis", "tulis"),
        ("memukul", "pukul"),
        ("mengambil", "ambil"),
        ("menggunakan", "guna"),
        ("berjalan", "jalan"),
        ("permainan", "main"),
        ("kebaikan", "baik"),
        ("pekerjaan", "kerja"),
        ("kebersihan", "bersih"),
        # possessive suffix + circumfix combination
        ("pengirimannya", "kirim"),
        ("kualitasnya", "kualitas"),
        # circumfix peN-/-an and ber-/-an
        ("pengiriman", "kirim"),
        ("perusahaan", "usaha"),
        ("pembelajaran", "belajar"),
        ("penjualan", "jual"),
    ],
)
def test_indonesian_builtin_stemmer(word, expected):
    assert stem_indonesian(word) == expected


@pytest.mark.parametrize(
    "word,expected",
    [
        ("running", "run"),
        ("played", "play"),
        ("studies", "study"),
        ("analyzing", "analyz"),
    ],
)
def test_english_stemmer(word, expected):
    assert stem_english(word) == expected


def test_protected_words_are_unchanged():
    for word in ("tidak", "bukan", "saya", "bagaimana", "kualitas"):
        assert stem_indonesian(word) == word


def test_lemmatization_irregular_english():
    stemmer = get_stemmer()
    assert stemmer.lemmatize("went", "en") == "go"
    assert stemmer.lemmatize("children", "en") == "child"
    assert stemmer.lemmatize("better", "en") == "good"


def test_stemmer_backend_selection(monkeypatch):
    builtin = Stemmer(backend="builtin")
    assert builtin.indonesian_backend == "builtin"
    assert builtin.stem("makanan", "id") == "makan"


def test_stemmer_info_reports_backend():
    info = get_stemmer().info()
    assert info["indonesian_backend"] in ("builtin", "sastrawi")
    assert "en" in info["supported_languages"]


def test_cache_returns_same_result():
    stemmer = Stemmer(backend="builtin")
    first = stemmer.stem("pelayanan", "id")
    second = stemmer.stem("pelayanan", "id")
    assert first == second == "layan"
