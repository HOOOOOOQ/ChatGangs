"""Normalisation, slang expansion and typo correction tests."""

from chatgangs.core.normalizer import TextNormalizer, get_dictionaries, get_normalizer, redact


def test_lowercase_and_whitespace():
    assert get_normalizer().normalize("  HALO   Dunia  ") == "halo dunia"


def test_slang_expansion_indonesian():
    normalizer = get_normalizer()
    assert "bagaimana" in normalizer.normalize("gmn caranya")
    assert "tidak" in normalizer.normalize("gak tau")
    assert normalizer.normalize("makasih") == "terima kasih"


def test_slang_expansion_english():
    assert "please" in get_normalizer().normalize("pls help")


def test_repeated_characters_collapsed():
    normalizer = get_normalizer()
    assert normalizer.normalize("bagussss") == "bagus"
    assert normalizer.normalize("mantappp") == "mantap"


def test_typo_correction_is_conservative():
    normalizer = get_normalizer()
    assert "bagaimana" in normalizer.normalize("bagaimna")
    # valid words must not be "corrected" into unrelated words
    assert normalizer.normalize("selamat tinggal") == "selamat tinggal"


def test_pii_redaction():
    text = "email saya budi@mail.com dan nomor 081234567890"
    redacted = redact(text)
    assert "budi@mail.com" not in redacted
    assert "081234567890" not in redacted
    assert "[EMAIL]" in redacted


def test_urls_and_hashtags_survive_normalisation():
    result = get_normalizer().normalize("Cek https://chatgangs.dev #promo di @admin")
    assert "https://chatgangs.dev" in result
    assert "#promo" in result


def test_dictionary_store_has_all_languages():
    store = get_dictionaries()
    for code in ("id", "en", "ms", "ja", "ko", "es", "de", "fr"):
        assert store.stopwords[code], f"missing stopwords for {code}"
    assert store.sentiment["id"]["positive"]
    assert store.sentiment["en"]["negative"]
    assert store.emoticons


def test_cache_is_used():
    normalizer = TextNormalizer()
    before = normalizer.cache_stats()["hits"]
    normalizer.normalize("halo apa kabar")
    normalizer.normalize("halo apa kabar")
    assert normalizer.cache_stats()["hits"] > before
