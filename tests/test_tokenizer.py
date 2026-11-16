"""Tokenizer tests."""

from chatgangs.core.tokenizer import Tokenizer, get_tokenizer


def test_basic_word_punctuation_split():
    tokens = Tokenizer().tokenize("Halo dunia! Apa kabar?")
    words = [t.text for t in tokens if t.is_word]
    assert words == ["Halo", "dunia", "Apa", "kabar"]
    punctuation = [t.text for t in tokens if t.is_punctuation]
    assert punctuation == ["!", "?"]


def test_offsets_point_into_original_text():
    text = "Halo dunia!"
    for token in Tokenizer().tokenize(text):
        assert text[token.start:token.end] == token.text


def test_urls_emails_are_single_tokens():
    tokens = Tokenizer().tokenize("Lihat https://chatgangs.dev/api dan kirim ke a@b.co.id ya")
    urls = [t.text for t in tokens if t.is_url]
    emails = [t.text for t in tokens if t.is_email]
    assert urls == ["https://chatgangs.dev/api"]
    assert emails == ["a@b.co.id"]


def test_hashtag_mention_emoji_and_numbers():
    tokens = Tokenizer().tokenize("Halo @admin #promo 😊 harga Rp 15.000,00 dan 3.14")
    assert any(t.is_mention for t in tokens)
    assert any(t.is_hashtag for t in tokens)
    assert any(t.is_emoji for t in tokens)
    numbers = [t.text for t in tokens if t.is_number]
    assert "15.000,00" in numbers
    assert "3.14" in numbers


def test_japanese_and_korean_tokens():
    ja = Tokenizer().tokenize("これはテストです")
    assert len(ja) >= 1 and ja[0].is_word
    ko = Tokenizer().tokenize("안녕하세요 여러분")
    assert [t.text for t in ko if t.is_word] == ["안녕하세요", "여러분"]


def test_sentence_segmentation():
    sentences = Tokenizer.sentences("Ini kalimat pertama. Ini kalimat kedua! Apa ini? Ya.")
    assert len(sentences) == 4


def test_sentence_segmentation_respects_abbreviations():
    sentences = Tokenizer.sentences("Dr. Smith pulang. Dia lelah.")
    assert len(sentences) == 2


def test_contraction_splitting():
    tokens = Tokenizer().tokenize("I can't do it", split_contractions=True)
    words = [t.text.lower() for t in tokens if t.is_word]
    assert words[:3] == ["i", "can", "not"]


def test_stats_and_counts():
    stats = get_tokenizer().count("Halo! Saya ingin bertanya. Bagaimana caranya?")
    assert stats["tokens"] == 9
    assert stats["sentences"] == 3
    assert stats["characters"] > 0


def test_empty_and_whitespace_input():
    tokenizer = get_tokenizer()
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("    ") == []
    assert tokenizer.sentences("") == []


def test_control_characters_are_stripped():
    tokens = get_tokenizer().tokenize("Halo\u200b\u0000dunia")
    words = [t.text for t in tokens if t.is_word]
    assert "Halo" in words and "dunia" in words
