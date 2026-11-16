"""Tokenizer module.

Three tokenisation strategies are provided behind one interface:

``word``   – Unicode-aware word/punctuation tokenisation with offsets.
``sentence`` – rule-based multilingual sentence segmentation.
``subword``  – byte-pair-encoded style subword units produced by a simple
               deterministic whitespace/character n-gram splitter, useful for
               TF-IDF character features and for inspecting OOV words.

Tokenisation is script-aware: Latin, Cyrillic, Greek, Arabic, Hangul and the
Han/Kana blocks are each handled, and emoji sequences are kept intact via the
``is_emoji`` flag instead of being torn into surrogate halves.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from chatgangs.core.text_utils import Token, clean_text, split_sentences

# ---------------------------------------------------------------------------
# Regex building blocks
# ---------------------------------------------------------------------------
_TLDS = (
    "com|co|id|net|org|io|ai|dev|app|xyz|me|info|biz|edu|gov|ac|sch|web|site|"
    "online|tech|store|shop|my|sg|jp|kr|de|fr|es|uk|us|tv|news|blog|cloud"
)
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+"
    r"|www\.[^\s<>\"']+"
    rf"|\b(?:[\w-]+\.)+(?:{_TLDS})\b(?:/[^\s<>\"']*)?"
    r"|\b[\w.+-]+@[\w-]+\.[\w.-]+\b",  # e-mail treated as URL-like token
    re.IGNORECASE,
)
_MENTION_RE = re.compile(r"(?<![\w@])@[\w.]{1,64}")
_HASHTAG_RE = re.compile(r"(?<![\w#])#[\w\u00C0-\u024F]{1,64}")
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?",
    re.IGNORECASE,
)
# Emoji blocks (pictographs, dingbats, emoticon ranges, flags, variation selectors)
_EMOJI_RE = re.compile(
    "(["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "\U0001F1E6-\U0001F1FF"
    "\U00002764"
    "\U0001F900-\U0001F9FF"
    "]+)",
)
_WORD_RE = re.compile(
    r"[\u00C0-\u024F\u1E00-\u1EFFA-Za-z\u0400-\u04FF\u0370-\u03FF]+"          # Latin/Cyrillic/Greek
    r"(?:['’\-][\u00C0-\u024F\u1E00-\u1EFFA-Za-z]+)*"
    r"|[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+"                # Kana + Han
    r"|[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]+"                            # Hangul
    r"|[\u0600-\u06FF\u0750-\u077F]+"                                          # Arabic
    r"|[\u0E00-\u0E7F]+",                                                      # Thai
)
_PUNCT_RE = re.compile(r"[^\w\s\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u0E00-\u0E7F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]")
_WS_RE = re.compile(r"\s+")

CONTRACTIONS = {
    "can't": ["can", "not"],
    "won't": ["will", "not"],
    "shan't": ["shall", "not"],
    "n't": ["not"],
    "'re": ["are"],
    "'ve": ["have"],
    "'ll": ["will"],
    "'d": ["would"],
    "'m": ["am"],
    "y'know": ["you", "know"],
    "gonna": ["going", "to"],
    "wanna": ["want", "to"],
}


def _match_kind(text: str) -> Optional[str]:
    if _URL_RE.fullmatch(text):
        return "url"
    if _MENTION_RE.fullmatch(text):
        return "mention"
    if _HASHTAG_RE.fullmatch(text):
        return "hashtag"
    if _EMOJI_RE.fullmatch(text):
        return "emoji"
    if _NUMBER_RE.fullmatch(text):
        return "number"
    if _WORD_RE.fullmatch(text):
        return "word"
    return None


def _iter_matches(text: str) -> Iterable[tuple]:
    """Yield ``(start, end, kind)`` for every non-whitespace unit in ``text``."""
    pos = 0
    length = len(text)
    patterns = (
        ("url", _URL_RE),
        ("mention", _MENTION_RE),
        ("hashtag", _HASHTAG_RE),
        ("emoji", _EMOJI_RE),
        ("number", _NUMBER_RE),
        ("word", _WORD_RE),
    )
    while pos < length:
        if text[pos].isspace():
            pos += 1
            continue
        best = None
        best_end = pos
        for kind, pattern in patterns:
            match = pattern.match(text, pos)
            if match and match.end() > best_end:
                best = (kind, match.start(), match.end())
                best_end = match.end()
        if best:
            kind, start, end = best
            if kind == "url":
                # Trailing sentence punctuation belongs to the sentence, not
                # the URL (e.g. "lihat tokopedia.com.").
                while end > start and text[end - 1] in ".,!?;:":
                    end -= 1
                    kind = "url"
            yield start, end, kind
            pos = end
            continue
        # punctuation or unknown single character
        char = text[pos]
        if char.isspace():
            pos += 1
            continue
        yield pos, pos + 1, "punct"
        pos += 1


class Tokenizer:
    """Stateless, thread-safe tokenizer."""

    def __init__(self, lowercase_normalized: bool = True) -> None:
        self.lowercase_normalized = lowercase_normalized

    # ---------------------------------------------------------------- words
    def tokenize(
        self,
        text: str,
        *,
        keep_punctuation: bool = True,
        split_contractions: bool = False,
        with_offsets: bool = True,
    ) -> List[Token]:
        """Tokenise ``text`` into :class:`Token` objects.

        The offsets always refer to the *original* (cleaned) string so callers
        can highlight matches in the user interface.
        """
        cleaned = clean_text(text or "")
        tokens: List[Token] = []
        idx = 0
        for start, end, kind in _iter_matches(cleaned):
            raw = cleaned[start:end]
            if kind == "word" and split_contractions and "n't" in raw.lower():
                pieces = self._split_contraction(raw)
                offset = start
                for piece in pieces:
                    piece_start = cleaned.find(piece, offset, end + len(piece))
                    if piece_start < 0:
                        piece_start = offset
                    tokens.append(
                        Token(
                            text=piece,
                            normalized=piece.lower() if self.lowercase_normalized else piece,
                            index=idx,
                            start=piece_start,
                            end=piece_start + len(piece),
                            is_word=True,
                        )
                    )
                    offset = piece_start + len(piece)
                    idx += 1
                continue
            token = Token(
                text=raw,
                normalized=raw.lower() if self.lowercase_normalized else raw,
                index=idx,
                start=start if with_offsets else 0,
                end=end if with_offsets else 0,
                is_word=kind in ("word", "number"),
                is_punctuation=kind == "punct",
                is_emoji=kind == "emoji",
                is_url=kind == "url" and "@" not in raw,
                is_email=kind == "url" and "@" in raw,
                is_mention=kind == "mention",
                is_hashtag=kind == "hashtag",
                is_number=kind == "number",
            )
            if not keep_punctuation and token.is_punctuation:
                continue
            tokens.append(token)
            idx += 1
        return tokens

    @staticmethod
    def _split_contraction(word: str) -> List[str]:
        """Split "can't" into ["can", "not"] without mangling the stem."""
        lower = word.lower()
        if lower.endswith("n't") and len(word) > 3:
            base = word[:-3]
            # "can't" → "can" + "not" (the n is part of "can")
            if base.lower().endswith("ca"):
                base = base + "n"
            return [base, "not"]
        for suffix, replacement in CONTRACTIONS.items():
            if suffix == "n't":
                continue
            if lower.endswith(suffix) and len(word) > len(suffix):
                return [word[: -len(suffix)]] + replacement
        return [word]

    # ------------------------------------------------------------- sentences
    @staticmethod
    def sentences(text: str) -> List[str]:
        return split_sentences(clean_text(text or ""))

    # -------------------------------------------------------------- subwords
    @staticmethod
    def subwords(word: str, min_n: int = 3, max_n: int = 5) -> List[str]:
        """Character n-gram decomposition (BPE-style inspection helper)."""
        word = (word or "").lower()
        if " " in word:
            return [word]
        if len(word) <= min_n:
            return [f"<{word}>"] if word else []
        padded = f"<{word}>"
        grams = set()
        for n in range(min_n, max_n + 1):
            for i in range(len(padded) - n + 1):
                grams.add(padded[i : i + n])
        return sorted(grams)

    # --------------------------------------------------------------- helpers
    def count(self, text: str) -> dict:
        tokens = self.tokenize(text)
        words = [t for t in tokens if t.is_word and not t.is_number]
        return {
            "tokens": len(tokens),
            "words": len(words),
            "unique_words": len({t.normalized for t in words}),
            "characters": len(clean_text(text or "")),
            "characters_no_spaces": len(re.sub(r"\s+", "", text or "")),
            "sentences": len(self.sentences(text)),
            "emojis": sum(1 for t in tokens if t.is_emoji),
            "urls": sum(1 for t in tokens if t.is_url),
            "emails": sum(1 for t in tokens if t.is_email),
            "hashtags": sum(1 for t in tokens if t.is_hashtag),
            "mentions": sum(1 for t in tokens if t.is_mention),
        }


_default_tokenizer: Optional[Tokenizer] = None


def get_tokenizer() -> Tokenizer:
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = Tokenizer()
    return _default_tokenizer
