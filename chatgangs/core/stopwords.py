"""Stopword removal module."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set

from chatgangs.core.normalizer import get_dictionaries


class StopwordRemover:
    """Removes high-frequency, low-information words for a given language.

    Negation words are intentionally preserved for sentiment analysis; pass
    ``keep_negations=True`` (the default) to retain them.
    """

    NEGATIONS = {
        "id": {"tidak", "bukan", "belum", "jangan", "tak", "nggak", "gak", "ga"},
        "ms": {"tidak", "bukan", "belum", "jangan", "tak"},
        "en": {"not", "no", "never", "without", "hardly", "barely", "cannot"},
        "es": {"no", "nunca", "sin", "nada", "tampoco"},
        "de": {"nicht", "kein", "keine", "nie", "ohne"},
        "fr": {"pas", "non", "jamais", "sans", "aucun"},
    }

    def __init__(self) -> None:
        self._store = get_dictionaries()

    def stopwords(self, language: str) -> Set[str]:
        return set(self._store.stopword_set(language))

    def remove(
        self,
        tokens: Iterable[str],
        language: str = "id",
        *,
        keep_negations: bool = True,
    ) -> List[str]:
        stop = self.stopwords(language)
        keep = self.NEGATIONS.get(language, set()) if keep_negations else set()
        return [
            token for token in tokens
            if token not in stop or token in keep
        ]

    def filter_tokens(self, tokens, language: str = "id", *, keep_negations: bool = True):
        """Token-object variant that also annotates ``is_stopword``."""
        stop = self.stopwords(language)
        keep = self.NEGATIONS.get(language, set()) if keep_negations else set()
        out = []
        for token in tokens:
            token.is_stopword = token.normalized in stop and token.normalized not in keep
            out.append(token)
        return out

    def is_stopword(self, word: str, language: str = "id") -> bool:
        return word.lower() in self.stopwords(language)

    def coverage(self, tokens: Sequence[str], language: str = "id") -> float:
        """Ratio of stopwords in a token list (text-complexity signal)."""
        if not tokens:
            return 0.0
        stop = self.stopwords(language)
        return round(sum(1 for t in tokens if t.lower() in stop) / len(tokens), 4)


_remover: StopwordRemover | None = None


def get_stopword_remover() -> StopwordRemover:
    global _remover
    if _remover is None:
        _remover = StopwordRemover()
    return _remover
