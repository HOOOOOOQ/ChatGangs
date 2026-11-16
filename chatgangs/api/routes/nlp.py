"""NLP endpoints — one per module, each usable standalone."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from chatgangs.api.schemas import (
    AnalyzeRequest,
    ClassificationRequest,
    EntitiesRequest,
    IntentRequest,
    QARequest,
    SentimentRequest,
    SimilarityRequest,
    TextRequest,
    TokenizeRequest,
)
from chatgangs.api.deps import get_engine_dep
from chatgangs.core.pipeline import NLPengine
from chatgangs.core.tokenizer import get_tokenizer

router = APIRouter(prefix="/api/nlp", tags=["NLP"])


@router.post("/analyze", summary="Full NLP analysis")
def analyze(payload: AnalyzeRequest, engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Run the complete understanding pipeline in a single call.

    Returns language detection, normalised text, tokens (with offset, POS,
    lemma and stem), stopwords, entities, intent, sentiment, classification,
    keywords and knowledge-base candidates.
    """
    result = engine.analyze(
        payload.text,
        language=payload.language,
        explain=payload.explain,
        include_tokens=payload.include_tokens,
        include_qa=payload.include_qa,
    )
    return result.to_dict()


@router.post("/tokenize", summary="Tokenize text")
def tokenize(payload: TokenizeRequest) -> Dict[str, Any]:
    """Word-level tokens with offsets, sentences and statistics.

    ``include_stems`` additionally returns stems/lemmas for every word token.
    """
    from chatgangs.core.languages import detect_language
    from chatgangs.core.stemmer import get_stemmer

    tokenizer = get_tokenizer()
    language = payload.language or detect_language(payload.text).language
    tokens = tokenizer.tokenize(payload.text)
    output = []
    stemmer = get_stemmer() if payload.include_stems else None
    for token in tokens:
        item = token.to_dict(include_offsets=payload.include_offsets)
        if not payload.lowercase:
            item["text"] = token.text
        if stemmer and token.is_word and not token.is_number:
            item["stem"] = stemmer.stem(token.normalized, language)
            item["lemma"] = stemmer.lemmatize(token.normalized, language)
        output.append(item)
    return {
        "language": language,
        "tokens": output,
        "sentences": tokenizer.sentences(payload.text),
        "stats": tokenizer.count(payload.text),
    }


@router.post("/sentiment", summary="Sentiment analysis")
def sentiment(payload: SentimentRequest, engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Lexicon + (optional) trained-model sentiment with evidence terms."""
    return engine.analyze_sentiment(payload.text, payload.language)


@router.post("/intent", summary="Intent detection")
def intent(payload: IntentRequest, engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Detect the user's intent using rules, the trained model and fuzzy matching."""
    result = engine.detect_intent(payload.text, payload.language)
    if not payload.include_candidates:
        result.pop("candidates", None)
    if payload.threshold is not None and result["confidence"] < payload.threshold:
        result["below_threshold"] = True
    return result


@router.post("/entities", summary="Named entity recognition")
def entities(payload: EntitiesRequest, engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Extract named entities (PERSON, ORG, GPE, MONEY, DATE, EMAIL ...)."""
    result = engine.extract_entities(payload.text, payload.language)
    if payload.labels:
        allowed = {label.upper() for label in payload.labels}
        result["entities"] = [e for e in result["entities"] if e["label"].upper() in allowed]
        summary: Dict[str, list] = {}
        for entity in result["entities"]:
            summary.setdefault(entity["label"], []).append(entity["text"])
        result["summary"] = summary
    if not payload.use_spacy:
        result["entities"] = [e for e in result["entities"] if e["source"] != "spacy"]
    return result


@router.post("/language", summary="Language detection")
def language(payload: TextRequest) -> Dict[str, Any]:
    """Detect the language of the text (8 supported languages)."""
    from chatgangs.core.languages import detect_language

    return detect_language(payload.text).to_dict()


@router.post("/similarity", summary="Semantic similarity")
def similarity(payload: SimilarityRequest, engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Pairwise cosine similarity between texts using TF-IDF vectors."""
    return engine.similarity(payload.texts, payload.language)


@router.post("/classify", summary="Text classification")
def classify(payload: ClassificationRequest, engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Classify text with the trained model or a custom keyword taxonomy."""
    if payload.labels is not None:
        from chatgangs.core.semantics import get_text_classifier

        return get_text_classifier().classify(
            payload.text,
            labels=payload.labels,
            language=payload.language or "id",
            task=payload.task,
        ).to_dict()
    return engine.classify(payload.text, payload.language, task=payload.task)


@router.post("/qa", summary="Question answering")
def qa(payload: QARequest, engine: NLPengine = Depends(get_engine_dep)) -> Dict[str, Any]:
    """Answer a question from the knowledge base with retrieval evidence."""
    return engine.answer_question(payload.question, payload.language, payload.top_k)


@router.post("/stem", summary="Stemming / lemmatization")
def stem(payload: TokenizeRequest) -> Dict[str, Any]:
    """Stems and lemmas for each word token (module-level endpoint)."""
    from chatgangs.core.languages import detect_language
    from chatgangs.core.stemmer import get_stemmer

    stemmer = get_stemmer()
    language = payload.language or detect_language(payload.text).language
    words = [t for t in get_tokenizer().tokenize(payload.text) if t.is_word]
    return {
        "language": language,
        "backend": stemmer.info(),
        "tokens": [
            {
                "text": token.text,
                "stem": stemmer.stem(token.normalized, language),
                "lemma": stemmer.lemmatize(token.normalized, language),
            }
            for token in words
        ],
    }


@router.post("/normalize", summary="Text normalisation")
def normalize(payload: TextRequest) -> Dict[str, Any]:
    """Normalise text: lowercase, slang expansion, typo correction, PII redaction."""
    from chatgangs.core.languages import detect_language
    from chatgangs.core.normalizer import get_normalizer

    language = payload.language or detect_language(payload.text).language
    normalizer = get_normalizer()
    return {
        "language": language,
        "original": payload.text,
        "normalized": normalizer.normalize(payload.text, language=language, keep=("emoji",)),
        "normalized_no_pii": normalizer.normalize(
            payload.text, language=language, keep=("emoji",), redact_pii=True
        ),
    }


@router.post("/stopwords", summary="Stopword removal")
def stopwords(payload: TextRequest) -> Dict[str, Any]:
    """Remove stopwords for the detected language (negations preserved)."""
    from chatgangs.core.languages import detect_language
    from chatgangs.core.stopwords import get_stopword_remover
    from chatgangs.core.tokenizer import get_tokenizer

    language = payload.language or detect_language(payload.text).language
    tokens = [t.normalized for t in get_tokenizer().tokenize(payload.text) if t.is_word]
    remover = get_stopword_remover()
    stop = remover.stopwords(language)
    kept = remover.remove(tokens, language)
    removed = [token for token in tokens if token in stop]
    return {
        "language": language,
        "tokens": tokens,
        "removed": removed,
        "content_tokens": kept,
        "stopword_ratio": remover.coverage(tokens, language),
    }
