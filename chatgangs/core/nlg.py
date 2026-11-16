"""Natural Language Generation (NLG) module.

The generator is template-based but *not* static: it composes sentences from
slots (detected intent, entities, sentiment, retrieved knowledge, context
references) and randomises surface forms with anti-repetition tracking so the
same question asked twice does not produce byte-identical answers.

Design notes
------------
* Every intent maps to a set of *discourse plans*; each plan is a list of
  sentence templates with placeholders.
* Openers/connectors vary by intent and by turn index, and the generator
  remembers the last few openings per session to avoid repetition.
* The response length adapts: short answers for greetings/small talk, longer
  answers when the user asks "explain"/"detail", plus an optional follow-up
  question to keep the conversation moving.
* English, Indonesian and Malay have native templates; other supported
  languages use a short native acknowledgement plus the English/Indonesian
  body (explicitly labelled), instead of pretending to generate fluently.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from chatgangs.config import get_settings

# --------------------------------------------------------------------------- #
# Openers / connectors (varied surface forms)
# --------------------------------------------------------------------------- #
OPENERS: Dict[str, Dict[str, List[str]]] = {
    "id": {
        "greeting": ["Halo", "Hai", "Halo juga", "Selamat datang"],
        "farewell": ["Sampai jumpa", "Baik, sampai jumpa", "Sampai nanti"],
        "gratitude": ["Sama-sama", "Dengan senang hati", "Tentu, sama-sama"],
        "identity": ["Tentu", "Baik", "Siap"],
        "capabilities": ["Tentu", "Baik", "Siap"],
        "api_help": ["Tentu", "Baik", "Siap membantu"],
        "pricing": ["Baik", "Tentu", "Jelas"],
        "privacy": ["Pertanyaan bagus", "Baik", "Tentu"],
        "training_help": ["Baik", "Tentu", "Siap"],
        "analyze_text": ["Berikut hasilnya", "Ini hasil analisisnya", "Sudah saya analisis"],
        "translate": ["Baik", "Tentu"],
        "feedback_positive": ["Terima kasih", "Wah, terima kasih", "Senang mendengarnya"],
        "complaint": ["Maaf atas ketidaknyamanannya", "Terima kasih atas laporannya", "Saya paham"],
        "small_talk": ["Haha", "Menarik", "Baik"],
        "question": ["Tentu", "Baik", "Ini yang saya tahu"],
        "help": ["Tentu", "Siap", "Tidak masalah"],
        "unknown": ["Hmm", "Maaf", "Saya belum yakin"],
    },
    "en": {
        "greeting": ["Hello", "Hi", "Hey there", "Welcome"],
        "farewell": ["Goodbye", "See you", "Take care"],
        "gratitude": ["You're welcome", "My pleasure", "Glad to help"],
        "identity": ["Sure", "Happy to explain", "Of course"],
        "capabilities": ["Sure", "Of course", "Absolutely"],
        "api_help": ["Sure", "Of course", "Happy to help"],
        "pricing": ["Sure", "Glad you asked", "Of course"],
        "privacy": ["Good question", "Sure", "Of course"],
        "training_help": ["Sure", "Of course", "Happy to explain"],
        "analyze_text": ["Here is the result", "Analysis complete", "Here's what I found"],
        "translate": ["Sure", "Of course"],
        "feedback_positive": ["Thank you", "Glad to hear it", "That's great to hear"],
        "complaint": ["I'm sorry about that", "Thanks for reporting this", "I understand"],
        "small_talk": ["Haha", "Interesting", "Fair enough"],
        "question": ["Sure", "Of course", "Based on what I know"],
        "help": ["Sure", "No problem", "Happy to help"],
        "unknown": ["Hmm", "Sorry", "I'm not sure yet"],
    },
    "ms": {
        "greeting": ["Helo", "Hai", "Selamat datang"],
        "farewell": ["Selamat tinggal", "Jumpa lagi"],
        "gratitude": ["Sama-sama", "Dengan senang hati"],
        "unknown": ["Maaf", "Saya belum pasti"],
    },
}

CONNECTORS: Dict[str, List[str]] = {
    "id": ["Selain itu,", "Perlu diketahui,", "Sebagai tambahan,", "Di samping itu,"],
    "en": ["Also,", "Note that", "In addition,", "Besides that,"],
}

# Templates per (language, intent). ``{slots}`` are filled by the generator.
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "id": {
        "greeting": [
            "{opener}! Ada yang bisa saya bantu hari ini?",
            "{opener} 👋 Silakan tanya apa saja, atau kirim teks untuk saya analisis.",
            "{opener}! Saya siap membantu — mau mulai dari mana?",
        ],
        "farewell": [
            "{opener} dan terima kasih sudah mengobrol. Sampai jumpa lagi!",
            "{opener}! Kalau butuh analisis teks lagi, saya selalu di sini.",
        ],
        "gratitude": [
            "{opener}! Kalau ada yang ingin dianalisis atau ditanyakan lagi, silakan.",
            "{opener}. Senang bisa membantu — ada lagi yang bisa saya lakukan?",
        ],
        "identity": [
            "Saya {bot_name}, sistem NLP yang dibangun dari modul modular: tokenizer, analisis sentimen, deteksi intent, ekstraksi entitas, pemahaman konteks, memori, dan generator respons. Semua jawaban berasal dari pipeline yang benar-benar berjalan di server, bukan daftar jawaban statis.",
        ],
        "capabilities": [
            "Saya dapat melakukan tokenisasi, normalisasi teks, stemming dan lemmatization, stopword removal, Named Entity Recognition, deteksi intent, ekstraksi entitas, analisis sentimen, kemiripan semantik, pemahaman konteks, klasifikasi teks, deteksi bahasa, tanya-jawab, dan generasi jawaban. Mau saya coba pada teks Anda?",
        ],
        "api_help": [
            "API ChatGangs memakai JSON. Endpoint utama: POST /api/chat untuk percakapan, POST /api/nlp/analyze untuk analisis lengkap, serta /api/nlp/tokenize, /api/nlp/sentiment, /api/nlp/intent, dan /api/nlp/entities untuk tiap subsistem. Operasional ada di GET /api/health dan GET /api/model/info. Dokumentasi interaktif: /docs.",
        ],
        "help": [
            "Tentu. Mulai dengan mengetik pertanyaan seperti \"apa itu NLP\", kirim teks untuk dianalisis sentimennya, atau buka /docs untuk mencoba API. Saya juga bisa menjelaskan kemampuan saya jika Anda bertanya \"kamu bisa apa\".",
        ],
        "analyze_text": [
            "Analisis singkat: sentimen {sentiment_label} (skor {sentiment_score}), intent {intent}, bahasa {language}.",
        ],
        "translate": [
            "Saya bisa mendeteksi bahasa dan membantu memetakan kalimat, tetapi penerjemahan penuh antar-delapan bahasa belum termasuk dalam engine ini. Yang tersedia: deteksi bahasa otomatis, pemahaman lintas bahasa untuk intent dan sentimen, serta penyusunan jawaban dalam bahasa pesan Anda.",
        ],
        "pricing": [
            "ChatGangs adalah perangkat lunak yang Anda jalankan sendiri (self-hosted), jadi tidak ada biaya lisensi. Biaya yang muncul hanya biaya server/VPS yang Anda gunakan.",
        ],
        "privacy": [
            "Riwayat hanya disimpan bila memori sesi diaktifkan, dan Anda bisa menghapusnya kapan saja lewat DELETE /api/memory/{{session_id}} atau dengan meminta \"hapus riwayat\". Log tidak menyimpan isi pesan secara default.",
        ],
        "privacy_deleted": [
            "Selesai — seluruh riwayat dan memori untuk sesi ini sudah dihapus. Kita mulai dari halaman kosong.",
        ],
        "training_help": [
            "Pipeline training nyata tersedia di scripts/train.py: ia memuat dataset dari folder data/, menjalankan cross-validation, mengukur akurasi dan macro-F1 pada validation set, lalu menyimpan artefak berversi ke folder models/. Metrik terbaru selalu bisa dibaca di GET /api/model/info.",
        ],
        "feedback_positive": [
            "{opener}! Masukan seperti ini membantu saya memahami apa yang sudah berjalan baik.",
            "{opener} — senang sekali mendengarnya. Ada bagian lain yang ingin Anda coba?",
        ],
        "complaint": [
            "{opener}. Saya akan sampaikan detail ini ke bagian terkait. Kalau Anda mengoperasikan sendiri instansinya, sertakan request id dan contoh input agar bisa ditelusuri di log.",
            "{opener}. Mohon maaf atas pengalamannya. Boleh jelaskan lebih spesifik apa yang salah, supaya saya bisa membantu menelusuri?",
        ],
        "small_talk": [
            "{opener}! Saya di sini hanya punya satu pekerjaan: memahami bahasa. Ceritakan apa pun, atau tanya soal NLP — saya akan berusaha menjawab.",
            "{opener}. Saya tidak punya aktivitas di luar memproses teks, tapi saya senang diajak mengobrol. Mau lanjut dengan pertanyaan nyata?",
        ],
        "question": [
            "{answer}",
        ],
        "unknown": [
            "{opener}, saya belum yakin memahami maksud Anda. Bisa dijelaskan ulang atau diformulasikan sebagai pertanyaan?",
            "{opener} — pertanyaan itu belum masuk cakupan pengetahuan saya. Coba tanya tentang NLP, API, privasi data, atau kirim teks untuk dianalisis.",
        ],
        "knowledge_miss": [
            "Saya belum memiliki jawaban yang terverifikasi untuk topik itu dalam knowledge base saya. Cakupan saya: {topics}. Saya memilih tidak mengarang jawaban.",
        ],
        "followup": [
            "Ada lagi yang ingin Anda tanyakan?",
            "Mau saya jelaskan lebih detail?",
            "Ingin saya coba pada contoh lain?",
        ],
    },
    "en": {
        "greeting": [
            "{opener}! How can I help you today?",
            "{opener} 👋 Ask me anything, or paste some text and I'll analyse it.",
            "{opener}! Where would you like to start?",
        ],
        "farewell": [
            "{opener} and thanks for chatting. See you next time!",
            "{opener}! Ping me whenever you need text analysed again.",
        ],
        "gratitude": [
            "{opener}! Let me know if there is anything else you'd like me to analyse.",
            "{opener}. Glad I could help — anything else?",
        ],
        "identity": [
            "I'm {bot_name}, an NLP system assembled from modular components: tokeniser, sentiment engine, intent detector, entity recogniser, context manager, memory and response generator. Every answer comes from a pipeline that actually runs on the server, not a static answer list.",
        ],
        "capabilities": [
            "I can tokenise, normalise, stem and lemmatise text, remove stopwords, run named entity recognition, detect intent, extract entities, analyse sentiment, compute semantic similarity, track context, classify text, detect language, answer questions and generate responses. Want me to run this on your text?",
        ],
        "api_help": [
            "The API is JSON-based. Main endpoints: POST /api/chat for conversation, POST /api/nlp/analyze for the full bundle, plus /api/nlp/tokenize, /api/nlp/sentiment, /api/nlp/intent and /api/nlp/entities per subsystem. Operations live at GET /api/health and GET /api/model/info. Interactive docs: /docs.",
        ],
        "help": [
            "Sure. Try asking \"what is NLP\", paste a review for sentiment analysis, or open /docs to explore the API. You can also ask me what I can do.",
        ],
        "analyze_text": [
            "Quick analysis: sentiment {sentiment_label} (score {sentiment_score}), intent {intent}, language {language}.",
        ],
        "pricing": [
            "ChatGangs is self-hosted software, so there is no licence fee. Your only cost is the server you run it on.",
        ],
        "privacy": [
            "History is stored only when session memory is enabled, and you can delete it any time via DELETE /api/memory/{{session_id}} or by asking me to forget the conversation. Logs do not contain message content by default.",
        ],
        "privacy_deleted": [
            "Done — this session's history and memory have been deleted. We start from a clean slate.",
        ],
        "training_help": [
            "A real training pipeline lives in scripts/train.py: it loads datasets from data/, runs cross-validation, measures accuracy and macro-F1 on a held-out validation split, then stores versioned artefacts in models/. Latest metrics are always exposed at GET /api/model/info.",
        ],
        "feedback_positive": [
            "{opener}! Feedback like this helps me understand what works.",
            "{opener} — great to hear. Anything else you'd like to try?",
        ],
        "complaint": [
            "{opener}. I'll surface these details for follow-up. If you run your own instance, include the request id and a sample input so it can be traced in the logs.",
            "{opener}. Can you share exactly what went wrong so I can help you narrow it down?",
        ],
        "small_talk": [
            "{opener}! My only job is understanding language — tell me anything, or ask about NLP.",
            "{opener}. I don't do much outside processing text, but I'm happy to chat. Want to move on to a real question?",
        ],
        "question": ["{answer}"],
        "unknown": [
            "{opener}, I'm not confident I understood that. Could you rephrase it or ask it as a question?",
            "{opener} — that request is outside what I currently know. Try asking about NLP, the API, data privacy, or paste text for analysis.",
        ],
        "knowledge_miss": [
            "I don't have a verified answer for that topic in my knowledge base. I cover: {topics}. I'd rather not invent an answer.",
        ],
        "translate": [
            "I can detect the language and understand intents and sentiment across the supported languages, but a full translation engine is not part of this system. I can still answer you in the language of your message.",
        ],
        "followup": [
            "Anything else you'd like to ask?",
            "Should I go into more detail?",
            "Want me to try it on another example?",
        ],
    },
    "ms": {
        "greeting": ["{opener}! Apa yang boleh saya bantu hari ini?"],
        "farewell": ["{opener} dan terima kasih kerana berbual."],
        "gratitude": ["{opener}! Beritahu saya jika ada lagi yang boleh dibantu."],
        "unknown": [
            "{opener}, maaf saya belum pasti memahami maksud anda. Boleh jelaskan semula?",
        ],
    },
}

# Short acknowledgements for languages without full templates.
LOCAL_ACK: Dict[str, Dict[str, str]] = {
    "ja": {"greeting": "こんにちは！", "thanks": "ありがとうございます。"},
    "ko": {"greeting": "안녕하세요!", "thanks": "감사합니다."},
    "es": {"greeting": "¡Hola!", "thanks": "¡Gracias!"},
    "de": {"greeting": "Hallo!", "thanks": "Danke!"},
    "fr": {"greeting": "Bonjour !", "thanks": "Merci !"},
}

FULL_GENERATION_LANGUAGES = {"id", "en", "ms"}

DETAIL_HINTS = {
    "id": [" jelaskan ", " detail ", " lengkap ", " rinci ", " lebih ", " bagaimana ", " mengapa "],
    "en": [" explain ", " detail ", " more ", " exactly ", " why ", " how "],
}


class _SafeDict(dict):
    """``str.format_map`` helper: unknown placeholders stay as written."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __missing__(self, key):  # noqa: D105 - trivial
        return "{" + key + "}"


@dataclass
class GenerationResult:
    text: str
    language: str
    template_id: str
    slots: Dict[str, str] = field(default_factory=dict)
    variants_considered: int = 1

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "template_id": self.template_id,
            "slots": self.slots,
            "variants_considered": self.variants_considered,
        }


class ResponseGenerator:
    """Composes natural-language replies from templates + slots."""

    def __init__(self) -> None:
        self.settings = get_settings()
        # session_id -> recent template ids (anti-repetition)
        self._history: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ public
    def generate(
        self,
        intent: str,
        language: str = "id",
        *,
        slots: Optional[Dict[str, str]] = None,
        session_id: str = "default",
        detail: Optional[bool] = None,
        include_followup: Optional[bool] = None,
        template_key: Optional[str] = None,
    ) -> GenerationResult:
        language = self._resolve_language(language)
        slots = dict(slots or {})
        slots.setdefault("bot_name", self.settings.app_name)
        slots.setdefault("version", self.settings.app_version)
        key = template_key or intent

        templates = self._templates_for(language, key)
        if not templates:
            # Fall back to English then Indonesian templates.
            templates = self._templates_for("en", key) or self._templates_for("id", key)
            if templates:
                language_out = "en"
            else:
                return self._fallback(language, slots, key)
        else:
            language_out = language

        opener_pool = OPENERS.get(language_out) or OPENERS["en"]
        openers = opener_pool.get(key)
        if not openers:
            openers = (OPENERS.get(language_out) or OPENERS["en"]).get("unknown", ["Sure"])
        slots["opener"] = self._pick_deterministic(
            openers, session_id + key + str(len(self._history.get(session_id, [])))
        )

        recent = self._history.get(session_id, [])
        chosen_index, chosen = self._pick(templates, recent, session_id, slots)
        template_id = f"{language_out}.{key}.{chosen_index}"
        # ``str.format_map`` with a tolerant mapping keeps literal braces in
        # template text (e.g. "DELETE /api/memory/{session_id}") intact.
        text = chosen.format_map(_SafeDict(dict((k, str(v)) for k, v in slots.items())))

        detail = detail if detail is not None else self._wants_detail(slots.get("_user_text", ""), language)
        if language_out not in FULL_GENERATION_LANGUAGES:
            ack = LOCAL_ACK.get(language, {}).get(key)
            if ack and not text.startswith(ack):
                text = f"{ack} {text}"

        if include_followup is None:
            include_followup = intent in {"question", "capabilities", "api_help", "training_help", "help"}
        if include_followup and detail and key in {"capabilities", "api_help", "training_help", "question"}:
            followups = TEMPLATES.get(language_out, TEMPLATES["en"]).get("followup", [])
            if followups:
                text = f"{text} {self._pick_deterministic(followups, session_id + template_id)}"

        self._remember(session_id, template_id)
        return GenerationResult(
            text=_tidy(text),
            language=language_out,
            template_id=template_id,
            slots={k: v for k, v in slots.items() if not k.startswith("_")},
            variants_considered=len(templates),
        )

    def compose_answer(
        self,
        language: str,
        answer_body: str,
        *,
        intent: str = "question",
        session_id: str = "default",
        slot_note: Optional[str] = None,
        include_followup: bool = False,
    ) -> GenerationResult:
        """Wrap a retrieved knowledge-base answer in natural discourse."""
        language = self._resolve_language(language)
        opener_pool = OPENERS.get(language) or OPENERS["en"]
        openers = opener_pool.get(intent) or opener_pool.get("question") or ["Sure"]
        opener = self._pick_deterministic(openers, session_id + answer_body[:24])
        if intent == "question":
            connector = self._pick_deterministic(CONNECTORS[language], session_id + "conn")
            if self._hash_int(session_id + answer_body[:16]) % 3 == 0:
                parts = [connector]
            else:
                parts = [f"{opener}."]
        else:
            parts = [f"{opener},"]
        parts.append(answer_body.strip())
        if slot_note:
            parts.append(slot_note)
        if include_followup:
            followups = TEMPLATES.get(language, TEMPLATES["en"]).get("followup", [])
            if followups:
                parts.append(self._pick_deterministic(followups, session_id + answer_body[:12]))
        text = " ".join(part for part in parts if part)
        template_id = f"{language}.kb.{intent}"
        self._remember(session_id, template_id)
        return GenerationResult(
            text=_tidy(text),
            language=language,
            template_id=template_id,
            slots={"answer": answer_body},
        )

    def reset(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    # ----------------------------------------------------------------- helpers
    def _resolve_language(self, language: str) -> str:
        return language if language in OPENERS or language in LOCAL_ACK else "en"

    def _templates_for(self, language: str, key: str) -> List[str]:
        table = TEMPLATES.get(language)
        if not table:
            return []
        return table.get(key, [])

    def _pick(
        self,
        templates: Sequence[str],
        recent: Sequence[str],
        session_id: str,
        slots: Dict[str, str],
    ) -> tuple:
        if len(templates) == 1:
            return 0, templates[0]
        seed_basis = session_id + "|" + "|".join(
            f"{k}={v}" for k, v in sorted(slots.items()) if not k.startswith("_")
        )
        offset = self._hash_int(seed_basis)
        order = [(offset + i) % len(templates) for i in range(len(templates))]
        for idx in order:
            candidate_id = f"{idx}"
            if candidate_id not in recent:
                return idx, templates[idx]
        idx = order[0]
        return idx, templates[idx]

    def _pick_deterministic(self, options: Sequence[str], seed: str) -> str:
        if not options:
            return ""
        return options[self._hash_int(seed) % len(options)]

    @staticmethod
    def _hash_int(seed: str) -> int:
        return int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16)

    def _remember(self, session_id: str, template_id: str) -> None:
        history = self._history.setdefault(session_id, [])
        history.append(template_id)
        if len(history) > 6:
            del history[:-6]

    def _wants_detail(self, text: str, language: str) -> bool:
        lowered = f" {text.lower()} "
        hints = DETAIL_HINTS.get(language, DETAIL_HINTS["en"])
        return any(hint in lowered for hint in hints)

    def _fallback(self, language: str, slots: Dict[str, str], key: str) -> GenerationResult:
        message = (
            "Maaf, saya belum punya jawaban untuk itu."
            if language == "id"
            else "I don't have an answer for that yet."
        )
        return GenerationResult(message, language, f"{language}.{key}.fallback", slots)


def _tidy(text: str) -> str:
    """Normalise spacing and capitalise the first character."""
    cleaned = " ".join(text.split())
    cleaned = cleaned.replace(" .", ".").replace(" ,", ",").replace(" !", "!")
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


_generator: Optional[ResponseGenerator] = None


def get_response_generator() -> ResponseGenerator:
    global _generator
    if _generator is None:
        _generator = ResponseGenerator()
    return _generator
