# Referensi API

Semua endpoint menerima dan mengembalikan **JSON**. Header respons selalu
memuat `X-Request-ID` dan `X-Response-Time-ms`.

**Autentikasi (opsional).** Bila `CHATGANGS_REQUIRE_API_KEY=true`, sertakan
`X-API-Key: <key>`. Nama header dapat diubah lewat `CHATGANGS_API_KEY_HEADER`.

**Batasan.** Maksimum 4.000 karakter per teks analisis, 2.000 karakter per
pesan chat, 64 teks per batch. Pelanggaran menghasilkan `422`.

**Error format.**

```json
{
  "error": "validation_error",
  "request_id": "9f1c…",
  "details": [{"field": "body.message", "message": "String should have at least 1 character", "type": "string_too_short"}]
}
```

| Status | Arti |
| --- | --- |
| 400 | request tidak valid secara semantik |
| 401 | API key salah/tidak ada |
| 404 | sesi tidak ditemukan |
| 422 | validasi skema gagal |
| 429 | rate limit terlampaui (lihat header `Retry-After`) |
| 500 | error internal (request id disertakan) |

---

## Percakapan

### `POST /api/chat`

```json
{
  "message": "Halo, apa saja kemampuanmu?",
  "session_id": "demo",
  "language": null,
  "store_memory": true,
  "user_profile": {"name": "Budi"}
}
```

| Field | Tipe | Wajib | Keterangan |
| --- | --- | --- | --- |
| `message` | string (1–2000) | ya | pesan pengguna |
| `session_id` | string (1–128, alfanumerik/`-`/`_`) | tidak | default `"default"`; menentukan konteks & memori |
| `language` | enum `id,en,ms,ja,ko,es,de,fr` | tidak | paksa bahasa; default auto-detect |
| `store_memory` | boolean | tidak | `false` untuk mode privat per giliran |
| `user_profile` | object | tidak | metadata non-sensitif, disimpan bersama sesi |

Respons:

```json
{
  "session_id": "demo",
  "reply": "Saya dapat melakukan tokenisasi, normalisasi teks, …",
  "language": "id",
  "intent": {"name": "capabilities", "confidence": 0.93, "method": "rule+model",
             "evidence": ["rule:capabilities:'apa saja kemampuanmu'"],
             "candidates": {"capabilities": 0.93, "identity": 0.02}},
  "sentiment": {"label": "neutral", "score": 0.0, "confidence": 0.35},
  "entities": [],
  "context": {"is_followup": false, "topic": "capabilities", "turn_count": 1,
              "language": "id", "language_confidence": 0.87, "sentiment_trend": null},
  "generation": {"text": "…", "language": "id", "template_id": "id.capabilities.1",
                 "slots": {"opener": "Tentu", "bot_name": "ChatGangs"},
                 "variants_considered": 1},
  "memory": {"enabled": true, "stored": true, "message_id": 2},
  "latency_ms": 17.63,
  "suggestions": ["Bagaimana cara memakai API?", "…"]
}
```

Contoh pertanyaan lanjutan:

```bash
curl -X POST localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"jelaskan lebih detail","session_id":"demo"}'
# → context.is_followup = true, topik & jawaban mengikuti giliran sebelumnya
```

Menghapus riwayat lewat chat:

```bash
curl -X POST localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"hapus riwayat percakapan saya","session_id":"demo"}'
# → intent privacy; database + context dihapus; balasan konfirmasi
```

### `POST /api/feedback`

```json
{"session_id": "demo", "rating": 1, "comment": "jawabannya membantu", "message_id": 2}
```

`rating`: -1 (buruk) … 5 (sangat baik). Data dipakai untuk evaluasi manual,
**tidak** otomatis dipakai untuk training.

---

## NLP

### `POST /api/nlp/analyze`

```json
{"text": "Pelayanan PT Maju Jaya bagus tapi pengirimannya lambat. Hubungi budi@mail.com di Jakarta.",
 "language": null, "explain": true, "include_tokens": true, "include_qa": true}
```

Mengembalikan:

| Field | Isi |
| --- | --- |
| `language` | `{code, name, confidence, method}` |
| `normalized` | teks setelah normalisasi (slang, typo, lower-case) |
| `tokens[]` | `text, normalized, index, start, end, is_word, is_stopword, is_punctuation, pos, stem, lemma` |
| `token_stats` | `tokens, words, unique_words, characters, sentences, emojis, urls, emails, hashtags, mentions` |
| `stems`, `lemmas`, `stopwords`, `content_terms` | daftar string |
| `entities[]` | `text, label, start, end, confidence, source` |
| `entity_summary` | `{label: [texts]}` |
| `intent` | `name, confidence, method, evidence, candidates` |
| `sentiment` | `label, score, confidence, evidence[], sentences[], model_used` |
| `classification` | `label, confidence, method, scores, rationale, speech_act, topic, topic_scores` |
| `keywords[]` | `term, frequency, score` |
| `qa_matches[]` | `id, topic, question, score` |
| `stats` | latensi per tahap (`normalize_ms`, `sentiment_ms`, `total_ms`, …) |

### `POST /api/nlp/tokenize`

```json
{"text": "Halo dunia! Ini tes.", "include_offsets": true, "include_stems": true, "lowercase": true}
```

```json
{
  "language": "id",
  "tokens": [{"text": "Halo", "normalized": "halo", "index": 0, "start": 0, "end": 4,
              "is_word": true, "is_stopword": false, "is_punctuation": false,
              "stem": "halo", "lemma": "halo"}],
  "sentences": ["Halo dunia!", "Ini tes."],
  "stats": {"tokens": 6, "words": 4, "sentences": 2, "...": "..."}
}
```

### `POST /api/nlp/sentiment`

```json
{"text": "Pelayanannya bagus tapi pengirimannya lambat banget", "explain": true}
```

```json
{
  "label": "negative", "score": -0.457, "confidence": 0.71, "language": "id",
  "lexicon_score": -0.457, "model_score": -0.82, "model_used": true,
  "evidence": [{"term": "bagus", "polarity": 0.8, "kind": "word", "negated": false},
               {"term": "lambat", "polarity": -0.88, "kind": "word", "negated": false}],
  "sentences": [{"text": "pelayanannya bagus", "label": "positive", "score": 0.8,
                 "contrast_clause": false},
                {"text": "pengirimannya lambat banget", "label": "negative",
                 "score": -0.88, "contrast_clause": true}]
}
```

### `POST /api/nlp/intent`

```json
{"text": "berapa harganya?", "threshold": 0.3, "include_candidates": true}
```

```json
{"name": "pricing", "confidence": 1.0, "method": "rule",
 "evidence": ["rule:pricing:'harga'"],
 "candidates": {"pricing": 1.0, "capabilities": 0.01},
 "description": "User asks about cost or licensing.",
 "response_hint": "pricing"}
```

### `POST /api/nlp/entities`

```json
{"text": "PT Maju Jaya di Bandung, email budi@mail.com, 5 unit laptop",
 "labels": null, "use_spacy": true}
```

```json
{
  "language": "id",
  "entities": [
    {"text": "PT Maju Jaya", "label": "ORGANIZATION", "start": 0, "end": 12, "confidence": 0.88, "source": "pattern"},
    {"text": "Bandung", "label": "GPE", "start": 16, "end": 23, "confidence": 0.82, "source": "gazetteer"},
    {"text": "budi@mail.com", "label": "EMAIL", "start": 31, "end": 44, "confidence": 0.99, "source": "regex"},
    {"text": "5 unit", "label": "QUANTITY", "start": 46, "end": 52, "confidence": 0.7, "source": "regex"}
  ],
  "summary": {"ORGANIZATION": ["PT Maju Jaya"], "GPE": ["Bandung"], "EMAIL": ["budi@mail.com"], "QUANTITY": ["5 unit"]}
}
```

Label yang mungkin: `PERSON, ORGANIZATION, GPE, LOCATION, PRODUCT, EVENT, DATE, TIME, MONEY, PERCENT, QUANTITY, EMAIL, PHONE, URL, LANGUAGE, NORP, CARDINAL`.

### `POST /api/nlp/language`

```json
{"text": "안녕하세요 오늘 날씨가 좋네요"}
```

```json
{"language": "ko", "name": "Korean", "native_name": "한국어",
 "confidence": 0.98, "method": "script", "scores": {"ko": 0.98}, "supported": true}
```

### `POST /api/nlp/similarity`

```json
{"texts": ["kucing makan ikan", "saya suka kucing", "harga mobil mahal"]}
```

```json
{
  "corpus": ["kucing makan ikan", "saya suka kucing", "harga mobil mahal"],
  "matrix": [[1.0, 0.2455, 0.0], [0.2455, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "most_similar": {"i": 0, "j": 1, "text_i": "kucing makan ikan", "text_j": "saya suka kucing"},
  "score": 0.2455
}
```

### `POST /api/nlp/classify`

Dua jenis label yang **tidak dicampur**:

* `speech_act` — peran percakapan (`pertanyaan`, `perintah`, `keluhan`,
  `pujian`, `sapaan`, `ucapan_terima_kasih`, `lainnya`) dari taksonomi kata
  kunci, selalu dihitung.
* `topic` — topik dari model terlatih (`teknologi`, `bisnis`, `kesehatan`,
  `pendidikan`, `hiburan`) bila artefak klasifikasi tersedia.

`task` memilih keluaran: `auto` (default, label = speech-act + topik
menyertai), `topic` (paksa model topik), `speech_act` (hanya taksonomi).

Bentuk respons **selalu sama**: field `speech_act`, `speech_act_scores`,
`topic`, dan `topic_scores` selalu ada. `topic` bernilai `null` bila belum ada
artefak model topik (mis. setelah clone baru sebelum `scripts.train`) —
klien tidak perlu menebak ada/tidaknya field.

```json
{"text": "Tolong jelaskan cara kerja sistem ini"}
```

```json
{
  "label": "perintah",
  "confidence": 0.97,
  "method": "keyword-taxonomy+trained-topic",
  "scores": {"perintah": 0.4714, "pertanyaan": 0.0, "keluhan": 0.0},
  "rationale": ["perintah: tolong, jelaskan"],
  "speech_act": "perintah",
  "speech_act_scores": {"perintah": 0.4714, "...": 0.0},
  "topic": "pendidikan",
  "topic_scores": {"pendidikan": 0.41, "teknologi": 0.22, "...": 0.0}
}
```

Model topik saja:

```json
{"text": "film horor itu pecahkan rekor", "task": "topic"}
```

```json
{"label": "hiburan", "confidence": 0.56, "method": "trained-model",
 "scores": {"hiburan": 0.56, "pendidikan": 0.1575, "teknologi": 0.1152},
 "speech_act": "lainnya", "speech_act_scores": {"keluhan": 0.0, "...": 0.0},
 "topic": "hiburan", "topic_scores": {"hiburan": 0.56, "...": 0.0}}
```

Taksonomi kustom (`task` diabaikan; label pengguna dipakai):

```json
{"text": "pengiriman lambat", "labels": {"logistik": ["kirim", "pengiriman", "kurir"], "pembayaran": ["bayar", "transfer"]}}
```

```json
{"label": "logistik", "confidence": 0.97, "method": "custom-taxonomy",
 "scores": {"logistik": 0.5774, "pembayaran": 0.0},
 "rationale": ["logistik: pengiriman"], "speech_act": "keluhan"}
```

### `POST /api/nlp/qa`

```json
{"question": "bagaimana cara memakai api", "top_k": 3}
```

```json
{
  "question": "bagaimana cara memakai api",
  "language": "id",
  "answer": "API ChatGangs memakai JSON. Endpoint utama: POST /api/chat …",
  "answered": true,
  "matches": [{"id": "kb.api", "topic": "api", "question": "Bagaimana cara memakai API ChatGangs?", "score": 0.504}],
  "kb_version": "1.0.0"
}
```

`answer` bernilai `null` bila tidak ada kecocokan yang cukup kuat — sistem
memilih tidak mengarang jawaban.

### `POST /api/nlp/normalize`, `/stem`, `/stopwords`, `/batch`

```bash
curl -X POST localhost:8000/api/nlp/normalize -H 'Content-Type: application/json' \
  -d '{"text":"gmn cara pkai API, email gw budi@mail.com"}'
# {"language":"id","original":"…","normalized":"bagaimana cara pakai api, email saya [email]",
#  "normalized_no_pii":"…"}
```

```bash
curl -X POST localhost:8000/api/nlp/batch -H 'Content-Type: application/json' \
  -d '{"texts":["halo","saya kecewa"],"tasks":["sentiment","intent","entities"]}'
```

---

## Memori & privasi

| Endpoint | Keterangan |
| --- | --- |
| `GET /api/memory?limit=20&offset=0` | daftar sesi tersimpan + statistik |
| `GET /api/memory/{session_id}` | detail sesi, pesan, dan konteks |
| `DELETE /api/memory/{session_id}` | hapus sesi + semua pesannya + konteks |
| `DELETE /api/memory` | hapus sesi yang sudah kedaluwarsa (TTL) |

```json
// DELETE /api/memory/demo
{"status": "deleted", "session_id": "demo", "persisted_deleted": true, "context_reset": true}
```

---

## Sistem

### `GET /api/health`

```json
{
  "status": "ok", "version": "1.0.0", "environment": "production", "uptime_seconds": 612.4,
  "checks": {
    "database": {"status": "ok", "backend": "postgresql"},
    "nlp_engine": {"status": "ok", "intents": 16, "kb_entries": 25, "spacy": true},
    "memory": {"status": "ok", "sessions": 12, "messages": 84, "context": {"sessions": 3}}
  }
}
```

### `GET /api/model/info`

```json
{
  "name": "ChatGangs", "version": "1.0.0", "environment": "production",
  "engines": {
    "tokenizer": {"backend": "regex-unicode"},
    "stemmer": {"indonesian_backend": "sastrawi", "english_backend": "porter-subset"},
    "sentiment": {"available": true, "metadata": {"version": "20261116.015014"}},
    "intent": {"available": true, "intent_count": 16, "algorithm": "tfidf+logistic-regression"},
    "entities": {"backends": ["regex", "gazetteer", "pattern", "spacy"],
                 "spacy": {"available": true, "model": "en_core_web_sm"}},
    "question_answering": {"backend": "tfidf-retrieval+template-nlg", "kb_entries": 25},
    "generation": {"backend": "slot-template-nlg", "languages": ["id", "en", "ms"]}
  },
  "artifacts": {
    "intent": {"trained": true, "version": "20261116.014947", "validation_accuracy": 0.7848},
    "sentiment": {"trained": true, "version": "20261116.015014", "validation_accuracy": 0.8082},
    "classification": {"trained": true, "validation_accuracy": 0.86}
  },
  "training": {"pipeline": "scripts/train.py", "intent_model_trained": true}
}
```

### `GET /api/metrics`

```json
{
  "requests_recent": 200, "error_rate_recent": 0.005,
  "latency_ms": {"p50": 11.2, "p90": 28.4, "p99": 61.9, "max": 120.3},
  "memory": {"sessions": 12, "messages": 84},
  "rate_limiter": {"tracked_clients": 4, "limit": 120, "window_seconds": 60},
  "caches": {"normalizer": {"hit_rate": 0.62}},
  "uptime_seconds": 612.4
}
```

### Katalog

* `GET /api/languages` → daftar 8 bahasa dengan prioritas `id`, `en`.
* `GET /api/intents` → nama, deskripsi, contoh tiap intent.
* `GET /api/knowledge-base` → versi, jumlah entri, topik, daftar pertanyaan.
