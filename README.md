# ChatGangs — Modular NLP & Conversational AI Engine

ChatGangs adalah engine **Natural Language Processing** yang modular dan
benar-benar berjalan: tokenisasi, normalisasi, stemming, lemmatization,
stopword removal, Named Entity Recognition, deteksi intent, ekstraksi entitas,
analisis sentimen, kemiripan semantik, pemahaman konteks, klasifikasi teks,
deteksi bahasa (8 bahasa), question answering, dan Natural Language
Generation — semuanya diekspos lewat REST API dan antarmuka web.

> **Prinsip kejujuran teknis.** ChatGangs **bukan** LLM dan tidak mengklaim
> begitu. Semua jawaban berasal dari pipeline yang dieksekusi di server
> (leksikon terbobot, TF‑IDF, model Logistic Regression yang dilatih nyata,
> retrieval knowledge base, dan template NLG dengan slot). Setiap komponen
> melaporkan backend yang benar-benar aktif di `GET /api/model/info`, dan
> endpoint training menjalankan training sungguhan beserta cross‑validation.

---

## 1. Demo cepat

```bash
# 1. instalasi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. latih model (opsional tetapi disarankan — metrik nyata, ±15 detik)
python -m scripts.train --task all

# 3. jalankan
python -m chatgangs --host 0.0.0.0 --port 8000
```

* Web client → <http://localhost:8000/>
* Swagger UI → <http://localhost:8000/docs>
* ReDoc → <http://localhost:8000/redoc>
* Health → <http://localhost:8000/api/health>

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Halo, apa saja kemampuanmu?","session_id":"demo"}'
```

```json
{
  "session_id": "demo",
  "reply": "Saya dapat melakukan tokenisasi, normalisasi teks, stemming ...",
  "language": "id",
  "intent": {"name": "capabilities", "confidence": 0.93, "method": "rule+model"},
  "sentiment": {"label": "neutral", "score": 0.0, "confidence": 0.35},
  "entities": [],
  "context": {"is_followup": false, "topic": "capabilities", "turn_count": 1},
  "memory": {"enabled": true, "stored": true, "message_id": 2},
  "latency_ms": 17.6
}
```

---

## 2. Arsitektur

```
                         ┌──────────────────────────────┐
  HTTP (JSON)  ────────► │  API Gateway (FastAPI)        │
                         │  auth · rate limit · req-id   │
                         └───────────────┬──────────────┘
                                         │
                 ┌───────────────────────▼────────────────────────┐
                 │              NLP ENGINE (pipeline.py)          │
                 │                                                │
   ┌─────────────┴─────────────┐        ┌───────────────────────┐ │
   │ Language Processor        │        │ Context Manager       │ │
   │ detect_language (8 lang)  │        │ follow-up resolution  │ │
   └─────────────┬─────────────┘        │ topic · sentiment arc │ │
                 │                      └───────────┬───────────┘ │
   ┌─────────────▼─────────────┐        ┌───────────▼───────────┐ │
   │ Tokenizer                 │        │ Memory Manager        │ │
   │ word / sentence / subword │        │ SQLite/PostgreSQL     │ │
   └─────────────┬─────────────┘        │ TTL · delete · consent│ │
                 │                      └───────────┬───────────┘ │
   ┌─────────────▼─────────────┐        ┌───────────▼───────────┐ │
   │ Normalizer                │        │ Response Generator    │ │
   │ slang · typo · PII redact │        │ slot templates · NLG  │ │
   └─────────────┬─────────────┘        │ anti-repetition       │ │
                 │                      └───────────┬───────────┘ │
   ┌─────────────▼───────┬───────────┬──────────────▼───────────┐ │
   │ Stemmer/Lemmatizer  │ Entities  │ Intent     │ Sentiment    │ │
   │ ID · EN · ES DE FR  │ NER       │ rules+ML   │ lexicon+ML   │ │
   └─────────────────────┴─────┬─────┴────────────┴──────────────┘ │
                               │                                  │
                 ┌─────────────▼──────────────┐  ┌───────────────┴─┐
                 │ Semantics (TF-IDF)         │  │ Knowledge Base  │
                 │ similarity · classify · QA │  │ retrieval + NLG │
                 └────────────────────────────┘  └─────────────────┘
```

### Modul dan tanggung jawab

| Folder | Komponen | Tanggung jawab |
| --- | --- | --- |
| `chatgangs/core/languages.py` | Language Processor | deteksi 8 bahasa (script + lexical + statistik), registry bahasa |
| `chatgangs/core/tokenizer.py` | Tokenizer | token word/punctuation + offset, segmen kalimat, subword n-gram, emoji/URL/hashtag/nomor |
| `chatgangs/core/normalizer.py` | Normalizer | NFKC, ekspansi slang ID/EN, koreksi typo konservatif, redaksi PII, cache LRU |
| `chatgangs/core/stemmer.py` | Stemmer / Lemmatizer | stemmer Indonesia (rule + kamus akar + Nazief style), Porter EN, tabel ES/DE/FR, lemma ireguler |
| `chatgangs/core/stopwords.py` | Stopword Remover | 8 bahasa, negasi dipertahankan untuk sentimen |
| `chatgangs/core/entities.py` | NER | regex (email/URL/telepon/uang/persen/tanggal/waktu/kuantitas), gazetteer, pola organisasi, spaCy opsional |
| `chatgangs/core/intent.py` | Intent Detection | aturan regex presisi tinggi + model TF-IDF/LogReg + fuzzy Jaccard, katalog dari `data/intents/intents.json` |
| `chatgangs/core/sentiment.py` | Sentiment Analysis | leksikon + negasi + amplifier/diminisher + klausa kontras + emoji, blending model terlatih |
| `chatgangs/core/semantics.py` | Semantic & QA | TF-IDF vectorizer sendiri (uni+bigram, sublinear tf, L2), similarity, classifier, knowledge base retrieval |
| `chatgangs/core/context.py` | Context Manager | follow-up detection, resolusi referensi, topik aktif, tren sentimen |
| `chatgangs/core/nlg.py` | Response Generator | template berslot, opener bervariasi, anti-repetisi per sesi, adaptasi panjang |
| `chatgangs/core/pipeline.py` | NLP Engine | orkestrasi semua modul; `analyze()` dan `chat()` |
| `chatgangs/core/spacy_pipeline.py` | spaCy bridge | opsional; melaporkan status nyata, fallback otomatis |
| `chatgangs/memory/` | Memory | SQLAlchemy models, manager, TTL purge, delete, audit log metadata |
| `chatgangs/api/` | API Gateway | skema Pydantic ketat, middleware request-id, auth key, rate limiter, error handler |
| `chatgangs/training/` | Learning System | dataset loader/validator, split, cross-validation, evaluasi, versioning |
| `chatgangs/monitoring/` | Observability | ringkasan metrik latensi/error dan cache |
| `scripts/train.py` | Training CLI | `--task intent|sentiment|classification|qa|all` |
| `web/` | Web Client | 3 tampilan: percakapan, analisis teks, model & sistem |

---

## 3. API

| Method | Endpoint | Deskripsi |
| --- | --- | --- |
| POST | `/api/chat` | percakapan sadar konteks & memori |
| POST | `/api/nlp/analyze` | analisis lengkap (semua modul sekaligus) |
| POST | `/api/nlp/tokenize` | tokenisasi + statistik + stem/lemma |
| POST | `/api/nlp/sentiment` | analisis sentimen + bukti |
| POST | `/api/nlp/intent` | deteksi intent + kandidat |
| POST | `/api/nlp/entities` | Named Entity Recognition |
| POST | `/api/nlp/language` | deteksi bahasa |
| POST | `/api/nlp/similarity` | kemiripan semantik berpasangan |
| POST | `/api/nlp/classify` | klasifikasi teks (model atau taksonomi kustom) |
| POST | `/api/nlp/qa` | tanya jawab knowledge base |
| POST | `/api/nlp/normalize` | normalisasi + redaksi PII |
| POST | `/api/nlp/stem` | stemming & lemmatization |
| POST | `/api/nlp/stopwords` | stopword removal |
| POST | `/api/nlp/batch` | analisis batch (≤64 teks) |
| GET | `/api/health` | liveness + dependency checks |
| GET | `/api/model/info` | backend tiap engine, artefak model, metrik |
| GET | `/api/metrics` | p50/p90/p99 latensi, error rate, cache |
| GET | `/api/languages` | daftar bahasa (prioritas: id, en) |
| GET | `/api/intents` | katalog intent |
| GET | `/api/knowledge-base` | topik & pertanyaan knowledge base |
| GET/DELETE | `/api/memory[/{session_id}]` | lihat & hapus riwayat percakapan |
| POST | `/api/feedback` | rating jawaban untuk evaluasi |

Panduan lengkap (contoh request/response tiap endpoint) ada di
[`docs/API.md`](docs/API.md).

---

## 4. Contoh penggunaan

### cURL

```bash
# analisis sentimen
curl -X POST localhost:8000/api/nlp/sentiment \
  -H 'Content-Type: application/json' \
  -d '{"text":"Pelayanannya bagus tapi pengirimannya lambat banget"}'
```

### Python

```python
import requests

BASE = "http://localhost:8000"

chat = requests.post(f"{BASE}/api/chat", json={
    "message": "Siapa kamu?", "session_id": "demo"
}).json()
print(chat["reply"])

analysis = requests.post(f"{BASE}/api/nlp/analyze", json={
    "text": "PT Maju Jaya di Bandung, hubungi budi@mail.com sebelum 12/03/2025"
}).json()
print(analysis["entities"])
```

Contoh lengkap: [`examples/python_client.py`](examples/python_client.py),
[`examples/javascript_client.js`](examples/javascript_client.js),
[`examples/curl_cheatsheet.sh`](examples/curl_cheatsheet.sh).

### Sebagai library (tanpa HTTP)

```python
from chatgangs.core.pipeline import get_engine

engine = get_engine()
print(engine.analyze("layanan sangat buruk").sentiment["label"])   # negative
print(engine.chat("halo", session_id="local").reply)
```

---

## 5. Training & evaluasi

```bash
python -m scripts.train --task intent          # dataset: data/intents/intents.json
python -m scripts.train --task sentiment       # dataset: data/sentiment/train.csv
python -m scripts.train --task classification  # dataset: data/classification/train.csv
python -m scripts.train --task all             # + evaluasi QA dan detektor intent end-to-end
```

Yang benar-benar terjadi pada setiap run:

1. dataset dimuat dan **divalidasi** (minimal 3 contoh/kelas, ≥2 kelas);
2. split stratifikasi train/validation (deterministik, seed 42);
3. **cross-validation** k-fold dengan scoring macro-F1 pada train split;
4. fit `TF-IDF(word 1-2 gram) + TF-IDF(char 3-5) → LogisticRegression`;
5. evaluasi pada validation set: accuracy, macro-F1, weighted-F1, laporan per
   kelas, confusion matrix;
6. artefak disimpan ke `models/<task>/` (`model.joblib`, `metrics.json`,
   `classification_report.txt`) dan langsung dipakai API.

Hasil referensi pada repo ini (dataset kecil, bukan benchmark produksi):

| Task | Dataset | Validation accuracy | Macro-F1 | CV (5-fold) |
| --- | --- | --- | --- | --- |
| intent | 395 contoh / 16 kelas | 0,785 | 0,786 | 0,719 ± 0,065 |
| sentiment | 362 contoh / 3 kelas | 0,808 | 0,812 | 0,766 ± 0,051 |
| klasifikasi topik | 222 contoh / 5 topik | 0,864 | 0,856 | 0,696 ± 0,078 |
| QA retrieval | 30 pertanyaan berlabel | top-1 0,933 | top-3 0,967 | — |
| intent end-to-end | 395 contoh | 0,952 (aturan+model) | — | — |

Angka ini **berubah** setiap kali dataset berubah — jalankan ulang perintah di
atas dan baca `GET /api/model/info` untuk metrik versi yang sedang aktif.

---

## 6. Konfigurasi

Semua pengaturan lewat environment variable berprefiks `CHATGANGS_`; lihat
[`.env.example`](.env.example) untuk daftar lengkap. Yang paling sering diubah:

| Variabel | Default | Fungsi |
| --- | --- | --- |
| `CHATGANGS_DATABASE_URL` | `sqlite:///./data/chatgangs.db` | SQLite (dev) atau PostgreSQL (produksi) |
| `CHATGANGS_REQUIRE_API_KEY` | `false` | wajibkan header `X-API-Key` |
| `CHATGANGS_API_KEYS` | *(kosong)* | daftar key dipisah koma |
| `CHATGANGS_RATE_LIMIT_REQUESTS` | `120` | batas request per klien per window |
| `CHATGANGS_MAX_INPUT_CHARS` | `4000` | batas ukuran input analisis |
| `CHATGANGS_MAX_MESSAGE_CHARS` | `2000` | batas ukuran pesan chat |
| `CHATGANGS_MEMORY_PERSIST` | `true` | simpan riwayat ke database |
| `CHATGANGS_MEMORY_TTL_HOURS` | `72` | masa simpan riwayat |
| `CHATGANGS_LOG_CHAT_CONTENT` | `false` | (jangan) tulis isi pesan ke log |
| `CHATGANGS_STEMMER_BACKEND` | `builtin` | `builtin` / `sastrawi` / `auto` |

---

## 7. Testing

```bash
pytest                      # 182 tes: unit + integrasi API
pytest --cov=chatgangs      # dengan laporan coverage
```

Cakupan tes: tokenizer, normalizer, stemmer, stopwords, sentimen (termasuk
negasi/amplifikasi/kontras), intent (termasuk typo dan frame sosial), NER,
semantik/QA, deteksi bahasa 8 bahasa, context & memory (termasuk penghapusan
data dan perluasan jawaban pada follow-up "jelaskan lebih detail"),
pipeline end-to-end, seluruh endpoint API, dan pipeline training.

---

## 8. Deployment

```bash
docker compose up --build            # API + web client pada :8000
docker compose --profile postgres up -d   # tambah PostgreSQL
```

Panduan VPS/Linux (systemd, Nginx, HTTPS, PostgreSQL, backup, scaling) ada di
[`docs/INSTALLATION.md`](docs/INSTALLATION.md) dan
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

---

## 9. Dokumentasi

| Dokumen | Isi |
| --- | --- |
| [`docs/INSTALLATION.md`](docs/INSTALLATION.md) | instalasi lokal, Docker, VPS, PostgreSQL, opsional NLP ekstra |
| [`docs/API.md`](docs/API.md) | referensi endpoint + contoh request/response |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | detail modul, alur data, keputusan desain |
| [`docs/NLP_COMPONENTS.md`](docs/NLP_COMPONENTS.md) | cara kerja tiap modul NLP & batasannya |
| [`docs/TRAINING.md`](docs/TRAINING.md) | format dataset, training, evaluasi, versi model |
| [`docs/SECURITY.md`](docs/SECURITY.md) | validasi, rate limit, auth, privasi, logging |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | health, metrik, log, backup, troubleshooting |

---

## 10. Batasan yang diakui secara terbuka

* **Bukan LLM generatif.** NLG berbasis template berslot; jawaban untuk
  pertanyaan di luar knowledge base akan menyatakan ketidakpastian, bukan
  mengarang.
* **Knowledge base terbatas pada domain produk.** Topik yang tersedia dapat
  dilihat di `GET /api/knowledge-base`.
* **Dataset kecil.** Akurasi intent/sentimen di atas adalah hasil dataset
  contoh di repo ini, bukan benchmark produksi; perluas dataset untuk domain
  Anda.
* **Terjemahan penuh antar 8 bahasa tidak termasuk.** Yang tersedia: deteksi
  bahasa, pemahaman lintas bahasa untuk intent/sentimen, dan jawaban dalam
  bahasa pesan.
* **spaCy/Transformer opsional.** Jika model tidak terpasang, sistem memakai
  fallback deterministik dan melaporkannya di `/api/model/info`.

---

## 11. Lisensi

MIT — lihat [`LICENSE`](LICENSE).
