# Arsitektur

ChatGangs dibangun sebagai kumpulan modul dengan satu titik orkestrasi
(`chatgangs/core/pipeline.py`). Setiap modul dapat diuji dan dipakai sendiri,
baik lewat Python langsung maupun lewat endpoint khusus.

---

## 1. Alur data

```
request ──► API gateway ──► NLPengine.analyze()/chat()
                              │
                              ├─ LanguageProcessor ──► kode bahasa (id/en/ms/ja/ko/es/de/fr)
                              ├─ Normalizer ─────────► teks bersih (slang, typo, PII)
                              ├─ Tokenizer ──────────► token + offset + statistik
                              ├─ Stemmer/Lemmatizer ─► stem & lemma per token
                              ├─ StopwordRemover ────► content terms
                              ├─ EntityRecognizer ───► entities + summary
                              ├─ IntentDetector ─────► intent + confidence + method
                              ├─ SentimentAnalyzer ──► label + skor + evidence
                              ├─ TextClassifier ─────► label + rationale
                              ├─ SemanticIndex ──────► kemiripan / ranking
                              └─ KnowledgeBase ──────► kandidat jawaban
                                        │
                     ContextManager ◄───┤  (follow-up, topik, tren sentimen)
                     MemoryManager  ◄───┤  (SQLite/PostgreSQL, TTL, delete)
                     ResponseGenerator ►┘  (NLG berslot, anti-repetisi)
```

`analyze()` mengembalikan seluruh hasil pemahaman sebagai satu objek tervalidasi
(`AnalysisResult`). `chat()` memakai hasil itu untuk memilih strategi jawaban:

1. permintaan hapus data → eksekusi penghapusan, konfirmasi (tidak disimpan);
2. intent percakapan (greeting, farewell, gratitude, identity, capabilities,
   api_help, pricing, privacy, training_help, translate, feedback, complaint,
   small_talk, help) → template NLG;
3. permintaan analisis eksplisit → ringkasan hasil analisis;
4. pertanyaan/unknown → retrieval knowledge base (dengan query yang diperluas
   bila ini follow-up) → jawaban + catatan entitas bila ada;
5. tidak ada kecocokan → jawaban jujur yang menyebut cakupan knowledge base.

---

## 2. Keputusan desain penting

**Tanpa ketergantungan berat secara default.** Tokenizer, normalizer, stemmer,
NER, sentimen, TF‑IDF, dan NLG diimplementasikan di repo sehingga sistem bisa
berjalan penuh tanpa unduhan model. Komponen opsional (Sastrawi, PyStemmer,
langdetect, spaCy) dideteksi saat runtime dan statusnya dilaporkan.

**Deteksi bahasa hibrida.** Skrip Unicode (kana/Hangul/Han) diprioritaskan,
lalu overlap leksikal dengan tabel kata fungsi per bahasa (deterministik dan
bisa diaudit), lalu `langdetect` hanya ketika sinyal cepat belum meyakinkan
(menghemat ±40 ms per request). Bahasa Indonesia vs Melayu dipisahkan dengan
penanda khusus Melayu (`awak`, `khabar`, `kerana`, …).

**Intent: aturan + model + fuzzy.** Aturan regex menangani kasus presisi
tinggi (sapaan, ucapan terima kasih, permintaan hapus data). Model TF‑IDF +
Logistic Regression menangani parafrase dan typo. Fuzzy Jaccard menjadi
fallback saat model belum dilatih. Aturan "social override" memastikan sapaan
tidak menutupi permintaan nyata ("hello, what can you do?" → `capabilities`).

**Sentimen yang bisa diaudit.** Skor berasal dari leksikon berbobot dengan
penanganan negasi (jendela 3 token), amplifier/diminisher (termasuk yang
mengikuti kata, khas Indonesia: "bagus **sekali**"), klausa kontras (`tapi`
memindahkan bobot ke klausa berikutnya), dan emoji. Setiap kontribusi
dilaporkan di `evidence`. Bila model terlatih ada, skor leksikon (35%) dan
model (65%) digabung.

**TF‑IDF sendiri, bukan `sklearn` di jalur inference.** Vectorizer kecil
(uni+bigram, sublinear tf, idf halus, L2) membuat index knowledge base bisa
dibangun sekali saat start (≈25 entri → <5 ms/query) dan mudah diuji. Indeks
KB juga memakai pencocokan varian persis sehingga "siapa kamu" tetap dikenali
walau seluruh tokennya stopword.

**Context != Memory.** `ContextManager` adalah state operasional cepat di
memori proses (maks. 20 giliran, TTL). `MemoryManager` adalah riwayat
persisten yang bisa dihapus pengguna. Pemisahan ini membuat penghapusan data
tidak mungkin "hidup lagi" dari cache konteks — setelah DELETE, konteks ikut
direset.

**Kegagalan yang aman.** Semua modul optional dibungkus `try/except` dan
menghasilkan fallback deterministik; health check melaporkan kondisi nyata;
training gagal keras bila dataset tidak valid (tidak ada model "pura-pura".

---

## 3. Struktur folder

```
ChatGangs/
├── chatgangs/
│   ├── config.py            # Settings (env, .env) + singleton
│   ├── logging_config.py    # logging terstruktur, redaksi, Timer
│   ├── core/
│   │   ├── languages.py     # registry + deteksi bahasa
│   │   ├── tokenizer.py     # word/sentence/subword tokenizer
│   │   ├── normalizer.py    # normalisasi, slang, typo, PII, kamus
│   │   ├── stemmer.py       # stemmer ID/EN/ES/DE/FR + lemmatizer
│   │   ├── stopwords.py     # stopword removal multibahasa
│   │   ├── entities.py      # NER regex + gazetteer + spaCy
│   │   ├── intent.py        # deteksi intent hibrida
│   │   ├── sentiment.py     # analisis sentimen hibrida
│   │   ├── semantics.py     # TF-IDF, similarity, classifier, knowledge base
│   │   ├── context.py       # context manager
│   │   ├── nlg.py           # response generator
│   │   ├── spacy_pipeline.py# bridge spaCy opsional
│   │   ├── pipeline.py      # orkestrasi analyze()/chat()
│   │   └── text_utils.py    # Token/Entity dataclass, LRU, split kalimat
│   ├── memory/
│   │   ├── models.py        # SQLAlchemy ORM
│   │   ├── database.py      # engine, session scope, WAL
│   │   └── manager.py       # CRUD, TTL, delete, audit log
│   ├── api/
│   │   ├── app.py           # factory, middleware, error handlers, static
│   │   ├── deps.py          # security dependency
│   │   ├── security.py      # API key, rate limiter, client hashing
│   │   ├── middleware.py    # request id, logging metadata, headers
│   │   ├── schemas.py       # skema Pydantic
│   │   └── routes/          # chat, nlp, memory, system
│   ├── training/pipeline.py # dataset, split, CV, evaluasi, artefak
│   └── monitoring/          # ringkasan metrik
├── data/                    # dataset & leksikon (versioned)
├── models/                  # artefak hasil training (git-ignored)
├── scripts/train.py         # CLI training
├── tests/                   # 182 tes unit + integrasi
├── web/                     # web client (vanilla JS/CSS)
└── docs/                    # dokumentasi
```

---

## 4. Kinerja

| Operasi | Latensi tipikal (CPU 2 vCPU) |
| --- | --- |
| `/api/nlp/sentiment` | 3–10 ms |
| `/api/nlp/intent` | 3–8 ms |
| `/api/nlp/tokenize` | 1–3 ms |
| `/api/nlp/entities` | 2–10 ms |
| `/api/nlp/analyze` (semua modul) | 10–25 ms |
| `/api/chat` | 10–30 ms |
| Start-up (muat model + warm-up) | 0,5–1,5 s |

Mekanisme yang dipakai: cache LRU untuk normalisasi, stemming, deteksi
intent dan konversi dokumen; indeks knowledge base dibangun sekali; deteksi
bahasa statistik hanya dipanggil bila perlu; koneksi database dengan pool
pre-ping dan WAL untuk SQLite.

**Warm-up.** `lifespan` memuat model terlatih, mengisi kosakata normalizer,
membangun indeks knowledge base, dan menjalankan satu analisis pemanasan —
sehingga request pertama pengguna tidak menanggung biaya kompilasi regex
(~1,2 s). Tanpa warm-up, p99 pada uji beban 8 thread menyentuh 2,3 s karena
beberapa request pertama berjalan paralel saat inisialisasi; setelah warm-up
p50 turun ke 10–50 ms.

**Cache opsional.** `CHATGANGS_RESPONSE_CACHE_TTL_SECONDS > 0` mengaktifkan
cache hasil `analyze()` dan balasan `chat()` (LRU, ukuran
`CHATGANGS_RESPONSE_CACHE_SIZE`). Reply cache hanya dipakai untuk giliran yang
tidak menulis memori (`store_memory=false`), agar riwayat percakapan tidak
pernah "terlewat" karena cache. Statistiknya ada di `GET /api/metrics`
(`caches.analysis`).

**Metrik tanpa blocking.** Middleware mencatat metadata request ke ring buffer
in-process (dibaca `/api/metrics` tanpa query database) dan menyimpannya ke
database lewat satu thread penulis dengan antrian terbatas — request tidak
pernah menunggu penulisan log.

**Inisialisasi DB yang aman.** Pembuatan engine/skema dijaga lock dan memakai
`create_all(checkfirst=True)`, sehingga beberapa request paralel pada detik
pertama tidak bisa saling menabrak tabel (bug yang ditemukan pada uji beban
dan sudah diperbaiki).

**Skalabilitas.** API stateless per request selain konteks in-process, jadi
beberapa worker/instance bisa dijalankan di belakang load balancer. Bila
beberapa instance melayani sesi yang sama, gunakan sticky session atau
pindahkan context store ke Redis (antarmuka `ContextManager` sudah terisolasi).
