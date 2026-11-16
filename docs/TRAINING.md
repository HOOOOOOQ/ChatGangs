# Training, dataset & versioning

ChatGangs menyertakan pipeline training **nyata**. Tidak ada model yang
"dianggap sudah dilatih": bila dataset tidak ada atau tidak valid, pipeline
berhenti dengan pesan jelas, dan API terus melaporkan `trained: false` sampai
artefak benar-benar dihasilkan.

---

## 1. Perintah

```bash
python -m scripts.train --task intent
python -m scripts.train --task sentiment
python -m scripts.train --task classification
python -m scripts.train --task qa            # hanya evaluasi retrieval
python -m scripts.train --task all           # semua + evaluasi end-to-end
python -m scripts.train --dry-run            # validasi dataset saja, tanpa menulis artefak
```

Opsi:

| Flag | Default | Keterangan |
| --- | --- | --- |
| `--task` | `all` | intent / sentiment / classification / qa / all / evaluate |
| `--version` | timestamp | label versi artefak |
| `--test-size` | `0.2` | proporsi validation split |
| `--cv-folds` | `5` | jumlah fold cross-validation (dibatasi oleh kelas minoritas) |
| `--dry-run` | off | hanya memuat & memvalidasi ketiga dataset, tidak melatih apa pun |

Exit code: `0` sukses, `1` dataset gagal validasi, `2` dataset tidak ditemukan.

---

## 2. Format dataset

### Intent — `data/intents/intents.json`

```json
{
  "version": "1.0.0",
  "intents": [
    {
      "name": "pricing",
      "description": "User asks about cost or licensing.",
      "response_hint": "pricing",
      "examples": ["berapa harganya", "apakah gratis", "pricing", "how much does it cost"]
    }
  ]
}
```

* Minimal **3 contoh per intent** (validator menolak kurang dari itu).
* Minimal **dua intent**.
* Tambahkan intent baru tanpa mengubah kode; NLG memakai `response_hint`
  (fallback ke template `question`/`knowledge_miss` bila belum ada template).

### Sentimen — `data/sentiment/train.csv`

```csv
text,label
"pelayanannya sangat memuaskan",positive
"barang datang terlambat",negative
"produk standar saja",neutral
```

Label valid: `positive`, `negative`, `neutral` (case-insensitive).

### Klasifikasi teks — `data/classification/train.csv`

Kolom `text,label` dengan **label bebas** (bukan hanya positive/negative).
Dataset bawaan berisi 222 kalimat pada 5 topik: `teknologi`, `bisnis`,
`kesehatan`, `pendidikan`, `hiburan`. Label sentiment yang tidak valid dibuang
otomatis oleh loader sentimen, sedangkan loader klasifikasi menerima label
apa pun yang tidak kosong.

### QA — `data/qa/knowledge_base.json` + `data/qa/evaluation_questions.jsonl`

Knowledge base berisi entri `{id, language, topic, question, variants,
keywords, answer, details?}`. Evaluasi retrieval memakai JSONL:

```json
{"question": "siapa kamu", "expected": "kb.identity"}
```

---

## 3. Apa yang terjadi saat training

1. **Validasi** — jumlah contoh per kelas, jumlah kelas, konsistensi panjang teks.
2. **Split stratifikasi** — deterministik (`seed=42`), tiap kelas terwakili.
3. **Cross-validation** — `StratifiedKFold(n=min(cv_folds, kelas minoritas))`
   dengan scoring `f1_macro` pada train split.
4. **Fit final** pada seluruh train split.
5. **Evaluasi** pada validation set: accuracy, macro-F1, weighted-F1,
   `classification_report` per kelas, confusion matrix, dan train accuracy
   (untuk mendeteksi overfitting).
6. **Persistensi** ke `models/<task>/`:

```
models/intent/
├── model.joblib                # dict: {model, vectorizer, pipeline, metadata}
├── metrics.json                # semua metrik + versi + timestamp
└── classification_report.txt
```

Pipeline yang dilatih:

```
ChatGangs normalizer  →  TF-IDF(word 1-2 gram)  ┐
                     →  TF-IDF(char_wb 3-5)     ┴→ FeatureUnion → LogisticRegression(C=8, balanced)
```

Normalizer ikut dibungkus sebagai transformer scikit-learn
(`TextNormalizerTransformer`), sehingga preprocessing saat training **identik**
dengan saat inference — inilah yang membuat classifier tahan terhadap slang
dan typo.

---

## 4. Evaluasi end-to-end

Selain evaluasi classifier, `--task all` menjalankan:

* **detektor intent lengkap** (aturan + model + fuzzy) pada seluruh katalog;
* **retrieval QA** pada set pertanyaan berlabel (top-1 dan top-3).

Contoh keluaran:

```
End-to-end intent accuracy: 95.19% (376/395)
  greeting        36/39
  small_talk      15/24

Question Answering evaluation
  Top-1 accuracy : 93.33% (28/30)
  Top-3 recall   : 96.67% (29/30)
```

Angka ini bukan benchmark produksi — dataset bawaan berukuran kecil dan
sengaja menyertakan contoh sulit. Gunakan sebagai baseline saat memperluas
dataset untuk domain Anda.

---

## 5. Runtime: bagaimana model baru dipakai

`IntentDetector`, `SentimentAnalyzer`, dan `TextClassifier` memuat artefak saat
diinisialisasi (start-up API). Alur update:

```bash
python -m scripts.train --task all          # 1. latih
sudo systemctl restart chatgangs            # 2. muat ulang model
curl -s localhost:8000/api/model/info | python -m json.tool | grep -A4 artifacts
```

`GET /api/model/info` memperlihatkan versi, waktu training, dan validation
accuracy artefak yang **sedang aktif** — bukan metrik historis.

---

## 6. Versioning & rollback

Setiap artefak membawa `metadata.version` (default: timestamp
`YYYYMMDD.HHMMSS`). Untuk versi eksplisit:

```bash
python -m scripts.train --task intent --version 2026.11.16-intent-v2
```

Rollback = pulihkan folder `models/<task>/` dari backup/copy sebelumnya lalu
restart layanan. Metrik historis tersimpan di `metrics.json` tiap folder;
praktik yang disarankan adalah mengarsipkan folder ber-versi:

```
models/intent/                        ← aktif
archive/intent-2026.11.16-intent-v1/  ← arsip
```

---

## 7. Monitoring kualitas

* `GET /api/model/info` → metrik artefak aktif.
* `GET /api/metrics` → latensi p50/p90/p99, error rate, hit-rate cache.
* `POST /api/feedback` → rating pengguna (`-1..5`). Data terkumpul di tabel
  `feedback` dan dapat diekspor untuk membuat dataset evaluasi baru:

```sql
SELECT session_id, rating, comment, created_at FROM feedback ORDER BY id DESC;
```

Feedback **tidak** otomatis dipakai untuk training; kurasi manual diperlukan
agar kualitas dataset tidak turun.

---

## 8. Memperluas dataset intent (resep praktis)

1. Kumpulkan 10–20 parafrase per intent untuk gaya bahasa nyata pengguna Anda.
2. Tambahkan ke `examples`, hindari duplikat persis antar intent.
3. Jalankan `python -m scripts.train --task intent` dan periksa
   `classification_report.txt`: kelas dengan recall rendah biasanya
   bertumpang tindih dengan intent lain → perjelas contoh atau definisinya.
4. Ulangi sampai macro-F1 stabil, lalu restart layanan.

Untuk domain khusus (mis. e-commerce), tambahkan intent seperti
`order_status`, `refund_request`, `shipping_eta` beserta template NLG di
`chatgangs/core/nlg.py` (`TEMPLATES[lang][intent]`) — atau biarkan fallback
knowledge base menjawab.
