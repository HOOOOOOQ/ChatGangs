# Komponen NLP — cara kerja dan batasannya

Dokumen ini menjelaskan **apa yang benar-benar dihitung** oleh setiap modul,
sehingga hasil API bisa ditafsirkan dengan tepat.

---

## 1. Tokenizer (`core/tokenizer.py`)

Tiga mode dalam satu kelas:

| Mode | Implementasi |
| --- | --- |
| `word` | pencocokan terpanjang atas pola URL/e-mail, mention, hashtag, emoji, angka, kata (Latin, Cyrillic, Yunani, Arab, Thai, kana+Han, Hangul), sisanya punctuation |
| `sentence` | pemecah kalimat multibahasa dengan pengecualian singkatan (`Dr.`, `Mr.`, `e.g.`) dan terminasi CJK (`。！？`) |
| `subword` | n-gram karakter (`<word>`, n=3..5) untuk inspeksi OOV |

Setiap token membawa `start`/`end` yang menunjuk ke **teks asli** (setelah
pembersihan karakter kontrol) sehingga UI bisa menyorot hasil. Emoji tetap utuh
(tidak dipecah menjadi surrogate). Angka dengan pemisah ribuan
(`15.000,00`, `3.14`) dikenali sebagai satu token.

## 2. Normalizer (`core/normalizer.py`)

Urutan langkah: NFKC → hapus karakter tak terlihat → redaksi PII opsional →
lowercase → perlindungan URL/mention/hashtag → per kata: ekspansi slang (ID/EN:
`gmn → bagaimana`, `gw → saya`, `pls → please`) → penciutan huruf berulang
(`bagussss → bagus`) → koreksi typo **konservatif** (hanya bila kata tidak ada
di kosakata, panjang ≥5, huruf awal sama, dan rasio kemiripan ≥0,90) →
normalisasi spasi.

Kosakata pembanding dibangun dari stopword, leksikon sentimen, kamus slang/typo,
contoh intent, dan varian knowledge base (±1.225 kata), lalu dicache di LRU.

## 3. Stemmer / Lemmatizer (`core/stemmer.py`)

* **Indonesia (builtin, default).** Menghasilkan kandidat akar dengan membuka
  sufiks (`-kan`, `-an`, `-i`), partikel (`-lah`, `-kah`, `-pun`), posesif
  (`-ku`, `-mu`, `-nya`), prefiks derivasional (`me-, mem-, men-, meng-, meny-,
  pe-, per-, ber-, ter-, di-, ke-, se-`) dan aturan asimilasi nasal
  (`menulis → tulis`, `memukul → pukul`, `menyapu → sapu`, `mengambil → ambil`).
  Kandidat kemudian dicocokkan dengan kamus akar terkurasi
  (`data/lexicons/roots_id.txt`): akar terkurasi paling tereduksi menang
  (`pelayanan → layan`), lalu heuristik silabis untuk kandidat lain.
  Hasil uji: 40/40 pasangan kata pada tes unit.
* **Sastrawi (opsional).** Bila terpasang dan `CHATGANGS_STEMMER_BACKEND=sastrawi`,
  Sastrawi dipakai (akurasi setara, ±9 ms/kata vs ±0,005 ms untuk builtin).
* **Inggris.** Porter subset (sufiks terurut, proteksi `-ss/-us/-is`, tabel
  lemma ireguler: `went → go`, `children → child`, `better → good`).
* **Melayu** memakai aturan Indonesia; **Spanyol/Jerman/Prancis** memakai
  pemangkasan sufiks ringan.

## 4. Stopword removal (`core/stopwords.py`)

Delapan daftar bahasa di `data/stopwords/`. Kata negasi (`tidak`, `bukan`,
`belum`, `not`, `no`, …) **dipertahankan** agar sentimen tidak berbalik makna.
`coverage()` melaporkan rasio stopword sebagai sinyal kompleksitas teks.

## 5. Named Entity Recognition (`core/entities.py`)

Dua lapis, masing-masing menyertakan `source` pada hasil:

1. **Regex** — `EMAIL, URL, PHONE, MONEY, PERCENT, DATE, TIME, QUANTITY`.
2. **Gazetteer & pola** — `ORGANIZATION` (PT/CV/Inc/Ltd/Corp + gazetteer
   perusahaan), `GPE/LOCATION` (negara & kota), `PRODUCT` (merek/produk),
   `PERSON` (gelar seperti Bapak/Mr. + daftar nama umum).
3. **spaCy** (opsional) — `doc.ents` dipetakan ke taksonomi internal
   (`ORG→ORGANIZATION`, `LOC→LOCATION`, …), confidence 0,9.

Entitas yang tumpang tindih dihapus dengan preferensi confidence tertinggi.
Batasan: NER tanpa spaCy bersifat leksikal; nama diri yang tidak ada di daftar
tidak akan dikenali kecuali berpola gelar.

## 6. Intent detection (`core/intent.py`)

| Lapis | Kekuatan | Batasan |
| --- | --- | --- |
| Aturan regex (24 pola) | presisi tinggi, dapat dijelaskan | hanya pola yang ditulis |
| Model TF-IDF + LogReg | parafrase & typo | butuh `scripts.train --task intent` |
| Fuzzy Jaccard ke contoh katalog | tetap jalan tanpa model | kasar untuk kalimat panjang |

Penggabungan: skor aturan berbobot 0,55–0,95; skor model `prob × 1,05`
(dibatasi 0,99); skor fuzzy `max(jaccard, 0,85 × containment)`. Aturan
"social override" menaikkan intent konten bila sebuah permintaan nyata
menyertai sapaan. Confidence di bawah `CHATGANGS_INTENT_CONFIDENCE_THRESHOLD`
(0,30) menghasilkan `unknown` alih-alih tebakan.

Evaluasi end-to-end pada katalog 395 contoh: **±91% akurasi**.

## 7. Sentiment (`core/sentiment.py`)

```
skor dokumen = Σ (polaritas klausa × bobot klausa)
polaritas klausa = clamp( Σ(polaritas kata × amplifier) / Σ|bobot| , -1, 1)
```

* Leksikon ID (56 positif / 61 negatif) dan EN (57/64) + 52 emoji.
* Negasi dalam jendela 3 token sebelum kata, atau kata langsung sesudahnya.
* Amplifier/diminisher sebelum **dan** sesudah kata (Indonesia: "bagus sekali").
* Klausa dipisah per kalimat lalu pada konjungsi kontras (`tapi`, `namun`,
  `but`, `however`); klausa setelah konjungsi berbobot lebih besar.
* Bila model terlatih tersedia: `0,35 × lexicon + 0,65 × model`, keduanya
  dilaporkan terpisah (`lexicon_score`, `model_score`).
* Input sangat pendek tanpa bukti leksikal (mis. "halo") dilaporkan `neutral`
  agar model tidak memberi polaritas palsu.

## 8. Semantik & QA (`core/semantics.py`)

TF-IDF sendiri: unigram + bigram kata, `tf = 1 + log(count)`, `idf =
log((1+N)/(1+df)) + 1`, normalisasi L2 → cosine = dot product.

* `tokenize_for_index`: normalisasi → hapus stopword → **stem** (sehingga
  "harganya" dan "harga" bertemu).
* Knowledge base membangun indeks sekali saat start, lalu setiap query hanya
  melakukan transformasi query + cosine atas 25 vektor; ditambah pencocokan
  varian persis/awalan untuk pertanyaan pendek.
* `TextClassifier` melaporkan **dua hal yang berbeda tanpa mencampurnya**:
  - `speech_act` (taksonomi kata kunci: `pertanyaan`, `perintah`, `keluhan`,
    `pujian`, `sapaan`, `ucapan_terima_kasih`, `lainnya`) — selalu dihitung;
  - `topic` (model terlatih 5 topik) — hanya bila artefak klasifikasi ada.
  Default `task="auto"` menjadikan speech-act sebagai `label` dan menambahkan
  `topic`; `task="topic"` memaksa prediksi topik; `task="speech_act"` memaksa
  taksonomi. Ini mencegah model topik menyembunyikan peran percakapan sebuah
  kalimat (mis. "tolong jelaskan ..." tetap `perintah`, bukan sebuah topik).

Evaluasi retrieval: top-1 **93%**, top-3 **97%** pada 30 pertanyaan berlabel.

## 9. Context (`core/context.py`)

Mendeteksi follow-up dari pola ("yang tadi", "itu", "jelaskan lagi", "what
about that", kalimat ≤4 token dengan pronomina). Saat follow-up terdeteksi,
query retrieval diperluas dengan pertanyaan pengguna sebelumnya dan topik aktif
dipertahankan. Tren sentimen dhitung dari tiga giliran terakhir
(`improving`, `deteriorating`, `consistently_positive`, …).

## 10. NLG (`core/nlg.py`)

Template berslot per intent × bahasa dengan 2–3 varian permukaan. Opener
dipilih deterministik dari hash (sesi, intent, isi) dan 6 template terakhir
per sesi diingat agar tidak berulang. Panjang jawaban menyesuaikan permintaan
("jelaskan lebih detail" → entri `details` knowledge base disertakan).
Bahasa Indonesia, Inggris dan Melayu punya template penuh; Jepang, Korea,
Spanyol, Jerman, Prancis mendapat penanda bahasa asli + badan jawaban, dan hal
ini dinyatakan apa adanya.
