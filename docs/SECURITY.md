# Keamanan & privasi

## 1. Validasi input

* Semua endpoint memakai model Pydantic dengan `extra="forbid"`: field tak
  dikenal ditolak (`422`), sehingga tidak ada parameter tak terduga yang masuk.
* Batas ukuran: 4.000 karakter (analisis), 2.000 karakter (chat), 64 teks per
  batch, 128 karakter `session_id` (hanya alfanumerik, `-`, `_`).
* `session_id` divalidasi polanya untuk mencegah path traversal / injeksi pada
  endpoint memori.
* Teks dibersihkan dari karakter kontrol dan zero-width sebelum diproses.
* Tidak ada endpoint yang menerima atau mengeksekusi kode/SQL dari pengguna;
  query database memakai SQLAlchemy ORM (parameterized).

## 2. Autentikasi API key

```ini
CHATGANGS_REQUIRE_API_KEY=true
CHATGANGS_API_KEYS=key1,key2
CHATGANGS_API_KEY_HEADER=X-API-Key
```

* Perbandingan memakai `hmac.compare_digest` (constant time).
* Key **tidak pernah** dikembalikan API dan tidak ditulis ke log; identitas
  klien disimpan sebagai `sha256(key)[:32]`.
* Bila autentikasi nonaktif (default development), endpoint tetap terlindungi
  rate limiting.

## 3. Rate limiting

* Sliding window per klien (key hash bila terautentikasi, selain itu IP).
* Default 120 request / 60 detik (`CHATGANGS_RATE_LIMIT_REQUESTS`,
  `CHATGANGS_RATE_LIMIT_WINDOW_SECONDS`).
* Respons `429` menyertakan `Retry-After`, `X-RateLimit-Limit`,
  `X-RateLimit-Remaining`.
* Jumlah klien yang dilacak dibatasi (`max_clients=50 000`) dengan eviction
  bucket basi agar memori tidak tumbuh tanpa batas.
* Di belakang proxy, set `CHATGANGS_TRUST_FORWARDED_FOR=true` **hanya** jika
  proxy Anda memang menimpa header `X-Forwarded-For` (kalau tidak, klien bisa
  memalsukan IP).

## 4. Privasi percakapan

| Data | Perlakuan |
| --- | --- |
| Isi pesan | disimpan **hanya** jika `MEMORY_PERSIST=true` dan sesi mengizinkan |
| Isi pesan di log | **tidak pernah** ditulis; log hanya metadata (path, status, latensi, ukuran) |
| E-mail & nomor telepon | diredaksi (`[EMAIL]`, `[NUMBER]`) sebelum analisis dan sebelum dipakai untuk retrieval |
| User profile | opsional, harus non-sensitif, disimpan bersama sesi |
| Penghapusan | `DELETE /api/memory/{session_id}` menghapus sesi + pesan + konteks; perintah chat "hapus riwayat" melakukan hal yang sama |
| Retensi | `MEMORY_TTL_HOURS` (default 72 jam) + endpoint purge untuk sesi kedaluwarsa |
| Consent | `store_memory=false` pada `/api/chat` membuat giliran tidak disimpan sama sekali |

Endpoint `GET /api/memory` dan `DELETE /api/memory/...` adalah kontrol pengguna
langsung: tidak ada riwayat yang "tersembunyi" dari pemilik sesi.

**Catatan deployment:** bila API dipublikasikan, batasi akses ke endpoint
memori (mis. di Nginx) karena `session_id` berfungsi sebagai kunci akses.
Aktifkan `REQUIRE_API_KEY` dan gunakan `session_id` acak (UUID) di klien.

## 5. Logging yang aman

* Default: teks pesan tidak pernah dicatat. `CHATGANGS_LOG_CHAT_CONTENT`
  disediakan untuk debugging lokal dan **tidak** disarankan di produksi.
* Log JSON opsional (`CHATGANGS_LOG_JSON=true`) dengan field tetap
  (`request_id, path, method, status, latency_ms, client`).
* Exception handler mengembalikan pesan generik di produksi
  (`CHATGANGS_ENVIRONMENT=production`) dan menyertakan `request_id` untuk
  korelasi log.

## 6. Header & transport

Middleware menambahkan `X-Content-Type-Options: nosniff`,
`X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: no-referrer`, serta
`X-Request-ID`/`X-Response-Time-ms`. CORS dibatasi lewat
`CHATGANGS_CORS_ORIGINS`. Selalu jalankan di belakang HTTPS (Nginx/Caddy) —
API tidak menangani TLS sendiri.

## 7. Ketahanan & DoS

* Batas ukuran input + batch mencegah payload raksasa.
* Cache LRU dengan kapasitas tetap (normalizer, stemmer, intent, KB) mencegah
  pertumbuhan memori akibat input unik.
* Timeout eksternal tidak relevan (tidak ada panggilan jaringan keluar).
* Rate limiter membatasi laju. Handler route didefinisikan sebagai fungsi
  sinkron sehingga FastAPI menjalankannya di threadpool — pekerjaan CPU
  (analisis NLP) tidak memblokir event loop dan request tetap dilayani paralel.

## 8. Kerentanan yang perlu diperhatikan operator

| Risiko | Mitigasi |
| --- | --- |
| `session_id` diketahui orang lain | gunakan UUID acak; batasi endpoint memori |
| API key bocor | rotasi `CHATGANGS_API_KEYS`, restart layanan |
| Akses database langsung | kredensial DB hanya untuk proses API; backup terenkripsi |
| Prompt/teks berbahaya | tidak ada eksekusi kode; input hanya dianalisis sebagai teks |
| Dokumen `/docs` terbuka | set `CHATGANGS_DOCS_ENABLED=false` di produksi |
