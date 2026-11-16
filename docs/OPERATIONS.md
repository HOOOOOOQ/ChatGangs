# Operasional

## 1. Health & readiness

```bash
curl -s localhost:8000/api/health | python -m json.tool
```

`status: ok` hanya jika database, engine NLP, dan memori sehat. Gunakan endpoint
ini untuk load balancer / Kubernetes probes:

```yaml
livenessProbe:  {httpGet: {path: /api/health, port: 8000}, periodSeconds: 30}
readinessProbe: {httpGet: {path: /api/health, port: 8000}, initialDelaySeconds: 10}
```

## 2. Metrik

```bash
curl -s localhost:8000/api/metrics
```

Menyediakan `requests_recent`, `error_rate_recent`, `server_error_rate_recent`,
latensi `p50/p90/p99/max`, `by_status` (2xx/3xx/4xx/5xx), statistik memori,
konteks, rate limiter, budget chat, dan cache.

Implementasi: middleware mencatat **metadata saja** ke ring buffer in-process
(2000 sampel terakhir, dijawab tanpa query database) dan mengirim salinan ke
satu thread penulis yang menyimpannya ke tabel `request_logs` secara batch.
Antrian dibatasi 10 000 sampel dan *tidak pernah* memblokir request; jika penuh,
sampel dibuang dan `dropped_samples` naik. Jadi `GET /api/metrics` tetap
responsif walau database lambat, dan riwayat tetap ada setelah restart.

Contoh keluaran nyata pada uji beban 8 thread × 160 request campuran
(`/api/nlp/analyze`, `/api/chat`, `/api/nlp/entities`, `/api/nlp/intent`):

```json
{
  "requests_recent": 160,
  "error_rate_recent": 0.0,
  "by_status": {"2xx": 160, "3xx": 0, "4xx": 0, "5xx": 0},
  "latency_ms": {"p50": 45.29, "p90": 212.23, "p99": 2267.83, "max": 5914.2},
  "writer_backlog": 0,
  "dropped_samples": 0
}
```

p99 yang tinggi berasal dari beberapa request pertama yang memicu inisialisasi
malas (muat model + kompilasi regex); setelah warm-up, latensi kembali ke
10–50 ms. Lihat catatan warm-up di `docs/ARCHITECTURE.md`.

Contoh query SQL:

```sql
-- latensi per jam, 24 jam terakhir
SELECT strftime('%Y-%m-%d %H', created_at, 'unixepoch') AS hour,
       COUNT(*) AS requests,
       ROUND(AVG(latency_ms), 1) AS avg_ms,
       SUM(status_code >= 500) AS server_errors
FROM request_logs
WHERE created_at > strftime('%s','now') - 86400
GROUP BY hour ORDER BY hour;
```

## 3. Log

* Human-readable (default) atau JSON (`CHATGANGS_LOG_JSON=true`).
* Field JSON: `ts, level, logger, msg, request_id, path, method, status,
  latency_ms, client`.
* Isi pesan tidak dicatat — untuk inspeksi konten aktifkan
  `CHATGANGS_LOG_CHAT_CONTENT=true` hanya di lingkungan aman.
* Rotasi log: gunakan `CHATGANGS_LOG_FILE` + `logrotate`, atau andalkan
  journald bila dijalankan via systemd.

## 4. Backup & retensi

**SQLite**

```bash
sqlite3 data/chatgangs.db ".backup 'backup/chatgangs-$(date +%F).db'"
```
Lakukan saat trafik rendah; WAL mode membuat backup `.backup` konsisten.

**PostgreSQL**

```bash
pg_dump -Fc chatgangs > backup/chatgangs-$(date +%F).dump
```

Jadwalkan harian, simpan terenkripsi di luar host, dan uji restore secara
berkala. Retensi data percakapan diatur `CHATGANGS_MEMORY_TTL_HOURS`; jalankan
`DELETE /api/memory` (atau `purge_expired()` lewat cron) untuk membersihkan
sesi kedaluwarsa.

## 5. Prosedur umum

| Tugas | Perintah |
| --- | --- |
| Update model | `python -m scripts.train --task all` lalu restart layanan |
| Ubah batas rate | ubah env, restart |
| Hapus seluruh riwayat | `DELETE /api/memory/:session_id` per sesi, atau `memory.delete_all()` untuk reset total |
| Rotasi API key | ganti `CHATGANGS_API_KEYS`, restart |
| Cek versi model aktif | `GET /api/model/info` → `artifacts` |
| Cek dependensi | `GET /api/health` → `checks` |

## 6. Troubleshooting

| Gejala | Diagnosis | Solusi |
| --- | --- | --- |
| Start-up > 5 s | muat model joblib besar / warm-up | normal; turunkan jumlah worker atau ukuran model |
| `database: error` di health | path SQLite salah / DB down | periksa `CHATGANGS_DATABASE_URL`, permission folder, `pg_isready` |
| 429 pada beban normal | limit terlalu rendah / proxy tidak meneruskan IP | naikkan limit, aktifkan `TRUST_FORWARDED_FOR` |
| Jawaban "tidak tahu" untuk topik yang jelas | parafrase terlalu jauh dari knowledge base | tambahkan `variants`/`keywords` ke `knowledge_base.json` |
| Intent sering `unknown` | confidence threshold tinggi / contoh kurang | turunkan `INTENT_CONFIDENCE_THRESHOLD`, tambah contoh, retrain |
| Sentimen meleset pada sarkasme | keterbatasan leksikon | tambahkan istilah domain ke leksikon; model terlatih membantu |
| Memori terus naik | banyak sesi unik | `ContextManager` dibatasi 2 000 sesi; `purge_expired()` untuk DB |

## 7. Scaling

* **Vertikal**: tambah CPU; latensi tiap request 10–30 ms sehingga satu vCPU
  sanggup ±30 req/s.
* **Horizontal**: jalankan beberapa instance (uvicorn workers atau container)
  di belakang load balancer. Catatan: konteks percakapan disimpan in-process,
  jadi gunakan sticky session per `session_id` **atau** pindahkan store konteks
  ke Redis (implementasi `ContextManager` terisolasi).
* **Database**: PostgreSQL untuk produksi; SQLite cukup untuk single-node
  hingga puluhan ribu pesan.
* **Latensi**: aktifkan cache normalizer/stemmer (default), hindari
  `--reload` di produksi, dan jaga `CHATGANGS_STEMMER_BACKEND=builtin` kecuali
  akurasi stemming lebih penting daripada throughput.

## 8. Checklist rilis

- [ ] `pytest -q` hijau
- [ ] `python -m scripts.train --task all` dijalankan dan metrik ditinjau
- [ ] `GET /api/model/info` menampilkan artefak yang diharapkan
- [ ] `.env` produksi: `ENVIRONMENT=production`, `DOCS_ENABLED=false`,
      `REQUIRE_API_KEY=true`, `LOG_JSON=true`
- [ ] HTTPS aktif dan `TRUST_FORWARDED_FOR` sesuai arsitektur
- [ ] Backup database terjadwal & teruji restore
