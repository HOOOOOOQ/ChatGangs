# Instalasi

ChatGangs berjalan di Python **3.11+** dan tidak memerlukan GPU. Semua langkah
di bawah sudah diverifikasi pada Linux; macOS bekerja sama, Windows memakai
PowerShell dengan perintah yang setara.

---

## 1. Instalasi lokal (development)

```bash
git clone https://github.com/HOOOOOOQ/ChatGangs.git
cd ChatGangs

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements-dev.txt  # runtime + tooling tes
```

### Latih model (disarankan)

```bash
python -m scripts.train --task all
```

Perintah ini benar-benar melatih tiga model (intent, sentimen, klasifikasi),
menjalankan cross-validation, dan menulis metrik ke `models/<task>/metrics.json`.
Tanpa langkah ini ChatGangs tetap berjalan dengan fallback aturan/leksikon dan
melaporkan `trained: false` di `GET /api/model/info`.

### Jalankan

```bash
cp .env.example .env                # opsional, sesuaikan pengaturan
python -m chatgangs --host 0.0.0.0 --port 8000
# atau dengan auto-reload saat development:
python -m chatgangs --reload
```

Buka:

* Web client: <http://localhost:8000/>
* Swagger UI: <http://localhost:8000/docs>
* Health: <http://localhost:8000/api/health>

---

## 2. Komponen NLP opsional

ChatGangs tidak bergantung pada komponen ini, tetapi kualitas meningkat bila
terpasang. Status nyata setiap komponen selalu terlihat di `GET /api/model/info`.

```bash
# Stemmer Indonesia berakurasi lebih tinggi (lebih lambat: ±9 ms/kata)
pip install Sastrawi
export CHATGANGS_STEMMER_BACKEND=sastrawi   # atau "auto"

# Stemmer Snowball untuk Spanyol/Jerman/Prancis, deteksi bahasa statistik
pip install PyStemmer langdetect

# POS tagging, dependency parsing, dan NER statistik untuk bahasa Inggris
pip install "spacy>=3.7,<4.0"
python -m spacy download en_core_web_sm
```

Jika spaCy terpasang tanpa model, sistem mencatat peringatan dan memakai
fallback regex + gazetteer. Tidak ada error, tidak ada klaim palsu.

---

## 3. Docker

```bash
docker compose up --build           # http://localhost:8000
```

Build tanpa komponen NLP opsional (image lebih kecil):

```bash
docker build --build-arg INSTALL_EXTRAS=0 -t chatgangs:slim .
docker run -p 8000:8000 chatgangs:slim
```

Data percakapan disimpan di volume `chatgangs-data` (SQLite default). Untuk
PostgreSQL:

```bash
docker compose --profile postgres up -d
# lalu jalankan API dengan:
export CHATGANGS_DATABASE_URL=postgresql+psycopg2://chatgangs:chatgangs@postgres:5432/chatgangs
docker compose up --build
```

> Driver PostgreSQL (`psycopg2-binary`) sudah disertakan pada image Docker
> ketika build arg `INSTALL_EXTRAS=1`. Untuk instalasi non-Docker, tambahkan
> `psycopg2-binary` ke environment Anda.

---

## 4. Deployment VPS (Ubuntu/Debian)

### 4.1 Siapkan sistem

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip \
     nginx postgresql postgresql-contrib git
sudo useradd --create-home --shell /bin/bash chatgangs
sudo su - chatgangs
git clone https://github.com/HOOOOOOQ/ChatGangs.git app
cd app
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt psycopg2-binary
python -m scripts.train --task all
```

### 4.2 PostgreSQL

```bash
sudo -u postgres psql <<'SQL'
CREATE USER chatgangs WITH PASSWORD 'ganti-password-kuat';
CREATE DATABASE chatgangs OWNER chatgangs;
GRANT ALL PRIVILEGES ON DATABASE chatgangs TO chatgangs;
SQL
```

Tabel dibuat otomatis saat API pertama kali start (`Base.metadata.create_all`).

### 4.3 Konfigurasi produksi (`.env`)

```ini
CHATGANGS_ENVIRONMENT=production
CHATGANGS_DEBUG=false
CHATGANGS_DOCS_ENABLED=false
CHATGANGS_DATABASE_URL=postgresql+psycopg2://chatgangs:PASSWORD@127.0.0.1:5432/chatgangs
CHATGANGS_API_KEYS=ganti-dengan-key-acak-panjang
CHATGANGS_REQUIRE_API_KEY=true
CHATGANGS_RATE_LIMIT_REQUESTS=120
CHATGANGS_RATE_LIMIT_WINDOW_SECONDS=60
CHATGANGS_LOG_JSON=true
CHATGANGS_LOG_CHAT_CONTENT=false
CHATGANGS_TRUST_FORWARDED_FOR=true
CHATGANGS_MEMORY_TTL_HOURS=72
```

Buat key acak:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4.4 systemd

`/etc/systemd/system/chatgangs.service`:

```ini
[Unit]
Description=ChatGangs NLP API
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=chatgangs
Group=chatgangs
WorkingDirectory=/home/chatgangs/app
EnvironmentFile=/home/chatgangs/app/.env
ExecStart=/home/chatgangs/app/.venv/bin/uvicorn chatgangs.api.app:app \
  --host 127.0.0.1 --port 8000 --workers 2 --proxy-headers
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now chatgangs
sudo systemctl status chatgangs
```

> Jumlah worker: `2 × core` adalah titik awal umum. Setiap worker memuat
> modelnya sendiri (±80 MB dengan model terlatih), jadi sesuaikan dengan RAM.

### 4.5 Nginx reverse proxy + HTTPS

```nginx
server {
    listen 80;
    server_name chatgangs.contoh.com;

    client_max_body_size 1m;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/chatgangs /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d chatgangs.contoh.com
```

Bila berada di belakang proxy, set `CHATGANGS_TRUST_FORWARDED_FOR=true` agar
rate limiting memakai IP asli klien.

---

## 5. Verifikasi instalasi

```bash
curl -s localhost:8000/api/health | python -m json.tool
curl -s localhost:8000/api/model/info | python -m json.tool | head -40
pytest -q
```

`/api/health` harus melaporkan `"status": "ok"` dengan `database` dan
`nlp_engine` ok, dan `/api/model/info` menampilkan artefak model yang sudah
dilatih (atau `trained: false` bila Anda melewatkan langkah training).

---

## 6. Masalah umum

| Gejala | Penyebab & solusi |
| --- | --- |
| `spaCy model 'en_core_web_sm' unavailable` | normal bila model belum diunduh. Jalankan `python -m spacy download en_core_web_sm`, atau biarkan — fallback aktif dan dilaporkan. |
| `sqlite3.OperationalError: unable to open database file` | folder `data/` tidak ada atau tidak bisa ditulis. `mkdir -p data` dan pastikan permission. |
| `401 Invalid or missing API key` | `CHATGANGS_REQUIRE_API_KEY=true` tetapi header `X-API-Key` tidak dikirim. |
| `429 Rate limit exceeded` | naikkan `CHATGANGS_RATE_LIMIT_REQUESTS` atau tunggu window selesai (`Retry-After`). |
| Model baru tidak terpakai | API memuat artefak saat start-up; restart service setelah `scripts.train`. |
| Latensi pertama tinggi (±1 s) | warm-up model/knowledge base; panggilan berikutnya ±5–20 ms. |
