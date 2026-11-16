# ChatGangs API & Web Client — deployment notes

Repository ini bisa langsung dideploy ke Vercel untuk **web client statis**
(`web/`), tetapi API Python/FastAPI membutuhkan host yang menjalankan proses
Python (VPS, Docker, Railway, Fly.io, Render, dll). Halaman statis memanggil API
lewat origin yang dikonfigurasi di `web/js/app.js`.

## 1. API saja (paling umum)

```bash
docker compose up --build          # http://localhost:8000
```

API menyajikan web client pada `/` sehingga satu host sudah cukup.

## 2. Web client di Vercel + API di host lain

1. Deploy API di VPS/host Python (lihat `docs/INSTALLATION.md`), aktifkan
   `CHATGANGS_CORS_ORIGINS=https://<proyek>.vercel.app`.
2. Arahkan web client ke API tersebut. Cara paling bersih: tambahkan satu baris
   sebelum `app.js` dimuat di `index.html`:

```html
<script>window.CHATGANGS_API_BASE = "https://api.contoh.com";</script>
```

   atau kirim key bila API mewajibkannya:

```html
<script>
  window.CHATGANGS_API_BASE = "https://api.contoh.com";
  window.CHATGANGS_API_KEY = "kunci-publik-terbatas";
</script>
```

3. Alternatif yang lebih aman untuk key rahasia: proxy `/api/*` di sisi server
   (Cloudflare Worker, Vercel Edge Function, Nginx) yang menambahkan header
   `X-API-Key`, lalu biarkan `CHATGANGS_API_BASE` kosong.

## 3. Catatan konfigurasi produksi

* Jangan expose `/docs` (`CHATGANGS_DOCS_ENABLED=false`) pada deployment publik.
* Wajibkan API key (`CHATGANGS_REQUIRE_API_KEY=true`) bila API terbuka ke
  internet; sesuaikan `web/js/app.js` untuk mengirim header `X-API-Key`, atau
  pasang proxy yang menambahkan header tersebut di sisi server.
* Gunakan HTTPS di kedua ujung dan batasi CORS hanya ke origin web client.
* Tidak ada data sensitif yang perlu disimpan di Vercel: aplikasi memakai
  `session_id` acak di klien dan riwayat tersimpan di database API.
