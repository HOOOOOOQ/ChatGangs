#!/usr/bin/env bash
# ChatGangs — cURL cheatsheet for every public endpoint.
#
#   ./examples/curl_cheatsheet.sh                 # uses http://localhost:8000
#   BASE=http://host:8000 KEY=secret ./examples/curl_cheatsheet.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
KEY="${KEY:-}"
SESSION="${SESSION:-curl-demo}"

if [[ -n "$KEY" ]]; then
  AUTH=(-H "X-API-Key: $KEY")
else
  AUTH=()
fi

json() { curl -sS -X "$1" "${BASE}$2" -H 'Content-Type: application/json' "${AUTH[@]}" ${3:+-d "$3"}; }
pretty() { json "$@" | python3 -m json.tool 2>/dev/null || json "$@"; }
section() { printf "\n== %s ==\n" "$1"; }

section "health"
pretty GET /api/health

section "model information"
pretty GET /api/model/info

section "runtime metrics"
pretty GET /api/metrics

section "chat: first turn"
pretty POST /api/chat "{\"message\":\"Halo, apa saja kemampuanmu?\",\"session_id\":\"$SESSION\"}"

section "chat: follow-up (context aware)"
pretty POST /api/chat "{\"message\":\"jelaskan lebih detail\",\"session_id\":\"$SESSION\"}"

section "chat: English"
pretty POST /api/chat '{"message":"How do I use the API?","session_id":"curl-en"}'

section "chat: delete conversation via natural language"
pretty POST /api/chat '{"message":"hapus riwayat percakapan saya","session_id":"curl-rm"}'

section "analyze (all modules)"
pretty POST /api/nlp/analyze \
  '{"text":"Pelayanan PT Maju Jaya bagus tapi pengirimannya lambat. Hubungi budi@mail.com di Jakarta sebelum 12/03/2025."}'

section "tokenize"
pretty POST /api/nlp/tokenize '{"text":"Halo dunia! Ini tes.","include_stems":true}'

section "sentiment (contrast clause)"
pretty POST /api/nlp/sentiment '{"text":"Makanannya enak tapi pelayanannya buruk sekali"}'

section "intent (typo tolerant)"
pretty POST /api/nlp/intent '{"text":"brp hargnya?"}'

section "entities"
pretty POST /api/nlp/entities '{"text":"PT Maju Jaya di Bandung, +62 812-3456-7890, 5 unit laptop"}'

section "language detection"
pretty POST /api/nlp/language '{"text":"この製品がとても好きです"}'

section "similarity"
pretty POST /api/nlp/similarity '{"texts":["kucing makan ikan","saya suka kucing","harga mobil mahal"]}'

section "classification"
pretty POST /api/nlp/classify '{"text":"Tolong jelaskan cara kerja sistem ini"}'

section "question answering"
pretty POST /api/nlp/qa '{"question":"bagaimana cara memakai api","top_k":3}'

section "normalize (slang + PII)"
pretty POST /api/nlp/normalize '{"text":"gmn cara pkai API, email gw budi@mail.com"}'

section "stem"
pretty POST /api/nlp/stem '{"text":"pengirimannya sangat memuaskan"}'

section "stopwords"
pretty POST /api/nlp/stopwords '{"text":"saya sangat suka produk ini"}'

section "batch"
pretty POST /api/nlp/batch '{"texts":["halo","saya kecewa"],"tasks":["sentiment","intent"]}'

section "catalogue"
pretty GET /api/languages
pretty GET /api/intents
pretty GET /api/knowledge-base

section "feedback & memory"
pretty POST /api/feedback "{\"session_id\":\"$SESSION\",\"rating\":5,\"comment\":\"membantu\"}"
pretty GET "/api/memory/$SESSION?limit=5"
pretty DELETE "/api/memory/$SESSION"
