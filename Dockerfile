# ---- build ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# Optional NLP enhancers (Sastrawi, PyStemmer, langdetect, spaCy + model).
# Disable with: docker build --build-arg INSTALL_EXTRAS=0 .
ARG INSTALL_EXTRAS=1
RUN if [ "$INSTALL_EXTRAS" = "1" ]; then \
        /opt/venv/bin/pip install "Sastrawi>=1.0.1" "PyStemmer>=2.2.0" "langdetect>=1.0.9" \
        && /opt/venv/bin/pip install "spacy>=3.7,<4.0" \
        && /opt/venv/bin/python -m spacy download en_core_web_sm; \
    fi || echo "optional NLP extras skipped"

# ---- runtime ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHATGANGS_ENVIRONMENT=production \
    CHATGANGS_DATABASE_URL=sqlite:////app/data/chatgangs.db

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 chatgangs

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=chatgangs:chatgangs chatgangs ./chatgangs
COPY --chown=chatgangs:chatgangs scripts ./scripts
COPY --chown=chatgangs:chatgangs data ./data
COPY --chown=chatgangs:chatgangs web ./web
COPY --chown=chatgangs:chatgangs models ./models
COPY --chown=chatgangs:chatgangs pytest.ini ./
RUN mkdir -p /app/data && chown -R chatgangs:chatgangs /app

USER chatgangs
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "chatgangs.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
