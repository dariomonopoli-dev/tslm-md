# Multi-stage build for the Trajecta inference service.
# Build context = repo root (set in docker-compose.yml) so we can COPY
# both inference-service/ and the editable OpenTSLM/ source tree.

# ---------- stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache layer for fat Python deps
COPY inference-service/requirements.txt /build/requirements.txt
RUN pip install --user -r /build/requirements.txt

# OpenTSLM as editable so its imports work without publishing
COPY OpenTSLM /build/OpenTSLM
RUN pip install --user -e /build/OpenTSLM

# ---------- stage 2: runtime ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/home/app/.local/bin:$PATH PYTHONPATH=/app:/build/OpenTSLM/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        autodock-vina \
        libxrender1 libxext6 libgl1 libgomp1 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

# Copy installed python packages from builder
COPY --from=builder /root/.local /home/app/.local
COPY --from=builder /build/OpenTSLM /build/OpenTSLM

WORKDIR /app
COPY inference-service/ /app/
RUN chown -R app:app /home/app /app && mkdir -p /app/data && chown app:app /app/data

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
