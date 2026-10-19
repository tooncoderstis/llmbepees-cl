# syntax=docker/dockerfile:1
# Image runtime untuk asisten BPS (FastAPI). Diaktifkan pada Fase 5 (lihat docs/runbooks/deploy.md).
# EasyPanel hanya pull image ini; build dilakukan di CI/lokal, bukan di server.

FROM python:3.11-slim AS runtime
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Pasang dependensi lebih dulu agar layer cache efisien.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[serve]"

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "llmbepees_cl.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
