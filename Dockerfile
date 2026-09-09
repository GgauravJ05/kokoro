# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
#
# Runs the Kokoro demo. Expects a built corpus and trained artifacts to be
# mounted or baked in — the image ships code, not data, because the corpus is
# derived from third-party dumps this project does not redistribute.
#
#   docker build -t kokoro .
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/data/processed:/app/data/processed:ro" \
#     -v "$PWD/artifacts/two_tower:/app/artifacts/two_tower:ro" kokoro

FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first, so a source edit does not re-resolve the whole tree.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

# CPU-only torch: the GPU wheels are ~2 GB and this serves on CPU by design.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" \
    && pip install ".[serve,index]" "sentence-transformers>=3.0"

# ------------------------------------------------------------------------------

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KOKORO_ARTIFACT_DIR=/app/artifacts/two_tower \
    KOKORO_CORPUS_DIR=/app/data/processed \
    KOKORO_INDEX=hnsw \
    KOKORO_EF_SEARCH=64 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY src ./src

# Bake the sentence encoder into the image so the first request does not spend
# seconds downloading it — the readiness probe can then honestly report ready.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

# Run unprivileged: nothing here needs root.
RUN useradd --create-home --uid 10001 kokoro \
    && mkdir -p /app/data/processed /app/artifacts \
    && chown -R kokoro:kokoro /app
USER kokoro

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "kokoro.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
