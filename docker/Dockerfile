# PyTorch wheels pull the CUDA runtime in themselves — no separate CUDA base needed.
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIGAAM_MODEL=v3_e2e_rnnt \
    GIGAAM_CACHE_DIR=/models/gigaam \
    GRANITE_CACHE_DIR=/models/huggingface \
    PYANNOTE_METRICS_ENABLED=0

# ffmpeg is needed by gigaam.load_audio (decoding any format),
# git — so pip can install the gigaam dependency from GitHub,
# curl — for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY giga_stt ./giga_stt
# With INSTALL_DIARIZATION=1 the image also gets the pyannote speaker-diarization
# dependencies (.[diarization]) — ~150 MB extra; enable GIGAAM_DIARIZATION to use it.
ARG INSTALL_DIARIZATION=0
# torch 2.8.0 separately — stabilizes the cache layer; the rest comes in via pip install .
RUN pip install torch==2.8.0 torchaudio==2.8.0 \
    && if [ "$INSTALL_DIARIZATION" = "1" ]; then pip install .[diarization]; else pip install .; fi

# Both language models are downloaded into the /models volume on first start.
# To make a self-contained image, uncomment the next layer
# (the build takes longer, but the container works offline):
# RUN python -c "import gigaam; gigaam.load_model('v3_e2e_rnnt', device='cpu')"

EXPOSE 8000
VOLUME /models

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"status":"ready"' || exit 1

CMD ["uvicorn", "giga_stt.app:app", "--host", "0.0.0.0", "--port", "8000"]