"""Service configuration. All parameters are set via ``GIGAAM_``-prefixed environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    """Service runtime settings."""

    # GigaAM model name (any *_ctc / *_rnnt; defaults to the e2e RNNT variant
    # with punctuation and text normalization).
    model_name: str = "v3_e2e_rnnt"
    # auto | cuda | cpu
    device: str = "auto"
    # Move the encoder to fp16 + autocast (speeds up inference on GPUs with
    # tensor cores). Disabled by default: on GTX 16xx GPUs (TU116/TU117 — no
    # tensor cores) fp16 cuDNN convolutions on long inputs produce NaN.
    fp16_encoder: bool = False
    # GigaAM checkpoint cache directory (defaults to ~/.cache/gigaam).
    cache_dir: str = ""
    # English IBM Granite CTC model and its Hugging Face cache directory.
    granite_model_name: str = "ibm-granite/granite-speech-5.0-470m-turboctc"
    granite_cache_dir: str = ""

    # HTTP
    host: str = "0.0.0.0"
    port: int = 8000

    # Request limits
    max_upload_mb: int = 100
    max_audio_sec: float = 2 * 60 * 60  # safety cap on audio duration

    # Inference parallelism (number of worker threads)
    max_concurrency: int = 1
    # How many seconds a request may wait in the queue before getting a 503
    queue_timeout_sec: float = 300.0

    # Long audio: maximum chunk length (model limit — 25 s per chunk)
    chunk_max_sec: float = 22.0
    # Batch size for batched chunk inference (tune to your VRAM size)
    fr_batch_size: int = 4

    # auto | pyannote | silero | energy
    vad_backend: str = "auto"

    # If set — requests must send the X-API-Key: <key> or Bearer <key> header
    api_key: str = ""

    # HF token for the official longform backend (pyannote/segmentation-3.0, gated)
    # and for the gated diarization model
    hf_token: str = ""

    # off | pyannote — speaker diarization backend (with_speakers=true requests)
    diarization: str = "off"
    # Gated pyannote diarization pipeline (accept the terms at hf.co, CC-BY-4.0)
    diarization_model: str = "pyannote/speaker-diarization-community-1"

    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir).expanduser() if self.cache_dir else Path(
            "~/.cache/gigaam"
        ).expanduser()

    @property
    def resolved_granite_cache_dir(self) -> Path:
        return Path(self.granite_cache_dir).expanduser() if self.granite_cache_dir else Path(
            "~/.cache/huggingface"
        ).expanduser()

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.model_name = _env("GIGAAM_MODEL", s.model_name)
        s.device = _env("GIGAAM_DEVICE", s.device).lower()
        s.fp16_encoder = _bool(_env("GIGAAM_FP16_ENCODER", "true" if s.fp16_encoder else "false"))
        s.cache_dir = _env("GIGAAM_CACHE_DIR", s.cache_dir)
        s.granite_model_name = _env("GRANITE_MODEL", s.granite_model_name)
        s.granite_cache_dir = _env("GRANITE_CACHE_DIR", s.granite_cache_dir)
        s.host = _env("GIGAAM_HOST", s.host)
        s.port = int(_env("GIGAAM_PORT", str(s.port)))
        s.max_upload_mb = int(_env("GIGAAM_MAX_UPLOAD_MB", str(s.max_upload_mb)))
        s.max_audio_sec = float(_env("GIGAAM_MAX_AUDIO_SEC", str(s.max_audio_sec)))
        s.max_concurrency = max(1, int(_env("GIGAAM_MAX_CONCURRENCY", str(s.max_concurrency))))
        s.queue_timeout_sec = float(_env("GIGAAM_QUEUE_TIMEOUT_SEC", str(s.queue_timeout_sec)))
        s.chunk_max_sec = float(_env("GIGAAM_CHUNK_MAX_SEC", str(s.chunk_max_sec)))
        s.chunk_max_sec = min(s.chunk_max_sec, 24.0)  # hard model limit — 25 s
        s.fr_batch_size = max(1, int(_env("GIGAAM_FR_BATCH_SIZE", str(s.fr_batch_size))))
        s.vad_backend = _env("GIGAAM_VAD_BACKEND", s.vad_backend).lower()
        s.api_key = _env("GIGAAM_API_KEY", s.api_key)
        s.hf_token = _env("HF_TOKEN", s.hf_token)
        s.diarization = _env("GIGAAM_DIARIZATION", s.diarization).lower()
        s.diarization_model = _env("GIGAAM_DIARIZATION_MODEL", s.diarization_model)
        return s
