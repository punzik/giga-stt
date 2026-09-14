"""FastAPI application: REST speech-to-text service based on GigaAM.

Endpoints:
* ``GET  /``              — service info
* ``GET  /health``        — model status
* ``POST /v1/transcribe`` — audio transcription (multipart/form-data)
* ``POST /v1/transcribe-raw`` — the same with a raw byte stream
  (``Content-Type: audio/*``, parameters in the query string)

Supported audio formats — anything ffmpeg can decode
(wav, mp3, ogg/opus, m4a, flac, webm, amr…), any number of channels,
any sample rate (resampled to 16 kHz).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile

from . import __version__
from .config import Settings
from .diarization import DiarizationBackend
from .engine import AudioDecodeError, AudioTooLongError
from .schemas import HealthResponse, ServiceInfo, SegmentOut, TranscriptionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gigaam-stt")

settings = Settings.from_env()
engines: dict[str, object] = {}
diarization_backend: DiarizationBackend | None = None

AUDIO_CT_SUFFIX = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/vnd.wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/amr": ".amr",
}


def _build_engine() -> "SpeechEngine":
    from .engine import SpeechEngine

    return SpeechEngine(settings)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    from .granite_engine import GraniteSpeechEngine

    global engines, diarization_backend
    engines = {
        "ru": _build_engine(),
        "en": GraniteSpeechEngine(settings),
    }
    for language, engine in engines.items():
        engine.start_loading()
        logger.info("Service starting; %s model is loading in the background", language)
    diarization_backend = DiarizationBackend(settings)
    diarization_backend.start_loading()
    if diarization_backend.enabled:
        logger.info("Service starting; diarization is loading in the background")
    try:
        yield
    finally:
        logger.info("Service stopped")


app = FastAPI(
    title="GigaAM Speech-to-Text",
    description=(
        "Speech-to-text REST service. Use language=ru (default) for GigaAM or "
        "language=en for IBM Granite Speech TurboCTC. Audio up to 25 s is "
        "transcribed in a single pass; longer audio is segmented by voice activity."
    ),
    version=__version__,
    lifespan=lifespan,
)

executor = ThreadPoolExecutor(max_workers=settings.max_concurrency, thread_name_prefix="stt")
_queue_sem = asyncio.Semaphore(settings.max_concurrency)


# ------------------------------------------------------------------ helpers


def require_api_key(request: Request) -> None:
    """Optional authentication: GIGAAM_API_KEY + X-API-Key / Bearer header."""
    if not settings.api_key:
        return
    provided = request.headers.get("x-api-key")
    if not provided:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if provided != settings.api_key:
        raise HTTPException(status_code=401, detail="A valid API key is required (X-API-Key)")


def _ensure_engine_ready(language: str) -> object:
    engine = engines.get(language)
    if engine is None:  # pragma: no cover — requests before lifespan are not served
        raise HTTPException(status_code=503, detail="Service is still initializing")
    if engine.loading_error:
        raise HTTPException(status_code=500, detail=f"{language} model failed to load: {engine.loading_error}")
    if not engine.ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading, please retry the request later",
            headers={"Retry-After": "10"},
        )
    return engine


async def _stream_to_tempfile(filename: Optional[str], chunks: AsyncIterator[bytes]) -> Path:
    """Save the upload to disk with size control (without bloating memory)."""
    suffix = (Path(filename or "").suffix or ".audio")[:10]
    fd, tmp_name = tempfile.mkstemp(prefix="gigaam_", suffix=suffix)
    tmp = Path(tmp_name)
    limit = settings.max_upload_mb * 1024 * 1024
    total = 0
    try:
        with os.fdopen(fd, "wb") as f:
            async for chunk in chunks:
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_mb} MB",
                    )
                f.write(chunk)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if total == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty audio file")
    return tmp


async def _upload_stream(upload: UploadFile) -> AsyncIterator[bytes]:
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        yield chunk


def _ensure_diarization_ready() -> None:
    """Reject with_speakers=true when the backend is off / loading / failed."""
    backend = diarization_backend
    if backend is None or backend.status() == "off":
        raise HTTPException(
            status_code=400,
            detail="Speaker diarization is not enabled "
            "(set GIGAAM_DIARIZATION=pyannote and install the .[diarization] dependencies)",
        )
    if backend.loading_error:
        raise HTTPException(
            status_code=500,
            detail=f"Diarization backend failed to load: {backend.loading_error}",
        )
    if not backend.ready:
        raise HTTPException(
            status_code=503,
            detail="Diarization model is still loading, please retry the request later",
            headers={"Retry-After": "10"},
        )


def _segment_out(seg, speaker: Optional[str] = None) -> SegmentOut:
    return SegmentOut(text=seg.text, start=round(seg.start, 3), end=round(seg.end, 3), speaker=speaker)


async def _run_inference(
    tmp: Path, language: str, with_segments: bool, with_speakers: bool
) -> TranscriptionResponse:
    """Queue the request for inference and run it in a worker."""
    loop = asyncio.get_running_loop()
    acquired = False
    try:
        try:
            async with asyncio.timeout(settings.queue_timeout_sec):
                await _queue_sem.acquire()
                acquired = True
        except TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Service is overloaded, all workers are busy. Please retry later.",
                headers={"Retry-After": "15"},
            )

        t0 = time.monotonic()
        diarization = diarization_backend if with_speakers else None
        try:
            result = await loop.run_in_executor(
                executor, engines[language].transcribe_file, str(tmp), diarization
            )
        except AudioDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Failed to decode audio: {exc}")
        except AudioTooLongError as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference error")
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
        inference = time.monotonic() - t0

        return TranscriptionResponse(
            text=result.text,
            duration=round(result.duration, 3),
            language=language,
            model=(settings.model_name if language == "ru" else settings.granite_model_name),
            vad_backend=result.vad_backend,
            processing_time=round(inference, 3),
            segments=(
                [
                    _segment_out(
                        s, result.speakers[i] if result.speakers is not None else None
                    )
                    for i, s in enumerate(result.segments)
                ]
                if with_segments
                else None
            ),
        )
    finally:
        if acquired:
            _queue_sem.release()


# ------------------------------------------------------------------- routes


@app.get("/", response_model=ServiceInfo)
async def root() -> ServiceInfo:
    return ServiceInfo(model=f"ru: {settings.model_name}; en: {settings.granite_model_name}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    import torch

    engine = engines.get("ru")
    if engine is None:
        return HealthResponse(status="loading", version=__version__)
    failed = next((e for e in engines.values() if e.loading_error), None)
    status = "error" if failed else "ready" if all(e.ready for e in engines.values()) else "loading"
    diarization_status: Optional[str] = None
    if diarization_backend is not None:
        diarization_status = diarization_backend.status()
        if diarization_backend.loading_error:
            diarization_status = f"error: {diarization_backend.loading_error}"
    return HealthResponse(
        status=status,
        model=f"ru: {settings.model_name}; en: {settings.granite_model_name}",
        device=str(engine.device) if engine.device is not None else None,
        torch=torch.__version__,
        vad_backend=engine.vad_backend,
        diarization=diarization_status,
        ffmpeg=str(engine.ffmpeg_path) if engine.ffmpeg_path else None,
        detail=failed.loading_error if failed else None,
    )


@app.post(
    "/v1/transcribe",
    response_model=TranscriptionResponse,
    responses={503: {"description": "Model is loading or overloaded"}},
    summary="Speech transcription (multipart/form-data)",
)
async def transcribe(
    _: None = Depends(require_api_key),
    file: UploadFile = File(..., description="Audio (wav, mp3, ogg, m4a, flac, webm…)"),
    language: Literal["ru", "en"] = Form("ru", description="Recognition language"),
    with_segments: bool = Form(True, description="Return the segment list"),
    with_speakers: bool = Form(
        False, description="Add the dominant speaker to segments (requires the diarization backend)"
    ),
) -> TranscriptionResponse:
    _ensure_engine_ready(language)
    if with_speakers:
        _ensure_diarization_ready()
    tmp = await _stream_to_tempfile(file.filename, _upload_stream(file))
    try:
        return await _run_inference(tmp, language, with_segments, with_speakers)
    finally:
        tmp.unlink(missing_ok=True)


@app.post(
    "/v1/transcribe-raw",
    response_model=TranscriptionResponse,
    responses={503: {"description": "Model is loading or overloaded"}},
    summary="Speech transcription (raw request body)",
    openapi_extra={
        "requestBody": {
            "content": {
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                "audio/ogg": {"schema": {"type": "string", "format": "binary"}},
            },
            "required": True,
        }
    },
)
async def transcribe_raw(
    request: Request,
    _: None = Depends(require_api_key),
    language: Literal["ru", "en"] = Query("ru", description="Recognition language"),
    with_segments: bool = Query(True, description="Return the segment list"),
    with_speakers: bool = Query(
        False, description="Add the dominant speaker to segments (requires the diarization backend)"
    ),
) -> TranscriptionResponse:
    _ensure_engine_ready(language)
    if with_speakers:
        _ensure_diarization_ready()
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    filename = (
        os.path.basename(request.headers.get("x-filename", ""))
        or f"audio{AUDIO_CT_SUFFIX.get(ctype, '.audio')}"
    )
    tmp = await _stream_to_tempfile(filename, request.stream())
    try:
        return await _run_inference(tmp, language, with_segments, with_speakers)
    finally:
        tmp.unlink(missing_ok=True)