"""English ASR backend powered by IBM Granite Speech TurboCTC."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from gigaam.types import Segment
from transformers import AutoModelForCTC, AutoProcessor

from .chunking import make_chunks
from .config import Settings
from .engine import (
    AudioDecodeError,
    AudioTooLongError,
    SAMPLE_RATE,
    SHORT_LIMIT_SEC,
    TranscriptionOutput,
    _find_binary,
    _load_audio_verbose,
    ensure_ffmpeg_on_path,
)

logger = logging.getLogger(__name__)


@dataclass
class GraniteSpeechEngine:
    """Load and run the English-only Granite Speech TurboCTC model."""

    settings: Settings
    model: object = field(init=False, default=None)
    processor: object = field(init=False, default=None)
    device: torch.device = field(init=False, default=None)
    vad_backend: Optional[str] = field(init=False, default=None)
    ready: bool = field(init=False, default=False)
    loading_error: Optional[str] = field(init=False, default=None)
    _thread: threading.Thread = field(init=False, default=None)

    def start_loading(self) -> None:
        self._thread = threading.Thread(target=self._load_in_thread, name="granite-loader", daemon=True)
        self._thread.start()

    def _load_in_thread(self) -> None:
        try:
            self.load()
            self.ready = True
            logger.info("Model %s ready (%s)", self.settings.granite_model_name, self.device)
        except Exception as exc:  # noqa: BLE001
            self.loading_error = str(exc)
            logger.exception("Granite model loading error")

    def load(self) -> None:
        ensure_ffmpeg_on_path()
        if self.settings.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.settings.device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("GIGAAM_DEVICE=cuda but CUDA is not available")

        cache_dir = str(self.settings.resolved_granite_cache_dir)
        self.processor = AutoProcessor.from_pretrained(self.settings.granite_model_name, cache_dir=cache_dir)
        self.model = AutoModelForCTC.from_pretrained(
            self.settings.granite_model_name, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def transcribe_file(self, audio_path: str) -> TranscriptionOutput:
        t0 = time.monotonic()
        duration = self._probe_duration(audio_path)
        if duration > self.settings.max_audio_sec:
            raise AudioTooLongError(
                f"Audio duration {duration:.0f} s exceeds the {self.settings.max_audio_sec:.0f} s limit"
            )
        wav = _load_audio_verbose(audio_path)

        if duration <= SHORT_LIMIT_SEC:
            segments = [Segment(text=self._decode(wav), start=0.0, end=round(duration, 3))]
            vad_backend = None
        else:
            requested = self.settings.vad_backend if self.settings.vad_backend in {"silero", "energy"} else "silero"
            (chunks, boundaries), vad_backend = make_chunks(
                wav, SAMPLE_RATE, requested, self.settings.chunk_max_sec
            )
            segments = [
                Segment(text=self._decode(chunk), start=round(start, 3), end=round(end, 3))
                for chunk, (start, end) in zip(chunks, boundaries)
            ]

        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return TranscriptionOutput(
            text=text,
            duration=duration,
            segments=segments,
            vad_backend=vad_backend,
            inference_time=time.monotonic() - t0,
        )

    def _decode(self, wav: torch.Tensor) -> str:
        try:
            inputs = self.processor(
                wav.cpu().numpy(),
                sampling_rate=SAMPLE_RATE,
                device=self.device,
                return_tensors="pt",
            )
            inputs.to(self.device, dtype=self.model.dtype)
            tokens = self.model.generate(**inputs)
            return self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
        except RuntimeError as exc:
            raise AudioDecodeError(str(exc)) from exc

    def _probe_duration(self, audio_path: str) -> float:
        ffprobe = _find_binary("ffprobe")
        if ffprobe is not None:
            try:
                out = subprocess.run(
                    [
                        str(ffprobe),
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        audio_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                return float(out.stdout.strip())
            except Exception as exc:  # noqa: BLE001
                logger.debug("ffprobe could not determine duration (%s) — decoding instead", exc)
        return _load_audio_verbose(audio_path).numel() / SAMPLE_RATE

    @property
    def status(self) -> str:
        if self.ready:
            return "ready"
        return "error" if self.loading_error else "loading"
