"""Optional speaker diarization backend (pyannote/speaker-diarization-community-1).

Enabled with ``GIGAAM_DIARIZATION=pyannote`` + the ``.[diarization]`` dependencies
and an HF token with accepted terms for the gated model. The pipeline loads in a
background thread alongside the ASR engine.

For every transcribed segment ``[start, end)`` the speaker with the largest
overlap in the *exclusive* diarization annotation (no overlapped speech) is
assigned — that annotation is designed for reconciliation with ASR timestamps.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class DiarizationBackend:
    """Wrapper around the pyannote speaker-diarization pipeline."""

    settings: Settings
    pipeline: object = field(init=False, default=None)
    device: torch.device = field(init=False, default=None)
    ready: bool = field(init=False, default=False)
    loading_error: Optional[str] = field(init=False, default=None)
    _thread: threading.Thread = field(init=False, default=None)

    # ------------------------------------------------------------------ state

    @property
    def enabled(self) -> bool:
        return self.settings.diarization == "pyannote"

    def status(self) -> str:
        """off | loading | ready | error"""
        if not self.enabled:
            return "off"
        if self.ready:
            return "ready"
        return "error" if self.loading_error else "loading"

    # ------------------------------------------------------------------ load

    def start_loading(self) -> None:
        """Start loading the pipeline in a separate thread (no-op when disabled)."""
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._load_in_thread, name="diarization-loader", daemon=True)
        self._thread.start()

    def _load_in_thread(self) -> None:
        try:
            self.load()
            self.ready = True
            logger.info("Diarization %s ready (%s)", self.settings.diarization_model, self.device)
        except Exception as exc:  # noqa: BLE001
            self.loading_error = str(exc)
            logger.exception("Diarization pipeline loading error")

    def load(self) -> None:
        from pyannote.audio import Pipeline

        if self.settings.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.settings.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("GIGAAM_DEVICE=cuda but CUDA is not available")
        self.device = device

        token = self.settings.hf_token
        if token:
            os.environ.setdefault("HF_TOKEN", token)

        t0 = time.monotonic()
        kwargs = {"token": token} if token else {}
        pipeline = Pipeline.from_pretrained(self.settings.diarization_model, **kwargs)
        pipeline.to(device)
        self.pipeline = pipeline
        logger.info(
            "Diarization %s loaded in %.1f s (device=%s)",
            self.settings.diarization_model,
            time.monotonic() - t0,
            device,
        )

    # ------------------------------------------------------------- inference

    def assign(self, waveform: torch.Tensor, sample_rate: int, segments) -> list[Optional[str]]:
        """Diarize the whole audio and return the dominant speaker per ASR segment.

        ``waveform`` — mono 1-D (N,) or (1, N) tensor at ``sample_rate`` (16 kHz).
        """
        if self.pipeline is None:
            raise RuntimeError("Diarization pipeline is not loaded")
        wav = waveform if waveform.dim() == 2 else waveform.unsqueeze(0)
        with torch.inference_mode():
            output = self.pipeline({"waveform": wav, "sample_rate": sample_rate})
        annotation = output.exclusive_speaker_diarization
        return [self._dominant_speaker(annotation, seg.start, seg.end) for seg in segments]

    @staticmethod
    def _dominant_speaker(annotation, start: float, end: float) -> Optional[str]:
        totals: dict[str, float] = {}
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > 0:
                totals[speaker] = totals.get(speaker, 0.0) + overlap
        if not totals:
            return None
        return max(totals, key=totals.get)
