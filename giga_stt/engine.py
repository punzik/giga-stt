"""GigaAM inference engine: model loading, short and long audio.

Duration logic:
* ≤ 25 s — single model pass;
* > 25 s — segmentation into chunks (VAD) and batched inference.
  Segmentation backends: pyannote (official VAD), silero-vad, energy
  (built-in fallback; see ``service.chunking``).

Key difference from the official ``model.transcribe``: inference runs in fp32
without ``torch.autocast``. On GTX 16xx GPUs (TU116/TU117 — no tensor cores)
cuDNN fp16 convolutions on long inputs (T ≳ 550 frames) produce NaN even at
batch=1, so fp32 is the reliable default
(``GIGAAM_FP16_ENCODER=false``). On other GPUs you can enable fp16 explicitly.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import gigaam
from gigaam import load_audio, load_model
from gigaam.model import LONGFORM_THRESHOLD
from gigaam.types import Segment
from gigaam.utils import AudioDataset

from .chunking import make_chunks
from .config import Settings
from .diarization import DiarizationBackend

logger = logging.getLogger(__name__)

SAMPLE_RATE = gigaam.preprocess.SAMPLE_RATE
SHORT_LIMIT_SEC = LONGFORM_THRESHOLD / SAMPLE_RATE  # 25 s — model.transcribe limit


class AudioDecodeError(Exception):
    """Audio could not be read/decoded (corrupt file or unsupported format)."""


def _load_audio_verbose(audio_path: str) -> torch.Tensor:
    """``gigaam.load_audio`` with useful diagnostics: on failure — ffmpeg stderr."""
    try:
        return load_audio(audio_path)
    except RuntimeError as exc:
        stderr = b""
        try:
            probe = subprocess.run(
                ["ffmpeg", "-nostdin", "-threads", "0", "-i", audio_path, "-f", "null", "-"],
                capture_output=True,
                timeout=120,
            )
            stderr = probe.stderr or b""
        except Exception:  # noqa: BLE001 — diagnostics must not mask the original error
            pass
        tail = stderr.decode(errors="replace").strip()[-1500:]
        if "No such file or directory" in tail and "ffmpeg" in tail:
            raise RuntimeError("ffmpeg not found in the service process PATH") from exc
        raise AudioDecodeError(tail or "ffmpeg failed to decode the file") from exc


class AudioTooLongError(Exception):
    """Audio is longer than the allowed safety limit."""


@dataclass
class TranscriptionOutput:
    text: str
    duration: float
    segments: list
    vad_backend: Optional[str]
    inference_time: float
    # Dominant speaker per segment (aligned with ``segments``), when diarized
    speakers: Optional[list] = None


def _find_binary(name: str) -> Optional[Path]:
    """Locate a binary: PATH → GIGAAM_FFMPEG_PATH → <repo>/bin → nix-store (for NixOS)."""
    which = shutil.which(name)
    if which:
        return Path(which)

    custom = os.environ.get("GIGAAM_FFMPEG_PATH")
    if custom:
        p = Path(custom)
        if p.is_file() and p.name == name:
            return p
        if p.is_dir() and (p / name).is_file():
            return p / name

    repo_bin = Path(__file__).resolve().parent.parent / "bin" / name
    if repo_bin.is_file():
        return repo_bin

    for candidate in glob.glob(f"/nix/store/*ffmpeg*-bin/bin/{name}"):
        return Path(candidate)
    return None


def ensure_ffmpeg_on_path() -> Optional[Path]:
    """Ensure ``ffmpeg``/``ffprobe`` are available in the process PATH. Returns the path to ffmpeg."""
    ffmpeg = _find_binary("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Install ffmpeg or set GIGAAM_FFMPEG_PATH."
        )
    bin_dir = str(ffmpeg.parent)
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path
    ffprobe = _find_binary("ffprobe")
    if ffprobe is None:
        logger.warning("ffprobe not found — duration will be computed via decoding")
    return ffmpeg


@dataclass
class SpeechEngine:
    """Wrapper around the GigaAM model for use in a REST service."""

    settings: Settings
    model: object = field(init=False, default=None)
    device: torch.device = field(init=False, default=None)
    vad_backend: Optional[str] = field(init=False, default=None)
    ffmpeg_path: Optional[Path] = field(init=False, default=None)

    # Loading state
    ready: bool = field(init=False, default=False)
    loading_error: Optional[str] = field(init=False, default=None)
    _thread: threading.Thread = field(init=False, default=None)

    # ------------------------------------------------------------------ load

    def start_loading(self) -> None:
        """Start loading the model in a separate thread (does not block startup)."""
        self._thread = threading.Thread(target=self._load_in_thread, name="gigaam-loader", daemon=True)
        self._thread.start()

    def _load_in_thread(self) -> None:
        try:
            self.load()
            self.ready = True
            logger.info("Model %s ready (%s)", self.settings.model_name, self.device)
        except Exception as exc:  # noqa: BLE001
            self.loading_error = str(exc)
            logger.exception("Model loading error")

    def load(self) -> None:
        self.ffmpeg_path = ensure_ffmpeg_on_path()

        if self.settings.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.settings.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("GIGAAM_DEVICE=cuda but CUDA is not available")
        self.device = device

        if self.settings.hf_token:
            os.environ.setdefault("HF_TOKEN", self.settings.hf_token)

        t0 = time.monotonic()
        self.model = load_model(
            self.settings.model_name,
            fp16_encoder=self.settings.fp16_encoder,
            device=device,
            download_root=str(self.settings.resolved_cache_dir),
        )
        logger.info(
            "Model %s loaded in %.1f s (device=%s, fp16_encoder=%s)",
            self.settings.model_name,
            time.monotonic() - t0,
            device,
            self.settings.fp16_encoder,
        )

        self._resolve_vad_backend()
        self._warmup()

    def _resolve_vad_backend(self) -> None:
        requested = self.settings.vad_backend
        if requested == "auto":
            if self._try_pyannote():
                self.vad_backend = "pyannote"
            elif self._try_silero():
                self.vad_backend = "silero"
            else:
                self.vad_backend = "energy"
        elif requested == "pyannote":
            if not self._try_pyannote():
                raise RuntimeError(
                    "GIGAAM_VAD_BACKEND=pyannote but pyannote.audio is unavailable "
                    "(the .[longform] dependencies, an HF token and access to pyannote/segmentation-3.0 are required)"
                )
            self.vad_backend = "pyannote"
        elif requested == "silero":
            if not self._try_silero():
                raise RuntimeError("GIGAAM_VAD_BACKEND=silero but the silero-vad package is unavailable")
            self.vad_backend = "silero"
        else:
            self.vad_backend = "energy"
        logger.info("Long audio segmentation backend: %s", self.vad_backend)

    def _try_pyannote(self) -> bool:
        try:
            from gigaam.vad_utils import get_pipeline

            get_pipeline(self.device)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pyannote unavailable (%s). The official longform requires the .[longform] dependencies, "
                "an HF token and access to the gated model pyannote/segmentation-3.0. "
                "Using a fallback backend.",
                exc,
            )
            return False

    def _try_silero(self) -> bool:
        try:
            import silero_vad  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            logger.info("silero-vad unavailable (%s)", exc)
            return False
        try:
            from silero_vad import load_silero_vad

            load_silero_vad()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load the silero-vad model (%s)", exc)
            return False

    def _warmup(self) -> None:
        """Warm up CUDA cores with a short fragment of silence so the first request is not slow.

        The file is written to a temp directory so the weights cache can be
        read-only (e.g. GIGAAM_CACHE_DIR pointing into /nix/store).
        """
        try:
            import tempfile

            import soundfile as sf

            warmup_path = Path(tempfile.gettempdir()) / "giga-stt-warmup.wav"
            if not warmup_path.exists():
                sf.write(warmup_path, torch.zeros(SAMPLE_RATE).numpy(), SAMPLE_RATE)
            self.transcribe_file(str(warmup_path))
            logger.info("Inference warmup done")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warmup failed (non-critical): %s", exc)

    # ------------------------------------------------------------- inference

    def _forward(self, wav: torch.Tensor, lengths: torch.Tensor):
        """Preprocessor + encoder. fp32 without autocast (see the module docstring);
        with GIGAAM_FP16_ENCODER=true — as in the official forward (with autocast)."""
        features, feature_lengths = self.model.preprocessor(wav, lengths)
        if self.settings.fp16_encoder and self.device.type == "cuda":
            with torch.autocast("cuda", torch.float16):
                return self.model.encoder(features, feature_lengths)
        return self.model.encoder(features, feature_lengths)

    @torch.inference_mode()
    def transcribe_file(
        self, audio_path: str, diarization: Optional[DiarizationBackend] = None
    ) -> TranscriptionOutput:
        t0 = time.monotonic()
        duration = self._probe_duration(audio_path)
        if duration > self.settings.max_audio_sec:
            raise AudioTooLongError(
                f"Audio duration {duration:.0f} s exceeds the "
                f"{self.settings.max_audio_sec:.0f} s limit"
            )

        if duration <= SHORT_LIMIT_SEC:
            try:
                segments, vad_backend = self._transcribe_short(audio_path, duration)
            except AudioDecodeError:
                raise
            except RuntimeError as exc:
                if "Failed to load audio" in str(exc):
                    raise AudioDecodeError(str(exc)) from exc
                raise
        else:
            segments, vad_backend = self._transcribe_long(audio_path)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        # Diarization runs over the whole audio (turns can cross ASR chunk boundaries)
        speakers = None
        if diarization is not None and segments:
            wav = _load_audio_verbose(audio_path)
            speakers = diarization.assign(wav, SAMPLE_RATE, segments)
        return TranscriptionOutput(
            text=text,
            duration=duration,
            segments=segments,
            vad_backend=vad_backend,
            inference_time=time.monotonic() - t0,
            speakers=speakers,
        )

    # -- short audio ---------------------------------------------------------

    def _transcribe_short(self, audio_path: str, duration: float):
        """Single model pass over audio ≤ 25 s (without model.transcribe — see _forward)."""
        wav = _load_audio_verbose(audio_path)
        wav = wav.to(self.device).unsqueeze(0)
        lengths = torch.tensor([wav.shape[-1]], device=self.device)
        encoded, encoded_len = self._forward(wav, lengths)
        text, _ = self.model._decode(encoded, encoded_len, lengths, False)[0]
        segment = Segment(text=text, start=0.0, end=round(duration, 3))
        return [segment], None

    # -- long audio ----------------------------------------------------------

    def _transcribe_long(self, audio_path: str):
        if self.vad_backend == "pyannote":
            try:
                # Official pyannote segmentation + our own batched decode
                # (model.transcribe_longform uses autocast and is unreliable on
                # some GPUs — see the module docstring).
                from gigaam.vad_utils import segment_audio_file

                segments, boundaries = segment_audio_file(
                    audio_path, SAMPLE_RATE, device=self.device
                )
                if not segments:
                    return [], "pyannote"
                return (
                    self._batched_decode(segments, boundaries),
                    "pyannote",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("pyannote longform failed (%s) — falling back to built-in chunking", exc)

        wav = _load_audio_verbose(audio_path)
        chosen = self.vad_backend if self.vad_backend in {"silero", "energy"} else "auto"
        if chosen == "auto":
            chosen = "silero"
        (segments_tensors, boundaries), used = make_chunks(
            wav, SAMPLE_RATE, chosen, self.settings.chunk_max_sec
        )
        if not segments_tensors:
            return [], used
        return self._batched_decode(segments_tensors, boundaries), used

    def _batched_decode(self, segment_tensors, boundaries):
        """Batched chunk inference (mirrors the internal loop of model.transcribe_longform).

        On NaN in a batch (fp16 numerical quirks on some GPUs), the request is
        automatically retried chunk by chunk (batch=1).
        """
        result_segments = []
        ds = AudioDataset(segment_tensors, tokenizer=None)
        dl = DataLoader(
            ds,
            batch_size=self.settings.fr_batch_size,
            shuffle=False,
            collate_fn=AudioDataset.collate,
            num_workers=0,
        )
        idx = 0
        for wav_pad, wav_lens in dl:
            wav_pad = wav_pad.to(self.device)
            wav_lens = wav_lens.to(self.device)
            encoded, encoded_len = self._forward(wav_pad, wav_lens)
            if torch.isnan(encoded).any():
                # fallback path: the same chunks one at a time
                logger.warning("NaN in batch inference (chunks %d-%d) — retrying one by one",
                               idx, idx + len(wav_lens) - 1)
                for k in range(idx, idx + len(wav_lens)):
                    result_segments.extend(
                        self._batched_decode([segment_tensors[k]], [boundaries[k]])
                    )
                idx += len(wav_lens)
                continue
            for text, _ in self.model._decode(encoded, encoded_len, wav_lens, False):
                seg_start, seg_end = boundaries[idx]
                idx += 1
                result_segments.append(Segment(text=text, start=seg_start, end=seg_end))
        return result_segments

    # ------------------------------------------------------------- utilities

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
                logger.debug("ffprobe could not determine the duration (%s) — decoding instead", exc)
        return _load_audio_verbose(audio_path).numel() / SAMPLE_RATE

    @property
    def status(self) -> str:
        if self.ready:
            return "ready"
        if self.loading_error:
            return "error"
        return "loading"