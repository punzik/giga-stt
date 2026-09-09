"""Splitting long audio into chunks for inference.

Backends (selected via the ``GIGAAM_VAD_BACKEND`` setting):

* ``pyannote`` — the official repository path: ``model.transcribe_longform``
  (requires ``pip install pyannote.audio`` and access to the gated model
  ``pyannote/segmentation-3.0`` via ``HF_TOKEN``). Implemented in engine.
* ``silero``   — lightweight VAD from the ``silero-vad`` package (no gated access).
* ``energy``   — built-in fallback without external models: chunks are cut at
  the "quietest" frame near the target boundary.

Returned chunks are contiguous intervals ``[start, end)`` in samples (16 kHz),
cut so speech inside is not severed (cuts go through pauses / quiet sections).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

ChunkerResult = Tuple[List[torch.Tensor], List[Tuple[float, float]]]  # (wav pieces, [start, end] seconds)


def _merge_regions_to_chunks(
    regions: List[Tuple[int, int]], sr: int, max_sec: float
) -> List[Tuple[int, int]]:
    """Merge speech regions into contiguous chunks no longer than ``max_sec`` (as in gigaam.vad_utils)."""
    min_sec = max_sec * 0.68
    min_keep_sec = 0.2
    chunks: List[Tuple[int, int]] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    for start, end in regions:
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        dur = (cur_end - cur_start) / sr
        add = (end - cur_end) / sr
        if dur > min_keep_sec and (dur + add > max_sec or dur > min_sec):
            chunks.append((cur_start, cur_end))
            cur_start = start
        cur_end = end

    if cur_start is not None and (cur_end - cur_start) / sr > min_keep_sec:
        chunks.append((cur_start, cur_end))

    # Single speech regions longer than max_sec are cut evenly (hard safety net)
    out: List[Tuple[int, int]] = []
    for start, end in chunks:
        dur = (end - start) / sr
        if dur <= max_sec:
            out.append((start, end))
            continue
        parts = math.ceil(dur / max_sec)
        step = (end - start) / parts
        for i in range(parts):
            out.append((int(start + i * step), int(start + (i + 1) * step)))
    return out


def silero_chunks(wav: torch.Tensor, sr: int, max_sec: float) -> ChunkerResult:
    """Chunking by the silero VAD (https://github.com/snakers4/silero-vad)."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad = load_silero_vad()
    speech = get_speech_timestamps(
        wav,
        vad,
        sampling_rate=sr,
        speech_pad_ms=60,
        min_speech_duration_ms=200,
        min_silence_duration_ms=100,
    )
    regions = [(s["start"], s["end"]) for s in speech]
    if not regions:
        # VAD found no speech — process the whole file as-is (plain chunking)
        regions = [(0, wav.numel())]
    return _to_chunks(regions, wav, sr, max_sec)


def _to_chunks(
    regions: List[Tuple[int, int]], wav: torch.Tensor, sr: int, max_sec: float
) -> ChunkerResult:
    chunks = _merge_regions_to_chunks(regions, sr, max_sec)
    segments = [wav[s:e] for s, e in chunks]
    boundaries = [(s / sr, e / sr) for s, e in chunks]
    return segments, boundaries


def energy_chunks(wav: torch.Tensor, sr: int, max_sec: float) -> ChunkerResult:
    """Fallback chunking without external models: cut at local minima of the signal energy.

    A short search window (~±2.5 s) around the target boundary avoids cutting
    words in half — the boundary is placed on the quietest 100 ms frame.
    """
    hop = sr // 10  # 100 ms
    n = wav.numel() // hop
    if n == 0:
        return [], []
    energy = torch.log10(wav[: n * hop].view(n, hop).pow(2).mean(dim=1) + 1e-10)
    k = 5  # smoothing ~0.5 s
    energy = torch.nn.functional.avg_pool1d(
        energy.view(1, 1, -1), kernel_size=k, stride=1, padding=k // 2
    ).view(-1)

    max_frames = int(max_sec * 10)
    search = int(2.5 * 10)
    cut_points = [0]
    cut = 0
    while n - cut > max_frames:
        target = cut + max_frames
        lo = max(cut + 1, target - search)
        hi = min(n, target + search)
        cut = lo + int(energy[lo:hi].argmin())
        cut_points.append(cut)
    cut_points.append(n)

    boundaries = [(a * hop / sr, b * hop / sr) for a, b in zip(cut_points, cut_points[1:])]
    segments = [wav[a * hop : b * hop] for a, b in zip(cut_points, cut_points[1:])]
    return segments, boundaries


def make_chunks(
    wav: torch.Tensor,
    sr: int,
    backend: str,
    max_sec: float,
) -> Tuple[ChunkerResult, str]:
    """Build chunks with the selected (or fallback) backend. Returns ((segments, boundaries), actual backend)."""
    if backend == "silero":
        try:
            return silero_chunks(wav, sr, max_sec), "silero"
        except Exception as exc:  # noqa: BLE001 — silero may fail to download the model, etc.
            logger.warning("silero-vad failed (%s), falling back to energy chunking", exc)
    return energy_chunks(wav, sr, max_sec), "energy"