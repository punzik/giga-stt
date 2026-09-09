"""Pydantic schemas for the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from . import __version__


class WordOut(BaseModel):
    text: str = Field(..., examples=["hello"])
    start: float = Field(..., description="Word start, seconds")
    end: float = Field(..., description="Word end, seconds")


class SegmentOut(BaseModel):
    text: str
    start: float = Field(..., description="Segment start, seconds (0 for short audio)")
    end: float = Field(..., description="Segment end, seconds")
    words: Optional[List[WordOut]] = Field(
        None, description="Words with timestamps (if word_timestamps=true)"
    )


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Final transcript (for e2e models — with punctuation)")
    duration: float = Field(..., description="Audio duration, seconds")
    language: str = Field("ru", description="Model language")
    model: str = Field(..., description="GigaAM model name")
    vad_backend: Optional[str] = Field(
        None, description="Segmentation backend (for audio longer than 25 s)"
    )
    processing_time: float = Field(..., description="Inference time, seconds")
    segments: Optional[List[SegmentOut]] = Field(
        None, description="Segments (for audio longer than 25 s)"
    )
    words: Optional[List[WordOut]] = Field(
        None, description="Flat list of words with timestamps (if word_timestamps=true)"
    )


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ready"])
    version: str = __version__
    model: Optional[str] = None
    device: Optional[str] = None
    torch: Optional[str] = None
    vad_backend: Optional[str] = None
    ffmpeg: Optional[str] = None
    detail: Optional[str] = Field(None, description="Loading error message, if any")


class ServiceInfo(BaseModel):
    service: str = "gigaam-stt"
    version: str = __version__
    model: str
    docs: str = "/docs"
    health: str = "/health"
    transcribe: str = "/v1/transcribe"