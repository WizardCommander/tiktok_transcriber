from functools import lru_cache
from pathlib import Path
from typing import Iterable
from typing import Protocol

from faster_whisper import BatchedInferencePipeline
from faster_whisper import WhisperModel
from openai import OpenAI
from openai.types.audio.transcription_segment import (
    TranscriptionSegment as OpenAITranscriptionSegment,
)

from tiktok_transcriber.models import SyncProfileSettings
from tiktok_transcriber.models import TranscriptSegment


class TimestampedSegment(Protocol):
    start: float
    end: float
    text: str


@lru_cache(maxsize=4)
def load_whisper_model(model_name: str) -> WhisperModel:
    return WhisperModel(model_name, device="cpu", compute_type="int8")


@lru_cache(maxsize=4)
def load_whisper_pipeline(model_name: str) -> BatchedInferencePipeline:
    return BatchedInferencePipeline(model=load_whisper_model(model_name))


@lru_cache(maxsize=2)
def load_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def transcribe_video(
    video_path: Path, settings: SyncProfileSettings
) -> tuple[str, list[TranscriptSegment]]:
    if settings.transcription_provider == "openai":
        return transcribe_video_with_openai(video_path, settings)
    return transcribe_video_with_local_whisper(video_path, settings)


def transcribe_video_with_openai(
    video_path: Path, settings: SyncProfileSettings
) -> tuple[str, list[TranscriptSegment]]:
    if settings.openai_api_key is None:
        raise ValueError("OpenAI API key is required for OpenAI transcription")
    client = load_openai_client(settings.openai_api_key)
    with video_path.open("rb") as audio_file:
        if settings.transcription_language is None:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=settings.openai_transcription_model,
                response_format="json",
            )
        else:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=settings.openai_transcription_model,
                language=settings.transcription_language,
                response_format="json",
            )
    return build_openai_transcript(transcription.text)


def transcribe_video_with_local_whisper(
    video_path: Path, settings: SyncProfileSettings
) -> tuple[str, list[TranscriptSegment]]:
    pipeline = load_whisper_pipeline(settings.whisper_model)
    raw_segments, _ = pipeline.transcribe(
        str(video_path),
        language=settings.transcription_language,
        vad_filter=True,
        batch_size=16,
        without_timestamps=False,
    )
    return build_whisper_transcript(raw_segments)


def build_openai_transcript(
    transcript_text: str,
    raw_segments: Iterable[TimestampedSegment] | None = None,
) -> tuple[str, list[TranscriptSegment]]:
    if raw_segments is None:
        return transcript_text.strip(), []
    transcript_segments = [
        build_transcript_segment(segment)
        for segment in raw_segments
        if segment.text.strip()
    ]
    return transcript_text.strip(), transcript_segments


def build_whisper_transcript(
    raw_segments: Iterable[TimestampedSegment],
) -> tuple[str, list[TranscriptSegment]]:
    transcript_segments: list[TranscriptSegment] = []
    transcript_parts: list[str] = []
    for raw_segment in raw_segments:
        text = raw_segment.text.strip()
        if not text:
            continue
        transcript_segments.append(
            TranscriptSegment(
                start_seconds=round(raw_segment.start, 2),
                end_seconds=round(raw_segment.end, 2),
                text=text,
            )
        )
        transcript_parts.append(text)
    return " ".join(transcript_parts).strip(), transcript_segments


def build_transcript_segment(
    segment: TimestampedSegment | OpenAITranscriptionSegment,
) -> TranscriptSegment:
    return TranscriptSegment(
        start_seconds=round(float(segment.start), 2),
        end_seconds=round(float(segment.end), 2),
        text=segment.text.strip(),
    )
