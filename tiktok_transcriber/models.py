from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CookieSettings:
    file_path: Path | None = None
    browser: str | None = None


@dataclass(frozen=True)
class ProfileVideo:
    video_id: str
    url: str
    profile_handle: str
    title: str
    description: str
    timestamp: int | None
    duration_seconds: int | None
    view_count: int | None


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class SceneDescription:
    timestamp_seconds: float
    summary: str
    visible_text: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VideoArtifacts:
    video: ProfileVideo
    video_path: Path | None
    transcript_text: str
    transcript_segments: list[TranscriptSegment]
    scene_descriptions: list[SceneDescription]


@dataclass(frozen=True)
class SyncProfileSettings:
    profile_url: str
    profile_handle: str
    output_dir: Path
    cookie_settings: CookieSettings
    gemini_api_key: str | None
    gemini_model: str
    transcription_provider: str
    openai_api_key: str | None
    openai_transcription_model: str
    whisper_model: str
    frame_samples: int
    limit: int | None
    skip_scene_descriptions: bool
    transcription_language: str | None
    overwrite: bool
