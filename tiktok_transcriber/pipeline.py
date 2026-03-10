from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Callable

from yt_dlp.utils import DownloadError

from tiktok_transcriber.corpus import build_clean_video_document
from tiktok_transcriber.corpus import build_retrieval_chunks
from tiktok_transcriber.corpus import build_skipped_video_record
from tiktok_transcriber.corpus import build_video_document
from tiktok_transcriber.corpus import build_video_markdown
from tiktok_transcriber.corpus import serialize_scene_description
from tiktok_transcriber.corpus import serialize_transcript_segment
from tiktok_transcriber.download import find_downloaded_video
from tiktok_transcriber.models import (
    ProfileVideo,
    SceneDescription,
    SyncProfileSettings,
    TranscriptSegment,
    VideoArtifacts,
)


DiscoverProfileVideos = Callable[[SyncProfileSettings], list[ProfileVideo]]
DownloadVideo = Callable[[ProfileVideo, Path, SyncProfileSettings], Path]
TranscribeVideo = Callable[
    [Path, SyncProfileSettings], tuple[str, list[TranscriptSegment]]
]
DescribeVideo = Callable[
    [Path, Path, ProfileVideo, SyncProfileSettings], list[SceneDescription]
]


@dataclass(frozen=True)
class PipelineDependencies:
    discover_profile_videos: DiscoverProfileVideos
    download_video: DownloadVideo
    transcribe_video: TranscribeVideo
    describe_video: DescribeVideo


def sync_profile(
    settings: SyncProfileSettings, dependencies: PipelineDependencies
) -> list[Path]:
    profile_dir = settings.output_dir / settings.profile_handle
    videos_dir = profile_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    documents: list[dict[str, object]] = []
    skipped_videos: list[dict[str, object]] = []
    discovered_videos = dependencies.discover_profile_videos(settings)
    write_json(
        profile_dir / "manifest.json",
        [
            {
                "video_id": video.video_id,
                "url": video.url,
                "profile_handle": video.profile_handle,
                "title": video.title,
                "description": video.description,
                "timestamp": video.timestamp,
                "duration_seconds": video.duration_seconds,
                "view_count": video.view_count,
            }
            for video in discovered_videos
        ],
    )
    for video in discovered_videos:
        video_dir = videos_dir / video.video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        skip_json_path = video_dir / "skip.json"
        document_json_path = video_dir / "document.json"
        document_markdown_path = video_dir / "document.md"
        transcript_json_path = video_dir / "transcript.json"
        scenes_json_path = video_dir / "scenes.json"
        if skip_json_path.exists() and not settings.overwrite:
            skip_payload = load_existing_skip(skip_json_path)
            skipped_videos.append(build_skipped_video_record(video, skip_payload))
            cleanup_video_processing_files(
                find_downloaded_video(video_dir, video.video_id),
                video_dir / "frames",
            )
            outputs.append(skip_json_path)
            continue
        if can_reuse_existing_outputs(
            document_json_path,
            document_markdown_path,
            transcript_json_path,
            scenes_json_path,
            settings,
        ):
            document = load_existing_document(document_json_path)
            cleanup_video_processing_files(
                find_downloaded_video(video_dir, video.video_id),
                video_dir / "frames",
            )
            outputs.extend(
                [
                    document_json_path,
                    document_markdown_path,
                    transcript_json_path,
                    scenes_json_path,
                ]
            )
            documents.append(document)
            continue
        try:
            video_path = dependencies.download_video(video, video_dir, settings)
        except (DownloadError, FileNotFoundError) as error:
            skip_payload = {"stage": "download", "message": str(error)}
            write_json(
                skip_json_path,
                skip_payload,
            )
            skipped_videos.append(build_skipped_video_record(video, skip_payload))
            cleanup_video_processing_files(
                find_downloaded_video(video_dir, video.video_id),
                video_dir / "frames",
            )
            outputs.append(skip_json_path)
            continue
        if skip_json_path.exists():
            skip_json_path.unlink()
        transcript_text, transcript_segments = load_or_create_transcript(
            transcript_json_path,
            video_path,
            settings,
            dependencies,
        )
        scene_descriptions = load_or_create_scene_descriptions(
            scenes_json_path,
            video_path,
            video_dir,
            video,
            settings,
            dependencies,
        )
        artifacts = VideoArtifacts(
            video=video,
            video_path=video_path,
            transcript_text=transcript_text,
            transcript_segments=transcript_segments,
            scene_descriptions=scene_descriptions,
        )
        document = build_video_document(artifacts)
        write_json(
            document_json_path,
            document,
        )
        document_markdown_path.write_text(build_video_markdown(artifacts))
        write_json(
            transcript_json_path,
            {
                "transcript_text": transcript_text,
                "transcript_segments": [
                    serialize_transcript_segment(segment)
                    for segment in transcript_segments
                ],
            },
        )
        write_json(
            scenes_json_path,
            [serialize_scene_description(scene) for scene in scene_descriptions],
        )
        outputs.extend(
            [
                document_json_path,
                document_markdown_path,
                transcript_json_path,
                scenes_json_path,
            ]
        )
        documents.append(document)
        cleanup_video_processing_files(video_path, video_dir / "frames")
    videos_jsonl_path = profile_dir / "videos.jsonl"
    videos_jsonl_path.write_text(
        "".join(
            json.dumps(document, separators=(",", ":")) + "\n" for document in documents
        )
    )
    cleaned_documents = [build_clean_video_document(document) for document in documents]
    cleaned_videos_jsonl_path = profile_dir / "videos.clean.jsonl"
    cleaned_videos_jsonl_path.write_text(
        "".join(
            json.dumps(document, separators=(",", ":")) + "\n"
            for document in cleaned_documents
        )
    )
    chunks_jsonl_path = profile_dir / "chunks.jsonl"
    chunks_jsonl_path.write_text(
        "".join(
            json.dumps(chunk, separators=(",", ":")) + "\n"
            for document in cleaned_documents
            for chunk in build_retrieval_chunks(document)
        )
    )
    skipped_jsonl_path = profile_dir / "skipped.json"
    write_json(skipped_jsonl_path, skipped_videos)
    outputs.extend(
        [
            videos_jsonl_path,
            cleaned_videos_jsonl_path,
            chunks_jsonl_path,
            skipped_jsonl_path,
        ]
    )
    return outputs


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")))


def load_or_create_transcript(
    transcript_json_path: Path,
    video_path: Path,
    settings: SyncProfileSettings,
    dependencies: PipelineDependencies,
) -> tuple[str, list[TranscriptSegment]]:
    if transcript_json_path.exists() and not settings.overwrite:
        payload = json.loads(transcript_json_path.read_text())
        if isinstance(payload, dict):
            transcript_text = str(payload.get("transcript_text") or "")
            raw_segments = payload.get("transcript_segments")
            if isinstance(raw_segments, list):
                return transcript_text, [
                    TranscriptSegment(
                        start_seconds=float(segment["start_seconds"]),
                        end_seconds=float(segment["end_seconds"]),
                        text=str(segment["text"]),
                    )
                    for segment in raw_segments
                    if isinstance(segment, dict)
                ]
    return dependencies.transcribe_video(video_path, settings)


def load_or_create_scene_descriptions(
    scenes_json_path: Path,
    video_path: Path,
    video_dir: Path,
    video: ProfileVideo,
    settings: SyncProfileSettings,
    dependencies: PipelineDependencies,
) -> list[SceneDescription]:
    if settings.skip_scene_descriptions:
        return []
    if scenes_json_path.exists() and not settings.overwrite:
        payload = json.loads(scenes_json_path.read_text())
        if isinstance(payload, list):
            return [
                SceneDescription(
                    timestamp_seconds=float(scene["timestamp_seconds"]),
                    summary=str(scene["summary"]),
                    visible_text=[
                        str(value) for value in scene.get("visible_text", [])
                    ],
                )
                for scene in payload
                if isinstance(scene, dict)
            ]
    return dependencies.describe_video(video_path, video_dir, video, settings)


def can_reuse_existing_outputs(
    document_json_path: Path,
    document_markdown_path: Path,
    transcript_json_path: Path,
    scenes_json_path: Path,
    settings: SyncProfileSettings,
) -> bool:
    if settings.overwrite:
        return False
    required_paths = [
        document_json_path,
        document_markdown_path,
        transcript_json_path,
    ]
    if not settings.skip_scene_descriptions:
        required_paths.append(scenes_json_path)
    return all(path.exists() for path in required_paths)


def load_existing_document(document_json_path: Path) -> dict[str, object]:
    payload = json.loads(document_json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Existing document payload is invalid: {document_json_path}")
    return payload


def load_existing_skip(skip_json_path: Path) -> dict[str, object]:
    payload = json.loads(skip_json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Existing skip payload is invalid: {skip_json_path}")
    return payload


def cleanup_video_processing_files(video_path: Path | None, frames_dir: Path) -> None:
    if video_path is not None and video_path.exists():
        video_path.unlink()
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
