from pathlib import Path

from tiktok_transcriber.models import ProfileVideo
from tiktok_transcriber.models import SceneDescription
from tiktok_transcriber.models import TranscriptSegment
from tiktok_transcriber.models import VideoArtifacts


def build_video_markdown(artifacts: VideoArtifacts) -> str:
    transcript_lines = "\n".join(
        format_transcript_segment(segment) for segment in artifacts.transcript_segments
    )
    scene_lines = "\n".join(
        format_scene_description(scene) for scene in artifacts.scene_descriptions
    )
    return (
        f"# {artifacts.video.title}\n\n"
        f"- Profile: @{artifacts.video.profile_handle}\n"
        f"- Video ID: {artifacts.video.video_id}\n"
        f"- URL: {artifacts.video.url}\n"
        f"- TikTok description: {artifacts.video.description}\n\n"
        "## Transcript\n\n"
        f"{artifacts.transcript_text}\n\n"
        "## Timestamped Transcript\n\n"
        f"{transcript_lines}\n\n"
        "## Scene Context\n\n"
        f"{scene_lines}\n"
    )


def build_video_document(artifacts: VideoArtifacts) -> dict[str, object]:
    markdown_path = (
        str(Path(artifacts.video_path).with_name("document.md"))
        if artifacts.video_path is not None
        else None
    )
    return {
        "id": f"{artifacts.video.profile_handle}:{artifacts.video.video_id}",
        "profile_handle": artifacts.video.profile_handle,
        "video_id": artifacts.video.video_id,
        "url": artifacts.video.url,
        "title": artifacts.video.title,
        "description": artifacts.video.description,
        "timestamp": artifacts.video.timestamp,
        "duration_seconds": artifacts.video.duration_seconds,
        "view_count": artifacts.video.view_count,
        "transcript_text": artifacts.transcript_text,
        "transcript_segments": [
            serialize_transcript_segment(segment)
            for segment in artifacts.transcript_segments
        ],
        "scene_descriptions": [
            serialize_scene_description(scene) for scene in artifacts.scene_descriptions
        ],
        "markdown_path": markdown_path,
        "video_path": (
            str(artifacts.video_path) if artifacts.video_path is not None else None
        ),
    }


def build_clean_video_document(document: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in document.items() if key != "video_path"}


def build_retrieval_chunks(
    document: dict[str, object],
    transcript_chunk_words: int = 220,
    transcript_chunk_overlap_words: int = 40,
) -> list[dict[str, object]]:
    transcript_chunks = build_transcript_chunks(
        document,
        transcript_chunk_words,
        transcript_chunk_overlap_words,
    )
    scene_chunks = build_scene_chunks(document)
    return transcript_chunks + scene_chunks


def build_skipped_video_record(
    video: ProfileVideo, skip_payload: dict[str, object]
) -> dict[str, object]:
    return {
        "id": f"{video.profile_handle}:{video.video_id}",
        "profile_handle": video.profile_handle,
        "video_id": video.video_id,
        "url": video.url,
        "title": video.title,
        "description": video.description,
        "timestamp": video.timestamp,
        "duration_seconds": video.duration_seconds,
        "view_count": video.view_count,
        "stage": str(skip_payload.get("stage") or ""),
        "message": str(skip_payload.get("message") or ""),
    }


def serialize_transcript_segment(segment: TranscriptSegment) -> dict[str, object]:
    return {
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "text": segment.text,
    }


def serialize_scene_description(scene: SceneDescription) -> dict[str, object]:
    return {
        "timestamp_seconds": scene.timestamp_seconds,
        "summary": scene.summary,
        "visible_text": scene.visible_text,
    }


def format_transcript_segment(segment: TranscriptSegment) -> str:
    return f"- {segment.start_seconds:.1f}-{segment.end_seconds:.1f}s: {segment.text}"


def format_scene_description(scene: SceneDescription) -> str:
    visible_text = "; ".join(scene.visible_text)
    if visible_text:
        return (
            f"- {scene.timestamp_seconds:.1f}s: {scene.summary} "
            f"Visible text: {visible_text}"
        )
    return f"- {scene.timestamp_seconds:.1f}s: {scene.summary}"


def build_transcript_chunks(
    document: dict[str, object],
    transcript_chunk_words: int,
    transcript_chunk_overlap_words: int,
) -> list[dict[str, object]]:
    transcript_text = str(document.get("transcript_text") or "").strip()
    if not transcript_text:
        return []
    words = transcript_text.split()
    chunks: list[dict[str, object]] = []
    for chunk_index, chunk_words in enumerate(
        chunk_words_iter(words, transcript_chunk_words, transcript_chunk_overlap_words)
    ):
        chunks.append(
            build_chunk_record(
                document,
                chunk_type="transcript",
                chunk_index=chunk_index,
                text=(
                    f"Title: {document.get('title')}\n"
                    f"Description: {document.get('description')}\n"
                    f"Transcript: {' '.join(chunk_words)}"
                ),
            )
        )
    return chunks


def build_scene_chunks(document: dict[str, object]) -> list[dict[str, object]]:
    scenes = document.get("scene_descriptions")
    if not isinstance(scenes, list):
        return []
    chunks: list[dict[str, object]] = []
    for chunk_index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue
        summary = str(scene.get("summary") or "").strip()
        if not summary:
            continue
        timestamp_seconds = float(scene.get("timestamp_seconds") or 0.0)
        visible_text = scene.get("visible_text")
        visible_text_items = (
            [str(item) for item in visible_text]
            if isinstance(visible_text, list)
            else []
        )
        text = (
            f"Title: {document.get('title')}\n"
            f"Description: {document.get('description')}\n"
            f"Scene at {timestamp_seconds:.1f}s: {summary}"
        )
        if visible_text_items:
            text += "\n" + f"Visible text: {'; '.join(visible_text_items)}"
        chunks.append(
            build_chunk_record(
                document,
                chunk_type="scene",
                chunk_index=chunk_index,
                text=text,
            )
        )
    return chunks


def chunk_words_iter(
    words: list[str], chunk_words: int, chunk_overlap_words: int
) -> list[list[str]]:
    if not words:
        return []
    normalized_chunk_words = max(chunk_words, 1)
    normalized_overlap_words = max(
        min(chunk_overlap_words, normalized_chunk_words - 1), 0
    )
    step = max(normalized_chunk_words - normalized_overlap_words, 1)
    chunks: list[list[str]] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + normalized_chunk_words]
        if not chunk:
            continue
        chunks.append(chunk)
        if start + normalized_chunk_words >= len(words):
            break
    return chunks


def build_chunk_record(
    document: dict[str, object],
    chunk_type: str,
    chunk_index: int,
    text: str,
) -> dict[str, object]:
    video_id = str(document.get("video_id") or "")
    profile_handle = str(document.get("profile_handle") or "")
    return {
        "id": f"{profile_handle}:{video_id}:{chunk_type}:{chunk_index}",
        "profile_handle": profile_handle,
        "video_id": video_id,
        "url": str(document.get("url") or ""),
        "title": str(document.get("title") or ""),
        "description": str(document.get("description") or ""),
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
        "text": text,
    }
