from pathlib import Path

from tiktok_transcriber.corpus import (
    build_clean_video_document,
    build_retrieval_chunks,
    build_video_document,
    build_video_markdown,
)
from tiktok_transcriber.models import (
    ProfileVideo,
    SceneDescription,
    TranscriptSegment,
    VideoArtifacts,
)


def build_fixture_artifacts() -> VideoArtifacts:
    return VideoArtifacts(
        video=ProfileVideo(
            video_id="123",
            url="https://www.tiktok.com/@deltatrendtrading/video/123",
            profile_handle="deltatrendtrading",
            title="Convexity example",
            description="Explaining convex payoff structures",
            timestamp=1_700_000_000,
            duration_seconds=90,
            view_count=1000,
        ),
        video_path=Path("data/profile/videos/123/123.mp4"),
        transcript_text="This is the cleaned transcript.",
        transcript_segments=[
            TranscriptSegment(0.0, 10.0, "This is the first segment."),
            TranscriptSegment(10.0, 20.0, "This is the second segment."),
        ],
        scene_descriptions=[
            SceneDescription(
                timestamp_seconds=5.0,
                summary="Candlestick chart with two highlighted levels.",
                visible_text=["NQ futures", "R:R 3.5"],
            ),
            SceneDescription(
                timestamp_seconds=15.0,
                summary="Spreadsheet view showing a Monte Carlo style payoff table.",
                visible_text=["pass rate 18%", "EV 142"],
            ),
        ],
    )


def test_build_video_document_includes_transcript_and_scene_context() -> None:
    artifacts = build_fixture_artifacts()

    assert build_video_document(artifacts) == {
        "id": "deltatrendtrading:123",
        "profile_handle": "deltatrendtrading",
        "video_id": "123",
        "url": "https://www.tiktok.com/@deltatrendtrading/video/123",
        "title": "Convexity example",
        "description": "Explaining convex payoff structures",
        "timestamp": 1_700_000_000,
        "duration_seconds": 90,
        "view_count": 1000,
        "transcript_text": "This is the cleaned transcript.",
        "transcript_segments": [
            {
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "text": "This is the first segment.",
            },
            {
                "start_seconds": 10.0,
                "end_seconds": 20.0,
                "text": "This is the second segment.",
            },
        ],
        "scene_descriptions": [
            {
                "timestamp_seconds": 5.0,
                "summary": "Candlestick chart with two highlighted levels.",
                "visible_text": ["NQ futures", "R:R 3.5"],
            },
            {
                "timestamp_seconds": 15.0,
                "summary": "Spreadsheet view showing a Monte Carlo style payoff table.",
                "visible_text": ["pass rate 18%", "EV 142"],
            },
        ],
        "markdown_path": "data/profile/videos/123/document.md",
        "video_path": "data/profile/videos/123/123.mp4",
    }


def test_build_video_markdown_formats_sections_for_skill_ingestion() -> None:
    artifacts = build_fixture_artifacts()

    assert (
        build_video_markdown(artifacts)
        == """# Convexity example

- Profile: @deltatrendtrading
- Video ID: 123
- URL: https://www.tiktok.com/@deltatrendtrading/video/123
- TikTok description: Explaining convex payoff structures

## Transcript

This is the cleaned transcript.

## Timestamped Transcript

- 0.0-10.0s: This is the first segment.
- 10.0-20.0s: This is the second segment.

## Scene Context

- 5.0s: Candlestick chart with two highlighted levels. Visible text: NQ futures; R:R 3.5
- 15.0s: Spreadsheet view showing a Monte Carlo style payoff table. Visible text: pass rate 18%; EV 142
"""
    )


def test_build_clean_video_document_removes_deleted_video_path() -> None:
    document = build_video_document(build_fixture_artifacts())

    assert build_clean_video_document(document) == {
        "id": "deltatrendtrading:123",
        "profile_handle": "deltatrendtrading",
        "video_id": "123",
        "url": "https://www.tiktok.com/@deltatrendtrading/video/123",
        "title": "Convexity example",
        "description": "Explaining convex payoff structures",
        "timestamp": 1_700_000_000,
        "duration_seconds": 90,
        "view_count": 1000,
        "transcript_text": "This is the cleaned transcript.",
        "transcript_segments": [
            {
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "text": "This is the first segment.",
            },
            {
                "start_seconds": 10.0,
                "end_seconds": 20.0,
                "text": "This is the second segment.",
            },
        ],
        "scene_descriptions": [
            {
                "timestamp_seconds": 5.0,
                "summary": "Candlestick chart with two highlighted levels.",
                "visible_text": ["NQ futures", "R:R 3.5"],
            },
            {
                "timestamp_seconds": 15.0,
                "summary": "Spreadsheet view showing a Monte Carlo style payoff table.",
                "visible_text": ["pass rate 18%", "EV 142"],
            },
        ],
        "markdown_path": "data/profile/videos/123/document.md",
    }


def test_build_retrieval_chunks_splits_transcript_and_scene_context() -> None:
    document = build_video_document(build_fixture_artifacts())

    assert build_retrieval_chunks(
        document,
        transcript_chunk_words=4,
        transcript_chunk_overlap_words=1,
    ) == [
        {
            "id": "deltatrendtrading:123:transcript:0",
            "profile_handle": "deltatrendtrading",
            "video_id": "123",
            "url": "https://www.tiktok.com/@deltatrendtrading/video/123",
            "title": "Convexity example",
            "description": "Explaining convex payoff structures",
            "chunk_type": "transcript",
            "chunk_index": 0,
            "text": (
                "Title: Convexity example\n"
                "Description: Explaining convex payoff structures\n"
                "Transcript: This is the cleaned"
            ),
        },
        {
            "id": "deltatrendtrading:123:transcript:1",
            "profile_handle": "deltatrendtrading",
            "video_id": "123",
            "url": "https://www.tiktok.com/@deltatrendtrading/video/123",
            "title": "Convexity example",
            "description": "Explaining convex payoff structures",
            "chunk_type": "transcript",
            "chunk_index": 1,
            "text": (
                "Title: Convexity example\n"
                "Description: Explaining convex payoff structures\n"
                "Transcript: cleaned transcript."
            ),
        },
        {
            "id": "deltatrendtrading:123:scene:0",
            "profile_handle": "deltatrendtrading",
            "video_id": "123",
            "url": "https://www.tiktok.com/@deltatrendtrading/video/123",
            "title": "Convexity example",
            "description": "Explaining convex payoff structures",
            "chunk_type": "scene",
            "chunk_index": 0,
            "text": (
                "Title: Convexity example\n"
                "Description: Explaining convex payoff structures\n"
                "Scene at 5.0s: Candlestick chart with two highlighted levels.\n"
                "Visible text: NQ futures; R:R 3.5"
            ),
        },
        {
            "id": "deltatrendtrading:123:scene:1",
            "profile_handle": "deltatrendtrading",
            "video_id": "123",
            "url": "https://www.tiktok.com/@deltatrendtrading/video/123",
            "title": "Convexity example",
            "description": "Explaining convex payoff structures",
            "chunk_type": "scene",
            "chunk_index": 1,
            "text": (
                "Title: Convexity example\n"
                "Description: Explaining convex payoff structures\n"
                "Scene at 15.0s: Spreadsheet view showing a Monte Carlo style payoff table.\n"
                "Visible text: pass rate 18%; EV 142"
            ),
        },
    ]
