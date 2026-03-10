from pathlib import Path

from yt_dlp.utils import DownloadError

from tiktok_transcriber.models import (
    CookieSettings,
    ProfileVideo,
    SceneDescription,
    SyncProfileSettings,
    TranscriptSegment,
)
from tiktok_transcriber.pipeline import PipelineDependencies, sync_profile


def build_settings(tmp_path: Path) -> SyncProfileSettings:
    return SyncProfileSettings(
        profile_url="https://www.tiktok.com/@deltatrendtrading",
        profile_handle="deltatrendtrading",
        output_dir=tmp_path,
        cookie_settings=CookieSettings(),
        gemini_api_key="test-key",
        gemini_model="gemini-2.5-flash",
        transcription_provider="openai",
        openai_api_key="openai-test-key",
        openai_transcription_model="gpt-4o-transcribe",
        whisper_model="small",
        frame_samples=3,
        limit=1,
        skip_scene_descriptions=False,
        transcription_language="en",
        overwrite=True,
    )


def test_sync_profile_writes_per_video_outputs_and_jsonl_corpus(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    discovered_video = ProfileVideo(
        video_id="123",
        url="https://www.tiktok.com/@deltatrendtrading/video/123",
        profile_handle="deltatrendtrading",
        title="Convexity example",
        description="Explaining convex payoff structures",
        timestamp=1_700_000_000,
        duration_seconds=90,
        view_count=1000,
    )

    def discover_profile_videos(_: SyncProfileSettings) -> list[ProfileVideo]:
        return [discovered_video]

    def download_video(
        video: ProfileVideo, video_dir: Path, _: SyncProfileSettings
    ) -> Path:
        video_path = video_dir / f"{video.video_id}.mp4"
        video_path.write_bytes(b"video-bytes")
        return video_path

    def transcribe_video(
        _: Path, __: SyncProfileSettings
    ) -> tuple[str, list[TranscriptSegment]]:
        return (
            "This is the cleaned transcript.",
            [TranscriptSegment(0.0, 10.0, "This is the first segment.")],
        )

    def describe_video(
        _: Path, __: Path, ___: ProfileVideo, ____: SyncProfileSettings
    ) -> list[SceneDescription]:
        return [
            SceneDescription(
                timestamp_seconds=5.0,
                summary="Chart with a highlighted breakout level.",
                visible_text=["breakout", "stop 10pts"],
            )
        ]

    outputs = sync_profile(
        settings,
        PipelineDependencies(
            discover_profile_videos=discover_profile_videos,
            download_video=download_video,
            transcribe_video=transcribe_video,
            describe_video=describe_video,
        ),
    )

    assert outputs == [
        tmp_path / "deltatrendtrading" / "videos" / "123" / "document.json",
        tmp_path / "deltatrendtrading" / "videos" / "123" / "document.md",
        tmp_path / "deltatrendtrading" / "videos" / "123" / "transcript.json",
        tmp_path / "deltatrendtrading" / "videos" / "123" / "scenes.json",
        tmp_path / "deltatrendtrading" / "videos.jsonl",
        tmp_path / "deltatrendtrading" / "videos.clean.jsonl",
        tmp_path / "deltatrendtrading" / "chunks.jsonl",
        tmp_path / "deltatrendtrading" / "skipped.json",
    ]

    assert (tmp_path / "deltatrendtrading" / "videos.jsonl").read_text() == (
        '{"id":"deltatrendtrading:123","profile_handle":"deltatrendtrading",'
        '"video_id":"123","url":"https://www.tiktok.com/@deltatrendtrading/video/123",'
        '"title":"Convexity example","description":"Explaining convex payoff structures",'
        '"timestamp":1700000000,"duration_seconds":90,"view_count":1000,'
        '"transcript_text":"This is the cleaned transcript.","transcript_segments":'
        '[{"start_seconds":0.0,"end_seconds":10.0,"text":"This is the first segment."}],'
        '"scene_descriptions":[{"timestamp_seconds":5.0,'
        '"summary":"Chart with a highlighted breakout level.",'
        '"visible_text":["breakout","stop 10pts"]}],'
        '"markdown_path":"'
        + str(tmp_path / "deltatrendtrading" / "videos" / "123" / "document.md")
        + '","video_path":"'
        + str(tmp_path / "deltatrendtrading" / "videos" / "123" / "123.mp4")
        + '"}\n'
    )
    assert (tmp_path / "deltatrendtrading" / "videos.clean.jsonl").read_text() == (
        '{"id":"deltatrendtrading:123","profile_handle":"deltatrendtrading",'
        '"video_id":"123","url":"https://www.tiktok.com/@deltatrendtrading/video/123",'
        '"title":"Convexity example","description":"Explaining convex payoff structures",'
        '"timestamp":1700000000,"duration_seconds":90,"view_count":1000,'
        '"transcript_text":"This is the cleaned transcript.","transcript_segments":'
        '[{"start_seconds":0.0,"end_seconds":10.0,"text":"This is the first segment."}],'
        '"scene_descriptions":[{"timestamp_seconds":5.0,'
        '"summary":"Chart with a highlighted breakout level.",'
        '"visible_text":["breakout","stop 10pts"]}],'
        '"markdown_path":"'
        + str(tmp_path / "deltatrendtrading" / "videos" / "123" / "document.md")
        + '"}\n'
    )
    assert (tmp_path / "deltatrendtrading" / "chunks.jsonl").read_text() == (
        '{"id":"deltatrendtrading:123:transcript:0","profile_handle":"deltatrendtrading",'
        '"video_id":"123","url":"https://www.tiktok.com/@deltatrendtrading/video/123",'
        '"title":"Convexity example","description":"Explaining convex payoff structures",'
        '"chunk_type":"transcript","chunk_index":0,"text":"Title: Convexity example\\nDescription: Explaining convex payoff structures\\nTranscript: This is the cleaned transcript."}\n'
        '{"id":"deltatrendtrading:123:scene:0","profile_handle":"deltatrendtrading",'
        '"video_id":"123","url":"https://www.tiktok.com/@deltatrendtrading/video/123",'
        '"title":"Convexity example","description":"Explaining convex payoff structures",'
        '"chunk_type":"scene","chunk_index":0,"text":"Title: Convexity example\\nDescription: Explaining convex payoff structures\\nScene at 5.0s: Chart with a highlighted breakout level.\\nVisible text: breakout; stop 10pts"}\n'
    )
    assert (tmp_path / "deltatrendtrading" / "skipped.json").read_text() == "[]"

    assert not (tmp_path / "deltatrendtrading" / "videos" / "123" / "123.mp4").exists()
    assert not (tmp_path / "deltatrendtrading" / "videos" / "123" / "frames").exists()


def test_sync_profile_reuses_existing_artifacts_without_redownloading(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path)
    settings = SyncProfileSettings(
        profile_url=settings.profile_url,
        profile_handle=settings.profile_handle,
        output_dir=settings.output_dir,
        cookie_settings=settings.cookie_settings,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        transcription_provider=settings.transcription_provider,
        openai_api_key=settings.openai_api_key,
        openai_transcription_model=settings.openai_transcription_model,
        whisper_model=settings.whisper_model,
        frame_samples=settings.frame_samples,
        limit=settings.limit,
        skip_scene_descriptions=settings.skip_scene_descriptions,
        transcription_language=settings.transcription_language,
        overwrite=False,
    )
    profile_dir = tmp_path / "deltatrendtrading"
    video_dir = profile_dir / "videos" / "123"
    video_dir.mkdir(parents=True)
    transcript_json_path = video_dir / "transcript.json"
    scenes_json_path = video_dir / "scenes.json"
    document_json_path = video_dir / "document.json"
    document_markdown_path = video_dir / "document.md"
    transcript_json_path.write_text(
        '{"transcript_text":"cached transcript","transcript_segments":[]}'
    )
    scenes_json_path.write_text(
        '[{"timestamp_seconds":5.0,"summary":"cached scene","visible_text":["level"]}]'
    )
    document_json_path.write_text(
        '{"id":"deltatrendtrading:123","profile_handle":"deltatrendtrading","video_id":"123","url":"https://www.tiktok.com/@deltatrendtrading/video/123","title":"Cached title","description":"Cached description","timestamp":1700000000,"duration_seconds":90,"view_count":1000,"transcript_text":"cached transcript","transcript_segments":[],"scene_descriptions":[{"timestamp_seconds":5.0,"summary":"cached scene","visible_text":["level"]}],"markdown_path":"'
        + str(document_markdown_path)
        + '","video_path":"'
        + str(video_dir / "123.mp4")
        + '"}'
    )
    document_markdown_path.write_text("# Cached title\n")
    discovered_video = ProfileVideo(
        video_id="123",
        url="https://www.tiktok.com/@deltatrendtrading/video/123",
        profile_handle="deltatrendtrading",
        title="Should not be used",
        description="Should not be used",
        timestamp=1_700_000_000,
        duration_seconds=90,
        view_count=1000,
    )
    calls: list[str] = []

    def discover_profile_videos(_: SyncProfileSettings) -> list[ProfileVideo]:
        return [discovered_video]

    def download_video(
        video: ProfileVideo, video_dir: Path, _: SyncProfileSettings
    ) -> Path:
        calls.append("download")
        return video_dir / f"{video.video_id}.mp4"

    def transcribe_video(
        _: Path, __: SyncProfileSettings
    ) -> tuple[str, list[TranscriptSegment]]:
        calls.append("transcribe")
        return ("", [])

    def describe_video(
        _: Path, __: Path, ___: ProfileVideo, ____: SyncProfileSettings
    ) -> list[SceneDescription]:
        calls.append("describe")
        return []

    outputs = sync_profile(
        settings,
        PipelineDependencies(
            discover_profile_videos=discover_profile_videos,
            download_video=download_video,
            transcribe_video=transcribe_video,
            describe_video=describe_video,
        ),
    )

    assert calls == []
    assert outputs == [
        document_json_path,
        document_markdown_path,
        transcript_json_path,
        scenes_json_path,
        profile_dir / "videos.jsonl",
        profile_dir / "videos.clean.jsonl",
        profile_dir / "chunks.jsonl",
        profile_dir / "skipped.json",
    ]
    assert (
        profile_dir / "videos.jsonl"
    ).read_text() == document_json_path.read_text() + "\n"
    assert (profile_dir / "videos.clean.jsonl").read_text() == (
        '{"id":"deltatrendtrading:123","profile_handle":"deltatrendtrading","video_id":"123","url":"https://www.tiktok.com/@deltatrendtrading/video/123","title":"Cached title","description":"Cached description","timestamp":1700000000,"duration_seconds":90,"view_count":1000,"transcript_text":"cached transcript","transcript_segments":[],"scene_descriptions":[{"timestamp_seconds":5.0,"summary":"cached scene","visible_text":["level"]}],"markdown_path":"'
        + str(document_markdown_path)
        + '"}\n'
    )
    assert (profile_dir / "chunks.jsonl").read_text() == (
        '{"id":"deltatrendtrading:123:transcript:0","profile_handle":"deltatrendtrading","video_id":"123","url":"https://www.tiktok.com/@deltatrendtrading/video/123","title":"Cached title","description":"Cached description","chunk_type":"transcript","chunk_index":0,"text":"Title: Cached title\\nDescription: Cached description\\nTranscript: cached transcript"}\n'
        '{"id":"deltatrendtrading:123:scene:0","profile_handle":"deltatrendtrading","video_id":"123","url":"https://www.tiktok.com/@deltatrendtrading/video/123","title":"Cached title","description":"Cached description","chunk_type":"scene","chunk_index":0,"text":"Title: Cached title\\nDescription: Cached description\\nScene at 5.0s: cached scene\\nVisible text: level"}\n'
    )
    assert (profile_dir / "skipped.json").read_text() == "[]"


def test_sync_profile_skips_missing_downloads_and_continues(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path)
    settings = SyncProfileSettings(
        profile_url=settings.profile_url,
        profile_handle=settings.profile_handle,
        output_dir=settings.output_dir,
        cookie_settings=settings.cookie_settings,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        transcription_provider=settings.transcription_provider,
        openai_api_key=settings.openai_api_key,
        openai_transcription_model=settings.openai_transcription_model,
        whisper_model=settings.whisper_model,
        frame_samples=settings.frame_samples,
        limit=2,
        skip_scene_descriptions=settings.skip_scene_descriptions,
        transcription_language=settings.transcription_language,
        overwrite=False,
    )
    missing_video = ProfileVideo(
        video_id="missing",
        url="https://www.tiktok.com/@deltatrendtrading/video/missing",
        profile_handle="deltatrendtrading",
        title="Missing video",
        description="This download should fail",
        timestamp=1_700_000_001,
        duration_seconds=45,
        view_count=100,
    )
    available_video = ProfileVideo(
        video_id="available",
        url="https://www.tiktok.com/@deltatrendtrading/video/available",
        profile_handle="deltatrendtrading",
        title="Available video",
        description="This download should succeed",
        timestamp=1_700_000_002,
        duration_seconds=60,
        view_count=200,
    )
    calls: list[str] = []

    def discover_profile_videos(_: SyncProfileSettings) -> list[ProfileVideo]:
        return [missing_video, available_video]

    def download_video(
        video: ProfileVideo, video_dir: Path, _: SyncProfileSettings
    ) -> Path:
        calls.append(f"download:{video.video_id}")
        if video.video_id == missing_video.video_id:
            raise FileNotFoundError(
                f"yt-dlp did not produce a video for {missing_video.url}"
            )
        video_path = video_dir / f"{video.video_id}.mp4"
        video_path.write_bytes(b"video-bytes")
        return video_path

    def transcribe_video(
        video_path: Path, _: SyncProfileSettings
    ) -> tuple[str, list[TranscriptSegment]]:
        calls.append(f"transcribe:{video_path.stem}")
        return ("Available transcript", [])

    def describe_video(
        video_path: Path, _: Path, __: ProfileVideo, ___: SyncProfileSettings
    ) -> list[SceneDescription]:
        calls.append(f"describe:{video_path.stem}")
        return []

    outputs = sync_profile(
        settings,
        PipelineDependencies(
            discover_profile_videos=discover_profile_videos,
            download_video=download_video,
            transcribe_video=transcribe_video,
            describe_video=describe_video,
        ),
    )

    skipped_json_path = (
        tmp_path / "deltatrendtrading" / "videos" / "missing" / "skip.json"
    )
    assert skipped_json_path.read_text() == (
        '{"stage":"download","message":"yt-dlp did not produce a video for '
        'https://www.tiktok.com/@deltatrendtrading/video/missing"}'
    )
    assert calls == [
        "download:missing",
        "download:available",
        "transcribe:available",
        "describe:available",
    ]
    assert outputs == [
        skipped_json_path,
        tmp_path / "deltatrendtrading" / "videos" / "available" / "document.json",
        tmp_path / "deltatrendtrading" / "videos" / "available" / "document.md",
        tmp_path / "deltatrendtrading" / "videos" / "available" / "transcript.json",
        tmp_path / "deltatrendtrading" / "videos" / "available" / "scenes.json",
        tmp_path / "deltatrendtrading" / "videos.jsonl",
        tmp_path / "deltatrendtrading" / "videos.clean.jsonl",
        tmp_path / "deltatrendtrading" / "chunks.jsonl",
        tmp_path / "deltatrendtrading" / "skipped.json",
    ]
    assert (tmp_path / "deltatrendtrading" / "videos.jsonl").read_text() == (
        '{"id":"deltatrendtrading:available","profile_handle":"deltatrendtrading",'
        '"video_id":"available","url":"https://www.tiktok.com/@deltatrendtrading/video/available",'
        '"title":"Available video","description":"This download should succeed",'
        '"timestamp":1700000002,"duration_seconds":60,"view_count":200,'
        '"transcript_text":"Available transcript","transcript_segments":[],'
        '"scene_descriptions":[],"markdown_path":"'
        + str(tmp_path / "deltatrendtrading" / "videos" / "available" / "document.md")
        + '","video_path":"'
        + str(tmp_path / "deltatrendtrading" / "videos" / "available" / "available.mp4")
        + '"}\n'
    )
    assert (tmp_path / "deltatrendtrading" / "videos.clean.jsonl").read_text() == (
        '{"id":"deltatrendtrading:available","profile_handle":"deltatrendtrading",'
        '"video_id":"available","url":"https://www.tiktok.com/@deltatrendtrading/video/available",'
        '"title":"Available video","description":"This download should succeed",'
        '"timestamp":1700000002,"duration_seconds":60,"view_count":200,'
        '"transcript_text":"Available transcript","transcript_segments":[],'
        '"scene_descriptions":[],"markdown_path":"'
        + str(tmp_path / "deltatrendtrading" / "videos" / "available" / "document.md")
        + '"}\n'
    )
    assert (tmp_path / "deltatrendtrading" / "chunks.jsonl").read_text() == (
        '{"id":"deltatrendtrading:available:transcript:0","profile_handle":"deltatrendtrading",'
        '"video_id":"available","url":"https://www.tiktok.com/@deltatrendtrading/video/available",'
        '"title":"Available video","description":"This download should succeed",'
        '"chunk_type":"transcript","chunk_index":0,"text":"Title: Available video\\nDescription: This download should succeed\\nTranscript: Available transcript"}\n'
    )
    assert (tmp_path / "deltatrendtrading" / "skipped.json").read_text() == (
        '[{"id":"deltatrendtrading:missing","profile_handle":"deltatrendtrading","video_id":"missing","url":"https://www.tiktok.com/@deltatrendtrading/video/missing","title":"Missing video","description":"This download should fail","timestamp":1700000001,"duration_seconds":45,"view_count":100,"stage":"download","message":"yt-dlp did not produce a video for https://www.tiktok.com/@deltatrendtrading/video/missing"}]'
    )


def test_sync_profile_skips_download_errors_and_continues(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings = SyncProfileSettings(
        profile_url=settings.profile_url,
        profile_handle=settings.profile_handle,
        output_dir=settings.output_dir,
        cookie_settings=settings.cookie_settings,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        transcription_provider=settings.transcription_provider,
        openai_api_key=settings.openai_api_key,
        openai_transcription_model=settings.openai_transcription_model,
        whisper_model=settings.whisper_model,
        frame_samples=settings.frame_samples,
        limit=2,
        skip_scene_descriptions=settings.skip_scene_descriptions,
        transcription_language=settings.transcription_language,
        overwrite=False,
    )
    blocked_video = ProfileVideo(
        video_id="blocked",
        url="https://www.tiktok.com/@deltatrendtrading/video/blocked",
        profile_handle="deltatrendtrading",
        title="Blocked video",
        description="This download should raise DownloadError",
        timestamp=1_700_000_003,
        duration_seconds=50,
        view_count=300,
    )
    available_video = ProfileVideo(
        video_id="available",
        url="https://www.tiktok.com/@deltatrendtrading/video/available",
        profile_handle="deltatrendtrading",
        title="Available video",
        description="This download should succeed",
        timestamp=1_700_000_004,
        duration_seconds=65,
        view_count=400,
    )
    calls: list[str] = []

    def discover_profile_videos(_: SyncProfileSettings) -> list[ProfileVideo]:
        return [blocked_video, available_video]

    def download_video(
        video: ProfileVideo, video_dir: Path, _: SyncProfileSettings
    ) -> Path:
        calls.append(f"download:{video.video_id}")
        if video.video_id == blocked_video.video_id:
            raise DownloadError(
                "ERROR: [TikTok] blocked: Your IP address is blocked from accessing this post"
            )
        video_path = video_dir / f"{video.video_id}.mp4"
        video_path.write_bytes(b"video-bytes")
        return video_path

    def transcribe_video(
        video_path: Path, _: SyncProfileSettings
    ) -> tuple[str, list[TranscriptSegment]]:
        calls.append(f"transcribe:{video_path.stem}")
        return ("Available transcript", [])

    def describe_video(
        video_path: Path, _: Path, __: ProfileVideo, ___: SyncProfileSettings
    ) -> list[SceneDescription]:
        calls.append(f"describe:{video_path.stem}")
        return []

    outputs = sync_profile(
        settings,
        PipelineDependencies(
            discover_profile_videos=discover_profile_videos,
            download_video=download_video,
            transcribe_video=transcribe_video,
            describe_video=describe_video,
        ),
    )

    skipped_json_path = (
        tmp_path / "deltatrendtrading" / "videos" / "blocked" / "skip.json"
    )
    assert skipped_json_path.read_text() == (
        '{"stage":"download","message":"ERROR: [TikTok] blocked: Your IP address is '
        'blocked from accessing this post"}'
    )
    assert calls == [
        "download:blocked",
        "download:available",
        "transcribe:available",
        "describe:available",
    ]
    assert outputs == [
        skipped_json_path,
        tmp_path / "deltatrendtrading" / "videos" / "available" / "document.json",
        tmp_path / "deltatrendtrading" / "videos" / "available" / "document.md",
        tmp_path / "deltatrendtrading" / "videos" / "available" / "transcript.json",
        tmp_path / "deltatrendtrading" / "videos" / "available" / "scenes.json",
        tmp_path / "deltatrendtrading" / "videos.jsonl",
        tmp_path / "deltatrendtrading" / "videos.clean.jsonl",
        tmp_path / "deltatrendtrading" / "chunks.jsonl",
        tmp_path / "deltatrendtrading" / "skipped.json",
    ]
    assert (tmp_path / "deltatrendtrading" / "skipped.json").read_text() == (
        '[{"id":"deltatrendtrading:blocked","profile_handle":"deltatrendtrading","video_id":"blocked","url":"https://www.tiktok.com/@deltatrendtrading/video/blocked","title":"Blocked video","description":"This download should raise DownloadError","timestamp":1700000003,"duration_seconds":50,"view_count":300,"stage":"download","message":"ERROR: [TikTok] blocked: Your IP address is blocked from accessing this post"}]'
    )
