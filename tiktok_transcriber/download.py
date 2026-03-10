import json
from pathlib import Path

from yt_dlp import YoutubeDL

from tiktok_transcriber.models import ProfileVideo
from tiktok_transcriber.models import SyncProfileSettings


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def download_video(
    video: ProfileVideo, video_dir: Path, settings: SyncProfileSettings
) -> Path:
    existing_video = find_downloaded_video(video_dir, video.video_id)
    if existing_video is not None and not settings.overwrite:
        return existing_video
    options: dict[str, object] = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "no_warnings": True,
        "outtmpl": str(video_dir / f"{video.video_id}.%(ext)s"),
        "quiet": True,
    }
    if settings.cookie_settings.file_path is not None:
        options["cookiefile"] = str(settings.cookie_settings.file_path)
    if settings.cookie_settings.browser is not None:
        options["cookiesfrombrowser"] = (
            settings.cookie_settings.browser,
            None,
            None,
            None,
        )
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(video.url, download=True)
    metadata_path = video_dir / "download-metadata.json"
    metadata_path.write_text(
        json.dumps(downloader.sanitize_info(info), separators=(",", ":"))
    )
    downloaded_video = find_downloaded_video(video_dir, video.video_id)
    if downloaded_video is None:
        raise FileNotFoundError(f"yt-dlp did not produce a video for {video.url}")
    return downloaded_video


def find_downloaded_video(video_dir: Path, video_id: str) -> Path | None:
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    return None
