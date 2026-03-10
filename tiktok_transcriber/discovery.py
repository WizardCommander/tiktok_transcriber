from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from tiktok_transcriber.models import ProfileVideo
from tiktok_transcriber.models import SyncProfileSettings


def discover_profile_videos(settings: SyncProfileSettings) -> list[ProfileVideo]:
    from yt_dlp import YoutubeDL

    options: dict[str, object] = {
        "extract_flat": True,
        "lazy_playlist": False,
        "no_warnings": True,
        "playlistend": settings.limit,
        "quiet": True,
        "skip_download": True,
    }
    options.update(build_cookie_options(settings))
    with YoutubeDL(options) as downloader:
        payload = downloader.extract_info(settings.profile_url, download=False)
    if not isinstance(payload, dict):
        raise ValueError("yt-dlp did not return playlist metadata for the profile URL")
    videos = parse_profile_playlist(payload)
    if settings.limit is None:
        return videos
    return videos[: settings.limit]


def build_cookie_options(settings: SyncProfileSettings) -> dict[str, object]:
    if settings.cookie_settings.file_path is not None:
        return {"cookiefile": str(settings.cookie_settings.file_path)}
    if settings.cookie_settings.browser is not None:
        return {
            "cookiesfrombrowser": (settings.cookie_settings.browser, None, None, None)
        }
    return {}


def normalize_profile_handle(profile_url: str) -> str:
    path = urlparse(profile_url).path.rstrip("/")
    path_parts = [part for part in path.split("/") if part]
    if not path_parts or not path_parts[0].startswith("@"):
        raise ValueError(f"Could not determine TikTok handle from {profile_url!r}")
    return path_parts[0].removeprefix("@")


def parse_profile_playlist(payload: dict[str, Any]) -> list[ProfileVideo]:
    entries = payload.get("entries")
    if not isinstance(entries, Iterable):
        raise ValueError("Playlist payload is missing entries")
    profile_handle = normalize_profile_handle(str(payload.get("webpage_url", "")))
    videos: list[ProfileVideo] = []
    seen_video_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("id", "")).strip()
        if not video_id or video_id in seen_video_ids:
            continue
        url = str(entry.get("url", "")).strip()
        if not url:
            continue
        videos.append(
            ProfileVideo(
                video_id=video_id,
                url=url,
                profile_handle=str(entry.get("uploader") or profile_handle),
                title=str(entry.get("title") or ""),
                description=str(entry.get("description") or ""),
                timestamp=(
                    int(entry["timestamp"])
                    if entry.get("timestamp") is not None
                    else None
                ),
                duration_seconds=(
                    int(entry["duration"])
                    if entry.get("duration") is not None
                    else None
                ),
                view_count=(
                    int(entry["view_count"])
                    if entry.get("view_count") is not None
                    else None
                ),
            )
        )
        seen_video_ids.add(video_id)
    return videos
