from tiktok_transcriber.discovery import (
    normalize_profile_handle,
    parse_profile_playlist,
)
from tiktok_transcriber.models import ProfileVideo


def test_normalize_profile_handle_trims_prefix_and_query() -> None:
    profile_url = "https://www.tiktok.com/@deltaTrendTrading?lang=en"

    assert normalize_profile_handle(profile_url) == "deltaTrendTrading"


def test_parse_profile_playlist_returns_unique_profile_videos() -> None:
    payload = {
        "webpage_url": "https://www.tiktok.com/@deltatrendtrading",
        "entries": [
            {
                "id": "100",
                "url": "https://www.tiktok.com/@deltatrendtrading/video/100",
                "title": "first title",
                "description": "first description",
                "timestamp": 1_700_000_000,
                "duration": 42,
                "view_count": 12,
                "uploader": "deltatrendtrading",
            },
            {
                "id": "100",
                "url": "https://www.tiktok.com/@deltatrendtrading/video/100",
                "title": "first title duplicate",
                "description": "duplicate should be ignored",
                "timestamp": 1_700_000_001,
                "duration": 43,
                "view_count": 14,
                "uploader": "deltatrendtrading",
            },
            {
                "id": "200",
                "url": "https://www.tiktok.com/@deltatrendtrading/video/200",
                "title": "second title",
                "description": "second description",
                "timestamp": 1_700_000_100,
                "duration": 84,
                "view_count": 99,
                "uploader": "deltatrendtrading",
            },
        ],
    }

    actual = parse_profile_playlist(payload)

    assert actual == [
        ProfileVideo(
            video_id="100",
            url="https://www.tiktok.com/@deltatrendtrading/video/100",
            profile_handle="deltatrendtrading",
            title="first title",
            description="first description",
            timestamp=1_700_000_000,
            duration_seconds=42,
            view_count=12,
        ),
        ProfileVideo(
            video_id="200",
            url="https://www.tiktok.com/@deltatrendtrading/video/200",
            profile_handle="deltatrendtrading",
            title="second title",
            description="second description",
            timestamp=1_700_000_100,
            duration_seconds=84,
            view_count=99,
        ),
    ]
