from pathlib import Path

import requests
from pytest import MonkeyPatch

from tiktok_transcriber.models import CookieSettings
from tiktok_transcriber.models import ProfileVideo
from tiktok_transcriber.models import SceneDescription
from tiktok_transcriber.models import SyncProfileSettings
from tiktok_transcriber.vision import SampledFrame
from tiktok_transcriber.vision import describe_sampled_frames
from tiktok_transcriber.vision import extract_gemini_text
from tiktok_transcriber.vision import GeminiFallbackError
from tiktok_transcriber.vision import should_fallback_to_openai


def build_settings() -> SyncProfileSettings:
    return SyncProfileSettings(
        profile_url="https://www.tiktok.com/@deltatrendtrading",
        profile_handle="deltatrendtrading",
        output_dir=Path("data"),
        cookie_settings=CookieSettings(),
        gemini_api_key="gemini-key",
        gemini_model="gemini-2.5-flash",
        transcription_provider="openai",
        openai_api_key="openai-key",
        openai_transcription_model="gpt-4o-transcribe",
        whisper_model="small",
        frame_samples=4,
        limit=1,
        skip_scene_descriptions=False,
        transcription_language="en",
        overwrite=False,
    )


def build_video() -> ProfileVideo:
    return ProfileVideo(
        video_id="123",
        url="https://www.tiktok.com/@deltatrendtrading/video/123",
        profile_handle="deltatrendtrading",
        title="Convexity example",
        description="Explaining convex payoff structures",
        timestamp=1_700_000_000,
        duration_seconds=90,
        view_count=1000,
    )


def build_http_error(status_code: int, body: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    response._content = body.encode("utf-8")
    return requests.HTTPError(response=response)


def test_should_fallback_to_openai_matches_quota_and_rate_limit_errors() -> None:
    quota_error = build_http_error(429, '{"error":{"message":"Quota exceeded"}}')
    exhausted_error = build_http_error(
        403, '{"error":{"message":"Resource has been exhausted"}}'
    )
    unavailable_error = build_http_error(
        503, '{"error":{"message":"Service Unavailable"}}'
    )
    auth_error = build_http_error(401, '{"error":{"message":"Invalid API key"}}')

    assert should_fallback_to_openai(quota_error) is True
    assert should_fallback_to_openai(exhausted_error) is True
    assert should_fallback_to_openai(unavailable_error) is True
    assert should_fallback_to_openai(auth_error) is False


def test_describe_sampled_frames_falls_back_to_openai_on_gemini_limit_error(
    monkeypatch: MonkeyPatch,
) -> None:
    settings = build_settings()
    sampled_frames = [
        SampledFrame(
            path=Path("frame_001.jpg"),
            timestamp_seconds=0.0,
        )
    ]
    expected = [
        SceneDescription(
            timestamp_seconds=0.0,
            summary="Chart view with highlighted support and resistance.",
            visible_text=["MNQ", "support"],
        )
    ]
    calls: list[str] = []

    def fake_gemini(
        _sampled_frames: list[SampledFrame],
        _video: ProfileVideo,
        _settings: SyncProfileSettings,
    ) -> list[SceneDescription]:
        calls.append("gemini")
        raise build_http_error(429, '{"error":{"message":"Rate limit exceeded"}}')

    def fake_openai(
        _sampled_frames: list[SampledFrame],
        _video: ProfileVideo,
        _settings: SyncProfileSettings,
    ) -> list[SceneDescription]:
        calls.append("openai")
        return expected

    monkeypatch.setattr(
        "tiktok_transcriber.vision.describe_frames_with_gemini", fake_gemini
    )
    monkeypatch.setattr(
        "tiktok_transcriber.vision.describe_frames_with_openai", fake_openai
    )

    actual = describe_sampled_frames(sampled_frames, build_video(), settings)

    assert actual == expected
    assert calls == ["gemini", "openai"]


def test_extract_gemini_text_raises_fallback_error_on_recitation() -> None:
    payload: dict[str, object] = {
        "candidates": [
            {
                "finishReason": "RECITATION",
                "index": 0,
            }
        ]
    }

    try:
        extract_gemini_text(payload)
    except GeminiFallbackError as error:
        assert "RECITATION" in str(error)
    else:
        raise AssertionError("Expected GeminiFallbackError")
