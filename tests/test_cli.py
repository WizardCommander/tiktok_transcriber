from pathlib import Path

from pytest import MonkeyPatch
from pytest import raises

from tiktok_transcriber.cli import build_parser
from tiktok_transcriber.cli import run
from tiktok_transcriber.config import resolve_sync_profile_settings
from tiktok_transcriber.models import SyncProfileSettings


def test_resolve_sync_profile_settings_requires_gemini_key_when_scene_descriptions_enabled() -> (
    None
):
    parser = build_parser()
    args = parser.parse_args(
        [
            "sync-profile",
            "--profile-url",
            "https://www.tiktok.com/@deltatrendtrading",
        ]
    )

    with raises(ValueError, match="GEMINI_API_KEY"):
        resolve_sync_profile_settings(args, {})


def test_resolve_sync_profile_settings_requires_openai_key_by_default() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "sync-profile",
            "--profile-url",
            "https://www.tiktok.com/@deltatrendtrading",
            "--skip-scene-descriptions",
        ]
    )

    with raises(ValueError, match="OPENAI_API_KEY"):
        resolve_sync_profile_settings(args, {})


def test_resolve_sync_profile_settings_allows_skip_scene_descriptions_without_key() -> (
    None
):
    parser = build_parser()
    args = parser.parse_args(
        [
            "sync-profile",
            "--profile-url",
            "https://www.tiktok.com/@deltatrendtrading",
            "--skip-scene-descriptions",
            "--transcription-provider",
            "whisper-local",
        ]
    )

    actual = resolve_sync_profile_settings(args, {})

    assert {
        "profile_url": actual.profile_url,
        "profile_handle": actual.profile_handle,
        "gemini_api_key": actual.gemini_api_key,
        "skip_scene_descriptions": actual.skip_scene_descriptions,
        "frame_samples": actual.frame_samples,
        "gemini_model": actual.gemini_model,
        "whisper_model": actual.whisper_model,
        "transcription_provider": actual.transcription_provider,
        "openai_api_key": actual.openai_api_key,
        "openai_transcription_model": actual.openai_transcription_model,
    } == {
        "profile_url": "https://www.tiktok.com/@deltatrendtrading",
        "profile_handle": "deltatrendtrading",
        "gemini_api_key": None,
        "skip_scene_descriptions": True,
        "frame_samples": 6,
        "gemini_model": "gemini-2.5-flash",
        "whisper_model": "small",
        "transcription_provider": "whisper-local",
        "openai_api_key": None,
        "openai_transcription_model": "gpt-4o-transcribe",
    }


def test_run_loads_gemini_api_key_from_dotenv(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=from-dotenv\nOPENAI_API_KEY=from-openai-dotenv\n"
    )
    captured: dict[str, object] = {}

    def fake_sync_profile(
        settings: SyncProfileSettings, dependencies: object
    ) -> list[Path]:
        captured["gemini_api_key"] = settings.gemini_api_key
        captured["openai_api_key"] = settings.openai_api_key
        captured["dependencies"] = dependencies
        return []

    monkeypatch.setattr("tiktok_transcriber.cli.sync_profile", fake_sync_profile)
    monkeypatch.setattr(
        "tiktok_transcriber.cli.build_pipeline_dependencies",
        lambda: "dependencies",
    )

    assert (
        run(
            [
                "sync-profile",
                "--profile-url",
                "https://www.tiktok.com/@deltatrendtrading",
            ]
        )
        == 0
    )
    assert captured == {
        "gemini_api_key": "from-dotenv",
        "openai_api_key": "from-openai-dotenv",
        "dependencies": "dependencies",
    }
