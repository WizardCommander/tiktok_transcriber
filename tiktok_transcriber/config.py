from argparse import Namespace
from pathlib import Path
from typing import Mapping

from tiktok_transcriber.discovery import normalize_profile_handle
from tiktok_transcriber.models import CookieSettings
from tiktok_transcriber.models import SyncProfileSettings


def resolve_sync_profile_settings(
    args: Namespace, environ: Mapping[str, str]
) -> SyncProfileSettings:
    cookie_file = Path(args.cookies_file).expanduser() if args.cookies_file else None
    cookie_browser = args.cookies_from_browser
    if cookie_file is not None and cookie_browser is not None:
        raise ValueError(
            "Use either --cookies-file or --cookies-from-browser, not both"
        )
    frame_samples = int(args.frame_samples)
    if frame_samples < 1:
        raise ValueError("--frame-samples must be at least 1")
    limit = int(args.limit) if args.limit is not None else None
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    gemini_api_key = args.gemini_api_key or environ.get("GEMINI_API_KEY")
    openai_api_key = args.openai_api_key or environ.get("OPENAI_API_KEY")
    if not args.skip_scene_descriptions and not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is required unless --skip-scene-descriptions is set"
        )
    if args.transcription_provider == "openai" and not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required when --transcription-provider is openai"
        )
    return SyncProfileSettings(
        profile_url=args.profile_url,
        profile_handle=normalize_profile_handle(args.profile_url),
        output_dir=Path(args.output_dir).expanduser(),
        cookie_settings=CookieSettings(
            file_path=cookie_file,
            browser=cookie_browser,
        ),
        gemini_api_key=gemini_api_key,
        gemini_model=args.gemini_model,
        transcription_provider=args.transcription_provider,
        openai_api_key=openai_api_key,
        openai_transcription_model=args.openai_transcription_model,
        whisper_model=args.whisper_model,
        frame_samples=frame_samples,
        limit=limit,
        skip_scene_descriptions=bool(args.skip_scene_descriptions),
        transcription_language=args.transcription_language,
        overwrite=bool(args.overwrite),
    )
