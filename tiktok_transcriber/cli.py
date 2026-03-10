from argparse import ArgumentParser
from collections.abc import Sequence
import os

from dotenv import find_dotenv
from dotenv import load_dotenv

from tiktok_transcriber.config import resolve_sync_profile_settings
from tiktok_transcriber.pipeline import sync_profile
from tiktok_transcriber.runtime import build_pipeline_dependencies


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="tiktok-transcriber")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_profile_parser = subparsers.add_parser("sync-profile")
    sync_profile_parser.add_argument("--profile-url", required=True)
    sync_profile_parser.add_argument("--output-dir", default="data")
    sync_profile_parser.add_argument("--gemini-api-key")
    sync_profile_parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    sync_profile_parser.add_argument(
        "--transcription-provider",
        choices=("openai", "whisper-local"),
        default="openai",
    )
    sync_profile_parser.add_argument("--openai-api-key")
    sync_profile_parser.add_argument(
        "--openai-transcription-model",
        default="gpt-4o-transcribe",
    )
    sync_profile_parser.add_argument("--whisper-model", default="small")
    sync_profile_parser.add_argument("--frame-samples", type=int, default=6)
    sync_profile_parser.add_argument("--limit", type=int)
    sync_profile_parser.add_argument("--transcription-language")
    sync_profile_parser.add_argument("--cookies-file")
    sync_profile_parser.add_argument("--cookies-from-browser")
    sync_profile_parser.add_argument("--skip-scene-descriptions", action="store_true")
    sync_profile_parser.add_argument("--overwrite", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "sync-profile":
        parser.error(f"Unsupported command: {args.command}")
    settings = resolve_sync_profile_settings(args, os.environ)
    outputs = sync_profile(settings, build_pipeline_dependencies())
    for output in outputs:
        print(output)
    return 0


def main() -> None:
    raise SystemExit(run())
