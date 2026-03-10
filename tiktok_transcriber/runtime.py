from tiktok_transcriber.discovery import discover_profile_videos
from tiktok_transcriber.download import download_video
from tiktok_transcriber.pipeline import PipelineDependencies
from tiktok_transcriber.transcription import transcribe_video
from tiktok_transcriber.vision import describe_video


def build_pipeline_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        discover_profile_videos=discover_profile_videos,
        download_video=download_video,
        transcribe_video=transcribe_video,
        describe_video=describe_video,
    )
