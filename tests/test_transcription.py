from types import SimpleNamespace

from tiktok_transcriber.models import TranscriptSegment
from tiktok_transcriber.transcription import build_openai_transcript


def test_build_openai_transcript_maps_verbose_response_segments() -> None:
    assert build_openai_transcript(
        "First sentence. Second sentence.",
        [
            SimpleNamespace(start=0.0, end=1.25, text=" First sentence. "),
            SimpleNamespace(start=1.25, end=2.5, text="Second sentence."),
        ],
    ) == (
        "First sentence. Second sentence.",
        [
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=1.25,
                text="First sentence.",
            ),
            TranscriptSegment(
                start_seconds=1.25,
                end_seconds=2.5,
                text="Second sentence.",
            ),
        ],
    )
