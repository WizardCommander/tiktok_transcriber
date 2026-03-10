import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import cv2
from openai import OpenAI
import requests

from tiktok_transcriber.models import ProfileVideo
from tiktok_transcriber.models import SceneDescription
from tiktok_transcriber.models import SyncProfileSettings


@dataclass(frozen=True)
class SampledFrame:
    path: Path
    timestamp_seconds: float


OPENAI_SCENE_DESCRIPTION_MODEL = "gpt-4.1-mini"


class GeminiFallbackError(ValueError):
    pass


def describe_video(
    video_path: Path,
    video_dir: Path,
    video: ProfileVideo,
    settings: SyncProfileSettings,
) -> list[SceneDescription]:
    if settings.gemini_api_key is None:
        raise ValueError("Gemini API key is required for scene descriptions")
    sampled_frames = sample_video_frames(
        video_path,
        settings.frame_samples,
        video_dir / "frames",
    )
    if not sampled_frames:
        return []
    return describe_sampled_frames(sampled_frames, video, settings)


def describe_sampled_frames(
    sampled_frames: list[SampledFrame],
    video: ProfileVideo,
    settings: SyncProfileSettings,
) -> list[SceneDescription]:
    try:
        return describe_frames_with_gemini(sampled_frames, video, settings)
    except (GeminiFallbackError, requests.HTTPError) as error:
        if should_fallback_to_openai(error):
            return describe_frames_with_openai(sampled_frames, video, settings)
        raise


def describe_frames_with_gemini(
    sampled_frames: list[SampledFrame],
    video: ProfileVideo,
    settings: SyncProfileSettings,
) -> list[SceneDescription]:
    if settings.gemini_api_key is None:
        raise ValueError("Gemini API key is required for scene descriptions")
    request_payload = build_gemini_request(sampled_frames, video)
    response = requests.post(
        (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_model}:generateContent"
        ),
        params={"key": settings.gemini_api_key},
        headers={"Content-Type": "application/json"},
        json=request_payload,
        timeout=120,
    )
    response.raise_for_status()
    response_payload = response.json()
    response_text = extract_gemini_text(response_payload)
    return parse_scene_descriptions(response_text)


def describe_frames_with_openai(
    sampled_frames: list[SampledFrame],
    video: ProfileVideo,
    settings: SyncProfileSettings,
) -> list[SceneDescription]:
    if settings.openai_api_key is None:
        raise ValueError(
            "OPENAI_API_KEY is required for OpenAI scene description fallback"
        )
    client = OpenAI(api_key=settings.openai_api_key)
    content: list[object] = [
        {
            "type": "text",
            "text": build_scene_prompt(video),
        }
    ]
    for frame in sampled_frames:
        content.append(
            {
                "type": "text",
                "text": (
                    f"Frame name: {frame.path.name}\n"
                    f"Timestamp seconds: {frame.timestamp_seconds}"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": build_data_url(frame.path),
                    "detail": "low",
                },
            }
        )
    completion = cast(Any, client.chat.completions).create(
        model=OPENAI_SCENE_DESCRIPTION_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )
    message_content = completion.choices[0].message.content
    if message_content is None:
        raise ValueError("OpenAI did not return scene description content")
    return parse_scene_descriptions(message_content)


def sample_video_frames(
    video_path: Path, frame_samples: int, frames_dir: Path
) -> list[SampledFrame]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or fps <= 0:
        capture.release()
        return []
    frame_indexes = build_frame_indexes(frame_count, frame_samples)
    sampled_frames: list[SampledFrame] = []
    for frame_number, frame_index in enumerate(frame_indexes, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        did_read, frame = capture.read()
        if not did_read:
            continue
        frame_path = frames_dir / f"frame_{frame_number:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        sampled_frames.append(
            SampledFrame(
                path=frame_path,
                timestamp_seconds=round(frame_index / fps, 2),
            )
        )
    capture.release()
    return sampled_frames


def build_frame_indexes(frame_count: int, frame_samples: int) -> list[int]:
    if frame_samples <= 1:
        return [max(frame_count // 2, 0)]
    if frame_count <= frame_samples:
        return list(range(frame_count))
    indexes = {
        round((frame_count - 1) * sample_index / (frame_samples - 1))
        for sample_index in range(frame_samples)
    }
    return sorted(indexes)


def build_gemini_request(
    sampled_frames: list[SampledFrame], video: ProfileVideo
) -> dict[str, object]:
    parts: list[dict[str, object]] = [{"text": build_scene_prompt(video)}]
    for frame in sampled_frames:
        parts.append(
            {
                "text": (
                    f"Frame name: {frame.path.name}\n"
                    f"Timestamp seconds: {frame.timestamp_seconds}"
                )
            }
        )
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": encode_image(frame.path),
                }
            }
        )
    return {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "frame_name": {"type": "string"},
                                "timestamp_seconds": {"type": "number"},
                                "summary": {"type": "string"},
                                "visible_text": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "frame_name",
                                "timestamp_seconds",
                                "summary",
                                "visible_text",
                            ],
                        },
                    }
                },
                "required": ["frames"],
            },
        },
    }


def build_scene_prompt(video: ProfileVideo) -> str:
    return (
        "Describe each frame from a TikTok video for downstream search and "
        "knowledge retrieval. Focus on what is on screen, especially charts, "
        "tables, indicators, annotations, ticker symbols, price levels, "
        "spreadsheets, and any trading context. Use visible_text only for "
        "important readable text, not every small UI label. Keep each summary "
        "to one or two sentences.\n"
        "Return a JSON object with a top-level 'frames' array. Each item must "
        "include frame_name, timestamp_seconds, summary, and visible_text.\n"
        f"Video title: {video.title}\n"
        f"Video description: {video.description}"
    )


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_data_url(path: Path) -> str:
    return f"data:image/jpeg;base64,{encode_image(path)}"


def parse_scene_descriptions(response_text: str) -> list[SceneDescription]:
    parsed_response = json.loads(response_text)
    frames = parsed_response.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Scene description response did not include a frames array")
    scene_descriptions: list[SceneDescription] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        visible_text = frame.get("visible_text")
        if not isinstance(visible_text, list):
            visible_text = []
        scene_descriptions.append(
            SceneDescription(
                timestamp_seconds=float(frame["timestamp_seconds"]),
                summary=str(frame["summary"]),
                visible_text=[str(item) for item in visible_text],
            )
        )
    return sorted(scene_descriptions, key=lambda scene: scene.timestamp_seconds)


def should_fallback_to_openai(error: Exception) -> bool:
    if isinstance(error, GeminiFallbackError):
        return True
    if not isinstance(error, requests.HTTPError):
        return False
    response = error.response
    if response is None:
        return False
    if response.status_code == 429:
        return True
    if 500 <= response.status_code < 600:
        return True
    if response.status_code != 403:
        return False
    response_text = response.text.lower()
    fallback_markers = (
        "quota",
        "rate limit",
        "resource has been exhausted",
        "resource_exhausted",
    )
    return any(marker in response_text for marker in fallback_markers)


def extract_gemini_text(response_payload: dict[str, object]) -> str:
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Gemini did not return candidates: {response_payload}")
    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise ValueError(f"Gemini candidate was not an object: {response_payload}")
    content = first_candidate.get("content")
    if not isinstance(content, dict):
        finish_reason = first_candidate.get("finishReason")
        if finish_reason in {"RECITATION", "SAFETY"}:
            raise GeminiFallbackError(
                f"Gemini returned no content with finish reason {finish_reason}"
            )
        raise ValueError(f"Gemini candidate content missing: {response_payload}")
    parts = content.get("parts")
    if not isinstance(parts, list):
        finish_reason = first_candidate.get("finishReason")
        if finish_reason in {"RECITATION", "SAFETY"}:
            raise GeminiFallbackError(
                f"Gemini returned no parts with finish reason {finish_reason}"
            )
        raise ValueError(f"Gemini candidate parts missing: {response_payload}")
    text_parts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    if not text_parts:
        finish_reason = first_candidate.get("finishReason")
        if finish_reason in {"RECITATION", "SAFETY"}:
            raise GeminiFallbackError(
                f"Gemini returned no text with finish reason {finish_reason}"
            )
        raise ValueError(f"Gemini returned no text parts: {response_payload}")
    return "".join(text_parts)
