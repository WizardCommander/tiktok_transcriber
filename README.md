# TikTok Transcriber

CLI for downloading every accessible TikTok from a profile, transcribing the audio with OpenAI by default or local `faster-whisper` behind a flag, generating on-screen scene descriptions with Gemini, and exporting both raw and retrieval-ready corpora for MCP servers or Claude skills.

## Install

```bash
pip3 install --user --break-system-packages -e .
```

## Configure

Set your Gemini and OpenAI API keys:

```bash
export GEMINI_API_KEY=your-key
export OPENAI_API_KEY=your-key
```

Or place it in a local `.env` file:

```bash
GEMINI_API_KEY=your-key
OPENAI_API_KEY=your-key
```

If TikTok requires authentication, pass either:

- `--cookies-file /path/to/cookies.txt`
- `--cookies-from-browser chrome`

## Run

Start small first:

```bash
python3 -m tiktok_transcriber sync-profile \
  --profile-url 'https://www.tiktok.com/@deltatrendtrading' \
  --limit 3 \
  --gemini-model gemini-2.5-flash
```

Use local Whisper instead of OpenAI transcription:

```bash
python3 -m tiktok_transcriber sync-profile \
  --profile-url 'https://www.tiktok.com/@deltatrendtrading' \
  --limit 3 \
  --transcription-provider whisper-local \
  --whisper-model small
```

Skip Gemini while testing the downloader/transcriber path:

```bash
python3 -m tiktok_transcriber sync-profile \
  --profile-url 'https://www.tiktok.com/@deltatrendtrading' \
  --limit 3 \
  --skip-scene-descriptions
```

## Output

For each video, the CLI writes:

- `data/<handle>/videos/<video_id>/download-metadata.json`
- `data/<handle>/videos/<video_id>/document.json`
- `data/<handle>/videos/<video_id>/document.md`
- `data/<handle>/videos/<video_id>/transcript.json`
- `data/<handle>/videos/<video_id>/scenes.json`

It also writes:

- `data/<handle>/manifest.json`
- `data/<handle>/videos.jsonl`
- `data/<handle>/videos.clean.jsonl`
- `data/<handle>/chunks.jsonl`
- `data/<handle>/skipped.json`

- `videos.jsonl` preserves the raw per-video export, including the original `video_path` value from processing time.
- `videos.clean.jsonl` removes stale `video_path` fields so the corpus reflects the retained artifacts on disk.
- `chunks.jsonl` expands each processed video into transcript and scene chunks for retrieval-oriented ingestion.
- `skipped.json` aggregates the videos that could not be downloaded or processed, with the failure reason for each entry.
