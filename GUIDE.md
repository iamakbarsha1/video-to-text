# How to Extract Text/Subtitles/Transcripts from Video

## Setup (one-time)

```bash
cd ~/Documents/code/video-to-text

# Create virtual env
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Need FFmpeg too (if not installed)
brew install ffmpeg
```

## Extract Plain Text

```bash
python3 transcribe.py video.mp4
# → creates video.txt with full transcript
```

## Extract Subtitles (SRT format)

```bash
python3 transcribe.py video.mp4 -f srt
# → creates video.srt with timestamped subtitles
# Works with VLC, YouTube uploads, video editors
```

## Extract Subtitles (VTT format)

```bash
python3 transcribe.py video.mp4 -f vtt
# → creates video.vtt (web-friendly subtitle format)
```

## Extract Structured JSON

```bash
python3 transcribe.py video.mp4 -f json
# → creates video.json with segments, timestamps, confidence scores
```

## Better Accuracy (slower)

```bash
# Use medium model instead of default "base"
python3 transcribe.py video.mp4 -m medium -f srt

# Available models: tiny, base, small, medium, large-v3
# tiny = fastest, large-v3 = most accurate
```

## Non-English Video

```bash
python3 transcribe.py video.mp4 -l ta    # Tamil
python3 transcribe.py video.mp4 -l hi    # Hindi
python3 transcribe.py video.mp4 -l es    # Spanish
```

## Batch Process Folder

```bash
python3 transcribe.py ./my-videos/ -o ./transcripts/ -f srt
# Transcribes every video in folder → outputs to transcripts/
```

## Custom Output Location

```bash
python3 transcribe.py video.mp4 -o ~/Desktop/transcript.txt
```

## Summary

| Goal | Command |
|------|---------|
| Plain text | `python3 transcribe.py video.mp4` |
| Subtitles (SRT) | `python3 transcribe.py video.mp4 -f srt` |
| Subtitles (VTT) | `python3 transcribe.py video.mp4 -f vtt` |
| JSON with timestamps | `python3 transcribe.py video.mp4 -f json` |
| Better accuracy | add `-m medium` or `-m large-v3` |
| Other language | add `-l <code>` (e.g. `-l ta`) |
| Batch folder | pass folder path instead of file |
