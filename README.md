# Video-to-Text Transcriber

Extract speech from any video/audio file and convert to text. Fully open source, runs locally, no API keys.

**Stack:** FFmpeg (audio extraction) + OpenAI Whisper (speech-to-text)

## Setup

```bash
# Install FFmpeg (if not already installed)
brew install ffmpeg          # macOS
# sudo apt install ffmpeg    # Ubuntu

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic — transcribe a video to text
python transcribe.py video.mp4

# Use a better model for accuracy
python transcribe.py video.mp4 -m medium

# Generate subtitles (SRT format)
python transcribe.py video.mp4 -f srt

# Force a specific language
python transcribe.py video.mp4 -l en

# Batch process a folder
python transcribe.py ./my-videos/ -o ./transcripts/

# Audio files work too
python transcribe.py podcast.mp3
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m` | Model: tiny, base, small, medium, large | base |
| `-f` | Format: txt, srt, vtt, json, tsv | txt |
| `-o` | Output directory | same as input |
| `-l` | Language code (auto-detected if omitted) | auto |

## Output Formats

- **txt** — plain text transcript
- **srt** — SubRip subtitles (for video players)
- **vtt** — WebVTT subtitles (for web)
- **json** — full Whisper output with timestamps and metadata
- **tsv** — tab-separated timestamps and text

## Supported Formats

**Video:** mp4, mkv, avi, mov, wmv, flv, webm, m4v, mpg, mpeg, 3gp, ogv, ts, vob

**Audio:** mp3, wav, flac, aac, ogg, wma, m4a, opus

## Model Comparison

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| tiny | 39MB | ~10x realtime | Quick drafts |
| base | 74MB | ~7x realtime | General use |
| small | 244MB | ~4x realtime | Good accuracy |
| medium | 769MB | ~2x realtime | High accuracy |
| large | 1.5GB | ~1x realtime | Best accuracy |

## License

MIT
