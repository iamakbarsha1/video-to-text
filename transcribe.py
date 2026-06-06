#!/usr/bin/env python3
"""
Video-to-Text Transcriber
Extracts audio from video files using FFmpeg, then transcribes using Whisper.
Fully open source. Runs locally. No API keys needed.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Fix duplicate OpenMP library issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed.")
    print("Run: pip install faster-whisper")
    sys.exit(1)


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv", ".ts", ".vob",
}

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus",
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

OUTPUT_FORMATS = ["txt", "srt", "vtt", "json", "tsv"]


def check_ffmpeg():
    """Verify FFmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio from video file using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", "16000",         # 16kHz sample rate (Whisper optimal)
        "-ac", "1",             # mono
        "-y",                   # overwrite
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg error: {result.stderr.splitlines()[-1] if result.stderr else 'unknown'}")
            return False
        return True
    except Exception as e:
        print(f"  FFmpeg failed: {e}")
        return False


def get_duration(file_path: str) -> float:
    """Get media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (ValueError, Exception):
        return 0.0


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def transcribe_file(
    file_path: str,
    model: WhisperModel,
    output_format: str = "txt",
    output_dir: str | None = None,
    language: str | None = None,
) -> str | None:
    """Transcribe a single video/audio file and return the output path."""

    file_path = os.path.abspath(file_path)
    path = Path(file_path)

    if not path.exists():
        print(f"File not found: {file_path}")
        return None

    ext = path.suffix.lower()
    is_audio = ext in SUPPORTED_AUDIO_EXTENSIONS
    is_video = ext in SUPPORTED_VIDEO_EXTENSIONS

    if not is_audio and not is_video:
        print(f"Unsupported format: {ext}")
        print(f"Supported video: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}")
        print(f"Supported audio: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}")
        return None

    # Determine output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_dir = output_dir
    else:
        out_dir = str(path.parent)

    output_path = os.path.join(out_dir, f"{path.stem}.{output_format}")

    duration = get_duration(file_path)
    duration_str = format_time(duration) if duration else "unknown"
    print(f"\n--- {path.name} (duration: {duration_str}) ---")

    # Extract audio if video file
    audio_path = file_path
    temp_audio = None

    if is_video:
        print("  [1/3] Extracting audio with FFmpeg...")
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        audio_path = temp_audio.name

        if not extract_audio(file_path, audio_path):
            os.unlink(audio_path)
            return None
        print("  Audio extracted.")
    else:
        print("  [1/3] Audio file — skipping extraction.")

    # Transcribe
    print("  [2/3] Transcribing...")
    start_time = time.time()

    transcribe_opts = {"beam_size": 5}
    if language:
        transcribe_opts["language"] = language

    segments_gen, info = model.transcribe(audio_path, **transcribe_opts)

    # Consume generator into list
    segments = list(segments_gen)

    elapsed = time.time() - start_time

    # Clean up temp audio
    if temp_audio:
        os.unlink(temp_audio.name)

    detected_lang = info.language
    lang_prob = f"{info.language_probability:.0%}"
    print(f"  Language: {detected_lang} ({lang_prob} confidence)")
    print(f"  Transcription done in {format_time(elapsed)}")

    # Write output
    print(f"  [3/3] Writing {output_format.upper()} output...")

    full_text = " ".join(seg.text.strip() for seg in segments)

    if output_format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

    elif output_format == "srt":
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = _format_srt_time(seg.start)
                end = _format_srt_time(seg.end)
                text = seg.text.strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    elif output_format == "vtt":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(segments, 1):
                start = _format_vtt_time(seg.start)
                end = _format_vtt_time(seg.end)
                text = seg.text.strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    elif output_format == "json":
        import json
        data = {
            "language": detected_lang,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "text": full_text,
            "segments": [
                {
                    "id": i,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
                for i, seg in enumerate(segments)
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    elif output_format == "tsv":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("start\tend\ttext\n")
            for seg in segments:
                start_ms = int(seg.start * 1000)
                end_ms = int(seg.end * 1000)
                text = seg.text.strip().replace("\t", " ")
                f.write(f"{start_ms}\t{end_ms}\t{text}\n")

    print(f"  Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Video/Audio to Text Transcriber (FFmpeg + Whisper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                            # Basic transcription
  %(prog)s video.mp4 -m medium                  # Better accuracy
  %(prog)s video.mp4 -f srt                     # Generate subtitles
  %(prog)s video.mp4 -f srt -l en               # Force English
  %(prog)s ./videos/ -m small -f txt -o out/    # Batch process folder
  %(prog)s audio.mp3                            # Works with audio too
        """,
    )

    parser.add_argument(
        "input",
        help="Video/audio file or directory of files to transcribe",
    )
    parser.add_argument(
        "-m", "--model",
        choices=WHISPER_MODELS,
        default="base",
        help="Whisper model size (default: base). Larger = slower but more accurate",
    )
    parser.add_argument(
        "-f", "--format",
        choices=OUTPUT_FORMATS,
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "-l", "--language",
        help="Force language (e.g., en, es, fr, ar). Auto-detected if not set",
    )

    args = parser.parse_args()

    # Check FFmpeg
    if not check_ffmpeg():
        print("Error: FFmpeg not found. Install it first:")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        sys.exit(1)

    input_path = os.path.abspath(args.input)

    # Collect files
    files = []
    if os.path.isdir(input_path):
        all_extensions = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS
        for f in sorted(os.listdir(input_path)):
            if Path(f).suffix.lower() in all_extensions:
                files.append(os.path.join(input_path, f))
        if not files:
            print(f"No supported media files found in: {input_path}")
            sys.exit(1)
        print(f"Found {len(files)} file(s) to process")
    elif os.path.isfile(input_path):
        files.append(input_path)
    else:
        print(f"Path not found: {input_path}")
        sys.exit(1)

    # Load model once (reused across files)
    print(f"Loading Whisper model: {args.model}...")
    model = WhisperModel(args.model, compute_type="int8")
    print("Model loaded.\n")

    # Process files
    results = []
    for file in files:
        output = transcribe_file(
            file,
            model=model,
            output_format=args.format,
            output_dir=args.output_dir,
            language=args.language,
        )
        if output:
            results.append(output)

    # Summary
    print(f"\n{'='*50}")
    print(f"Done! {len(results)}/{len(files)} file(s) transcribed.")
    for r in results:
        print(f"  -> {r}")


if __name__ == "__main__":
    main()
