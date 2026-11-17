"""Transcribe an MP3 with separate title and link arguments."""

import argparse
import subprocess

import numpy as np
from faster_whisper import WhisperModel


def timestamp(seconds):
    seconds = int(seconds)
    return f"{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mp3")
    parser.add_argument("output")
    parser.add_argument("title")
    parser.add_argument("url")
    args = parser.parse_args()

    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8)
    raw = args.mp3.rsplit(".", 1)[0] + ".raw"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", args.mp3,
        "-ac", "1", "-ar", "16000", "-f", "s16le", raw,
    ], check=True)
    audio = np.fromfile(raw, dtype=np.int16).astype(np.float32) / 32768
    segments, info = model.transcribe(audio, vad_filter=True, beam_size=1)
    print("lang", info.language, info.language_probability, "dur", info.duration, flush=True)

    with open(args.output, "w", encoding="utf-8") as output:
        output.write(
            f"# {args.title} ({args.url}) - faster-whisper small int8, lang={info.language}\n"
        )
        for segment in segments:
            output.write(
                f"[{timestamp(segment.start)}-{timestamp(segment.end)}] {segment.text.strip()}\n"
            )
            output.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
