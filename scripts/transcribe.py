#!/usr/bin/env python3
"""Transcribe one broadcast mp3 with faster-whisper small / CPU int8.
Decode via ffmpeg to s16le raw to avoid PyAV metadata_errors incompatibility.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


def ts(s):
    s = int(s)
    return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"


def main():
    if len(sys.argv) < 4:
        print("usage: transcribe.py <mp3> <out_txt> <header_title_and_link>")
        sys.exit(2)
    mp3, out_txt, header = sys.argv[1], sys.argv[2], sys.argv[3]
    raw = str(Path(mp3).with_suffix(".raw"))
    print(f"decoding {mp3} -> {raw}", flush=True)
    subprocess.check_call([
        "ffmpeg", "-y", "-i", mp3,
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        raw,
    ])
    audio = np.fromfile(raw, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        os.remove(raw)
    except OSError:
        pass
    print(f"audio samples={audio.shape[0]} duration={audio.shape[0]/16000:.1f}s", flush=True)
    print("loading model", flush=True)
    m = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8)
    print("model loaded, transcribing...", flush=True)
    segs, info = m.transcribe(audio, vad_filter=True, beam_size=1)
    print(f"lang {info.language} {info.language_probability}", flush=True)
    out = Path(out_txt)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {header} - faster-whisper small int8, lang={info.language}\n")
        for s in segs:
            f.write(f"[{ts(s.start)}-{ts(s.end)}] {s.text.strip()}\n")
            f.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
