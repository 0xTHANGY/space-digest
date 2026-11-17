# Space Digest

Turn a local X Space MP3 into a timestamped text transcript. Both scripts use
Python, NumPy, and faster-whisper's small model with INT8 inference on CPU.
FFmpeg decodes audio to 16 kHz mono PCM.

## Setup

Install Python 3 and FFmpeg, with `ffmpeg` available on your PATH.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdir -p transcripts
```

The first run may download the model. Obtain the audio separately and respect
X's terms and the creators' rights when obtaining, transcribing, or sharing it.

## Usage

Use `scripts/transcribe.py` for day-to-day runs. It takes a single header
argument and cleans up the intermediate raw audio after reading it.

```sh
python3 scripts/transcribe.py space.mp3 transcripts/space.txt 'Example Space (https://example.com/space)'
```

Use `scripts/tx_easy.py` if your workflow passes the title and link as separate
arguments. It leaves the intermediate `space.raw` alongside the MP3.

```sh
python3 scripts/tx_easy.py space.mp3 transcripts/space.txt 'Example Space' 'https://example.com/space'
```
