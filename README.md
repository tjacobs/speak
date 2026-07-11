# speak

Text to speech from the command line, using the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model. Each run picks a random American or British voice, generates the audio, saves it as wav files, and plays it.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install kokoro soundfile
```

On Linux, install `espeak-ng` for phonemization and `alsa-utils` for playback:

```bash
sudo apt install espeak-ng alsa-utils
```

## Run

```bash
./speak.py
```

Edit the `TEXT` constant in `speak.py` to change what is spoken, or set a specific voice from the `VOICES` list.

## Notes

- The model and voice files are downloaded from Hugging Face on first use and cached in `cache/` next to the script, so later runs work offline.
- Audio plays with `afplay` on macOS and `aplay` on Linux.
- Generated audio is written as `0.wav`, `1.wav`, etc, one file per text chunk.
