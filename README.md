# Speak

Offline text to speech generation, using the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model.

Two tools:

- `speak.py` — speak a fixed phrase once, with timing stats
- `say.py` — interactive keyboard control over SSH, with preset phrases, voice and speed control

Both use CUDA when available. Pass `--cpu` to force CPU inference.

```bash
./speak.py
./speak.py --cpu
./say.py
./say.py --cpu
```

On startup both print import time, CPU info, GPU info, and device. With `--cpu`, GPU shows as `disabled`.

## Setup

On first run, the model and all voices download into `cache/`.

## speak.py

Speak a single block of text. Edit the `TEXT` constant in `speak.py` to change what is spoken.

Generated audio files are saved in `audio/` as `001.wav`, `002.wav`, etc.

## say.py

Interactive speech tool. Press keys to speak phrases.

Single keypresses work without Enter. Preset phrases are in `phrases.json`, triggered by keys `1`–`9`.

### Controls

| Key | Action |
|-----|--------|
| `t` | Type a custom phrase |
| `r` | Repeat last custom phrase |
| `c` | Cancel current speech |
| `x` | Clear queued speech |
| `+` / `-` | Speed up / down |
| `v` | Next voice |
| `h` | Show help |
| `q` | Quit |

Default speed is 1.5x. Use `+` / `-` to adjust, `v` to change voice.

Generated audio files are saved in `audio/`.

## Tools

- `tools-offline.sh` — block internet for offline testing, `./tools-offline.sh --fix` to restore
- `tools-perf.sh` — show CPU info

## Notes

- First `say.py` launch downloads all voices and takes longer. Later launches are faster.
