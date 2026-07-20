# Speak

Offline text to speech generation, using the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model.

Three tools:

- `speak.py` — speak a fixed phrase once, with timing stats
- `say.py` — interactive keyboard control over SSH, with preset phrases, voice and speed control
- `listen.py` — live speech to text from the microphone, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

Both use CUDA when available. Pass `--cpu` to force CPU inference.

```bash
./speak.py
./speak.py --cpu
./say.py
./say.py --cpu
```

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

Pass `--test` to speak the first two preset phrases and exit.

## listen.py

Live transcription from the microphone. Speak and lines print as you talk, CTRL-C to stop.

```bash
./install_listen.sh
./listen.py
```

`install_listen.sh` clones and builds [CTranslate2](https://github.com/OpenNMT/CTranslate2) with CUDA for the Jetson GPU, then installs it and faster-whisper into `.venv`. The build takes around 30 minutes and only runs once.

Records from the first USB soundcard with a mic. Uses the whisper `base` model on GPU with voice activity detection.

## Testing

```bash
./test.py
./test.py --fresh
```

Runs `speak.py` and `say.py --test` online, with `--cpu`, and offline. Pass `--fresh` to clear `cache/` and `audio/` first. Requires internet when the model is not cached.

## Tools

- `tools-audio.sh` — route audio to any USB soundcard, disable HDMI and APE outputs. Run `./tools-audio.sh --install` once, then `./tools-audio.sh` after plugging in a new adapter
- `tools-offline.sh` — block internet for offline testing, `./tools-offline.sh --fix` to restore
- `tools-perf.sh` — show CPU info

## Notes

- First `say.py` launch downloads all voices and takes longer. Later launches are faster.
