# Speak

Text to speech from the command line, using the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model.

Two tools:

- `speak.py` — speak a fixed phrase once, with timing stats
- `talk.py` — interactive keyboard control over SSH, with preset phrases, voice and speed control

## Setup

On first run, the model and all voices download from Hugging Face into `cache/`. After that, `talk.py` works offline.

## speak.py

Speak a single block of text. Edit the `TEXT` constant in `speak.py` to change what is spoken, or set a voice from the `VOICES` list.

```bash
./speak.py
```

Generated audio is written as `0.wav`, `1.wav`, etc, one file per text chunk. Audio plays with `afplay` on macOS and `aplay` on Linux.

## talk.py

Interactive speech tool for controlling a robot over SSH. Phrases queue in the background so you can keep typing while it speaks.

```bash
./talk.py
```

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

On speech errors, playback stops, the queue is cleared, and a short error message is shown. The worker keeps running so you can continue. Voice changes skip unavailable voices without clearing the queue.

Generated audio files are saved in `audio/`.

## Notes

- First `talk.py` launch downloads all voices and takes longer. Later launches are faster.
- If a voice is missing, run once with network access to download it.
- On Jetson, USB audio is configured in `/etc/asound.conf`. Both normal and `sudo aplay` use the USB card.
- `speak.py` disables CUDA probing on CPU-only devices to avoid driver warnings on Jetson.
