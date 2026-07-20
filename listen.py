#!.venv/bin/python

# Imports
import os
import subprocess
import sys

import numpy as np

# Config
MODEL_SIZE = 'base'
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3

# Main
def main():
    # Find microphone
    card = find_capture_card()
    if card is None:
        print('No microphone found. Plug in a USB mic and try again.')
        sys.exit(1)

    # Silence onnxruntime GPU discovery warning from the VAD
    import_onnxruntime_quietly()

    # Load model and transcribe the microphone until stopped
    model = load_model()
    run_transcribe_loop(model, card)

# Return card index for the first device with capture support
def find_capture_card():
    result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith('card') and 'USB' in line:
            return int(line.split(':')[0].split()[1])
    return None

# Import onnxruntime with stderr muted, it warns during gpu discovery on jetson
def import_onnxruntime_quietly():
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            import onnxruntime
            onnxruntime.set_default_logger_severity(3)
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)

# Load whisper model on gpu when available
def load_model():
    # Pick device
    print(f'Loading {MODEL_SIZE} model...', flush=True)
    import ctranslate2
    from faster_whisper import WhisperModel
    device = 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
    compute_type = 'float16' if device == 'cuda' else 'float32'

    # Load model
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
    print(f'Whisper {MODEL_SIZE} on {device.upper()} {compute_type}, VAD on CPU')
    return model

# Record chunks from the microphone and print transcripts
def run_transcribe_loop(model, card):
    recorder = start_recorder(card)
    chunk_bytes = SAMPLE_RATE * 2 * CHUNK_SECONDS
    print('Listening, speak now, CTRL-C to stop.', flush=True)
    try:
        while True:
            # Read one chunk of raw audio
            data = recorder.stdout.read(chunk_bytes)
            if not data:
                break

            # Transcribe and print each spoken segment
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = model.transcribe(audio, language='en', vad_filter=True)
            for segment in segments:
                print(segment.text.strip(), flush=True)
    except KeyboardInterrupt:
        print('\nDone.')
    finally:
        recorder.terminate()

# Start arecord streaming raw audio to stdout
def start_recorder(card):
    command = ['arecord', '-D', f'plughw:{card},0', '-f', 'S16_LE', '-r', str(SAMPLE_RATE), '-c', '1', '-t', 'raw', '-q']
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# Main
if __name__ == '__main__':
    main()
