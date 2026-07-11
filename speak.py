#!.venv/bin/python

# Imports needed before cache setup and timing
import os
import time

# Start script timer
STARTUP_START = time.perf_counter()

# Cache model downloads next to this script, must be set before importing kokoro
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models--hexgrad--Kokoro-82M', 'snapshots')
os.environ['HF_HUB_CACHE'] = CACHE_DIR

# Use local cache only when the model is already downloaded
if os.path.isdir(MODEL_CACHE_DIR) and os.listdir(MODEL_CACHE_DIR):
    os.environ['HF_HUB_OFFLINE'] = '1'

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils.weight_norm')

# Imports
KOKORO_START = time.perf_counter()
import kokoro
KOKORO_SECONDS = time.perf_counter() - KOKORO_START

import soundfile
import subprocess
import platform
import random

# Model repo
REPO_ID = 'hexgrad/Kokoro-82M'

# Timing format
SECONDS_DECIMALS = 2
AUDIO_SAMPLE_RATE = 24000
CHUNK_COL_WIDTH = 5
TIMING_COL_WIDTH = 8
SPEED_COL_WIDTH = 6

# Text to speak
TEXT = '''
Hi there!
I'm a robot, and I'm here to help you.
'''

# American and british voices from the model repo
VOICES = [
    'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa',
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
]

# Speak the text with a random voice
def main():
    # Print kokoro import timing
    print_import_timing()

    # Start run timer
    run_start = time.perf_counter()

    # Pick a random voice
    voice = "am_echo"
    #voice = random.choice(VOICES)
    print("Voice: " + voice)

    # Generate audio and play it
    generate_and_play(voice, TEXT)

    # Print total time
    log_timing("run total", run_start)
    log_timing("script total", STARTUP_START)

# Print kokoro import timing
def print_import_timing():
    log_elapsed("import kokoro", KOKORO_SECONDS)

# Print elapsed seconds for a stored duration
def log_elapsed(label, seconds):
    print(f"{label}: {format_seconds(seconds)}")

# Generate audio for the text and play each chunk
def generate_and_play(voice, text):
    # Load the model and pipeline
    pipeline_start = time.perf_counter()
    pipeline = kokoro.KPipeline(lang_code=voice[0], repo_id=REPO_ID)
    log_timing("load pipeline", pipeline_start)

    # Audio player, afplay on mac, aplay on linux
    player = 'afplay' if platform.system() == 'Darwin' else 'aplay'

    # Generate, write, and play each chunk
    print_chunk_timing_header()
    chunk_start = time.perf_counter()
    generator = pipeline(text, voice=voice)
    for index, (graphemes, phonemes, audio) in enumerate(generator):
        # Time chunk generation
        generate_seconds = time.perf_counter() - chunk_start
        audio_seconds = len(audio) / AUDIO_SAMPLE_RATE

        # Write wav file
        soundfile.write(f'{index}.wav', audio, AUDIO_SAMPLE_RATE)

        # Play wav file
        play_start = time.perf_counter()
        subprocess.run([player, f'{index}.wav'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        play_seconds = time.perf_counter() - play_start
        log_chunk_timing(index, generate_seconds, play_seconds, audio_seconds)

        # Start timer for next chunk generation
        chunk_start = time.perf_counter()

# Print chunk timing column headers
def print_chunk_timing_header():
    print(f"{'chunk':>{CHUNK_COL_WIDTH}}  {'generate':>{TIMING_COL_WIDTH}}  {'play':>{TIMING_COL_WIDTH}}  {'speed':>{SPEED_COL_WIDTH}}")

# Print one chunk row with aligned generate and play times
def log_chunk_timing(index, generate_seconds, play_seconds, audio_seconds):
    speed = audio_seconds / generate_seconds if generate_seconds > 0 else 0
    print(f"{index:>{CHUNK_COL_WIDTH}}  {format_seconds(generate_seconds):>{TIMING_COL_WIDTH}}  {format_seconds(play_seconds):>{TIMING_COL_WIDTH}}  {format_speed(speed):>{SPEED_COL_WIDTH}}")

# Print elapsed seconds for a timed step
def log_timing(label, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"{label}: {format_seconds(elapsed)}")

# Format seconds for timing output
def format_seconds(seconds):
    return f"{seconds:.{SECONDS_DECIMALS}f}s"

# Format realtime speed for timing output
def format_speed(speed):
    return f"{speed:.{SECONDS_DECIMALS}f}x"

# Main
if __name__ == '__main__':
    main()
