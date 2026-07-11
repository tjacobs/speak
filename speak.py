#!.venv/bin/python

# Imports needed before cache setup and timing
import os
import sys
import time

# Start script timer
STARTUP_START = time.perf_counter()

# Cache model downloads next to this script, must be set before importing kokoro
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models--hexgrad--Kokoro-82M', 'snapshots')
os.environ['HF_HUB_CACHE'] = CACHE_DIR
os.environ['HF_HUB_VERBOSITY'] = 'error'

# Default voice for speak.py
DEFAULT_VOICE = 'bm_fable'

# Parse command line arguments
def parse_args():
    force_cpu = False
    for argument in sys.argv[1:]:
        if argument == '--cpu':
            force_cpu = True
        else:
            print(f"Unknown argument: {argument}")
            sys.exit(1)
    return force_cpu

FORCE_CPU = parse_args()
if FORCE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Use local cache only when model and voice are already downloaded
def enable_offline_if_cached(voice):
    if not os.path.isdir(MODEL_CACHE_DIR) or not os.listdir(MODEL_CACHE_DIR):
        return
    for snapshot_name in os.listdir(MODEL_CACHE_DIR):
        voice_path = os.path.join(MODEL_CACHE_DIR, snapshot_name, 'voices', voice + '.pt')
        if os.path.isfile(voice_path):
            os.environ['HF_HUB_OFFLINE'] = '1'
            return

enable_offline_if_cached(DEFAULT_VOICE)

# Limit torch thread pools before kokoro imports torch
TORCH_THREADS = '4'
os.environ['OMP_NUM_THREADS'] = TORCH_THREADS
os.environ['MKL_NUM_THREADS'] = TORCH_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = TORCH_THREADS

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils.weight_norm')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

# Print before heavy import
print("Loading...")

# Time kokoro import
KOKORO_START = time.perf_counter()
import kokoro
import torch
KOKORO_SECONDS = time.perf_counter() - KOKORO_START

# Pick cuda when available, else cpu
def pick_device():
    if FORCE_CPU:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

DEVICE = pick_device()

# Pin torch to all pi cores
torch.set_num_threads(int(TORCH_THREADS))

import soundfile
import subprocess
import platform
import random
import glob

# Model repo
REPO_ID = 'hexgrad/Kokoro-82M'

# Speed tweaks
DISABLE_COMPLEX = True
SPEECH_SPEED = 1.2
CPU_PERF_MODE = 'performance'
CPU_SCALING_FILE = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'

# Timing format
SECONDS_DECIMALS = 1
AUDIO_SAMPLE_RATE = 24000
AUDIO_DIR = os.path.join(SCRIPT_DIR, 'audio')
AUDIO_NAME_WIDTH = 3
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

    # Set cpu perf mode to performance, restore when done
    saved_cpu_mode = get_cpu_mode()
    perf_set = set_cpu_mode(CPU_PERF_MODE)
    print_system_info(perf_set)

    # Start run timer
    run_start = time.perf_counter()

    try:
        # Pick default voice
        voice = DEFAULT_VOICE
        #voice = random.choice(VOICES)
        print("Voice: " + voice)

        # Generate audio and play it
        generate_and_play(voice, TEXT)

        # Print total time
        log_timing("Run total", run_start)
        log_timing("Script total", STARTUP_START)
    finally:
        if saved_cpu_mode:
            set_cpu_mode(saved_cpu_mode)

# Print kokoro import timing
def print_import_timing():
    log_elapsed("Import kokoro", KOKORO_SECONDS)

# Print elapsed seconds for a stored duration
def log_elapsed(label, seconds):
    print(f"{label}: {format_seconds(seconds)}")

# Read cpu scaling mode for the first core
def get_cpu_mode():
    if not os.path.exists(CPU_SCALING_FILE):
        return None
    with open(CPU_SCALING_FILE) as scaling_file:
        return scaling_file.read().strip()

# Set cpu scaling mode for all cores, return true on success
def set_cpu_mode(mode):
    for cpu_index in range(64):
        scaling_path = f'/sys/devices/system/cpu/cpu{cpu_index}/cpufreq/scaling_governor'
        if not os.path.exists(scaling_path):
            break
        try:
            with open(scaling_path, 'w') as scaling_file:
                scaling_file.write(mode)
        except OSError:
            return False
    return True

# Print cpu, gpu, and device info
def print_system_info(perf_set):
    current_cpu_mode = get_cpu_mode() or 'unknown'
    if perf_set and current_cpu_mode == CPU_PERF_MODE:
        print(f"CPU: {current_cpu_mode}")
    else:
        print(f"CPU: {current_cpu_mode} (run with sudo to change)")
    if DEVICE == 'cuda':
        properties = torch.cuda.get_device_properties(0)
        frequency = read_gpu_frequency_mhz()
        clock_text = f"{frequency / 1000:.1f}GHz" if frequency else "unknown"
        memory_gigabytes = properties.total_memory / 1024 / 1024 / 1024
        print(f"GPU: {properties.name}, {memory_gigabytes:.1f}GB memory, clock {clock_text}")
    elif FORCE_CPU:
        print("GPU: disabled")
    else:
        print("GPU: not available")
    print(f"Device: {DEVICE}")

# Read jetson gpu clock in mhz from sysfs
def read_gpu_frequency_mhz():
    frequency_paths = glob.glob('/sys/class/devfreq/*gpu*/cur_freq')
    if not frequency_paths:
        return None
    with open(frequency_paths[0]) as frequency_file:
        hertz = int(frequency_file.read().strip())
    return hertz / 1_000_000

# Generate audio for the text and play each chunk
def generate_and_play(voice, text):
    # Load the model and pipeline
    pipeline_start = time.perf_counter()
    model = kokoro.KModel(repo_id=REPO_ID, disable_complex=DISABLE_COMPLEX).to(DEVICE).eval()
    pipeline = kokoro.KPipeline(lang_code=voice[0], repo_id=REPO_ID, model=model)
    pipeline.load_voice(voice)
    log_timing("Load pipeline", pipeline_start)

    # Audio player, afplay on mac, aplay on linux
    player = 'afplay' if platform.system() == 'Darwin' else 'aplay'

    # Generate, write, and play each chunk
    print_chunk_timing_header()
    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_counter = 1
    chunk_start = time.perf_counter()
    generator = pipeline(text, voice=voice, speed=SPEECH_SPEED)
    for index, (graphemes, phonemes, audio) in enumerate(generator):
        # Time chunk generation
        generate_seconds = time.perf_counter() - chunk_start
        audio_seconds = len(audio) / AUDIO_SAMPLE_RATE

        # Write wav file
        wav_name = str(audio_counter).zfill(AUDIO_NAME_WIDTH) + '.wav'
        audio_counter += 1
        wav_path = os.path.join(AUDIO_DIR, wav_name)
        soundfile.write(wav_path, audio, AUDIO_SAMPLE_RATE)

        # Play wav file
        play_start = time.perf_counter()
        subprocess.run([player, wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        play_seconds = time.perf_counter() - play_start
        log_chunk_timing(index, generate_seconds, play_seconds, audio_seconds)

        # Start timer for next chunk generation
        chunk_start = time.perf_counter()

# Print chunk timing column headers
def print_chunk_timing_header():
    print(f"{'Wav':>{CHUNK_COL_WIDTH}}  {'Generate':>{TIMING_COL_WIDTH}}  {'Play':>{TIMING_COL_WIDTH}}  {'Speed':>{SPEED_COL_WIDTH}}")

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
