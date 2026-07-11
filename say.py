#!.venv/bin/python

# Imports needed before cache setup
import os
import sys
import time
import json
import threading
import subprocess
import platform
import queue
import termios
import tty
import glob

# Cache model downloads next to this script, must be set before importing kokoro
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models--hexgrad--Kokoro-82M', 'snapshots')
PHRASES_PATH = os.path.join(SCRIPT_DIR, 'phrases.json')
os.environ['HF_HUB_CACHE'] = CACHE_DIR
os.environ['HF_HUB_VERBOSITY'] = 'error'

# Limit torch thread pools before kokoro imports torch
TORCH_THREADS = '4'
os.environ['OMP_NUM_THREADS'] = TORCH_THREADS
os.environ['MKL_NUM_THREADS'] = TORCH_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = TORCH_THREADS

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

# Load preset phrases from json file
def load_phrases():
    with open(PHRASES_PATH) as phrases_file:
        return json.load(phrases_file)

# Print startup banner and key help
def print_banner(phrases):
    print("Say — interactive speech over SSH")
    print("Preset keys:")
    for key in sorted(phrases.keys()):
        print(f"  {key}  {phrases[key]}")
    print("Controls:")
    print("  t  type a custom phrase")
    print("  r  repeat last custom phrase")
    print("  c  cancel current speech")
    print("  x  clear queued speech")
    print("  +  faster")
    print("  -  slower")
    print("  v  next voice")
    print("  h  show help")
    print("  q  quit")
    print()

# Print banner before heavy imports
print_banner(load_phrases())
IMPORT_START = time.perf_counter()

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils.weight_norm')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

# Imports
import kokoro
import torch
import soundfile

IMPORT_SECONDS = time.perf_counter() - IMPORT_START
print(f"Import kokoro: {IMPORT_SECONDS:.1f}s")

torch.set_num_threads(int(TORCH_THREADS))

# Pick cuda when available, else cpu
def pick_device():
    if FORCE_CPU:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

DEVICE = pick_device()

CPU_PERF_MODE = 'performance'
CPU_SCALING_FILE = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'

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

# Read jetson gpu clock in mhz from sysfs
def read_gpu_frequency_mhz():
    frequency_paths = glob.glob('/sys/class/devfreq/*gpu*/cur_freq')
    if not frequency_paths:
        return None
    with open(frequency_paths[0]) as frequency_file:
        hertz = int(frequency_file.read().strip())
    return hertz / 1_000_000

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

CPU_PERF_SET = set_cpu_mode(CPU_PERF_MODE)
print_system_info(CPU_PERF_SET)

# Model repo
REPO_ID = 'hexgrad/Kokoro-82M'

# Speech defaults
DISABLE_COMPLEX = True
DEFAULT_SPEED = 1.5
SPEED_STEP = 0.1
SPEED_MIN = 0.5
SPEED_MAX = 2.0
AUDIO_SAMPLE_RATE = 24000
REALTIME_DECIMALS = 1
AUDIO_DIR = os.path.join(SCRIPT_DIR, 'audio')
AUDIO_NAME_WIDTH = 3

# All voices for manual cycling
VOICES = [
    'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa',
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
]

# Status column widths for aligned output
STATUS_STATE_WIDTH = 8
STATUS_QUEUE_WIDTH = 2
STATUS_VOICE_WIDTH = 11
STATUS_SPEED_WIDTH = 4
STATUS_REALTIME_WIDTH = 5

# Audio player command
PLAYER = 'afplay' if platform.system() == 'Darwin' else 'aplay'

# Lock all terminal output so worker and input threads do not interleave
OUTPUT_LOCK = threading.Lock()

# Run the interactive say tool
def main():
    # Start the speech engine
    engine = SpeechEngine()
    engine.start()

    try:
        # Handle keyboard input until quit
        run_input_loop(engine, load_phrases())
    finally:
        engine.stop()

# Update status in place on one line
def write_status(text):
    with OUTPUT_LOCK:
        sys.stdout.write('\r\033[K' + text)
        sys.stdout.flush()

# Write a scrolling line to the terminal
def write_line(text):
    with OUTPUT_LOCK:
        sys.stdout.write('\r\033[K' + text + '\n')
        sys.stdout.flush()

# Read single keypress without waiting for enter
def read_key():
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            return 'q'
        return line.strip()[0]

    file_descriptor = sys.stdin.fileno()
    old_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        key = sys.stdin.read(1)
        if key == '\x1b':
            key += sys.stdin.read(2)
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_settings)
    return key

# Read a full line for custom phrases
def read_line(prompt):
    with OUTPUT_LOCK:
        sys.stdout.write('\n' + prompt)
        sys.stdout.flush()
    return input()

# Handle keyboard commands
def run_input_loop(engine, phrases):
    while True:
        key = read_key()

        # Quit
        if key in ('q', 'Q', '\x03'):
            write_line("Quit.")
            break

        # Custom phrase
        if key in ('t', 'T', ':'):
            text = read_line("Say> ").strip()
            if text:
                engine.set_last_custom(text)
                engine.enqueue(text)
            continue

        # Repeat last custom phrase
        if key in ('r', 'R'):
            text = engine.get_last_custom()
            if text:
                engine.enqueue(text)
                print_status(engine, f"Repeat: {text}")
            else:
                print_status(engine, "No custom phrase to repeat.")
            continue

        # Cancel current speech
        if key in ('c', 'C'):
            if engine.cancel_current():
                print_status(engine, "Cancelled current speech.")
            continue

        # Clear queue
        if key in ('x', 'X'):
            engine.clear_queue()
            print_status(engine, "Queue cleared.")
            continue

        # Speed up
        if key in ('+', '='):
            engine.change_speed(SPEED_STEP)
            print_status(engine, f"Speed {engine.speed:.1f}.")
            continue

        # Speed down
        if key in ('-', '_'):
            engine.change_speed(-SPEED_STEP)
            print_status(engine, f"Speed {engine.speed:.1f}.")
            continue

        # Next voice
        if key in ('v', 'V'):
            if engine.next_voice():
                print_status(engine, f"Voice {engine.voice}.")
            continue

        # Help
        if key in ('h', 'H', '?'):
            print_banner(phrases)
            continue

        # Preset phrase keys
        if key in phrases:
            engine.enqueue(phrases[key])
            print_status(engine, f"Queued: {phrases[key]}")
            continue

        # Ignore other keys
        if key not in ('\r', '\n'):
            print_status(engine, f"Unknown key: {repr(key)}")

# Format exception text for status lines
def format_error(error):
    text = str(error).strip()
    if 'offline mode is enabled' in text:
        return 'voice not cached, run say once online to download voices'
    if 'Cannot reach' in text:
        return 'network unavailable, voice not cached'
    if len(text) > 80:
        return text[:77] + '...'
    return text

# Format realtime generation speed for status output
def format_realtime(speed):
    return f"{speed:.{REALTIME_DECIMALS}f}x"

# Format aligned status prefix
def format_status(engine, state=None, message=''):
    state = state or engine.state_label()
    if engine.last_realtime_speed is None:
        realtime = f"{'--':>{STATUS_REALTIME_WIDTH}}"
    else:
        realtime = f"{format_realtime(engine.last_realtime_speed):>{STATUS_REALTIME_WIDTH}}"
    prefix = (
        f"[{state:<{STATUS_STATE_WIDTH}} | queue {engine.queue_size():>{STATUS_QUEUE_WIDTH}} | "
        f"{engine.voice:<{STATUS_VOICE_WIDTH}} | "
        f"{engine.speed:>{STATUS_SPEED_WIDTH}.1f}x | {realtime}]"
    )
    if message:
        return f"{prefix} {message}"
    return prefix

# Print a status line
def print_status(engine, message):
    write_status(format_status(engine, message=message))

# Background speech engine with queue and playback control
class SpeechEngine:

    # Create engine with default voice and speed
    def __init__(self):
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.worker = None
        self.running = False
        self.state = 'idle'
        self.player_process = None
        self.cancel_flag = threading.Event()
        self.voice_index = VOICES.index('bm_fable')
        self.voice = VOICES[self.voice_index]
        self.speed = DEFAULT_SPEED
        self.last_realtime_speed = None
        self.model = None
        self.pipeline = None
        self.audio_counter = 1
        self.last_custom = ''
        self.available_voices = set()
        self.pipelines = {}

    # Store last typed custom phrase
    def set_last_custom(self, text):
        with self.lock:
            self.last_custom = text

    # Return last typed custom phrase
    def get_last_custom(self):
        with self.lock:
            return self.last_custom

    # Start background worker thread
    def start(self):
        os.makedirs(AUDIO_DIR, exist_ok=True)
        self.running = True
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()
        print_status(self, "Loading model, please wait...")
        self.queue.put(('__load__', None, None, None))
        while self.model is None and self.running:
            time.sleep(0.1)

    # Stop worker and cancel playback
    def stop(self):
        self.running = False
        self.cancel_current()
        self.clear_queue()
        self.queue.put(None)
        if self.worker:
            self.worker.join(timeout=5)

    # Add phrase to speech queue
    def enqueue(self, text):
        with self.lock:
            voice = self.voice
            speed = self.speed
        self.queue.put((text, voice, speed, time.time()))

    # Cancel current generation or playback
    def cancel_current(self):
        was_busy = self.state in ('speaking', 'loading') or self.player_process is not None
        self.cancel_flag.set()
        self.stop_player()
        if self.state == 'speaking':
            self.state = 'idle'
        return was_busy

    # Stop speech, clear queue, and report error
    def handle_error(self, error, context='', clear_queue=True):
        self.cancel_flag.set()
        self.stop_player()
        self.state = 'idle'
        cleared = 0
        if clear_queue:
            cleared = self.clear_queue()
        detail = format_error(error)
        if context:
            write_line(format_status(self, state='error', message=f"{context}: {detail}"))
        else:
            write_line(format_status(self, state='error', message=detail))
        if clear_queue and cleared:
            write_status(format_status(self, message='Queue cleared after error.'))

    # Remove all queued phrases, return count removed
    def clear_queue(self):
        cleared = 0
        while True:
            try:
                item = self.queue.get_nowait()
                if item is None:
                    self.queue.put(None)
                    break
                if item[0] != '__load__':
                    cleared += 1
            except queue.Empty:
                break
        return cleared

    # Change speech speed within limits
    def change_speed(self, delta):
        with self.lock:
            self.speed = max(SPEED_MIN, min(SPEED_MAX, round(self.speed + delta, 2)))

    # Cycle to next voice, return true on success
    def next_voice(self):
        start_index = self.voice_index
        for offset in range(len(VOICES)):
            index = (start_index + 1 + offset) % len(VOICES)
            voice = VOICES[index]
            if self.try_load_voice(voice):
                with self.lock:
                    self.voice_index = index
                    self.voice = voice
                return True
        write_line(format_status(self, state='error', message='Voice change failed: no voices available.'))
        return False

    # Get pipeline for a language code
    def get_pipeline(self, lang_code):
        if self.model is None:
            return None
        if lang_code not in self.pipelines:
            self.pipelines[lang_code] = kokoro.KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=self.model)
        return self.pipelines[lang_code]

    # Try to load a voice, return true when ready
    def try_load_voice(self, voice):
        if voice in self.available_voices:
            return True
        if self.model is None:
            return True
        try:
            pipeline = self.get_pipeline(voice[0])
            pipeline.load_voice(voice)
            self.available_voices.add(voice)
            return True
        except Exception as error:
            if 'HF_HUB_OFFLINE' in os.environ:
                try:
                    del os.environ['HF_HUB_OFFLINE']
                    pipeline = self.get_pipeline(voice[0])
                    pipeline.load_voice(voice)
                    self.available_voices.add(voice)
                    if len(self.available_voices) == len(VOICES):
                        os.environ['HF_HUB_OFFLINE'] = '1'
                    return True
                except Exception as retry_error:
                    write_line(format_status(self, state='error', message=f"Voice {voice} unavailable: {format_error(retry_error)}"))
                    return False
            write_line(format_status(self, state='error', message=f"Voice {voice} unavailable: {format_error(error)}"))
            return False

    # Load voice weights if needed, return false when unavailable
    def ensure_voice(self, voice):
        return self.try_load_voice(voice)

    # Return queue size excluding control messages
    def queue_size(self):
        return self.queue.qsize()

    # Return current state label
    def state_label(self):
        return self.state

    # Background loop that loads model and speaks queued phrases
    def worker_loop(self):
        while self.running:
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if item is None:
                    break

                # Load model on first request
                text, voice, speed, queued_at = item
                if text == '__load__':
                    self.load_model()
                    continue

                # Skip stale queue entries older than 60 seconds
                if queued_at and time.time() - queued_at > 60:
                    continue

                # Speak the phrase
                self.speak_phrase(text, voice, speed)
            except Exception as error:
                self.handle_error(error, context='Worker error')

    # Load kokoro model once
    def load_model(self):
        try:
            self.state = 'loading'

            # Allow voice downloads on first run
            if 'HF_HUB_OFFLINE' in os.environ:
                del os.environ['HF_HUB_OFFLINE']

            self.model = kokoro.KModel(repo_id=REPO_ID, disable_complex=DISABLE_COMPLEX).to(DEVICE).eval()
            lang_code = self.voice[0]
            self.pipeline = self.get_pipeline(lang_code)
            self.preload_voices()
            self.state = 'idle'
            print_status(self, "Ready.")
        except Exception as error:
            self.model = None
            self.pipeline = None
            self.handle_error(error, context='Model load failed')

    # Preload all voices so switching works offline later
    def preload_voices(self):
        voices_loaded = 0
        for lang_code in ('a', 'b'):
            pipeline = self.get_pipeline(lang_code)
            for voice in VOICES:
                if voice[0] != lang_code:
                    continue
                try:
                    pipeline.load_voice(voice)
                    self.available_voices.add(voice)
                    voices_loaded += 1
                except Exception as error:
                    write_line(format_status(self, state='loading', message=f"Skipped voice {voice}: {format_error(error)}"))
        lang_code = self.voice[0]
        self.pipeline = self.get_pipeline(lang_code)

        # Use offline mode after voices are cached
        if voices_loaded == len(VOICES):
            os.environ['HF_HUB_OFFLINE'] = '1'

    # Generate and play one phrase
    def speak_phrase(self, text, voice, speed):
        if self.pipeline is None or self.model is None:
            self.handle_error('Speech engine not ready.', context=f"Skipped '{text}'")
            return

        if not self.try_load_voice(voice):
            self.handle_error(f"Voice {voice} unavailable.", context=f"Skipped '{text}'")
            return

        self.cancel_flag.clear()
        self.state = 'speaking'
        print_status(self, f"Speaking: {text}")

        total_audio_seconds = 0
        total_generate_seconds = 0

        try:
            # Generate and play each chunk
            pipeline = self.get_pipeline(voice[0])
            self.pipeline = pipeline
            generator = pipeline(text, voice=voice, speed=speed)
            chunk_start = time.perf_counter()
            for chunk_index, (graphemes, phonemes, audio) in enumerate(generator):
                if self.cancel_flag.is_set():
                    break

                generate_seconds = time.perf_counter() - chunk_start
                audio_seconds = len(audio) / AUDIO_SAMPLE_RATE
                total_audio_seconds += audio_seconds
                total_generate_seconds += generate_seconds

                # Write wav file
                wav_name = str(self.audio_counter).zfill(AUDIO_NAME_WIDTH) + '.wav'
                self.audio_counter += 1
                wav_path = os.path.join(AUDIO_DIR, wav_name)
                soundfile.write(wav_path, audio, AUDIO_SAMPLE_RATE)

                # Play wav file
                if self.cancel_flag.is_set():
                    break
                self.play_wav(wav_path)
                chunk_start = time.perf_counter()
        except Exception as error:
            self.handle_error(error, context=f"Failed while speaking '{text}'")
            return

        self.stop_player()
        self.state = 'idle'
        if not self.cancel_flag.is_set() and total_generate_seconds > 0:
            self.last_realtime_speed = total_audio_seconds / total_generate_seconds
            print_status(self, "Done.")

    # Play wav file and allow cancellation
    def play_wav(self, wav_path):
        self.stop_player()
        self.player_process = subprocess.Popen([PLAYER, wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while True:
            if self.cancel_flag.is_set():
                self.stop_player()
                return
            if self.player_process.poll() is not None:
                self.player_process = None
                return
            time.sleep(0.05)

    # Stop current audio playback
    def stop_player(self):
        if self.player_process and self.player_process.poll() is None:
            self.player_process.terminate()
            try:
                self.player_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.player_process.kill()
        self.player_process = None

# Main
if __name__ == '__main__':
    main()
