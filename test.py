#!.venv/bin/python

import glob
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAK = os.path.join(SCRIPT_DIR, 'speak.py')
SAY = os.path.join(SCRIPT_DIR, 'say.py')
OFFLINE_TOOL = os.path.join(SCRIPT_DIR, 'tools-offline.sh')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
AUDIO_DIR = os.path.join(SCRIPT_DIR, 'audio')
ONLINE_TIMEOUT_SECONDS = 600
OFFLINE_TIMEOUT_SECONDS = 180
MIN_WAV_COUNT = 1

def main():
    parse_args()
    clean_dirs()
    failed = False
    failed |= not run_check('online network', check_online)
    if failed:
        print_fail('online network')
        sys.exit(1)
    online_needles = online_device_needles()
    failed |= not run_step('online speak.py', lambda: run_speak([], *online_needles))
    failed |= not run_step('online say.py --test', lambda: run_say(['--test'], *online_needles))
    failed |= not run_step('cpu speak.py --cpu', lambda: run_speak(['--cpu'], 'Device: cpu', 'GPU: disabled'))
    failed |= not run_step('cpu say.py --test --cpu', lambda: run_say(['--test', '--cpu'], 'Device: cpu', 'GPU: disabled'))
    blocked = try_block_internet()
    if blocked:
        failed |= not run_step('offline speak.py', lambda: run_speak([], timeout=OFFLINE_TIMEOUT_SECONDS))
        failed |= not run_step('offline say.py --test', lambda: run_say(['--test'], timeout=OFFLINE_TIMEOUT_SECONDS))
        try_restore_internet()
    else:
        print_skip('firewall offline tests, using HF_HUB_OFFLINE instead')
        failed |= not run_step('offline speak.py', lambda: run_speak([], env=offline_env(), timeout=OFFLINE_TIMEOUT_SECONDS))
        failed |= not run_step('offline say.py --test', lambda: run_say(['--test'], env=offline_env(), timeout=OFFLINE_TIMEOUT_SECONDS))
    if failed:
        print_fail('one or more tests')
        sys.exit(1)
    print_pass('all tests')

# Parse command line arguments
def parse_args():
    for argument in sys.argv[1:]:
        print(f"Unknown argument: {argument}")
        sys.exit(1)

# Remove cache and audio dirs for a fresh run
def clean_dirs():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)

# Run one named check
def run_check(name, function):
    print(f"== {name} ==")
    return function()

# Run one named test step
def run_step(name, function):
    print(f"== {name} ==")
    return function()

# Return true when public internet responds to ping
def check_online():
    result = subprocess.run(['ping', '-c', '1', '-W', '2', '1.1.1.1'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        print('Need internet for first-run cache download.')
        return False
    return True

# Return output needles for default online cuda runs
def online_device_needles():
    if cuda_available():
        return ('Device: cuda',)
    return ()

# Return true when torch sees cuda
def cuda_available():
    result = subprocess.run([sys.executable, '-c', "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True, cwd=SCRIPT_DIR)
    return result.stdout.strip() == 'True'

# Run speak.py with args and check output
def run_speak(args, *needles, env=None, timeout=ONLINE_TIMEOUT_SECONDS):
    return run_script(SPEAK, args, needles, env, timeout)

# Run say.py with args and check output
def run_say(args, *needles, env=None, timeout=ONLINE_TIMEOUT_SECONDS):
    return run_script(SAY, args, needles, env, timeout)

# Run a script and verify output and audio files
def run_script(script, args, needles, env, timeout):
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run([script] + args, capture_output=True, text=True, cwd=SCRIPT_DIR, env=run_env, timeout=timeout)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print(output)
        return False
    for needle in needles:
        if needle not in output:
            print(f"Missing in output: {needle}")
            print(output)
            return False
    if len(glob.glob(os.path.join(AUDIO_DIR, '*.wav'))) < MIN_WAV_COUNT:
        print(f"Missing wav files in {AUDIO_DIR}")
        return False
    return True

# Return env dict for huggingface offline mode
def offline_env():
    return {'HF_HUB_OFFLINE': '1'}

# Block internet with tools-offline.sh when sudo works
def try_block_internet():
    result = subprocess.run([OFFLINE_TOOL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    return result.returncode == 0

# Restore internet after tools-offline.sh
def try_restore_internet():
    subprocess.run([OFFLINE_TOOL, '--fix'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)

# Print pass line
def print_pass(message):
    print(f"PASS: {message}")

# Print fail line
def print_fail(message):
    print(f"FAIL: {message}")

# Print skip line
def print_skip(message):
    print(f"SKIP: {message}")

# Main
if __name__ == '__main__':
    main()
