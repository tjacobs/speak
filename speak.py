#!.venv/bin/python

# Cache model downloads next to this script, must be set before importing kokoro
import os
os.environ['HF_HUB_CACHE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils.weight_norm')

# Imports
import kokoro
import soundfile
import subprocess
import platform
import random

# Model repo
REPO_ID = 'hexgrad/Kokoro-82M'

# American and british voices from the model repo
VOICES = [
    'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa',
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
]

# Text to speak
TEXT = '''
Hi there, what's up?
'''

# Speak the text with a random voice
def main():
    # Pick a random voice
    voice = random.choice(VOICES)
    print(voice)

    # Generate audio and play it
    generate_and_play(voice, TEXT)

# Generate audio for the text and play each chunk
def generate_and_play(voice, text):
    # Lang code is the first letter of the voice, a for american, b for british
    pipeline = kokoro.KPipeline(lang_code=voice[0], repo_id=REPO_ID)

    # Audio player, afplay on mac, aplay on linux
    player = 'afplay' if platform.system() == 'Darwin' else 'aplay'

    # Write each chunk to a wav file and play it
    generator = pipeline(text, voice=voice)
    for i, (gs, ps, audio) in enumerate(generator):
        soundfile.write(f'{i}.wav', audio, 24000)
        subprocess.run([player, f'{i}.wav'])

# Main
if __name__ == '__main__':
    main()
