import re

from ollama import generate
import vosk
import pyaudio
import json
from TTS.api import TTS
import numpy as np

from services.config_service import settings


def clean_text(text: str) -> str:
    # Remove emojis and non-ASCII characters
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()


def safe_tts(tts, text):
    clean = clean_text(text)
    if len(clean) < 5:
        print(f"[TTS Skipped]: Text too short after cleaning -> '{clean}'")
        return None
    try:
        wav = tts.tts(clean)
        return (np.array(wav) * 32767).astype(np.int16)
    except Exception as e:
        print(f"[TTS error]: {e}")
        return None


def main():
    print('Initializing models')

    model_path = f'{settings.vosk_models_path}/vosk-model-en-us-0.42-gigaspeech'
    model = vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(model, 16000)

    # TODO: reduce latency in core conversation loop
    p = pyaudio.PyAudio()
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    tts_sample_rate = 22050  # Matches Tacotron2-DDC
    input_stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True,
                          frames_per_buffer=4096)
    output_stream = p.open(format=pyaudio.paInt16,
                           channels=1,
                           rate=tts_sample_rate,
                           output=True)

    with open('talos_prefixed_instructions.txt', 'r', encoding='utf-8') as f:
        system_instruction = f.read().strip()

    print("Listening for speech. Say 'Terminate' to stop.")
    try:
        while True:
            data = input_stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                recognized_text = result.get('text', '').strip()

                if not recognized_text:
                    continue

                print(f"\n[Recognized]: {recognized_text}")

                if "terminate" in recognized_text.lower():
                    print("\nTermination keyword detected. Stopping...")
                    break

                # Pause input during response generation
                input_stream.stop_stream()

                try:
                    prefixed_prompt = f"{system_instruction}\n\nUser: {recognized_text}\nAssistant:"
                    response = generate(model='gemma3', prompt=prefixed_prompt, stream=True)
                    print("[Response]: ", end='', flush=True)
                    full_response = ''
                    for chunk in response:
                        text = chunk['response']
                        print(text, end='', flush=True)
                        full_response += text

                    print("\n[Synthesizing...]")
                    wav_int16 = safe_tts(tts, full_response)
                    if wav_int16 is not None:
                        output_stream.write(wav_int16.tobytes())

                except Exception as e:
                    print(f"\n[Ollama error]: {e}")

                finally:
                    input_stream.start_stream()
    finally:
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()


if __name__ == '__main__':
    main()
