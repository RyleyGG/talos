import ollama
import vosk
import pyaudio
import json

from services.config_service import settings


def main():
    # Initialize Vosk model
    # TODO: Make this flexible
    model_path = f'{settings.vosk_models_path}/vosk-model-en-us-0.42-gigaspeech'
    model = vosk.Model(model_path)

    rec = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4096)

    print("Listening for speech. Say 'Terminate' to stop.")

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                recognized_text = result.get('text', '').strip()

                if not recognized_text:
                    continue

                print(f"\n[Recognized]: {recognized_text}")

                # Temporarily stop audio input while generating response
                stream.stop_stream()
                try:
                    response = ollama.generate(model='gemma3', prompt=recognized_text, stream=True)
                    for chunk in response:
                        print(chunk['response'], end='', flush=True)
                except Exception as e:
                    print(f"\n[Ollama error]: {e}")
                finally:
                    stream.start_stream()

                if "terminate" in recognized_text.lower():
                    print("\nTermination keyword detected. Stopping...")
                    break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    main()
