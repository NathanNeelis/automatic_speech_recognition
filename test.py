import speech_recognition as sr
from faster_whisper import WhisperModel
from transformers import pipeline
import torch
import wave
import tempfile
import os

def transcribe_audio_stream():
    # Load the pre-trained ASR pipeline from transformers
    print("Loading model...")
    # asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-medium")
    model = WhisperModel("medium", device="cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the recognizer and microphone
    recognizer = sr.Recognizer()
    mic = sr.Microphone(chunk_size=4096)

    print("Model loaded. Adjusting for ambient noise, please wait...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print(f"Adjusted for ambient noise. Energy threshold set to {recognizer.energy_threshold}.")

    print("Listening... (Press Ctrl+C to stop)")

    try:
        while True:
            with mic as source:
                print("Recording...")
                audio = recognizer.listen(source)

            # Convert audio to a WAV file
            print("Processing audio...")
            temp_wav = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    with wave.open(temp_wav.name, "wb") as wf:
                        wf.setnchannels(1) 
                        wf.setsampwidth(2)
                        wf.setframerate(44100)  
                        wf.writeframes(audio.get_raw_data())

                    # Use the pipeline to transcribe the WAV file
                    segments, info = model.transcribe(temp_wav.name)
                    for segment in segments:
                        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

                    # alternative option, using transformers pipeline
                    # transcription = asr_pipeline(temp_wav.name)
                    # print(f"Transcription: {transcription['text']}")
            except Exception as e:
                print(f"Error during transcription: {e}")
            finally:
                if temp_wav and os.path.exists(temp_wav.name):
                    print('deleting...', temp_wav.name)
                    os.remove(temp_wav.name)

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    transcribe_audio_stream()
