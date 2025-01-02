import pyaudio
import wave
import speech_recognition as sr
import os
import tempfile

# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# frames = []

# try:
#     while True:
#         data = stream.read(1024)
#         frames.append(data)
# except KeyboardInterrupt:
#     pass

# stream.stop_stream()
# stream.close()
# p.terminate()

# sound_file = wave.open("myrecording.wav", "wb")
# sound_file.setnchannels(1)
# sound_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
# sound_file.setframerate(44100)
# sound_file.writeframes(b''.join(frames))
# sound_file.close()


def record_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone(chunk_size=1024) # change chunking size to bigger/smaller for performance

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print(f"Adjusted for ambient noise. Energy threshold set to {recognizer.energy_threshold}.")

    print("Listening... (Press Ctrl+C to stop)")

    try:
        while True:
            with mic as source:
                print("Recording...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=2)

            # Convert audio to a WAV file
            print("Processing audio...")
            temp_wav = None # initialize temp_wav variable

            # Get the directory to save the temp audio files
            script_dir = os.path.dirname(os.path.abspath(__file__))
            temp_audio_dir = os.path.join(script_dir, "temp_audio")

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_audio_dir, delete=False) as temp_wav: # file is deleted in finally statement
                    with wave.open(temp_wav.name, "wb") as wf:
                        wf.setnchannels(1) 
                        wf.setsampwidth(2)
                        wf.setframerate(44100) # 44100 is correct frequency to my mic. If this does not work, see of 16000 works. 
                        wf.writeframes(audio.get_raw_data())
                        wf.close()
            except Exception as e:
                    print(f"Error during transcription: {e}")
            finally:
                if temp_wav and os.path.exists(temp_wav.name):
                    print('saving...', temp_wav.name)

    except KeyboardInterrupt:
        print("\n Recording & transcription stopped...")


if __name__ == "__main__":
    record_audio()
