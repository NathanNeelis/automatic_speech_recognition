import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# Initialize the Faster Whisper model
model = WhisperModel("medium", device="cpu", compute_type="float32")

# Parameters for PyAudio
RATE = 16000  # Whisper works well with 16kHz
CHANNELS = 1  # Mono audio
FORMAT = pyaudio.paInt16  # 16-bit audio
CHUNK_SIZE = 1024  # Small chunk size for real-time audio
BUFFER_SIZE = 16000 * 2  # Buffer to accumulate 2 seconds of audio for better transcription

# Set up PyAudio for microphone input
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

print("Listening...")

# Function to convert raw audio to float32 for Whisper
def preprocess_audio(audio_data):
    return audio_data.astype(np.float32) / 32768.0

# Accumulated audio buffer for longer transcription
buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

try:
    while True:
        # Read small chunk of audio
        audio_chunk = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)

        # Preprocess the audio chunk (normalize it)
        audio_chunk = preprocess_audio(audio_chunk)

        # Add the chunk to the buffer
        buffer = np.roll(buffer, -len(audio_chunk))  # Shift old data
        buffer[-len(audio_chunk):] = audio_chunk  # Append new data

        # If the buffer is sufficiently filled (e.g., 2 seconds of audio), transcribe it
        if np.count_nonzero(buffer) >= RATE / 2:  # If buffer has 2 seconds of data
            # Transcribe the accumulated audio
            segments, info = model.transcribe(buffer)
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()



