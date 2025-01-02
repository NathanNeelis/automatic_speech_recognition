import speech_recognition as sr
from faster_whisper import WhisperModel
from transformers import pipeline
import torch
import wave
import tempfile
import os

def transcribe_audio_stream():
    """
    records audio through the mic
    transcribes it near real time using a whisper model. This model can be changed for something else, e.g, NL languange
    prints the transcribation as output.

    todo;
    1. make 2 functions. one for recording, one for transcribing.
    2. function for recording, save audio in a folder somewere
    3. function for transcribing, transcribe the audio files in the folder, delete after recording, queue all files and check for new files all the time. Async with recording.


    save transcribations in a log file
    check languange on interpunction and readability. if bad, use LLM to upgrade?

    bug in whisper medium model: 
    Sometimes it transcibes 'thanks for watching!' ?? This was in an audio file i used previously in the day. 
    I do not understand why it still comes up sometimes. It is not in my tempfiles.
    
    """

    # model = "jonatasgrosman/wav2vec2-large-xlsr-53-dutch"
    # model = "openai/whisper-medium"


    # Load the pre-trained ASR pipeline from transformers
    print("Loading model...")
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    # model = WhisperModel("medium", device="cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the recognizer and microphone
    recognizer = sr.Recognizer()
    mic = sr.Microphone(chunk_size=1024) # change chunking size to bigger/smaller for performance

    print("Model loaded. Adjusting for ambient noise, please wait...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print(f"Adjusted for ambient noise. Energy threshold set to {recognizer.energy_threshold}.")

    print("Listening... (Press Ctrl+C to stop)")

    try:
        while True:
            with mic as source:
                print("Recording...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)

            # Convert audio to a WAV file
            print("Processing audio...")
            temp_wav = None # initialize temp_wav variable

            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Path to the 'temp_audio' folder
            temp_audio_dir = os.path.join(script_dir, "temp_audio")

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_audio_dir, delete=False) as temp_wav: # file is deleted in finally statement
                    with wave.open(temp_wav.name, "wb") as wf:
                        wf.setnchannels(1) 
                        wf.setsampwidth(2)
                        wf.setframerate(44100) # 44100 is correct frequency to my mic. If this does not work, see of 16000 works. 
                        wf.writeframes(audio.get_raw_data())
                        wf.close()

                    # Transcribe the WAV file
                    # segments, info = model.transcribe(temp_wav.name)
                    # for segment in segments:
                    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

                    # Alternative option, using transformers pipeline to transcribe the WAV file
                    transcription = asr_pipeline(temp_wav.name)
                    print(f"Transcription: {transcription['text']}")
            except Exception as e:
                print(f"Error during transcription: {e}")
            finally:
                if temp_wav and os.path.exists(temp_wav.name):
                    print('deleting...', temp_wav.name)
                    # os.remove(temp_wav.name)

    except KeyboardInterrupt:
        print("\n Recording & transcription stopped...")

if __name__ == "__main__":
    transcribe_audio_stream()
