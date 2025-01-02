# Near Real-Time Speech Recognition

This Python script transcribes audio captured by your microphone in near real-time. Simply speak into your microphone and watch the transcription appear in your terminal.

![image](https://github.com/user-attachments/assets/a8a43178-69dc-4077-938d-236408b42b23)

## Getting Started

1. Install the required packages using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

2. Once the dependencies are installed, run the following command in your terminal:

   ```bash
   python main.py
   ```

   - On the first run, the script will download the necessary model for transcribing audio input.
   - Each time the script is executed, it calibrates ambient noise and adjusts the input settings accordingly.

3. The script will activate your microphone (default device in your system settings). You can then start speaking, and the transcription will be displayed in your terminal.

## Model

The script currently uses the [faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) model for transcription.

## Future work

- Save transciptions to a log file
- Improve the readability of transciptions
- If necessary, enhance readability using advanced techniques (e.g., LLMs).
- Experiment with support for multiple languages
- Test and evaluate lightweight models for performance improvements.
