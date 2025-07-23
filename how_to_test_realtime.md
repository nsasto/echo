# Setting Up and Running Real-Time Transcription with Whisper on Windows

This guide provides step-by-step instructions to set up a fine-tuned Whisper model for real-time speech-to-text transcription on a Windows machine using Python. The process involves downloading the model from Kaggle, installing dependencies, and running a Python script for real-time transcription.

## Prerequisites

Before you begin, ensure you have the following:

- **A working microphone** connected to your Windows computer.
- **Python 3.8 or higher** installed. Check your Python version by running:
  ```cmd
  python --version
  ```
  If Python is not installed, download and install it from [python.org](https://www.python.org/downloads/). Ensure you check "Add Python to PATH" during installation.
- A **Kaggle account** to download the fine-tuned Whisper model dataset (`nathansasto/whisper-echo`).
- **Optional: A GPU** with CUDA support for faster transcription. If you don’t have a GPU, the script will run on CPU but may be slower.
- **Command Prompt** or **PowerShell** for running commands. These instructions use Command Prompt (`cmd`) for simplicity.

## Step 1: Set Up Your Environment

### 1.1 Create a Python Virtual Environment

To avoid conflicts with other Python projects, create a virtual environment:

1. Open Command Prompt:
   - Press `Win + R`, type `cmd`, and press Enter.
2. Create a virtual environment:
   ```cmd
   python -m venv whisper_env
   ```
3. Activate the virtual environment:
   ```cmd
   whisper_env\Scripts\activate
   ```
   You should see `(whisper_env)` in your Command Prompt.

### 1.2 Install Python Dependencies

Install the required Python packages within the virtual environment:

```cmd
pip install pyaudio librosa transformers torch numpy
```

- **Note**: On Windows, `pyaudio` typically installs without additional system dependencies.
- To ensure the latest `transformers` library (to avoid deprecated warnings):
  ```cmd
  pip install --upgrade transformers
  ```
- **Optional: GPU Support**:
  If you have a compatible NVIDIA GPU and want faster transcription, install PyTorch with CUDA support:
  ```cmd
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
  Verify GPU availability after setup:
  ```cmd
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  This should output `True` if CUDA is configured correctly.

## Step 2: Download the Fine-Tuned Whisper Model

The real-time transcription script uses a fine-tuned Whisper model from the Kaggle dataset `nathansasto/whisper-echo`.

### 2.1 Obtain a Kaggle API Token

1. Log in to your [Kaggle account](https://www.kaggle.com) or create one.
2. Go to your profile (click your profile picture > "Account").
3. Scroll to the "API" section and click "Create New API Token."
4. Download the `kaggle.json` file, which contains your API credentials.
5. Move the `kaggle.json` file to the Kaggle configuration directory:
   ```cmd
   mkdir %USERPROFILE%\.kaggle
   move kaggle.json %USERPROFILE%\.kaggle\
   ```

### 2.2 Install the Kaggle CLI

Install the Kaggle command-line tool:

```cmd
pip install kaggle
```

### 2.3 Download and Extract the Model

1. Download the `nathansasto/whisper-echo` dataset:
   ```cmd
   kaggle datasets download -d nathansasto/whisper-echo
   ```
   This downloads a file named `whisper-echo.zip` to your current directory.
2. Create a directory for the model and extract the dataset:
   ```cmd
   mkdir whisper_echo
   unzip whisper-echo.zip -d whisper_echo
   ```
   - If `unzip` is not available, install a tool like [7-Zip](https://www.7-zip.org/) or extract the ZIP file manually using File Explorer.
   - The `whisper_echo` directory should contain model files (e.g., `pytorch_model.bin`, `config.json`).

## Step 3: Save the Real-Time Transcription Script

Create a file named `real_time_transcription.py` in your working directory (e.g., `C:\dev\echo`) and copy the following code into it:

```python
import pyaudio
import numpy as np
import torch
import librosa
import logging
import os
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from queue import Queue
from threading import Thread
from time import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio settings
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate (Hz), standard for Whisper
CHUNK_DURATION = CHUNK / RATE  # Duration of each chunk in seconds
MODEL_PATH = "./whisper_echo"  # Path to fine-tuned Whisper model
TRANSCRIPTION_INTERVAL = 3.0  # Seconds of audio to accumulate before transcribing
SILENCE_THRESHOLD = 0.01  # Amplitude threshold for silence detection

def load_whisper_model(model_path):
    """Loads the fine-tuned Whisper model and sets up the pipeline."""
    try:
        logging.info("Loading fine-tuned model...")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        # Optimize generation config to avoid redundant logits processors
        model.generation_config.update(
            suppress_tokens=None,
            begin_suppress_tokens=None,
            task="transcribe",
            language="en"  # Assuming English audio
        )
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"task": "transcribe", "language": "en"}
        )
        logging.info(f"Model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        return pipe
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

def audio_stream(queue):
    """Captures audio from the microphone and puts chunks into a queue."""
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        logging.info("Starting audio stream...")
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            queue.put(data)
    except Exception as e:
        logging.error(f"Audio stream error: {e}")
        raise
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.info("Audio stream terminated")

def transcribe_audio(pipe, audio_queue, transcription_interval):
    """Processes audio chunks from the queue and transcribes them."""
    audio_buffer = []
    buffer_duration = 0.0
    try:
        while True:
            # Get audio chunk from queue
            if not audio_queue.empty():
                data = audio_queue.get()
                # Convert raw audio to numpy array
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(audio_array)
                buffer_duration += CHUNK_DURATION

                # Transcribe when enough audio is accumulated
                if buffer_duration >= transcription_interval:
                    # Concatenate buffer into a single array
                    audio_data = np.concatenate(audio_buffer)
                    # Check for silence
                    if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
                        logging.info("Skipping silent audio chunk")
                        audio_buffer = []
                        buffer_duration = 0.0
                        continue
                    # Resample to ensure 16kHz (should already be 16kHz, but included for robustness)
                    audio_data = librosa.resample(audio_data, orig_sr=RATE, target_sr=16000)
                    # Transcribe
                    try:
                        start_time = time()
                        result = pipe(audio_data)
                        transcription = result['text'].strip()
                        if transcription:
                            print(f"Transcription: {transcription} (processed in {time() - start_time:.2f}s)")
                        else:
                            print("No transcription (silent or empty audio)")
                    except Exception as e:
                        logging.error(f"Transcription error: {e}")
                    # Reset buffer
                    audio_buffer = []
                    buffer_duration = 0.0
    except KeyboardInterrupt:
        logging.info("Transcription stopped by user")
    except Exception as e:
        logging.error(f"Transcription thread error: {e}")
        raise

def real_time_transcription(max_duration_seconds=60):
    """Runs real-time transcription using audio streaming."""
    # Load the model
    pipe = load_whisper_model(MODEL_PATH)

    # Initialize audio queue
    audio_queue = Queue()

    # Start audio streaming in a separate thread
    audio_thread = Thread(target=audio_stream, args=(audio_queue,))
    audio_thread.daemon = True  # Thread exits when main program exits
    audio_thread.start()

    print(f"Starting real-time transcription (max {max_duration_seconds} seconds). Press Ctrl+C to stop.")
    start_time = time()

    try:
        # Run transcription in the main thread
        transcribe_audio(pipe, audio_queue, TRANSCRIPTION_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        elapsed_time = time() - start_time
        print(f"Total duration: {elapsed_time:.2f} seconds")
        logging.info("Real-time transcription terminated")

if __name__ == "__main__":
    # Run real-time transcription for up to 60 seconds
    real_time_transcription(max_duration_seconds=60)
```

You can save this code using a text editor like Notepad or an IDE (e.g., VS Code). Ensure it’s saved in the same directory as `whisper_echo` (e.g., `C:\dev\echo\real_time_transcription.py`).

## Step 4: Run the Real-Time Transcription Script

1. Navigate to your working directory in Command Prompt:
   ```cmd
   cd C:\dev\echo
   ```
   Replace `C:\dev\echo` with the path where `real_time_transcription.py` and `whisper_echo` are located.
2. Ensure the virtual environment is activated:
   ```cmd
   whisper_env\Scripts\activate
   ```
3. Run the script:
   ```cmd
   python real_time_transcription.py
   ```
4. Speak into your microphone. The script will transcribe your speech every ~3 seconds and display the results. For example:
   ```
   Starting real-time transcription (max 60 seconds). Press Ctrl+C to stop.
   Transcription: Hello, this is a test of real-time transcription. (processed in 2.34s)
   ```
5. Press `Ctrl+C` to stop the script.

## Troubleshooting

- **No Transcriptions Printed**:
  - **Microphone Issue**: Ensure your microphone is connected and working. Test it with a simple recording tool (e.g., Windows Voice Recorder).
  - **Silent Audio**: If you see "Skipping silent audio chunk," lower the `SILENCE_THRESHOLD` in the script (e.g., from `0.01` to `0.005`) to make it more sensitive to quiet audio.
  - **Model Path**: Verify the `whisper_echo` directory contains model files (`pytorch_model.bin`, `config.json`). Update `MODEL_PATH` in the script if the directory is elsewhere.
- **Slow Transcription**:
  - If running on CPU, transcription may take 5-10 seconds per 3-second chunk, depending on the model size. Consider using a GPU or a smaller model (e.g., `tiny` or `base` if available).
  - Increase `TRANSCRIPTION_INTERVAL` (e.g., to `5.0`) to reduce processing frequency:
    ```python
    TRANSCRIPTION_INTERVAL = 5.0
    ```
- **Errors Loading Model**:
  - Ensure the `whisper_echo` directory contains valid model files. If the dataset download failed, re-run:
    ```cmd
    kaggle datasets download -d nathansasto/whisper-echo
    unzip whisper-echo.zip -d whisper_echo
    ```
  - Check for Kaggle API issues by running:
    ```cmd
    kaggle datasets list
    ```
    If you get an authentication error, verify `kaggle.json` is in `%USERPROFILE%\.kaggle\`.
- **Warnings or Deprecated Messages**:
  - If you see warnings about `forced_decoder_ids` or `inputs`, ensure `transformers` is updated:
    ```cmd
    pip install --upgrade transformers
    ```
  - The provided script is optimized to avoid these warnings by using `task="transcribe"` and `language="en"`.

## Optional: Optimize for Performance

- **Use a GPU**:
  - If you have an NVIDIA GPU, ensure CUDA is installed and PyTorch is configured for GPU:
    ```cmd
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```
  - Verify GPU usage:
    ```cmd
    python -c "import torch; print(torch.cuda.is_available())"
    ```
  - GPU transcription is 2-5x faster than CPU.
- **Use a Smaller Model**:
  - If the fine-tuned model is large (e.g., `medium` or `large`), check if a smaller variant (`tiny` or `base`) is available in the dataset. Update `MODEL_PATH` to point to the smaller model.
- **Use `faster-whisper`**:
  - For significantly faster transcription, use the `faster-whisper` library, which is optimized for real-time use:
    ```cmd
    pip install faster-whisper
    ```
  - You’ll need to convert the fine-tuned model to CTranslate2 format and modify the script. Contact your project administrator or refer to the `faster-whisper` documentation for guidance.

## Notes

- The script assumes English audio (`language="en"`) for optimal performance. If your audio is in another language, update the `language` parameter in the script (e.g., `language="es"` for Spanish).
- The script processes audio in 3-second chunks to balance real-time feedback and transcription quality. Adjust `TRANSCRIPTION_INTERVAL` if needed (e.g., `2.0` for faster feedback, `5.0` for less frequent but more accurate transcriptions).
- The script skips silent audio to avoid unnecessary processing. If it skips too much, lower `SILENCE_THRESHOLD` (e.g., to `0.005`).

For additional help or customizations (e.g., saving transcriptions to a file, changing chunk duration, or integrating with other tools), consult your project documentation or contact your administrator.
