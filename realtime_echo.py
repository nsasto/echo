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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Audio settings
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate (Hz), standard for Whisper
CHUNK_DURATION = CHUNK / RATE  # Duration of each chunk in seconds
MODEL_PATH = "./model"  # Path to fine-tuned Whisper model
TRANSCRIPTION_INTERVAL = 3.0  # Increased to 3 seconds for CPU processing
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
            language="en",  # Assuming English audio
        )
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"task": "transcribe", "language": "en"},
        )
        logging.info(
            f"Model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}"
        )
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
            frames_per_buffer=CHUNK,
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
                audio_array = (
                    np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                )
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
                    audio_data = librosa.resample(
                        audio_data, orig_sr=RATE, target_sr=16000
                    )
                    # Transcribe
                    try:
                        start_time = time()
                        result = pipe(audio_data)
                        transcription = result["text"].strip()
                        if transcription:
                            print(
                                f"Transcription: {transcription} (processed in {time() - start_time:.2f}s)"
                            )
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

    print(
        f"Starting real-time transcription (max {max_duration_seconds} seconds). Press Ctrl+C to stop."
    )
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
