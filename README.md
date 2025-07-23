# Echo 🎤 ➡️ 📝

Echo is an advanced speech-to-text system built on OpenAI's Whisper model, specifically fine-tuned to improve accessibility for people who are hard of hearing. This project uses state-of-the-art machine learning to provide accurate, real-time transcription of spoken words.

## Features

- **Real-time Speech Recognition**: Convert spoken words to text in real-time using a fine-tuned Whisper model
- **Interactive Recording Interface**: Easy-to-use widget interface with visual feedback
- **Custom Model Training**: Fine-tuned on specific speech patterns to improve accuracy
- **Visual Feedback**: Real-time waveform visualization during recording
- **Performance Metrics**: Built-in testing and validation with WER (Word Error Rate) and CER (Character Error Rate) measurements

## Requirements

- Python 3.12+
- PyTorch
- Transformers library
- PyAudio
- librosa
- ipywidgets (for Jupyter interface)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd echo
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Using the Interactive Widget (Still buggy)

1. Open `Test_Echo_widget.ipynb` in Jupyter Notebook
2. Run all cells
3. Press and hold the spacebar to record
4. Release the spacebar to stop recording and see the transcription
5. The visualization will show a red waveform while recording and blue when idle

### Using the Basic Interface (Recommended)

1. Open `go_Echo.ipynb`
2. Run all cells
3. Enter the desired recording duration
4. Speak into your microphone
5. View the transcription results

## Model Information

The project uses a fine-tuned version of OpenAI's Whisper model, specifically optimized for:

- Clear speech recognition
- Improved accuracy for various speech patterns
- Real-time transcription capabilities

### Training

The model can be trained on custom datasets using `echo.py`. The training process includes:

- Audio preprocessing with 16kHz sampling rate
- Text tokenization using Whisper's processor
- Model fine-tuning with custom parameters
- Performance validation against baseline model

Training metrics include:

- Word Error Rate (WER)
- Character Error Rate (CER)
- Comparative analysis with baseline Whisper model
- Per-sample improvement tracking

## Project Structure

- `echo.py`: Core implementation and training code
- `Test_Echo_widget.ipynb`: Interactive Jupyter widget interface
- `Test_Echo.ipynb`: Basic testing interface
- `goEcho.ipynb`: Google Colab compatible notebook
- `model/`: Directory for storing the fine-tuned model
- `requirements.txt`: Project dependencies

## For Developers

The project offers comprehensive tools for model development and testing:

### Model Training

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-finetuned",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    max_steps=1000,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    predict_with_generate=True
)
```

### Performance Testing

The system includes tools for:

- Model performance comparison with baseline Whisper
- Detailed error rate analysis
- Visual performance metrics
- Real-time audio processing validation

### Key Components

- Audio processing (16kHz, mono channel)
- Real-time visualization
- Custom data preprocessing
- Model validation suite

## Support and Contribution

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For issues and feature requests, please use the GitHub issue tracker.

## Acknowledgments

- Based on OpenAI's Whisper model
- Uses HuggingFace's Transformers library
