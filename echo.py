# %% [markdown]
# <a href="https://colab.research.google.com/github/nsasto/echo/blob/main/Echo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # 📢Echo

# %%
# if not installed run
#!pip install -U transformers

# %% [markdown]
# Model page: https://huggingface.co/openai/whisper-small
# 
# ⚠️ If the generated code snippets do not work, please open an issue on either the [model repo](https://huggingface.co/openai/whisper-small)
# 			and/or on [huggingface.js](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts) 🙏

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%


# %%
from google.colab import userdata

hf_token = userdata.get('HF_TOKEN')

# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# %% [markdown]
# Load Libraries

# %%
import os
import torch
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import torch

# %%
# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")

# %% [markdown]
# ### Configuration

# %%
CSV_PATH = "data.csv"
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "text"
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "./whisper-small-finetuned"
MAX_STEPS = 1000  # adjust based on your dataset
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
USE_FP16 = torch.cuda.is_available()


# %% [markdown]
# ### Load dataset

# %%
import pandas as pd
import os
# Ensure you have the 'datasets' library installed: pip install datasets
from datasets import Dataset, Audio

# Define constants for file paths and column names
# IMPORTANT: You will need to create the './audio_data' directory
# and place dummy .m4a files (e.g., data001.m4a, data002.m4a, etc.) inside it
# for the audio file existence check to pass and for the Audio feature to load correctly.
# Updated DATA_PATH based on the FileNotFoundError traceback
DATA_PATH = "/content/drive/MyDrive/model_training/training_data"
AUDIO_COLUMN = 'Filename' # This column in the DataFrame will store paths to audio files
TEXT_COLUMN = 'Phrases'   # This column in the DataFrame will store associated text
TRAINING_PATH = "/content/training"

# Data provided by the user to create the initial DataFrame
data = {
    'Filename': ['data001.m4a', 'data002.m4a', 'data003.m4a', 'Data004.m4a',
                 'data005.m4a', 'data006.m4a', 'data007.m4a', 'data008.m4a',
                 'data009.m4a', 'data010.m4a', 'data011.m4a', 'data012.m4a',
                 'data013.m4a', 'data014.m4a', 'data015.m4a', 'data016.mp3',
                 'data017.mp3', 'data018.mp3', 'data019.mp3', 'data020.mp3',
                 'data021.mp3', 'data022.mp3', 'data023.mp3', 'data024.mp3',
                 'data025.mp3', 'data026.mp3', 'data027.mp3', 'data028.mp3',
                 'data029.mp3', 'data030.mp3'],
    'Phrases': [
        'I’d like to book an appointment, please.',
        'Do you have anything available this week?',
        'Can I come in on Tuesday morning?',
        'What times are available for Friday?',
        'I need to see a doctor urgently.',
        'Please book me in for the earliest slot.',
        'Can I book a virtual appointment?',
        'I prefer afternoons if possible.',
        'I’d like to cancel my appointment.',
        'I need to reschedule.',
        'Can I move my appointment to next week?',
        'Is 3 PM still okay?',
        'Just calling to confirm my appointment.',
        'I’d like to double-check the time.',
        'I forgot the date—can you remind me?',
        'Will you send me a confirmation text?',
        'Do you have any other slots?',
        'I have a change of plans.',
        'Can we push it to later in the day?',
        'Please leave a message for them.',
        'Can you tell them I called?',
        'I’ll try again later.',
        'Please make sure that everything is alright.',
        'Please can I buy this?',
        'What time will you be home?',
        'Please can I book a flight?',
        'Sorry that I cannot make it.',
        'Are you available to meet up?',
        'What would you like for dinner?',
        'What are you doing tomorrow?'
    ]
}

# Create the DataFrame from the dictionary
df = pd.DataFrame(data)
print(df)

# %%

# Create the DataFrame from the dictionary
df = pd.DataFrame(data)

# Prepend the DATA_PATH and append the .m4a extension to the filenames.
# This assumes the 'Filename' column initially contains only the base name (e.g., 'data001').
df[AUDIO_COLUMN] = df[AUDIO_COLUMN].apply(lambda x: os.path.join(DATA_PATH, f"{x}"))

# Print a loading message
print("🔹 checking all files exist ...")

# Check if the required columns exist in the DataFrame
if AUDIO_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
    raise ValueError(f"DataFrame must have '{AUDIO_COLUMN}' and '{TEXT_COLUMN}' columns.")

# Check that all audio files specified in the DataFrame actually exist on disk.
# This is crucial before attempting to load them into the Dataset.
missing_files = [path for path in df[AUDIO_COLUMN] if not os.path.exists(path)]
if missing_files:
    # If files are missing, raise an error with a list of the missing paths.
    # For testing, you may need to manually create empty .m4a files in the DATA_PATH directory
    # (e.g., touch ./audio_data/data001.m4a).
    raise FileNotFoundError(f"Missing audio files: {missing_files}. Please ensure these files exist in the '{DATA_PATH}' directory.")
else:
  print("✅ All files exist.")


# %%
# Print a loading message
print("🔹 converting to wav ...")

for filename in df[AUDIO_COLUMN]:
    input_path = os.path.join(DATA_PATH, filename)
    #output filename should have only filename, no path
    output_filename = os.path.basename(filename)
    output_filename = os.path.splitext(output_filename)[0] + ".wav"
    output_path = os.path.join(TRAINING_PATH, output_filename)
    #if TRAINING_PATH doesnt exist, create it
    if not os.path.exists(TRAINING_PATH):
        os.makedirs(TRAINING_PATH)

    output_path = os.path.join(TRAINING_PATH, output_filename)
    print(output_path)
    # Use ffmpeg to convert to WAV silently
    if os.path.exists(input_path):
        !ffmpeg -i "$input_path" -acodec pcm_s16le -ar 16000 -ac 1 "$output_path" -y -loglevel quiet
    else:
        print(f"Warning: Input file not found: {input_path}")

print(f"\n✅ Conversion complete. Converted files are in: {TRAINING_PATH}")

# %%
# for each file in df[AUDIO_COLUMN], change file extension to wav and folder to TRAINING_PATH
# heres an example of filename   /content/drive/MyDrive/model_training/training_data/data001.m4a
updated_filenames = []
for filename in df[AUDIO_COLUMN]:
    # Extract the base filename without the original path and extension
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    # Construct the new path with the TRAINING_PATH and .wav extension
    new_path = os.path.join(TRAINING_PATH, base_filename + ".wav")
    updated_filenames.append(new_path)

# Update the 'Filename' column in the DataFrame with the new paths
df[AUDIO_COLUMN] = updated_filenames

display(df)

# %%
#
# Convert the Pandas DataFrame into a Hugging Face Dataset object.
dataset = Dataset.from_pandas(df)

# Cast the audio column to the Audio feature type.
# This will load the actual audio data when an item from the dataset is accessed.
# sampling_rate=16000 is a common standard for audio processing.
dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))

# Print a confirmation and a preview of the created dataset
print("\nDataset created successfully:")
print(dataset)
print("\nFirst entry of the dataset (note the 'audio' feature loaded):")
print(dataset[0])

# %% [markdown]
# ### Model and processor

# %%
print("🔹 Loading Whisper model and processor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)



# %% [markdown]
# ## Pre-processing
# 
# This function, preprocess, will be used to prepare each audio-text pair in the dataset for training the speech-to-text model. For each data sample (or batch), it will perform the following steps:
# 
# 1. Extract the audio data from the batch using the key specified by the AUDIO_COLUMN variable (which is "audio"). The extracted audio variable contains a dictionary with the audio sample's raw data ("array") and its sampling rate ("sampling_rate").
# 2. Uses the processor object, which is associated with the specific speech-to-text model being used, to transform the raw audio data into a format that the model can understand.  This transformation includes processing the audio array at the specified sampling rate and formatting the output as PyTorch tensors (return_tensors="pt").
# 4. Extracts the primary input features generated by the processor and adds them to the batch under the key "input_features". The [0] is used to select the first element, assuming the processor output is a batch of size one for a single input sample.
# 5. Uses the tokenizer associated with the processor to convert the text transcript for the audio sample (accessed using the TEXT_COLUMN variable, which is "text") into a sequence of numerical IDs. These IDs represent the tokenized version of the text, which serves as the target output for the model during training. The resulting list of IDs is added to the batch under the key "labels".
# 6. Finally, the function returns the modified batch dictionary, which now includes the input_features and labels necessary for training, in addition to the original data.

# %%
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

print("🔹 Setting training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
    predict_with_generate=True,
    fp16=USE_FP16,
    report_to="none",
)

# %%
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received and labels.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Fixed preprocessing function
def preprocess(batch):
    """
    Preprocess function for Whisper fine-tuning
    """
    # Extract audio data
    audio = batch[AUDIO_COLUMN]

    # Process audio input
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )

    # For Whisper, we need input_features
    batch["input_features"] = inputs.input_features[0]

    # Process text labels using the tokenizer directly
    labels = processor.tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding=False,
        return_tensors="pt"
    ).input_ids[0]

    # Convert to list for easier handling
    batch["labels"] = labels.tolist()

    return batch

# Apply preprocessing
print("🔹 Preprocessing audio and text...")
dataset = dataset.map(preprocess, remove_columns=[AUDIO_COLUMN, TEXT_COLUMN])

# Create the custom data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Update your trainer with the correct data collator
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(min(5, len(dataset)))),
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)

print("✅ Trainer setup complete with custom data collator")
print("🔹 Starting training...")
# Now you can run trainer.train()

# %% [markdown]
# ### Training setup
# 
# Define the training arguments using the Seq2SeqTrainingArguments class from the transformers library. We can tweak these later to see if it improves results but will go for standard initially

# %% [markdown]
# ### Train

# %%
trainer.train()

# %% [markdown]
# 

# %%

print(f"✅ Training complete. Saving to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("✅ Model saved and ready to use!")

# %%
print("\nDataset columns after preprocessing:")
print(dataset.column_names)

print("\nSample dataset entry after preprocessing:")
print(dataset[0])

# %% [markdown]
# # Let's show results
# 
# 📈 Visualizations:
# 
# * WER Comparison: Bar chart comparing baseline vs fine-tuned WER
# * CER Comparison: Bar chart comparing baseline vs fine-tuned CER
# * Improvement Distribution: Shows per-sample improvements (green=better, red=worse)
# * Overall Statistics: Summary of average error rates
# 
# 
# 

# %%
#!pip install jiwer

# %%
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import librosa
import numpy as np
from jiwer import wer, cer  # pip install jiwer for Word Error Rate calculation
import warnings
import os
warnings.filterwarnings('ignore')

def test_model_on_validation(model_path, processor, test_audio_files, ground_truth_texts, max_samples=10):
    """
    Test the fine-tuned Whisper model on validation data and visualize results

    Args:
        model_path: Path to the fine-tuned model
        processor: WhisperProcessor object
        test_audio_files: List of audio file paths for testing
        ground_truth_texts: List of ground truth transcriptions
        max_samples: Maximum number of samples to test
    """

    # First check if files exist
    print("🔹 Checking audio files...")
    valid_files = []
    valid_texts = []

    for audio_file, text in zip(test_audio_files[:max_samples], ground_truth_texts[:max_samples]):
        if os.path.exists(audio_file):
            valid_files.append(audio_file)
            valid_texts.append(text)
        else:
            print(f"⚠️ File not found: {audio_file}")

    if not valid_files:
        print("❌ No valid audio files found!")
        return pd.DataFrame()

    print(f"✅ Found {len(valid_files)} valid audio files")

    # Load the fine-tuned model and processor
    print("🔹 Loading fine-tuned model...")
    try:
        # Load model and processor directly
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)

        # Create pipeline with specific configuration to avoid forced_decoder_ids conflict
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"forced_decoder_ids": None}  # Explicitly set to None
        )
        print("✅ Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading fine-tuned model: {e}")
        return pd.DataFrame()

    # Also create a baseline model for comparison
    print("🔹 Loading baseline Whisper model...")
    try:
        baseline_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"forced_decoder_ids": None}  # Explicitly set to None
        )
        print("✅ Baseline model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading baseline model: {e}")
        return pd.DataFrame()

    # Store results
    results = []

    print(f"🔹 Testing on {len(valid_files)} samples...")

    for i, (audio_file, true_text) in enumerate(zip(valid_files, valid_texts)):
        print(f"Processing sample {i+1}/{len(valid_files)}: {os.path.basename(audio_file)}")

        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_file, sr=16000)

            # Generate predictions
            fine_tuned_result = pipe(audio)
            baseline_result = baseline_pipe(audio)

            fine_tuned_text = fine_tuned_result['text'].strip()
            baseline_text = baseline_result['text'].strip()

            # Calculate metrics (handle empty predictions)
            if fine_tuned_text and baseline_text and true_text:
                ft_wer = wer(true_text, fine_tuned_text)
                baseline_wer = wer(true_text, baseline_text)
                ft_cer = cer(true_text, fine_tuned_text)
                baseline_cer = cer(true_text, baseline_text)
            else:
                print(f"⚠️ Empty prediction for sample {i+1}")
                ft_wer = baseline_wer = ft_cer = baseline_cer = 1.0

            # Store results
            results.append({
                'sample_id': i + 1,
                'audio_file': os.path.basename(audio_file),
                'ground_truth': true_text,
                'fine_tuned_prediction': fine_tuned_text,
                'baseline_prediction': baseline_text,
                'fine_tuned_wer': ft_wer,
                'baseline_wer': baseline_wer,
                'fine_tuned_cer': ft_cer,
                'baseline_cer': baseline_cer,
                'wer_improvement': baseline_wer - ft_wer,
                'cer_improvement': baseline_cer - ft_cer
            })

            print(f"  ✅ Ground truth: {true_text}")
            print(f"  🔵 Fine-tuned:  {fine_tuned_text}")
            print(f"  🔴 Baseline:    {baseline_text}")
            print(f"  📊 WER: {ft_wer:.3f} vs {baseline_wer:.3f}")

        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            continue

    if not results:
        print("❌ No successful predictions!")
        return pd.DataFrame()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    print("\n" + "="*80)
    print("🎯 VALIDATION RESULTS")
    print("="*80)

    # Summary statistics
    avg_ft_wer = results_df['fine_tuned_wer'].mean()
    avg_baseline_wer = results_df['baseline_wer'].mean()
    avg_ft_cer = results_df['fine_tuned_cer'].mean()
    avg_baseline_cer = results_df['baseline_cer'].mean()

    print(f"\n📊 OVERALL METRICS:")
    print(f"Fine-tuned WER: {avg_ft_wer:.3f}")
    print(f"Baseline WER:   {avg_baseline_wer:.3f}")
    print(f"WER Improvement: {avg_baseline_wer - avg_ft_wer:.3f}")
    print(f"\nFine-tuned CER: {avg_ft_cer:.3f}")
    print(f"Baseline CER:   {avg_baseline_cer:.3f}")
    print(f"CER Improvement: {avg_baseline_cer - avg_ft_cer:.3f}")

    # Show sample improvements
    improvements = results_df['wer_improvement'].tolist()
    better_count = sum(1 for x in improvements if x > 0)
    worse_count = sum(1 for x in improvements if x < 0)
    same_count = sum(1 for x in improvements if x == 0)

    print(f"\n🎯 Performance Summary:")
    print(f"  Improved: {better_count}/{len(improvements)} samples ({better_count/len(improvements)*100:.1f}%)")
    print(f"  Worse: {worse_count}/{len(improvements)} samples ({worse_count/len(improvements)*100:.1f}%)")
    print(f"  Same: {same_count}/{len(improvements)} samples ({same_count/len(improvements)*100:.1f}%)")

    return results_df

def visualize_results(results_df):
    """
    Create visualizations of the validation results
    """
    if results_df.empty:
        print("❌ No results to visualize!")
        return

    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Whisper Fine-tuning Validation Results', fontsize=16, fontweight='bold')

    # 1. WER Comparison
    ax1 = axes[0, 0]
    x = range(len(results_df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], results_df['baseline_wer'], width,
            label='Baseline', alpha=0.7, color='lightcoral')
    ax1.bar([i + width/2 for i in x], results_df['fine_tuned_wer'], width,
            label='Fine-tuned', alpha=0.7, color='lightblue')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Word Error Rate (WER)')
    ax1.set_title('WER Comparison: Baseline vs Fine-tuned')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{i+1}' for i in x])

    # 2. CER Comparison
    ax2 = axes[0, 1]
    ax2.bar([i - width/2 for i in x], results_df['baseline_cer'], width,
            label='Baseline', alpha=0.7, color='lightcoral')
    ax2.bar([i + width/2 for i in x], results_df['fine_tuned_cer'], width,
            label='Fine-tuned', alpha=0.7, color='lightblue')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Character Error Rate (CER)')
    ax2.set_title('CER Comparison: Baseline vs Fine-tuned')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'S{i+1}' for i in x])

    # 3. Improvement Distribution
    ax3 = axes[1, 0]
    improvements = results_df['wer_improvement']
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
    bars = ax3.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('WER Improvement')
    ax3.set_title('Per-Sample WER Improvement\n(Positive = Better, Negative = Worse)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(len(improvements)))
    ax3.set_xticklabels([f'S{i+1}' for i in range(len(improvements))])

    # Add value labels on bars
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    # 4. Overall Statistics
    ax4 = axes[1, 1]
    metrics = ['WER', 'CER']
    baseline_scores = [results_df['baseline_wer'].mean(), results_df['baseline_cer'].mean()]
    finetuned_scores = [results_df['fine_tuned_wer'].mean(), results_df['fine_tuned_cer'].mean()]

    x_pos = np.arange(len(metrics))
    bars1 = ax4.bar(x_pos - 0.2, baseline_scores, 0.4, label='Baseline', alpha=0.7, color='lightcoral')
    bars2 = ax4.bar(x_pos + 0.2, finetuned_scores, 0.4, label='Fine-tuned', alpha=0.7, color='lightblue')
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Error Rate')
    ax4.set_title('Average Error Rates')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add values on bars
    for bars, scores in zip([bars1, bars2], [baseline_scores, finetuned_scores]):
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    # Create a detailed comparison table
    print("\n📋 DETAILED RESULTS TABLE:")
    display_df = results_df[['sample_id', 'audio_file', 'ground_truth', 'fine_tuned_prediction',
                            'fine_tuned_wer', 'baseline_wer', 'wer_improvement']].copy()
    display_df.columns = ['ID', 'File', 'Ground Truth', 'Fine-tuned Prediction',
                         'FT WER', 'Base WER', 'WER Δ']
    display_df = display_df.round(3)

    # Truncate long text for better display
    display_df['Ground Truth'] = display_df['Ground Truth'].str[:40] + '...' if any(len(str(x)) > 40 for x in display_df['Ground Truth']) else display_df['Ground Truth']
    display_df['Fine-tuned Prediction'] = display_df['Fine-tuned Prediction'].str[:40] + '...' if any(len(str(x)) > 40 for x in display_df['Fine-tuned Prediction']) else display_df['Fine-tuned Prediction']

    print(display_df.to_string(index=False, max_colwidth=45))

def transcribe_with_model(model_path, audio_file, processor=None):
    """
    Simple function to transcribe a single audio file with the fine-tuned model
    """
    try:
        # Load model and processor
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        if processor is None:
            processor = WhisperProcessor.from_pretrained(model_path)

        # Create pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"forced_decoder_ids": None}
        )

        # Load and transcribe audio
        audio, sr = librosa.load(audio_file, sr=16000)
        result = pipe(audio)

        return result['text'].strip()

    except Exception as e:
        print(f"Error transcribing {audio_file}: {e}")
        return None

# Example usage with your validation data
def run_validation_test(model_path, processor=None):
    """
    Run the validation test using your training data as a quick test
    """
    # Use some of your training data for quick testing
    # In practice, you'd want separate validation data
    test_files = [
        '/content/training/data001.wav',
        '/content/training/data002.wav',
        '/content/training/data003.wav',
        '/content/training/data005.wav'  # Skip data004.wav as it wasn't found
    ]

    test_texts = [
        "I'd like to book an appointment, please.",
        "Do you have anything available this week?",
        "Can I come in on Tuesday morning?",
        "I need to see a doctor urgently."
    ]

    # Test the model
    results = test_model_on_validation(
        model_path=model_path,
        processor=processor,
        test_audio_files=test_files,
        ground_truth_texts=test_texts,
        max_samples=len(test_files)
    )

    # Visualize results if we have any
    if not results.empty:
        visualize_results(results)

    return results

# Alternative method using direct model inference (if pipeline still has issues)
def test_with_direct_inference(model_path, test_audio_files, ground_truth_texts):
    """
    Alternative testing method using direct model inference
    """
    print("🔹 Using direct model inference method...")

    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)

    if torch.cuda.is_available():
        model = model.to("cuda")

    results = []

    for i, (audio_file, true_text) in enumerate(zip(test_audio_files, ground_truth_texts)):
        if not os.path.exists(audio_file):
            continue

        try:
            # Load and process audio
            audio, sr = librosa.load(audio_file, sr=16000)

            # Process with the processor
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

            if torch.cuda.is_available():
                input_features = input_features.to("cuda")

            # Generate prediction
            with torch.no_grad():
                predicted_ids = model.generate(input_features)

            # Decode prediction
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            # Calculate WER
            wer_score = wer(true_text, transcription) if transcription and true_text else 1.0

            results.append({
                'sample_id': i + 1,
                'audio_file': os.path.basename(audio_file),
                'ground_truth': true_text,
                'prediction': transcription,
                'wer': wer_score
            })

            print(f"Sample {i+1}: WER = {wer_score:.3f}")
            print(f"  True: {true_text}")
            print(f"  Pred: {transcription}")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

    return pd.DataFrame(results)

# Usage example:
# Replace 'OUTPUT_DIR' with your actual model path
# results = run_validation_test(OUTPUT_DIR, processor)

# %%

results = run_validation_test(OUTPUT_DIR, processor)

# %%



