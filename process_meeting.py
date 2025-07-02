import os
import torch
from dotenv import load_dotenv
from openai import OpenAI
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import wave
import contextlib
from datetime import timedelta

# --- Setup ---
# Load environment variables
load_dotenv()

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Main Functions ---

def diarize_audio(audio_path: str) -> Annotation:
    """
    Performs speaker diarization on an audio file.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        Annotation: A pyannote Annotation object with speaker segments.
    """
    print("Step 1: Starting speaker diarization...")
    # Make sure you have accepted the user conditions on Hugging Face Hub
    # for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your .env file.")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    pipeline.to(torch.device(DEVICE))
    
    # Get the duration of the audio file to show progress
    with contextlib.closing(wave.open(audio_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration_seconds = frames / float(rate)
    
    print(f"Audio duration: {timedelta(seconds=duration_seconds)}")
    
    diarization_result = pipeline(audio_path)
    print("Diarization complete.")
    return diarization_result


def transcribe_with_word_timestamps(audio_path: str) -> dict:
    """
    Transcribes an audio file using OpenAI's Whisper API with word-level timestamps.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        dict: The full transcription result from Whisper.
    """
    print("Step 2: Starting transcription with Whisper...")
    client = OpenAI() # API key is loaded automatically from OPENAI_API_KEY env var
    with open(audio_path, "rb") as audio_file:
        transcript_result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    print("Transcription complete.")
    return transcript_result


def combine_results(diarization: Annotation, transcription: dict) -> str:
    """
    Combines diarization and transcription results into a formatted string.

    Args:
        diarization (Annotation): The result from pyannote.
        transcription (dict): The result from Whisper.

    Returns:
        str: A formatted string of the transcript with speaker labels.
    """
    print("Step 3: Combining diarization and transcription...")
    final_transcript = ""
    transcribed_words = transcription.words
    
    # Get speaker turns from the diarization result
    speaker_turns = list(diarization.itertracks(yield_label=True))
    
    # Map words to speakers
    word_index = 0
    for turn, _, speaker in speaker_turns:
        # Format the timestamp
        start_time = timedelta(seconds=turn.start)
        formatted_time = f"[{int(start_time.total_seconds() // 60):02d}:{int(start_time.total_seconds() % 60):02d}]"
        
        final_transcript += f"\n{formatted_time} {speaker}:"
        
        # Find all words that fall within the current speaker's turn
        while word_index < len(transcribed_words):
            word_info = transcribed_words[word_index]
            # Check if the start of the word is within the current speaker's turn
            if word_info.start >= turn.start and word_info.start <= turn.end:
                final_transcript += f" {word_info.word}"
                word_index += 1
            # If the word starts after the current turn, break to move to the next speaker
            elif word_info.start > turn.end:
                break
            # If the word starts before the turn, it might be an unassigned word; skip it.
            else:
                 word_index += 1
                 
    print("Combination complete.")
    return final_transcript.strip()


if __name__ == '__main__':
    # --- Configuration ---
    # Make sure to place a meeting audio file here. It must be a .wav file for pyannote.
    AUDIO_FILE_PATH = "sample_voice.wav" 

    # --- Execution ---
    # 1. Diarize to find out who spoke when
    speaker_diarization = diarize_audio(AUDIO_FILE_PATH)
    
    # 2. Transcribe with word-level timestamps
    transcription_result = transcribe_with_word_timestamps(AUDIO_FILE_PATH)
    
    # 3. Combine the results
    final_output = combine_results(speaker_diarization, transcription_result)
    
    # 4. Print and save the final transcript
    print("\n--- FINAL TRANSCRIPT ---\n")
    print(final_output)
    
    with open("final_transcript.txt", "w") as f:
        f.write(final_output)
    
    print("\nFinal transcript saved to 'final_transcript.txt'")