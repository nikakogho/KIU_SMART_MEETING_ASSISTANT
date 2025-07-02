import os
import json
import requests
import torch
import numpy as np
from openai import OpenAI
from datetime import timedelta
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import wave
import contextlib
from pydub import AudioSegment

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"

load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSCRIBE_MODEL = "whisper-1"
IMAGE_GENERATION_MODEL = "dall-e-3"
LANGUAGE_MODEL = "gpt-4-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"

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
            model=TRANSCRIBE_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    print("Transcription complete.")
    return transcript_result

def combine_transcript_and_diarize_results(diarization: Annotation, transcription: dict) -> str:
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

def analyze_meeting_transcript(transcript_text: str, client: OpenAI) -> dict:
    """
    Analyzes a meeting transcript using GPT-4-turbo to extract summary, decisions, and action items.

    Args:
        transcript_text (str): The full text of the meeting transcript.

    Returns:
        dict: A dictionary containing the structured analysis.
    """
    # This is the core of the request. The system prompt defines the AI's role
    # and the exact JSON structure we want. This makes the output highly reliable.
    system_prompt = """
    You are an expert AI assistant specializing in analyzing business meeting transcripts.
    Your task is to process the provided transcript and extract the following information
    in a clean, structured JSON format.

    The JSON object must have the following three keys:
    1. "summary": A concise, one-paragraph summary of the meeting's purpose and key discussions.
    2. "decisions_made": A list of key decisions that were finalized during the meeting. Each item in the list should be a clear, unambiguous string. If no decisions were made, return an empty list [].
    3. "action_items": A list of tasks assigned to individuals. Each item in the list must be a JSON object with three keys: "task" (the specific action to be taken), "owner" (the person responsible), and "due_date" (the deadline, if mentioned). If an owner or due_date is not mentioned for a task, set its value to "Not specified". If no action items were assigned, return an empty list [].
    """

    print("🤖 Sending transcript to GPT-4 for analysis...")

    try:
        response = client.chat.completions.create(
            model=LANGUAGE_MODEL,
            # This crucial parameter tells the API to guarantee the output is valid JSON
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript_text}
            ]
        )
        
        # Extract the JSON string from the response
        analysis_json_string = response.choices[0].message.content
        
        # Convert the JSON string into a Python dictionary
        analysis_result = json.loads(analysis_json_string)
        
        print("✅ Analysis complete.")
        return analysis_result

    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None

def pretty_print_transcript_analysis(analysis: dict):
    """Prints the analysis in a human-readable format."""
    if not analysis:
        print("Cannot print analysis, as it is empty.")
        return

    print("\n--- MEETING ANALYSIS ---")
    
    # Print Summary
    print("\n## 📝 Summary")
    print(analysis.get("summary", "No summary provided."))
    
    # Print Decisions
    print("\n## ⚖️ Decisions Made")
    decisions = analysis.get("decisions_made", [])
    if decisions:
        for i, decision in enumerate(decisions, 1):
            print(f"{i}. {decision}")
    else:
        print("No specific decisions were recorded.")
        
    # Print Action Items
    print("\n## 🚀 Action Items")
    action_items = analysis.get("action_items", [])
    if action_items:
        for i, item in enumerate(action_items, 1):
            print(f"{i}. Task: {item.get('task', 'N/A')}")
            print(f"   Owner: {item.get('owner', 'N/A')}")
            print(f"   Due Date: {item.get('due_date', 'N/A')}\n")
    else:
        print("No action items were assigned.")
        
    print("------------------------\n")

def create_embedding(text_to_embed: str, client: OpenAI) -> list[float]:
    """Creates an embedding vector for a given text."""
    response = client.embeddings.create(
        input=text_to_embed,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2) -> float:
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def generate_answer(query: str, context: str, client: OpenAI) -> str:
    """
    Generates an answer using GPT-4 based on the provided context.
    """
    system_prompt = """
    You are an AI assistant who answers questions based *only* on the provided meeting context.
    - Your task is to analyze the user's question and the meeting text.
    - Formulate a direct and concise answer using only the information from the meeting text.
    - Do not use any external knowledge.
    - If the answer is not present in the provided context, you MUST respond with:
    "I could not find a specific answer to your question in the provided meeting text."
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Here is the meeting text:\n\n---\n{context}\n---\n\nPlease answer this question: {query}"
        }
    ]

    try:
        response = client.chat.completions.create(
            model=LANGUAGE_MODEL,
            messages=messages,
            temperature=0.2, # Lower temperature for more factual, less creative answers
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

def create_dalle_prompt(analysis_data: dict, client: OpenAI) -> str:
    """
    Uses GPT-4 to generate a descriptive DALL-E prompt from meeting analysis.
    """
    print("🤖 Using GPT-4 to generate a creative prompt for DALL-E...")

    # Convert the analysis data to a string for the prompt
    analysis_text = f"""
    Meeting Summary: {analysis_data.get('summary', 'N/A')}
    Decisions Made: {', '.join(analysis_data.get('decisions_made', []))}
    Action Items: {', '.join([item.get('task', '') for item in analysis_data.get('action_items', [])])}
    """

    system_prompt = """
    You are an AI assistant who is an expert at creating visual concepts.
    Your task is to transform a text-based meeting summary into a single, detailed, and creative prompt
    for an image generation model like DALL-E 3.

    The prompt should describe a single, coherent image.
    The image should be a metaphorical or symbolic representation of the meeting's key outcomes.
    Describe the desired art style (e.g., 'digital art infographic', 'minimalist vector illustration', 'abstract painting').
    Be specific about the visual elements, colors, and composition.
    Do not ask questions. Only output the final, descriptive prompt.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the meeting analysis:\n\n{analysis_text}"}
    ]

    try:
        response = client.chat.completions.create(
            model=LANGUAGE_MODEL,
            messages=messages,
            temperature=0.7,
        )
        creative_prompt = response.choices[0].message.content
        print("✅ Creative prompt generated.")
        return creative_prompt
    except Exception as e:
        print(f"❌ Error generating DALL-E prompt: {e}")
        return None

def generate_and_save_image(prompt: str, client: OpenAI, output_filename: str):
    """
    Generates an image using DALL-E 3 and saves it locally.
    """
    if not prompt:
        print("Cannot generate image without a prompt.")
        return

    print("🎨 Sending prompt to DALL-E 3 for image generation...")
    try:
        response = client.images.generate(
            model=IMAGE_GENERATION_MODEL,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        print("✅ Image generated. Downloading...")

        # Download the image from the URL
        image_response = requests.get(image_url)
        image_response.raise_for_status()  # Will raise an exception for bad status codes

        # Save the image to a file
        with open(output_filename, 'wb') as f:
            f.write(image_response.content)
        print(f"🖼️ Image saved successfully as '{output_filename}'")

    except Exception as e:
        print(f"❌ An error occurred during image generation or download: {e}")

def convert_audio_to_wav(filepath: str) -> str:
    """Converts an audio file to WAV format if it's not already."""
    path, ext = os.path.splitext(filepath)
    if ext.lower() == '.wav':
        return filepath # It's already a WAV file

    print(f"Converting {ext} to .wav...")
    try:
        sound = AudioSegment.from_file(filepath)
        wav_filepath = path + ".wav"
        sound.export(wav_filepath, format="wav")
        return wav_filepath
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        raise # Re-raise the exception to be caught by the main handler
