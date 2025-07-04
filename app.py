import os
import json
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from openai import OpenAI

# Import all your service functions from your 'services.py' file
# Ensure you have a 'services.py' file in the same directory with all the
# necessary functions defined.
from services import (
    chunk_and_transcribe_audio, diarize_audio, transcribe_with_word_timestamps, combine_transcript_and_diarize_results,
    analyze_meeting_transcript, create_dalle_prompt, generate_and_save_image,
    create_embedding, answer_from_meetings, convert_audio_to_wav
)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ANALYSIS_FOLDER = 'analyses'  # To store analysis JSONs
STATIC_FOLDER = 'static'      # To store generated images
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'

# --- Helper Function ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file was selected.')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a .wav, .mp3, or .m4a file.')
            return redirect(request.url)

        job_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(filepath)

        flash('Processing started... This may take several minutes.')
        client = OpenAI()  # Initialize the OpenAI client once

        try:
            flash('Step 1/7: Converting audio to WAV format...')
            wav_filepath = convert_audio_to_wav(filepath)
        except Exception as e:
            flash(f'An error occurred during audio conversion: {e}')
            return redirect(request.url)

        try:
            flash('Step 2/7: Identifying speakers (diarization)...')
            diarization = diarize_audio(wav_filepath)
        except Exception as e:
            flash(f'An error occurred during audio diarization: {e}')
            return redirect(request.url)

        try:
            flash('Step 3/7: Chunking and transcribing audio to text...')
            transcription = chunk_and_transcribe_audio(wav_filepath, client)
        except Exception as e:
            flash(f'An error occurred during audio transcription: {e}')
            return redirect(request.url)

        try:
            flash('Step 4/7: Combining transcription and speaker data...')
            full_transcript = combine_transcript_and_diarize_results(diarization, transcription)
        except Exception as e:
            flash(f'An error occurred while combining results: {e}')
            return redirect(request.url)

        try:
            flash('Step 5/7: Analyzing meeting content...')
            analysis_data = analyze_meeting_transcript(full_transcript, client)
            analysis_filepath = os.path.join(ANALYSIS_FOLDER, f"{job_id}.json")
            with open(analysis_filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=4)
        except Exception as e:
            flash(f'An error occurred during content analysis: {e}')
            return redirect(request.url)

        image_filename = f"{job_id}.png"
        image_url = None
        try:
            flash('Step 6/7: Generating visual summary...')
            image_save_path = os.path.join(STATIC_FOLDER, image_filename)
            dalle_prompt = create_dalle_prompt(analysis_data, client)
            generate_and_save_image(dalle_prompt, client, image_save_path)
            image_url = url_for('static', filename=image_filename)
        except Exception as e:
            flash(f'An error occurred during image generation: {e}')
            # This step is non-critical, so we can proceed without an image.
            # If you want to stop processing on image failure, use: return redirect(request.url)

        try:
            flash('Step 7/7: Indexing meeting for future search...')
            summary = analysis_data.get("summary", "")
            decisions = " ".join(analysis_data.get("decisions_made", []))
            effectiveness_rating = analysis_data.get("effectiveness_rating", "Not specified")
            text_for_embedding = f"Summary: {summary}\n\nDecisions: {decisions}\n\nEffectiveness Rating: {effectiveness_rating}"
            embedding = create_embedding(text_for_embedding, client)

            kb_file = "meetings_kb.json"
            knowledge_base = []
            if os.path.exists(kb_file):
                with open(kb_file, 'r', encoding='utf-8') as f:
                    try:
                        knowledge_base = json.load(f)
                    except json.JSONDecodeError:
                        knowledge_base = [] # Reset if file is corrupt or empty

            new_entry = {"meeting_id": job_id, "text": text_for_embedding, "embedding": embedding}
            knowledge_base.append(new_entry)
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=4)
        except Exception as e:
            flash(f'An error occurred while indexing the meeting: {e}')
            return redirect(request.url)

        results = {
            "transcript": full_transcript,
            "summary": analysis_data.get("summary", ""),
            "decisions": analysis_data.get("decisions_made", []),
            "action_items": analysis_data.get("action_items", []),
            "effectiveness_rating": analysis_data.get("effectiveness_rating", "Not specified"),
            "image_url": image_url
        }

        flash('Processing complete!')
        return render_template('results.html', results=results)

    return render_template('index.html')

@app.route('/search')
def search():
    """Handles search queries against the knowledge base."""
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('upload_and_process'))

    try:
        client = OpenAI()
        answer = answer_from_meetings(query, client)
        return render_template('search_results.html', query=query, answer=answer)
    except Exception as e:
        flash(f"An error occurred during search: {e}")
        return redirect(url_for('upload_and_process'))


if __name__ == '__main__':
    # Create necessary folders on startup
    for folder in [UPLOAD_FOLDER, ANALYSIS_FOLDER, STATIC_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    app.run(debug=True)