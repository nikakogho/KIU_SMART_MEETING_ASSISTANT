import os
import json
import uuid # Add this import
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from openai import OpenAI

# Import all your service functions
from services import (
    diarize_audio, transcribe_with_word_timestamps, combine_transcript_and_diarize_results,
    analyze_meeting_transcript, create_dalle_prompt, generate_and_save_image,
    create_embedding, answer_from_meetings, convert_audio_to_wav
)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ANALYSIS_FOLDER = 'analyses' # To store analysis JSONs
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'

# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            job_id = str(uuid.uuid4()) # << CREATE A UNIQUE ID FOR THIS JOB
            original_filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(filepath)

            flash('Processing started... This may take several minutes.')

            try:
                # 1. Convert audio to WAV if necessary
                wav_filepath = convert_audio_to_wav(filepath)

                # 2. Process Audio
                diarization = diarize_audio(wav_filepath)
                transcription = transcribe_with_word_timestamps(wav_filepath)
                full_transcript = combine_transcript_and_diarize_results(diarization, transcription)

                # 3. Analyze Content
                client = OpenAI()
                analysis_data = analyze_meeting_transcript(full_transcript, client)
                
                # Save the analysis JSON with the unique job_id
                analysis_filepath = os.path.join(ANALYSIS_FOLDER, f"{job_id}.json")
                with open(analysis_filepath, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=4)

                # 4. Create Visuals
                image_filename = f"{job_id}.png"
                image_save_path = os.path.join('static', image_filename)
                dalle_prompt = create_dalle_prompt(analysis_data, client)
                generate_and_save_image(dalle_prompt, client, image_save_path)

                # 5. Index for Search
                summary = analysis_data.get("summary", "")
                decisions = " ".join(analysis_data.get("decisions_made", []))
                effectiveness_rating = analysis_data.get("effectiveness_rating", "Not specified")
                text_for_embedding = f"Summary: {summary}\n\nDecisions: {decisions}\n\nEffectiveness Rating: {effectiveness_rating}"
                embedding = create_embedding(text_for_embedding, client)

                kb_file = "meetings_kb.json"
                knowledge_base = []
                if os.path.exists(kb_file):
                    with open(kb_file, 'r', encoding='utf-8') as f:
                        knowledge_base = json.load(f)
                
                # Use the job_id as the unique identifier
                new_entry = {"meeting_id": job_id, "text": text_for_embedding, "embedding": embedding}
                knowledge_base.append(new_entry)
                with open(kb_file, 'w', encoding='utf-8') as f:
                    json.dump(knowledge_base, f, indent=4)

                # 6. Prepare results for the template
                results = {
                    "transcript": full_transcript,
                    "summary": summary,
                    "decisions": analysis_data.get("decisions_made", []),
                    "action_items": analysis_data.get("action_items", []),
                    "effectiveness_rating": effectiveness_rating,
                    "image_url": url_for('static', filename=image_filename)
                }
                
                flash('Processing complete!')
                return render_template('results.html', results=results)

            except Exception as e:
                flash(f'An error occurred: {e}')
                return redirect(request.url)

    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('upload_and_process')) # Redirect home if no query
    
    client = OpenAI()
    answer = answer_from_meetings(query, client)
    
    return render_template('search_results.html', query=query, answer=answer)

if __name__ == '__main__':
    # Create necessary folders on startup
    for folder in [UPLOAD_FOLDER, ANALYSIS_FOLDER, 'static']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    app.run(debug=True)