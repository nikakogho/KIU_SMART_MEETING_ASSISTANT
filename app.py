import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from services import diarize_audio, transcribe_with_word_timestamps, combine_transcript_and_diarize_results, analyze_meeting_transcript
from services import create_embedding, cosine_similarity, generate_answer, create_dalle_prompt, generate_and_save_image, convert_audio_to_wav
from openai import OpenAI
import json

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key' # Needed for flash messages

# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # --- START: AI Processing Pipeline ---
            flash('Processing started... This may take several minutes.')
            
            try:
                # 0. Convert audio to WAV if necessary
                if not filename.lower().endswith('.wav'):
                    filepath = convert_audio_to_wav(filepath)

                # 1. Process Audio (Diarize + Transcribe)
                print("Step 1: Processing Audio...")
                diarization = diarize_audio(filepath)
                transcription = transcribe_with_word_timestamps(filepath)
                full_transcript = combine_transcript_and_diarize_results(diarization, transcription)
                
                # 2. Analyze Content (Summarize, etc.)
                print("Step 2: Analyzing Content...")
                client = OpenAI()
                analysis_data = analyze_meeting_transcript(full_transcript, client)

                # 3. Create Visuals
                print("Step 3: Creating Visuals...")
                # The image must be saved in the 'static' folder to be displayed on the web page.
                image_filename = "visual_summary.png"
                image_save_path = os.path.join('static', image_filename)
                
                dalle_prompt = create_dalle_prompt(analysis_data, client)
                generate_and_save_image(dalle_prompt, client, image_save_path)

                # 4. Index for Search (Simplified for integration)
                print("Step 4: Indexing for Search...")
                summary = analysis_data.get("summary", "")
                decisions = " ".join(analysis_data.get("decisions_made", []))
                text_for_embedding = f"Summary: {summary}\n\nDecisions: {decisions}"
                embedding = create_embedding(text_for_embedding, client)
                
                # Load/Save to Knowledge Base
                kb_file = "meetings_kb.json"
                if os.path.exists(kb_file):
                    with open(kb_file, 'r', encoding='utf-8') as f:
                        knowledge_base = json.load(f)
                else:
                    knowledge_base = []
                
                new_entry = {"meeting_id": filename, "text": text_for_embedding, "embedding": embedding}
                knowledge_base.append(new_entry)
                with open(kb_file, 'w', encoding='utf-8') as f:
                    json.dump(knowledge_base, f, indent=4)

                # 5. Prepare results for the template
                print("Step 5: Preparing Results...")
                results = {
                    "transcript": full_transcript,
                    "summary": analysis_data.get("summary", "N/A"),
                    "decisions": analysis_data.get("decisions_made", []),
                    "action_items": analysis_data.get("action_items", []),
                    "image_url": url_for('static', filename=image_filename) # Generate URL for the image
                }
                
                flash('Processing complete!')
                return render_template('results.html', results=results)

            except Exception as e:
                flash(f'An error occurred: {e}')
                return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)