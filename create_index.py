import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
ANALYSIS_FILE = "meeting_analysis.json"  # Input from Phase 2
KNOWLEDGE_BASE_FILE = "meetings_kb.json" # Output for this phase
EMBEDDING_MODEL = "text-embedding-3-small"

def create_embedding(text_to_embed: str, client: OpenAI) -> list[float]:
    """Creates an embedding vector for a given text."""
    response = client.embeddings.create(
        input=text_to_embed,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def main():
    """
    Main function to create and store embeddings for meeting analyses.
    """
    load_dotenv()
    client = OpenAI()

    # 1. Check if the analysis file exists
    if not os.path.exists(ANALYSIS_FILE):
        print(f"❌ Error: Analysis file not found at '{ANALYSIS_FILE}'")
        print("Please run the Phase 2 script first.")
        return

    # 2. Read the meeting analysis data
    with open(ANALYSIS_FILE, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    # 3. Prepare the text for embedding
    # We'll combine the summary and decisions for a comprehensive embedding.
    summary = analysis_data.get("summary", "")
    decisions = " ".join(analysis_data.get("decisions_made", []))
    text_for_embedding = f"Summary: {summary}\n\nDecisions: {decisions}"
    
    if not text_for_embedding.strip():
        print("❌ Error: No text found in the analysis file to create an embedding.")
        return

    print(f"✅ Text prepared for embedding from '{ANALYSIS_FILE}'.")
    print("🤖 Creating embedding...")

    # 4. Create the embedding
    try:
        embedding = create_embedding(text_for_embedding, client)
        print("🎉 Embedding created successfully.")
    except Exception as e:
        print(f"❌ Error creating embedding: {e}")
        return

    # 5. Load existing knowledge base or create a new one
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
    else:
        knowledge_base = []
    
    # 6. Add the new meeting data to the knowledge base
    # Using the analysis filename as a unique ID for this meeting
    new_entry = {
        "meeting_id": ANALYSIS_FILE,
        "text": text_for_embedding,
        "embedding": embedding
    }
    
    # Avoid duplicate entries
    knowledge_base = [entry for entry in knowledge_base if entry["meeting_id"] != new_entry["meeting_id"]]
    knowledge_base.append(new_entry)

    # 7. Save the updated knowledge base
    with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=4)
    
    print(f"✅ Meeting knowledge base updated and saved to '{KNOWLEDGE_BASE_FILE}'.")


if __name__ == '__main__':
    main()