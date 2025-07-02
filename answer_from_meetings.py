import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
KNOWLEDGE_BASE_FILE = "meetings_kb.json"
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4-turbo"
SIMILARITY_THRESHOLD = 0.5

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
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.2, # Lower temperature for more factual, less creative answers
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

def main(search_query: str):
    """
    Main function to retrieve relevant context and generate a direct answer.
    """
    load_dotenv()
    client = OpenAI()

    # 1. Load the knowledge base
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        print(f"❌ Error: Knowledge base file '{KNOWLEDGE_BASE_FILE}' not found.")
        return

    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    if not knowledge_base:
        print("⚠️ Knowledge base is empty.")
        return

    # 2. Create an embedding for the search query
    try:
        response = client.embeddings.create(input=search_query, model=EMBEDDING_MODEL)
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"❌ Error creating query embedding: {e}")
        return

    # 3. Find the most relevant meeting (Retrieval)
    results = []
    for entry in knowledge_base:
        similarity = cosine_similarity(np.array(query_embedding), np.array(entry["embedding"]))
        results.append({"meeting_id": entry["meeting_id"], "text": entry["text"], "similarity": similarity})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    top_result = results[0] if results else None

    # 4. Check if the top result is relevant enough
    if not top_result or top_result["similarity"] < SIMILARITY_THRESHOLD:
        print("\n💡 I couldn't find a sufficiently relevant meeting to answer your question.")
        return

    print(f"✅ Found relevant context in '{top_result['meeting_id']}' (Similarity: {top_result['similarity']:.4f})")
    print("🤖 Generating a direct answer...")

    # 5. Generate the final answer (Augmented Generation)
    context_text = top_result["text"]
    final_answer = generate_answer(search_query, context_text, client)

    print("\n--- ANSWER ---")
    print(final_answer)
    print("--------------\n")

if __name__ == '__main__':
    # You can change the search query here to test different questions
    query = "was there a test meeting"
    main(query)