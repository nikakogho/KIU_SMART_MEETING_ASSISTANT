import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
ANALYSIS_FILE = "meeting_analysis.json"
OUTPUT_IMAGE_FILE = "meeting_visual_summary.png"
PROMPT_GENERATION_MODEL = "gpt-4-turbo"
IMAGE_GENERATION_MODEL = "dall-e-3"

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
            model=PROMPT_GENERATION_MODEL,
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

def main():
    """
    Main function to run the visual synthesis process.
    """
    load_dotenv()
    client = OpenAI()

    # 1. Load the analysis file from Phase 2
    if not os.path.exists(ANALYSIS_FILE):
        print(f"❌ Error: Analysis file '{ANALYSIS_FILE}' not found.")
        return

    with open(ANALYSIS_FILE, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    # 2. Generate a creative DALL-E prompt
    dalle_prompt = create_dalle_prompt(analysis, client)

    # 3. Generate and save the final image
    if dalle_prompt:
        generate_and_save_image(dalle_prompt, client, OUTPUT_IMAGE_FILE)

if __name__ == '__main__':
    main()