import json
from openai import OpenAI
from dotenv import load_dotenv

def analyze_meeting_transcript(transcript_text: str) -> dict:
    """
    Analyzes a meeting transcript using GPT-4-turbo to extract summary, decisions, and action items.

    Args:
        transcript_text (str): The full text of the meeting transcript.

    Returns:
        dict: A dictionary containing the structured analysis.
    """
    # Load environment variables (for the OpenAI API key)
    load_dotenv()
    client = OpenAI()

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
            model="gpt-4-turbo",
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

def pretty_print_analysis(analysis: dict):
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


if __name__ == '__main__':
    # The input file from Phase 1
    input_filename = "final_transcript.txt"
    # The output file for this phase
    output_filename = "meeting_analysis.json"

    try:
        print(f"Reading transcript from '{input_filename}'...")
        with open(input_filename, 'r', encoding='utf-8') as f:
            transcript = f.read()

        if transcript:
            analysis_data = analyze_meeting_transcript(transcript)
            
            if analysis_data:
                # Print the formatted results to the console
                pretty_print_analysis(analysis_data)
                
                # Save the structured JSON output to a file
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=4)
                print(f"✅ Structured analysis saved to '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
        print("Please make sure you have run the Phase 1 script first and the file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")