from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

audio_file = open("sample_voice.mp3", "rb")

transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  response_format="verbose_json"
)

print(transcript.text)