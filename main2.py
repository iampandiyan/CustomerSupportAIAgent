import os
import asyncio
import threading
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
import httpx
from google.cloud import speech
from google.cloud import texttospeech
import pyaudio # <-- Use PyAudio for streaming playback

#With Single Audio
# Load environment variables
load_dotenv()

app = FastAPI()

# --- AI Service Configurations ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

GYM_LEAD_AGENT_PROMPT = """
You are a friendly and encouraging AI assistant for 'Fitness First Gym' based in Chennai.
Your primary goal is to understand the caller's fitness goals and persuade them to schedule a free gym visit.
DO NOT discuss pricing. Defer all price questions by saying, "We have some great offers that we can discuss in person during your visit."
Keep your responses short and conversational.
If the user's message is in Tamil, you MUST respond in fluent, conversational Tamil.
"""

# --- Google Cloud Clients ---
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# --- (ThreadSafeAudioQueue class remains the same) ---
class ThreadSafeAudioQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()

    def put(self, item):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self):
        return await self._queue.get()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # ... (This entire function remains exactly the same as before) ...
    await websocket.accept()
    print("Microphone client connected.")

    audio_queue = ThreadSafeAudioQueue()

    async def audio_receiver():
        try:
            while True:
                audio_chunk = await websocket.receive_bytes()
                audio_queue.put(audio_chunk)
        except Exception:
            audio_queue.put(None)
            print("Client disconnected.")

    def stt_consumer():
        def audio_generator():
            while True:
                chunk = asyncio.run_coroutine_threadsafe(audio_queue.get(), audio_queue._loop).result()
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-IN",
                alternative_language_codes=["ta-IN"],
                model="telephony",
            ),
            interim_results=True,
        )

        responses = speech_client.streaming_recognize(config=streaming_config, requests=audio_generator())

        try:
            for response in responses:
                if not response.results or not response.results[0].alternatives:
                    continue
                
                transcript = response.results[0].alternatives[0].transcript
                if response.results[0].is_final:
                    print(f"\nUser said: {transcript}")
                    asyncio.run_coroutine_threadsafe(process_final_transcript(transcript), audio_queue._loop)
        except Exception as e:
            print(f"Error processing STT responses: {e}")

    consumer_thread = threading.Thread(target=stt_consumer)
    consumer_thread.start()
    await audio_receiver()
    consumer_thread.join()
    print("Connection closed and resources released.")


# --- FINAL, MODIFIED FUNCTION FOR AUDIO STREAMING ---
async def process_final_transcript(transcript: str):
    """Processes transcript, gets LLM response, and streams audio to speakers."""
    llm_response = await get_llm_response(transcript)
    print(f"AI says: {llm_response}\n")

    synthesis_input = texttospeech.SynthesisInput(text=llm_response)

    is_tamil = any(0x0B80 <= ord(c) <= 0x0BFF for c in llm_response)

    if is_tamil:
        voice = texttospeech.VoiceSelectionParams(language_code="ta-IN", name="ta-IN-Wavenet-D")
    else:
        voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name="en-IN-Wavenet-A")

    # --- IMPORTANT: Request raw audio (LINEAR16) instead of MP3 ---
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000 # A standard sample rate for WaveNet voices
    )
    
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # --- NEW: Stream audio directly to speakers using PyAudio ---
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True)
    
    print("AI is speaking...")
    stream.write(response.audio_content)

    # Clean up the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Playback finished.")


async def get_llm_response(user_text: str) -> str:
    # ... (This function remains exactly the same as before) ...
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    data = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": GYM_LEAD_AGENT_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": 150
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data, timeout=30)
            response.raise_for_status()
            llm_data = response.json()
            return llm_data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            print(f"ERROR: Received status code {e.response.status_code} from Together AI. Response: {e.response.text}")
            return "Sorry, I'm having trouble connecting to the AI service right now."
        except Exception as e:
            print(f"An unexpected error occurred while contacting Together AI: {e}")
            return "Sorry, there was an unexpected error."
