import os
import asyncio
import threading
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv
import httpx
from google.cloud import speech

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

# --- Google Cloud Client ---
speech_client = speech.SpeechClient()

# --- A Thread-Safe Queue to bridge asyncio and threads ---
class ThreadSafeAudioQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()

    def put(self, item):
        # This method can be called from any thread
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self):
        # This method must be awaited in the asyncio event loop
        return await self._queue.get()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Microphone client connected.")

    audio_queue = ThreadSafeAudioQueue()

    # --- Task 1 (async): Receive audio from WebSocket and put in queue ---
    async def audio_receiver():
        try:
            while True:
                audio_chunk = await websocket.receive_bytes()
                audio_queue.put(audio_chunk)
        except Exception:
            # When client disconnects, signal the end to the consumer
            audio_queue.put(None)
            print("Client disconnected.")

    # --- Task 2 (sync, runs in a separate thread): Consumes audio and calls Google STT ---
    def stt_consumer():
        
        # This is a synchronous generator that the Google client can understand
        def audio_generator():
            while True:
                # We use a blocking call here because this runs in a separate thread
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
                    # We need to run the async LLM call back in the main event loop
                    asyncio.run_coroutine_threadsafe(process_final_transcript(transcript), audio_queue._loop)
        except Exception as e:
            print(f"Error processing STT responses: {e}")

    # Start the blocking STT consumer in a separate thread
    consumer_thread = threading.Thread(target=stt_consumer)
    consumer_thread.start()

    # Run the async audio receiver in the main thread
    await audio_receiver()

    # Wait for the consumer thread to finish
    consumer_thread.join()
    print("Connection closed and resources released.")

async def process_final_transcript(transcript: str):
    """Processes the final transcript with the LLM."""
    llm_response = await get_llm_response(transcript)
    print(f"AI says: {llm_response}\n")

async def get_llm_response(user_text: str) -> str:
    """Sends text to Together AI and gets a response."""
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
            response.raise_for_status()  # This will raise an error for 4xx or 5xx responses
            llm_data = response.json()
            return llm_data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            print(f"ERROR: Received status code {e.response.status_code} from Together AI. Response: {e.response.text}")
            return "Sorry, I'm having trouble connecting to the AI service right now."
        except Exception as e:
            print(f"An unexpected error occurred while contacting Together AI: {e}")
            return "Sorry, there was an unexpected error."

