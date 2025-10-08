import os
import asyncio
import threading
from dotenv import load_dotenv
import httpx
from google.cloud import speech
from google.cloud import texttospeech
from collections import deque
import time
from langdetect import detect, LangDetectException
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pyaudio

# --- Basic Setup ---
load_dotenv()
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- AI and Audio Configurations ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
GYM_LEAD_AGENT_PROMPT = """
You are 'Fitness First Gym's' AI Customer Support & Lead Conversion Voice Agent based in Chennai. 
You are friendly, professional, and encouraging, like a fitness coach + receptionist.
## Your Goals
1. Greet every caller warmly and introduce yourself as the Fitness First Gym assistant.
2. Understand the caller’s fitness goals, needs, or concerns.
3. Provide information about gym facilities, trainers, working hours, and special programs.
4. Persuade the caller to schedule a free gym visit and trial session.
5. Collect and confirm appointment details (name, phone, preferred date & time).
6. Defer all pricing questions politely: say, "We have some great offers that we can discuss in person during your visit."
7. Keep responses short, natural, and conversational (not robotic).
8. Always maintain a positive, motivating, and helpful tone.
## Language
- If the caller speaks in English → reply in fluent, simple English.
- If the caller speaks in Tamil → reply in fluent, conversational Tamil.
- Do NOT switch languages unnecessarily.
## Conversation Style
- Always start with a warm greeting: “Hello! This is Fitness First Gym in Chennai. How can I help you today?”
- Use short sentences (like natural speech).
- Encourage the user: “That’s a great goal!”, “You’re making a healthy choice!”, “We’ll help you achieve it.”
- If the caller seems unsure, reassure them about the benefits of visiting the gym.
- Confirm details clearly: “So you’d like to visit on Tuesday at 6 PM, correct?”
- If something is unclear, politely ask for clarification.
- If interrupted, stop speaking and listen.
## Safety & Trust
- Never share private or financial details.
- Do not promise results (e.g., weight loss guarantees).
- If you don’t know an answer, say: “I’ll check with the gym staff and let you know during your visit.”
## Call Closing
- Once appointment is booked, repeat details back to confirm.
- End with gratitude and motivation:  
  - English: “Thank you! We look forward to seeing you at Fitness First Gym. Let’s begin your fitness journey!”  
  - Tamil: “நன்றி! உங்களை Fitness First Gym-ல் சந்திக்க ஆவலாக இருக்கிறோம். உங்கள் ஆரோக்கிய பயணம் இங்கிருந்து தொடங்கும்!”  
"""
RATE = 16000

# --- Clients, Queues, and Playback Management ---
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
audio_queue = deque()
current_playback_thread = None
stop_requested = threading.Event()

# --- Voice AI Logic ---
def audio_generator():
    """A synchronous generator that yields audio chunks from the global queue."""
    while True:
        if len(audio_queue) > 0:
            yield speech.StreamingRecognizeRequest(audio_content=audio_queue.popleft())
        else:
            time.sleep(0.01) # A shorter sleep can improve responsiveness

def play_audio_on_server(audio_content):
    """Plays audio content on the server using PyAudio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2), # 16-bit PCM
                      channels=1,
                      rate=24000,
                      output=True)
    
    chunk_size = 1024
    try:
        for i in range(0, len(audio_content), chunk_size):
            if stop_requested.is_set():
                break
            stream.write(audio_content[i:i+chunk_size])
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def stop_current_playback():
    """Signals any current playback thread to stop and waits for it to exit."""
    global current_playback_thread
    if current_playback_thread and current_playback_thread.is_alive():
        print(f"--- [{datetime.now().strftime('%H:%M:%S')}] User interrupted. Signaling playback to stop. ---")
        stop_requested.set()
        current_playback_thread.join()
        current_playback_thread = None

async def get_llm_response(user_text: str) -> str:
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    data = {"model": TOGETHER_MODEL, "messages": [{"role": "system", "content": GYM_LEAD_AGENT_PROMPT}, {"role": "user", "content": user_text}], "max_tokens": 150}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"LLM Error: {e}"

async def handle_final_transcript(transcript: str, websocket: WebSocket):
    """Handles barge-in, gets LLM response, and initiates new playback on the server."""
    global current_playback_thread

    stop_current_playback()
    
    await websocket.send_json({"type": "user_transcript", "data": transcript})
    
    llm_response = await get_llm_response(transcript)
    await websocket.send_json({"type": "ai_response", "data": llm_response})

    try:
        lang = detect(llm_response)
    except LangDetectException:
        lang = 'en'

    voice = texttospeech.VoiceSelectionParams(language_code="ta-IN" if lang == 'ta' else "en-IN", name="ta-IN-Wavenet-D" if lang == 'ta' else "en-IN-Wavenet-A")
    synthesis_input = texttospeech.SynthesisInput(text=llm_response)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=24000)
    
    loop = asyncio.get_running_loop()
    tts_response = await loop.run_in_executor(None, lambda: tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config))
    
    stop_requested.clear()
    current_playback_thread = threading.Thread(target=play_audio_on_server, args=(tts_response.audio_content,))
    current_playback_thread.start()

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    loop = asyncio.get_running_loop()

    def stt_thread_func():
        """This function runs in a separate thread, handling the blocking STT call."""
        streaming_config = speech.StreamingRecognitionConfig(config=speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=RATE, language_code="en-IN", alternative_language_codes=["ta-IN"], model="telephony"), interim_results=False)
        requests = audio_generator()
        try:
            responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                if response.results and response.results[0].alternatives and response.results[0].is_final:
                    transcript = response.results[0].alternatives[0].transcript
                    if transcript.strip(): # Ensure we don't process empty transcripts
                        asyncio.run_coroutine_threadsafe(handle_final_transcript(transcript, websocket), loop)
        except Exception as e:
            print(f"STT thread error: {e}. Client likely disconnected.")
    
    audio_receiver_task = None
    try:
        async def receive_audio():
            """Receives audio from the client and puts it into the global queue."""
            while True:
                audio_data = await websocket.receive_bytes()
                audio_queue.append(audio_data)

        audio_receiver_task = asyncio.create_task(receive_audio())
        stt_thread = threading.Thread(target=stt_thread_func, daemon=True)
        stt_thread.start()

        # Keep the connection alive while the STT thread is running
        while stt_thread.is_alive():
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred in websocket endpoint: {e}")
    finally:
        if audio_receiver_task and not audio_receiver_task.done():
            audio_receiver_task.cancel()
        audio_queue.clear()
        print("Websocket connection closed and cleaned up.")
