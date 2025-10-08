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
import json
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
You are 'Black Bull Fitness Studio's' AI Customer Support & Lead Conversion Voice Agent...
(Use the full, detailed prompt you created previously)
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
    """A synchronous generator that yields audio chunks with proper delay."""
    while True:
        if len(audio_queue) > 0:
            yield speech.StreamingRecognizeRequest(audio_content=audio_queue.popleft())
        else:
            time.sleep(0.1)

# --- FINAL, CORRECTED PLAYBACK FUNCTION ---
def play_audio(audio_content):
    """Plays audio content with the correct format to prevent noise."""
    p = pyaudio.PyAudio()
    
    # Use get_format_from_width(2) for 16-bit audio (2 bytes).
    # This is the key to fixing the noise issue.
    stream = p.open(format=p.get_format_from_width(2),
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
    """Handles barge-in, gets LLM response, and initiates new playback."""
    global current_playback_thread

    stop_current_playback()
    
    await websocket.send_json({"type": "user_transcript", "data": f"[{datetime.now().strftime('%H:%M:%S')}] User: {transcript}"})
    
    llm_response = await get_llm_response(transcript)
    await websocket.send_json({"type": "ai_response", "data": f"[{datetime.now().strftime('%H:%M:%S')}] AI: {llm_response}"})

    try:
        lang = detect(llm_response)
    except LangDetectException:
        lang = 'en'

    if lang == 'ta':
        voice = texttospeech.VoiceSelectionParams(language_code="ta-IN", name="ta-IN-Wavenet-D")
    else:
        voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name="en-IN-Wavenet-A")
    
    # Request LINEAR16 so we have raw data for PyAudio
    synthesis_input = texttospeech.SynthesisInput(text=llm_response)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=24000)
    
    loop = asyncio.get_running_loop()
    tts_response = await loop.run_in_executor(None, lambda: tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config))
    
    await websocket.send_json({"type": "ai_voice_start", "data": f"[{datetime.now().strftime('%H:%M:%S')}] AI voice start"})
    
    stop_requested.clear()
    current_playback_thread = threading.Thread(target=play_audio, args=(tts_response.audio_content,))
    current_playback_thread.start()
    
    # We play the audio on the server, so we don't send the bytes to the client
    # await websocket.send_bytes(tts_response.audio_content)

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    stt_thread = None
    audio_receiver_task = None
    loop = asyncio.get_running_loop()

    def stt_thread_func():
        while True:
            streaming_config = speech.StreamingRecognitionConfig(config=speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=RATE, language_code="en-IN", alternative_language_codes=["ta-IN"], model="telephony"), interim_results=False)
            requests = audio_generator()
            try:
                responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)
                for response in responses:
                    if response.results and response.results[0].alternatives and response.results[0].is_final:
                        asyncio.run_coroutine_threadsafe(handle_final_transcript(response.results[0].alternatives[0].transcript, websocket), loop)
            except Exception as e:
                print(f"STT thread error: {e}. Client likely disconnected.")
                break 

    try:
        async def receive_audio():
            while True:
                audio_data = await websocket.receive_bytes()
                audio_queue.append(audio_data)

        audio_receiver_task = asyncio.create_task(receive_audio())
        stt_thread = threading.Thread(target=stt_thread_func, daemon=True)
        stt_thread.start()

        while stt_thread.is_alive():
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        if audio_receiver_task:
            audio_receiver_task.cancel()
        audio_queue.clear()
