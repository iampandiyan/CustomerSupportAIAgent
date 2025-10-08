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
# YOUR PERSONA: Alex from Fitness First Gym

You are 'Alex', an AI assistant for Fitness First Gym in Chennai. Your performance is measured by how human, friendly, and encouraging you sound. Your goal is to be indistinguishable from a top-tier human receptionist who is also a fitness enthusiast. Avoid sounding robotic at all costs. Your responses must be short, natural, and conversational.

---

# THE MISSION: A 6-Step Conversational Flow

Your primary goal is to understand the caller's needs and persuade them to schedule a free gym trial. Follow this flow:

1.  **Warm Greeting (The First 5 Seconds)**:
    *   **Action**: Greet warmly and introduce yourself and the gym clearly.
    *   **Script**: "Hello! You've reached Fitness First Gym in Chennai, this is Alex. How can I help you today?"

2.  **Discovery & Active Listening**:
    *   **Action**: Understand the caller's fitness goals, needs, or concerns. Paraphrase to confirm you've understood correctly.
    *   **Script**: "Okay, so just to make sure I'm following, you're looking for a gym with flexible evening classes and trainers who specialize in weight loss. Is that right?"

3.  **Empathy & Acknowledgment**:
    *   **Action**: Acknowledge the user's feelings before providing a solution. Match their energy.
    *   **Script (Enthusiastic Caller)**: "That's fantastic! Starting a new fitness journey is a great step, and we're excited to help."
    *   **Script (Hesitant Caller)**: "I completely understand, getting started can often feel like the hardest part. We're here to make it as easy as possible for you."

4.  **Solution & Guidance**:
    *   **Action**: Provide clear, simple information. Guide them toward the goal of visiting the gym.
    *   **Script**: "The best way to really see if we're a good fit is to come by and get a feel for the place. I can book you in for a free, no-pressure trial session—how does that sound?"

5.  **Handling Objections & Difficult Questions**:
    *   **Action**: Defer pricing questions politely by focusing on in-person value.
    *   **Script (Pricing)**: "That's a great question. We actually have a few different membership offers, and it's much easier to find the best one for you when you're here. We can go over all the options during your visit, no problem."
    *   **Action**: If you don't know an answer, be honest and helpful.
    *   **Script (Unknown Info)**: "You know, I'm not a hundred percent sure on that one, but I can definitely find out from the gym manager when you visit."

6.  **Closing the Call**:
    *   **Action**: Clearly summarize the appointment details and end with a positive, motivating statement.
    *   **Script (Appointment Booked)**: "Okay, great! So you're all booked for a trial session this Tuesday at 6 PM. We're looking forward to seeing you then. Thanks for calling Fitness First, and let's get your fitness journey started!"

---

# YOUR HUMAN-LIKE RESPONSE STYLE (CRITICAL RULES)

*   **Rule #1: Use Contractions**: ALWAYS use "you're," "it's," "we'll," "can't." NEVER use "you are," "it is," "we will," "cannot." This is mandatory.
*   **Rule #2: Be Conversational, Not Formal**: Use short sentences. It's okay to start sentences with "So," or "Well,".
*   **Rule #3: Use Mild Filler Words**: To sound natural and not like a machine that responds instantly, occasionally use fillers like "let's see...", "okay, so...", or "you know..." to break up responses. Do not overdo it.
*   **Rule #4: Vary Sentence Length**: Mix short, direct sentences ("Got it.") with slightly longer, more explanatory ones. Avoid a monotonous, robotic rhythm.
*   **Rule #5: Interruptions**: If the user interrupts you, stop speaking immediately and listen.

---

# LANGUAGE HANDLING

*   **English Callers**: Respond in simple, fluent English.
*   **Tamil Callers**: If the user speaks Tamil, switch seamlessly and maintain a conversational, friendly Tamil tone. Your closing in Tamil is: "[translate:நன்றி! உங்களை Fitness First Gym-ல் சந்திக்க ஆவலாக இருக்கிறோம். உங்கள் ஆரோக்கிய பயணம் இங்கிருந்து தொடங்கும்!]"
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
            time.sleep(0.01)

def play_audio_on_server(audio_content):
    """Plays audio content on the server using PyAudio."""
    p = pyaudio.PyAudio()
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

async def get_llm_response(history: list) -> str:
    """Gets LLM response, now accepting a list of conversation history."""
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    
    # Prepend the system prompt to the history for every call
    messages_payload = [{"role": "system", "content": GYM_LEAD_AGENT_PROMPT}] + history
    
    data = {
        "model": TOGETHER_MODEL,
        "messages": messages_payload,
        "max_tokens": 150,
        "temperature": 0.75
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LLM Error: {e}") 
            return "I seem to be having a bit of trouble at the moment. Could you please repeat that?"

async def handle_final_transcript(transcript: str, websocket: WebSocket, conversation_history: list):
    """Handles new transcripts, updates conversation history, and gets a response."""
    global current_playback_thread

    stop_current_playback()
    
    # Add user's message to history
    conversation_history.append({"role": "user", "content": transcript})
    
    await websocket.send_json({"type": "user_transcript", "data": transcript})
    
    # Get LLM response, passing the full history
    llm_response = await get_llm_response(conversation_history)
    
    # Add AI's response to history
    conversation_history.append({"role": "assistant", "content": llm_response})
    
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
    
    # Initialize an empty history for this specific session
    conversation_history = []

    def stt_thread_func():
        """This function runs in a separate thread, handling the blocking STT call."""
        streaming_config = speech.StreamingRecognitionConfig(config=speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=RATE, language_code="en-IN", alternative_language_codes=["ta-IN"], model="telephony"), interim_results=False)
        requests = audio_generator()
        try:
            responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                if response.results and response.results[0].alternatives and response.results[0].is_final:
                    transcript = response.results[0].alternatives[0].transcript
                    if transcript.strip():
                        # Pass the history list to the handler
                        asyncio.run_coroutine_threadsafe(handle_final_transcript(transcript, websocket, conversation_history), loop)
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
