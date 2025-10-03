import os
import asyncio
import threading
from dotenv import load_dotenv
import httpx
from google.cloud import speech
from google.cloud import texttospeech
import pyaudio
from collections import deque
import time
from langdetect import detect, LangDetectException
from datetime import datetime

# Load environment variables
load_dotenv()

# --- AI and Audio Configurations ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
GYM_LEAD_AGENT_PROMPT = """
You are 'Black Bull Fitness Studio's' AI Customer Support & Lead Conversion Voice Agent. The gym is located in West Mambalam, Chennai.
You are friendly, professional, and encouraging, like a fitness coach + receptionist.

## Gym Information & Talking Points
- **Name:** Black Bull Fitness Studio
- **Location:** West Mambalam, Chennai [633].
- **Tagline/Vibe:** A premium, luxurious fitness hub redefining fitness in West Mambalam [632][636].
- **Key Equipment:** Over 70 pieces of world-class, top-tier equipment [635].
- **Specialty Classes & Programs:** We offer a wide range of activities including Hyrox training, Zumba, and Yoga [635]. We have programs for weight loss, muscle gain, and general wellness.
- **Unique Facilities:** We have premium amenities including a Jacuzzi, an Ice Bath, a Steam room, and even an on-site Salon & Cafe [635].
- **Trainers:** Our trainers are experienced, certified, and friendly. Customers have praised trainers like Saravanan, Kishore, and Arun [634].
- **Operating Hours:** We are open 7 days a week, from 5:30 AM to 9:30 PM [631][634].
- **Target Audience:** Open to all age groups [634]. We provide customized programs for everyone, including students, seniors, and members with disabilities [633].

## Your Goals
1.  Greet every caller warmly and introduce yourself as the Black Bull Fitness Studio assistant.
2.  Understand the caller’s fitness goals (e.g., weight loss, muscle gain, general wellness).
3.  Provide specific, accurate information about our facilities, trainers (mention they are highly rated), working hours (5:30 AM to 9:30 PM, all days), and unique offerings like Hyrox, Zumba, Jacuzzi, and Ice Baths.
4.  Persuade the caller to schedule a free gym visit and trial session to experience our premium atmosphere.
5.  Collect and confirm appointment details (name, phone, preferred date & time).
6.  Politely defer all pricing questions: say, "We have some fantastic membership plans and special offers. The best way to discuss them is in person during your free visit, where we can tailor a plan just for you."
7.  Keep responses short, natural, and conversational.
8.  Always maintain a positive, motivating, and helpful tone.

## Language
- If the caller speaks in English → reply in fluent, simple English.
- If the caller speaks in Tamil → reply in fluent, conversational Tamil.
- Do NOT switch languages unnecessarily.

## Conversation Style
- Always start with a warm greeting: “Hello! This is Black Bull Fitness Studio in West Mambalam. How can I help you today?”
- Use short sentences.
- Encourage the user: “That’s a great goal!”, “An ice bath after a workout is an amazing recovery tool!”, “We can definitely help you on your fitness journey.”
- If the caller seems unsure, highlight our unique features: "Our Jacuzzi and steam room are perfect for relaxing after a workout. You can experience it all during your free trial!"
- Confirm details clearly: “So you’d like to visit on Tuesday at 6 PM, correct?”
- If something is unclear, politely ask for clarification.
- If interrupted, stop speaking and listen.

## Safety & Trust
- Never share private or financial details.
- Do not promise specific results (e.g., "you will lose 10kg").
- If you don’t know an answer, say: “That’s a great question. I’ll make a note, and our gym manager can give you all the details during your visit.”

## Call Closing
- Once an appointment is booked, repeat the details back for confirmation.
- End with gratitude and motivation:  
  - English: “Thank you! We look forward to seeing you at Black Bull Fitness Studio. Your journey to redefining fitness starts here!”  
  - Tamil: “[translate:நன்றி! உங்களை Black Bull Fitness Studio-வில் சந்திக்க ஆவலாக இருக்கிறோம். உங்கள் ஆரோக்கிய பயணம் இங்கே தொடங்குகிறது!]”  
"""



# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
STREAM_LIMIT_SECONDS = 280

# --- Google Cloud Clients & Audio Queue ---
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
audio_queue = deque()

# --- Playback Management ---
playback_lock = threading.Lock()
current_playback_thread = None
stop_requested = threading.Event()

# --- PyAudio Callback ---
def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.append(in_data)
    return (None, pyaudio.paContinue)

def audio_generator():
    """A synchronous generator that yields audio chunks."""
    while True:
        if len(audio_queue) > 0:
            yield speech.StreamingRecognizeRequest(audio_content=audio_queue.popleft())
        else:
            time.sleep(0.1)

# --- Interruptible Playback Function ---
def play_audio(audio_content):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    
    chunk_size = 1024
    try:
        for i in range(0, len(audio_content), chunk_size):
            # Before writing each chunk, check if a stop has been requested
            if stop_requested.is_set():
                break
            stream.write(audio_content[i:i+chunk_size])
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# --- Function to stop playback ---
def stop_current_playback():
    global current_playback_thread
    if current_playback_thread and current_playback_thread.is_alive():
        print(f"--- [{datetime.now().strftime('%H:%M:%S')}] User interrupted. Signaling playback to stop. ---")
        stop_requested.set() # Signal the thread to stop
        current_playback_thread.join() # Wait for the thread to finish cleanly
        current_playback_thread = None

async def get_llm_response(user_text: str) -> str:
    # (This function remains the same)
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    data = {"model": TOGETHER_MODEL, "messages": [{"role": "system", "content": GYM_LEAD_AGENT_PROMPT}, {"role": "user", "content": user_text}], "max_tokens": 150}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Sorry, I had an issue thinking of a response."

def stt_thread_func(loop):
    """This function runs in a separate thread and handles the STT stream, including restarts."""
    while True:
        print(f"\n--- [{datetime.now().strftime('%H:%M:%S')}] Starting new STT stream ---")
        streaming_config = speech.StreamingRecognitionConfig(config=speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=RATE, language_code="en-IN", alternative_language_codes=["ta-IN"], model="telephony"), interim_results=False)
        requests = audio_generator()
        stream_start_time = time.time()
        
        try:
            responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                if time.time() - stream_start_time > STREAM_LIMIT_SECONDS:
                    print(f"--- [{datetime.now().strftime('%H:%M:%S')}] Stream time limit reached. Restarting. ---")
                    break
                
                if response.results and response.results[0].alternatives and response.results[0].is_final:
                    transcript = response.results[0].alternatives[0].transcript
                    asyncio.run_coroutine_threadsafe(handle_final_transcript(transcript), loop)
        except Exception as e:
            print(f"STT thread error: {e}. Restarting...")
            time.sleep(1)

# --- FINALIZED TRANSCRIPT HANDLER WITH BARGE-IN ---
async def handle_final_transcript(transcript: str):
    """Stops current playback, gets a new response, and starts new playback."""
    global current_playback_thread
    
    # 1. BARGE-IN: Stop any currently playing audio immediately.
    stop_current_playback()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] User said: {transcript}")
    
    llm_response = await get_llm_response(transcript)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] AI says: {llm_response}")

    try:
        lang = detect(llm_response)
    except LangDetectException:
        lang = 'en'

    if lang == 'ta':
        voice = texttospeech.VoiceSelectionParams(language_code="ta-IN", name="ta-IN-Wavenet-D")
    else:
        voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name="en-IN-Wavenet-A")
    
    synthesis_input = texttospeech.SynthesisInput(text=llm_response)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=24000)
    
    loop = asyncio.get_running_loop()
    tts_response = await loop.run_in_executor(None, lambda: tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config))
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] AI voice start")
    
    # Start new playback
    stop_requested.clear() # Clear the stop flag before starting
    current_playback_thread = threading.Thread(target=play_audio, args=(tts_response.audio_content,))
    current_playback_thread.start()

async def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_callback)
    
    stream.start_stream()
    print("Microphone stream started. The agent is now listening.")

    loop = asyncio.get_running_loop()
    stt_thread = threading.Thread(target=stt_thread_func, args=(loop,), daemon=True)
    stt_thread.start()

    try:
        while stt_thread.is_alive():
            await asyncio.sleep(1)
    finally:
        print("Shutting down stream...")
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
