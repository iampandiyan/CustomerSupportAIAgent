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

#COnversational is working fine but we have TTS limit upto 5 min
# Load environment variables
load_dotenv()

# --- AI and Audio Configurations ---
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
GYM_LEAD_AGENT_PROMPT = """
You are a friendly and encouraging AI assistant for 'Fitness First Gym' based in Chennai.
Your primary goal is to understand the caller's fitness goals and persuade them to schedule a free gym visit.
DO NOT discuss pricing. Defer all price questions by saying, "We have some great offers that we can discuss in person during your visit."
Keep your responses very short and conversational.
If the user's message is in Tamil, you MUST respond in fluent, conversational Tamil.
"""

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# --- Google Cloud Clients & Audio Queue ---
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
audio_queue = deque()

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

def play_audio(audio_content):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream.write(audio_content)
    stream.stop_stream()
    stream.close()
    p.terminate()

async def get_llm_response(user_text: str) -> str:
    # (This function remains the same)
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    data = { "model": TOGETHER_MODEL, "messages": [{"role": "system", "content": GYM_LEAD_AGENT_PROMPT}, {"role": "user", "content": user_text}], "max_tokens": 150}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Sorry, I had an issue thinking of a response."

def stt_thread_func(loop):
    """This function runs in a separate thread and handles the STT stream."""
    streaming_config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-IN",
            alternative_language_codes=["ta-IN"],
            model="telephony"
        ),
        interim_results=False
    )
    
    requests = audio_generator()
    responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)

    try:
        for response in responses:
            if response.results and response.results[0].alternatives and response.results[0].is_final:
                transcript = response.results[0].alternatives[0].transcript
                # Use call_soon_threadsafe to schedule the async function in the main loop
                asyncio.run_coroutine_threadsafe(handle_final_transcript(transcript), loop)
    except Exception as e:
        print(f"STT thread error: {e}")

async def handle_final_transcript(transcript: str):
    """Handles the final transcript from the STT thread."""
    print(f"\nUser said: {transcript}")
    
    llm_response = await get_llm_response(transcript)
    print(f"AI says: {llm_response}")

    synthesis_input = texttospeech.SynthesisInput(text=llm_response)
    is_tamil = any(0x0B80 <= ord(c) <= 0x0BFF for c in llm_response)
    voice = texttospeech.VoiceSelectionParams(language_code="ta-IN" if is_tamil else "en-IN", name="ta-IN-Wavenet-D" if is_tamil else "en-IN-Wavenet-A")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=24000)
    
    # Run blocking TTS call in executor
    loop = asyncio.get_running_loop()
    tts_response = await loop.run_in_executor(None, lambda: tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config))
    
    # Run blocking audio playback in another thread to not block this one
    threading.Thread(target=play_audio, args=(tts_response.audio_content,)).start()

async def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=audio_callback)
    
    stream.start_stream()
    print("Microphone stream started. The agent is now listening.")

    # Get the current asyncio event loop
    loop = asyncio.get_running_loop()
    
    # Start the STT processing in a separate thread
    stt_thread = threading.Thread(target=stt_thread_func, args=(loop,), daemon=True)
    stt_thread.start()

    try:
        # Keep the main thread alive
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

