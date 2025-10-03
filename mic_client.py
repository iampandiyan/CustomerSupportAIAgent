import asyncio
import websockets
import pyaudio

# Configuration
WEBSOCKET_URI = "ws://localhost:8000/ws"
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Must match the sample rate in main.py

async def stream_microphone():
    """Captures audio from the microphone and streams it to the server."""
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Connecting to server...")
    async with websockets.connect(WEBSOCKET_URI) as websocket:
        print("Connected! Start speaking into your microphone (press Ctrl+C to stop).")
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                await websocket.send(data)
                await asyncio.sleep(0.01)
        except websockets.exceptions.ConnectionClosed:
            print("Server connection closed.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(stream_microphone())
    except KeyboardInterrupt:
        print("Stopped by user.")
