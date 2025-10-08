const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const chatLog = document.getElementById('chat-log');

let socket;
let mediaStream;
let audioContext;
let scriptProcessor;
let audioPlayerContext;
let currentAudioSource = null; // <-- NEW: To manage the currently playing audio

function appendMessage(message, type) {
    const p = document.createElement('p');
    p.textContent = message;
    
    // Add a class for styling based on the message type
    if (type === 'user') {
        p.className = 'user-message';
    } else if (type === 'ai') {
        p.className = 'ai-message';
    } else { // System messages
        p.className = 'system-message';
    }

    chatLog.appendChild(p);
    chatLog.scrollTop = chatLog.scrollHeight;
}

startBtn.onclick = async () => {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    appendMessage('Starting call...', 'system');

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        audioContext = new AudioContext({ sampleRate: 16000 });
        scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);
        const source = audioContext.createMediaStreamSource(mediaStream);
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        socket = new WebSocket(`ws://${window.location.host}/ws`);

        socket.onopen = () => {
            appendMessage('Connection established. You can start speaking.', 'system');
            scriptProcessor.onaudioprocess = (event) => {
                const inputData = event.inputBuffer.getChannelData(0);
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    pcmData[i] = inputData[i] * 32767;
                }
                if (socket.readyState === WebSocket.OPEN) {
                    socket.send(pcmData.buffer);
                }
            };
        };

        socket.onmessage = async (event) => {
            if (event.data instanceof Blob) {
                // --- NEW: BARGE-IN LOGIC ---
                // 1. If audio is already playing, stop it immediately.
                if (currentAudioSource) {
                    currentAudioSource.stop();
                    currentAudioSource = null;
                }

                const audioData = await event.data.arrayBuffer();
                if (!audioPlayerContext || audioPlayerContext.state === 'closed') {
                    audioPlayerContext = new (window.AudioContext || window.webkitAudioContext)();
                }
                
                // 2. Decode and play the new audio
                const audioBuffer = await audioPlayerContext.decodeAudioData(audioData);
                const sourceNode = audioPlayerContext.createBufferSource();
                sourceNode.buffer = audioBuffer;
                sourceNode.connect(audioPlayerContext.destination);
                sourceNode.start(0);

                // 3. Keep a reference to the new audio source
                currentAudioSource = sourceNode;
                sourceNode.onended = () => {
                    currentAudioSource = null; // Clear the reference when playback finishes
                };

            } else {
                // This is text data
                const message = JSON.parse(event.data);
                if (message.type === 'user_transcript') {
                    appendMessage(message.data, 'user');
                } else if (message.type === 'ai_response' || message.type === 'ai_voice_start') {
                    appendMessage(message.data, 'ai');
                }
            }
        };

        socket.onclose = () => {
            appendMessage('Connection closed.', 'system');
            stopCall();
        };

        socket.onerror = (error) => {
            console.error('WebSocket Error:', error);
            appendMessage('Connection error.', 'system');
            stopCall();
        };

    } catch (err) {
        console.error('Error starting call:', err);
        appendMessage('Could not access microphone.', 'system');
        stopCall();
    }
};

function stopCall() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    if (socket) {
        socket.close();
    }
    if (currentAudioSource) {
        currentAudioSource.stop();
        currentAudioSource = null;
    }
    if (audioPlayerContext) {
        audioPlayerContext.close();
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
}

stopBtn.onclick = stopCall;
