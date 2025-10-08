// This script runs in a separate thread (the AudioWorklet).
class AudioRecorderProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);
        // The buffer size we want to accumulate before sending
        this.bufferSize = 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
        this.port.onmessage = (event) => {};
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const channelData = input[0];

        if (channelData) {
            // Downsample the incoming audio chunk
            const downsampled = this.downsampleBuffer(channelData, sampleRate, 16000);

            // Add the downsampled chunk to our internal buffer
            for (let i = 0; i < downsampled.length; i++) {
                if (this.bufferIndex < this.bufferSize) {
                    this.buffer[this.bufferIndex++] = downsampled[i];
                }
            }

            // If our buffer is full, process and send it
            if (this.bufferIndex >= this.bufferSize) {
                const pcm16 = this.floatTo16BitPCM(this.buffer);
                
                // Post the complete, buffered PCM data back to the main thread
                this.port.postMessage(pcm16, [pcm16.buffer]);
                
                // Reset the buffer for the next round
                this.buffer = new Float32Array(this.bufferSize);
                this.bufferIndex = 0;
            }
        }

        // Return true to keep the processor alive.
        return true;
    }

    downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
        if (inputSampleRate === outputSampleRate) {
            return buffer;
        }
        const sampleRateRatio = inputSampleRate / outputSampleRate;
        const newLength = Math.round(buffer.length / sampleRateRatio);
        const result = new Float32Array(newLength);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < result.length) {
            const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
            let accum = 0, count = 0;
            for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                accum += buffer[i];
                count++;
            }
            result[offsetResult] = accum / count;
            offsetResult++;
            offsetBuffer = nextOffsetBuffer;
        }
        return result;
    }

    floatTo16BitPCM(input) {
        const output = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]));
            output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return output;
    }
}

registerProcessor('audio-recorder-processor', AudioRecorderProcessor);
