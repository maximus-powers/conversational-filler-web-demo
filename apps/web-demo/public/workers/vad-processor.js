const MIN_CHUNK_SIZE = 512;
let globalPointer = 0;
let globalBuffer = new Float32Array(MIN_CHUNK_SIZE);

class VADProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
  }
  
  process(inputs, outputs, parameters) {
    const buffer = inputs[0][0];
    if (!buffer) {
      console.log('VAD processor: no buffer received');
      return true; // buffer is null when the stream ends
    }

    if (buffer.length > MIN_CHUNK_SIZE) {
      // if buffer is larger than the minimum chunk size, send the entire buffer
      this.port.postMessage({ type: 'audio', audio: buffer });
    } else {
      const remaining = MIN_CHUNK_SIZE - globalPointer;
      if (buffer.length >= remaining) {
        // if buffer is larger than (or equal to) the remaining space in the global buffer, copy the remaining space
        globalBuffer.set(buffer.subarray(0, remaining), globalPointer);

        this.port.postMessage({ type: 'audio', audio: globalBuffer });

        // reset the global buffer and set the remaining buffer
        globalBuffer.fill(0);
        globalBuffer.set(buffer.subarray(remaining), 0);
        globalPointer = buffer.length - remaining;
      } else {
        // if buffer is smaller than the remaining space in the global buffer, copy the buffer to the global buffer
        globalBuffer.set(buffer, globalPointer);
        globalPointer += buffer.length;
      }
    }

    return true; 
  }
}

registerProcessor("vad-processor", VADProcessor);