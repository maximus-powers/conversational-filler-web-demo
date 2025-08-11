import { AutoModel, Tensor, pipeline } from "@huggingface/transformers";

import { KokoroTTS, TextSplitterStream } from "kokoro-js";
console.log('KokoroTTS imported:', typeof KokoroTTS, 'TextSplitterStream:', typeof TextSplitterStream);

// wrap everything in async IIFE to handle top-level await
(async () => {

// Audio constants - matching webgpu-demo exactly
const INPUT_SAMPLE_RATE = 16000;
const INPUT_SAMPLE_RATE_MS = INPUT_SAMPLE_RATE / 1000;
const SPEECH_THRESHOLD = 0.3;
const EXIT_THRESHOLD = 0.1;
const MIN_SILENCE_DURATION_MS = 1000; // Increased from 400ms to 1 second for more lenient pause detection
const MIN_SILENCE_DURATION_SAMPLES = MIN_SILENCE_DURATION_MS * INPUT_SAMPLE_RATE_MS;
const SPEECH_PAD_MS = 80;
const SPEECH_PAD_SAMPLES = SPEECH_PAD_MS * INPUT_SAMPLE_RATE_MS;
const MIN_SPEECH_DURATION_SAMPLES = 250 * INPUT_SAMPLE_RATE_MS; // 250 ms
const MAX_BUFFER_DURATION = 30;
const NEW_BUFFER_SIZE = 512;
const MAX_NUM_PREV_BUFFERS = Math.ceil(SPEECH_PAD_SAMPLES / NEW_BUFFER_SIZE);

const device = "webgpu";

self.postMessage({ type: "info", message: `Using device: "${device}"` });
self.postMessage({
  type: "info",
  message: "Loading models...",
  duration: "until_next",
});

// Initialize TTS
const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
let voice; // Will be set later or use default
let tts;

try {
  console.log('Initializing TTS with model:', model_id);
  
  // Check if WebGPU is available in worker
  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
  console.log('WebGPU available in worker:', hasWebGPU);
  
  // Try with wasm if webgpu isn't available
  const ttsDevice = hasWebGPU ? "webgpu" : "wasm";
  console.log('Using device for TTS:', ttsDevice);
  
  tts = await KokoroTTS.from_pretrained(model_id, {
    dtype: "fp32",
    device: ttsDevice,
  });
  console.log('TTS initialized successfully');
  console.log('Available voices:', tts.voices ? Object.keys(tts.voices) : 'No voices found');
  
  
} catch (error) {
  console.error('Failed to initialize TTS:', error);
  self.postMessage({ type: "error", error: `TTS initialization failed: ${error.message}` });
}

// Load VAD model
const silero_vad = await AutoModel.from_pretrained(
  "onnx-community/silero-vad",
  {
    config: { model_type: "custom" },
    dtype: "fp32", // Full-precision
  },
).catch((error) => {
  self.postMessage({ error });
  throw error;
});

// Load Whisper for transcription
// Use whisper-small for better accuracy (244M params vs 74M in base)
const DEVICE_DTYPE_CONFIGS = {
  webgpu: {
    encoder_model: "fp32",
    decoder_model_merged: "fp32",
  },
  wasm: {
    encoder_model: "fp32",
    decoder_model_merged: "q8",
  },
};

console.log('Loading Whisper large v3 turbo model for better accuracy...');
self.postMessage({ 
  type: "info", 
  message: "Loading Whisper Large v3 Turbo model (this may take 30-60 seconds)...",
  duration: "until_next"
});

const transcriber = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/whisper-base",
  {
    device,
    dtype: DEVICE_DTYPE_CONFIGS[device],
  },
).catch((error) => {
  self.postMessage({ error });
  throw error;
});

self.postMessage({ 
  type: "info", 
  message: "Whisper model loaded successfully"
});

await transcriber(new Float32Array(INPUT_SAMPLE_RATE)); // Compile shaders

// Load SmolLM for conversation (OUR MODEL)
const llm_model_id = "maximuspowers/smollm-convo-filler-onnx-official";
const llm = await pipeline("text-generation", llm_model_id, {
  dtype: "fp32", 
  device: "webgpu",
});

await llm("test", { max_new_tokens: 1 }); // Compile shaders

let messages = [];

// Set default voice if not already set
if (!voice && tts.voices) {
  voice = Object.keys(tts.voices)[0] || "af_heart";
  console.log('Setting default voice:', voice);
}

self.postMessage({
  type: "status",
  status: "ready",
  message: "Ready!",
  voices: tts.voices,
});

// Global audio buffer to store incoming audio - EXACTLY LIKE WEBGPU-DEMO
const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;

// Initial state for VAD - EXACTLY LIKE WEBGPU-DEMO
const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);

// Whether we are in the process of adding audio to the buffer
let isRecording = false;
let isPlaying = false;

/**
 * Perform Voice Activity Detection (VAD) - EXACTLY FROM WEBGPU-DEMO
 * @param {Float32Array} buffer The new audio buffer
 * @returns {Promise<boolean>} `true` if the buffer is speech, `false` otherwise.
 */
async function vad(buffer) {
  const input = new Tensor("float32", buffer, [1, buffer.length]);

  const { stateN, output } = await silero_vad({ input, sr, state });
  state = stateN; // Update state

  const isSpeech = output.data[0];

  // Use heuristics to determine if the buffer is speech or not
  return (
    // Case 1: We are above the threshold (definitely speech)
    isSpeech > SPEECH_THRESHOLD ||
    // Case 2: We are in the process of recording, and the probability is above the negative (exit) threshold
    (isRecording && isSpeech >= EXIT_THRESHOLD)
  );
}

/**
 * Generate and process thoughts from OpenAI - streaming version
 */
const generateAndProcessThoughts = async (conversationHistory, userInput, immediateResponse, splitter) => {
  // Use our Next.js API endpoint
  const response = await fetch('/api/chat-thoughts', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      messages: conversationHistory
    }),
  });

  if (!response.ok) {
    console.warn('Failed to get thoughts from OpenAI');
    return [];
  }

  // Stream thoughts as they arrive
  const reader = response.body?.getReader();
  if (!reader) return [];

  const decoder = new TextDecoder();
  let buffer = '';
  const thoughts = [];
  let thoughtIndex = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value, { stream: true });
    buffer += chunk;

    // Extract thoughts from buffer with [bt] and [et] markers
    let startIndex = buffer.indexOf('[bt]');
    while (startIndex !== -1) {
      const endIndex = buffer.indexOf('[et]', startIndex);
      if (endIndex !== -1) {
        const thought = buffer.substring(startIndex + 4, endIndex).trim();
        if (thought && !thoughts.includes(thought)) {
          thoughts.push(thought);
          
          // Send thought immediately as it's found
          self.postMessage({ type: "thought", thought, index: thoughtIndex++ });
          
          // Process this thought immediately to generate enhanced response
          let contextPrompt = `<|im_start|>user\n${userInput}<|im_end|>\n`;
          if (immediateResponse) {
            contextPrompt += `<|im_start|>assistant\n${immediateResponse}<|im_end|>\n`;
          }
          contextPrompt += `<|im_start|>knowledge\n${thought}<|im_end|>\n<|im_start|>assistant\n`;

          const result = await llm(contextPrompt, {
            max_new_tokens: 50,
            temperature: 1,
            do_sample: false,
            return_full_text: false,
            repetition_penalty: 1.2,
            top_p: 0.9,
            top_k: 50,
            pad_token_id: 2,
            eos_token_id: 2,
          });

          let response = "";
          if (Array.isArray(result) && result[0]?.generated_text) {
            response = result[0].generated_text;
          } else if (result?.generated_text) {
            response = result.generated_text;
          }
          
          response = response
            .replace(/<\|im_start\|>/g, "")
            .replace(/<\|im_end\|>/g, "")
            .replace(/^assistant\s*/i, "")
            .split("\n")[0]
            .trim();

          if (response) {
            // Send enhanced response message regardless of TTS
            self.postMessage({ type: "enhanced_response", response });
            
            // Add to TTS stream if available
            if (splitter) {
              console.log('Pushing enhanced response to TTS splitter:', response);
              splitter.push(" " + response);
            }
          }
        }
        // Remove processed thought from buffer
        buffer = buffer.substring(endIndex + 4);
        startIndex = buffer.indexOf('[bt]');
      } else {
        // No complete thought yet, wait for more chunks
        break;
      }
    }

    // Check for [done] token
    if (buffer.includes('[done]')) {
      break;
    }
  }

  return thoughts;
};

// processThoughts function removed - now integrated into generateAndProcessThoughts

/**
 * Transcribe the audio buffer and generate responses - FOLLOWING WEBGPU-DEMO
 * @param {Float32Array} buffer The audio buffer
 */
const speechToSpeech = async (buffer) => {
  isPlaying = true;

  // 1. Transcribe the audio from the user
  const text = await transcriber(buffer).then(({ text }) => text.trim());
  if (["", "[BLANK_AUDIO]"].includes(text)) {
    // If the transcription is empty or a blank audio, we skip the rest of the processing
    isPlaying = false;
    return;
  }
  messages.push({ role: "user", content: text });
  self.postMessage({ type: "transcription", text });

  // Set up text-to-speech streaming
  if (!tts) {
    console.error('TTS not initialized, cannot speak response');
    return;
  }
  
  const splitter = new TextSplitterStream();
  console.log('Creating TTS stream with voice:', voice, 'TTS object:', tts);
  
  // Create stream - note: don't pass voice if it's undefined
  const streamOptions = voice ? { voice } : {};
  const stream = tts.stream(splitter, streamOptions);
  console.log('TTS stream created successfully with options:', streamOptions);
  
  // Start TTS processing in background
  (async () => {
    let chunkCount = 0;
    self.postMessage({ type: "tts_start", text: "Starting TTS" });
    
    try {
      for await (const chunk of stream) {
        chunkCount++;
        console.log(`TTS chunk ${chunkCount}:`, chunk);
        
        // Extract audio from RawAudio object
        let audioData;
        const text = chunk.text || chunk.content || '';
        
        if (chunk.audio) {
          // Check if it's a RawAudio object with nested audio property
          if (chunk.audio.audio && chunk.audio.audio instanceof Float32Array) {
            audioData = chunk.audio.audio;
            console.log(`  - Extracted audio from RawAudio object, length=${audioData.length}, sample_rate=${chunk.audio.sampling_rate}`);
          } else if (chunk.audio instanceof Float32Array) {
            audioData = chunk.audio;
            console.log(`  - Audio is direct Float32Array, length=${audioData.length}`);
          } else {
            console.log(`  - Unknown audio format:`, chunk.audio);
          }
        }
        
        console.log(`  - text="${text}", audio extracted=${!!audioData}, audio length=${audioData?.length}`);
        
        if (audioData && audioData.length > 0) {
          // Don't resample - let the audio context handle the sample rate
          // The play-worklet will play at the correct rate
          console.log(`  - Sending audio at original sample rate: ${chunk.audio.sampling_rate || 'unknown'}Hz`);
          self.postMessage({ type: "output", text: text, result: audioData });
        }
      }
    } catch (error) {
      console.error('Error in TTS stream:', error);
    }
    
    console.log('TTS stream completed, total chunks:', chunkCount);
    self.postMessage({ type: "tts_end", text: "TTS complete" });
  })();

  // 2. Generate immediate response using the SmolLM conversation filler model
  const simplePrompt = `User: ${text}\nAssistant:`;
  
  const immediateResult = await llm(simplePrompt, {
    max_new_tokens: 25,
    temperature: 0.7,
    do_sample: true,
    return_full_text: false,
    repetition_penalty: 1.1,
    top_p: 0.9,
  });

  let immediateResponse = "";
  if (Array.isArray(immediateResult) && immediateResult[0]?.generated_text) {
    immediateResponse = immediateResult[0].generated_text;
  } else if (immediateResult?.generated_text) {
    immediateResponse = immediateResult.generated_text;
  }

  // Clean up the response
  immediateResponse = immediateResponse
    .replace(/User:.*$/gi, "")
    .replace(/Assistant:\s*/gi, "")
    .split("\n")[0]
    .trim();

  if (immediateResponse) {
    messages.push({ role: "assistant", content: immediateResponse });
    console.log('Pushing immediate response to TTS splitter:', immediateResponse);
    
    // Push text to TTS
    try {
      splitter.push(immediateResponse);
      console.log('Text pushed to splitter successfully:', immediateResponse);
    } catch (error) {
      console.error('Error pushing text to splitter:', error);
    }
    
    self.postMessage({ type: "immediate_response", response: immediateResponse });
  }

  // 3. Generate and process thoughts as they stream in
  try {
    await generateAndProcessThoughts(messages, text, immediateResponse, splitter);
  } catch (error) {
    console.warn("Failed to generate thoughts:", error);
  }

  // Finally, close the stream to signal that no more text will be added.
  splitter.close();
  // Don't set isPlaying = false here, wait for playback_ended message
};

// Track the number of samples after the last speech chunk - EXACTLY FROM WEBGPU-DEMO
let postSpeechSamples = 0;
const resetAfterRecording = (offset = 0) => {
  self.postMessage({
    type: "status",
    status: "recording_end",
    message: "Transcribing...",
    duration: "until_next",
  });
  BUFFER.fill(0, offset);
  bufferPointer = offset;
  isRecording = false;
  postSpeechSamples = 0;
};

const dispatchForTranscriptionAndResetAudioBuffer = (overflow) => {
  // Get start and end time of the speech segment, minus the padding
  const now = Date.now();
  const end =
    now - ((postSpeechSamples + SPEECH_PAD_SAMPLES) / INPUT_SAMPLE_RATE) * 1000;
  const start = end - (bufferPointer / INPUT_SAMPLE_RATE) * 1000;
  const duration = end - start;
  const overflowLength = overflow?.length ?? 0;

  // Send the audio buffer to the worker
  const buffer = BUFFER.slice(0, bufferPointer + SPEECH_PAD_SAMPLES);

  const prevLength = prevBuffers.reduce((acc, b) => acc + b.length, 0);
  const paddedBuffer = new Float32Array(prevLength + buffer.length);
  let offset = 0;
  for (const prev of prevBuffers) {
    paddedBuffer.set(prev, offset);
    offset += prev.length;
  }
  paddedBuffer.set(buffer, offset);
  speechToSpeech(paddedBuffer);

  // Set overflow (if present) and reset the rest of the audio buffer
  if (overflow) {
    BUFFER.set(overflow, 0);
  }
  resetAfterRecording(overflowLength);
};

// Previous buffers FIFO queue - EXACTLY FROM WEBGPU-DEMO
let prevBuffers = [];

// For text mode processing
async function processTextMode(text, enableTTS = false) {
  isPlaying = true;
  messages.push({ role: "user", content: text });
  
  // Generate immediate response
  const simplePrompt = `User: ${text}\nAssistant:`;
  const immediateResult = await llm(simplePrompt, {
    max_new_tokens: 25,
    temperature: 0.7,
    do_sample: true,
    return_full_text: false,
    repetition_penalty: 1.1,
    top_p: 0.9,
  });

  let immediateResponse = "";
  if (Array.isArray(immediateResult) && immediateResult[0]?.generated_text) {
    immediateResponse = immediateResult[0].generated_text;
  } else if (immediateResult?.generated_text) {
    immediateResponse = immediateResult.generated_text;
  }

  immediateResponse = immediateResponse
    .replace(/User:.*$/gi, "")
    .replace(/Assistant:\s*/gi, "")
    .split("\n")[0]
    .trim();

  if (immediateResponse) {
    messages.push({ role: "assistant", content: immediateResponse });
    self.postMessage({ type: "immediate_response", response: immediateResponse });
    
    if (enableTTS) {
      const splitter = new TextSplitterStream();
      const stream = tts.stream(splitter, { voice });
      (async () => {
        for await (const { text: chunkText, audio } of stream) {
          self.postMessage({ type: "output", text: chunkText, result: audio });
        }
      })();
      splitter.push(immediateResponse);
      
      // Generate and speak thoughts
      try {
        await generateAndProcessThoughts(messages, text, immediateResponse, splitter);
      } catch (error) {
        console.warn("Failed to generate thoughts:", error);
      }
      
      splitter.close();
    } else {
      // Just generate thoughts without TTS
      try {
        await generateAndProcessThoughts(messages, text, immediateResponse, null);
      } catch (error) {
        console.warn("Failed to generate thoughts:", error);
      }
    }
  }
  
  isPlaying = false;
}

// Message handler - FOLLOWING WEBGPU-DEMO STRUCTURE
self.onmessage = async (event) => {
  const { type } = event.data;

  // refuse new audio while playing back
  if (type === "audio" && isPlaying) return;

  switch (type) {
    case "init":
      // Already initialized
      self.postMessage({ 
        type: "status", 
        status: "ready", 
        voices: tts.voices,
        message: "All models loaded" 
      });
      return;
      
    case "set_voice":
      voice = event.data.voice;
      return;
      
    case "playback_ended":
      isPlaying = false;
      return;
      
    case "process_text":
      // For text mode
      const text = event.data.text;
      const enableTTS = event.data.enableTTS || false;
      if (text) {
        await processTextMode(text, enableTTS);
      }
      return;
      
    case "end_call":
      messages = [];
      return;
  }

  // Audio processing - EXACTLY FROM WEBGPU-DEMO
  // The vad-processor sends { type: 'audio', audio: Float32Array }
  // The unified-pipeline forwards as { type: 'audio', buffer: Float32Array }
  const buffer = event.data.buffer || event.data.audio;
  if (type !== "audio" || !buffer) return;
  
  const wasRecording = isRecording; // Save current state
  const isSpeech = await vad(buffer);

  if (!wasRecording && !isSpeech) {
    // We are not recording, and the buffer is not speech,
    // so we will probably discard the buffer. So, we insert
    // into a FIFO queue with maximum size of MAX_NUM_PREV_BUFFERS
    if (prevBuffers.length >= MAX_NUM_PREV_BUFFERS) {
      // If the queue is full, we discard the oldest buffer
      prevBuffers.shift();
    }
    prevBuffers.push(buffer);
    return;
  }

  const remaining = BUFFER.length - bufferPointer;
  if (buffer.length >= remaining) {
    // The buffer is larger than (or equal to) the remaining space in the global buffer,
    // so we perform transcription and copy the overflow to the global buffer
    BUFFER.set(buffer.subarray(0, remaining), bufferPointer);
    bufferPointer += remaining;

    // Dispatch the audio buffer
    const overflow = buffer.subarray(remaining);
    dispatchForTranscriptionAndResetAudioBuffer(overflow);
    return;
  } else {
    // The buffer is smaller than the remaining space in the global buffer,
    // so we copy it to the global buffer
    BUFFER.set(buffer, bufferPointer);
    bufferPointer += buffer.length;
  }

  if (isSpeech) {
    if (!isRecording) {
      // Indicate start of recording
      self.postMessage({
        type: "status",
        status: "recording_start",
        message: "Listening...",
        duration: "until_next",
      });
    }
    // Start or continue recording
    isRecording = true;
    postSpeechSamples = 0; // Reset the post-speech samples
    return;
  }

  postSpeechSamples += buffer.length;

  // At this point we're confident that we were recording (wasRecording === true), but the latest buffer is not speech.
  // So, we check whether we have reached the end of the current audio chunk.
  if (postSpeechSamples < MIN_SILENCE_DURATION_SAMPLES) {
    // There was a short pause, but not long enough to consider the end of a speech chunk
    // (e.g., the speaker took a breath), so we continue recording
    return;
  }

  if (bufferPointer < MIN_SPEECH_DURATION_SAMPLES) {
    // The entire buffer (including the new chunk) is smaller than the minimum
    // duration of a speech chunk, so we can safely discard the buffer.
    resetAfterRecording();
    return;
  }

  dispatchForTranscriptionAndResetAudioBuffer();
};

})().catch(error => {
  console.error('Worker initialization error:', error);
  self.postMessage({ error: error.message });
});