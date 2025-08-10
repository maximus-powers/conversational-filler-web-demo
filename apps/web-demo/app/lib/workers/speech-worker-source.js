// Worker implementation following webgpu-demo structure exactly
// Only differences: using our LLM model and OpenAI API endpoint

import {
  // VAD
  AutoModel,
  
  // Speech recognition  
  Tensor,
  pipeline,
} from "@huggingface/transformers";

import { KokoroTTS, TextSplitterStream } from "kokoro-js";

// Wrap everything in async IIFE to handle top-level await
(async () => {

// Audio constants - matching webgpu-demo exactly
const INPUT_SAMPLE_RATE = 16000;
const INPUT_SAMPLE_RATE_MS = INPUT_SAMPLE_RATE / 1000;
const SPEECH_THRESHOLD = 0.3;
const EXIT_THRESHOLD = 0.1;
const MIN_SILENCE_DURATION_MS = 400;
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
let voice = "af_heart"; // Default voice
const tts = await KokoroTTS.from_pretrained(model_id, {
  dtype: "fp32",
  device: "webgpu",
});

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

await transcriber(new Float32Array(INPUT_SAMPLE_RATE)); // Compile shaders

// Load SmolLM for conversation (OUR MODEL)
const llm_model_id = "maximuspowers/smollm-convo-filler-onnx-official";
const llm = await pipeline("text-generation", llm_model_id, {
  dtype: "fp32", 
  device: "webgpu",
});

await llm("test", { max_new_tokens: 1 }); // Compile shaders

let messages = [];
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
 * Generate thoughts from OpenAI for the conversation - OUR API ENDPOINT
 */
const generateThoughts = async (conversationHistory) => {
  // Build conversation lines for OpenAI
  const conversationLines = [];
  for (const msg of conversationHistory) {
    if (msg.role === 'user') {
      conversationLines.push(`User: ${msg.content}`);
    } else if (msg.role === 'assistant') {
      conversationLines.push(`Responder: ${msg.content}`);
    }
  }
  
  const conversationText = conversationLines.join('\n');

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

  // Our API returns JSON with thoughts array
  const data = await response.json();
  const thoughts = data.thoughts || [];
  
  // Send each thought as it's received
  thoughts.forEach(thought => {
    self.postMessage({ type: "thought", thought });
  });
  
  return thoughts;
};

/**
 * Process thoughts and generate enhanced responses
 */
const processThoughts = async (thoughts, userInput) => {
  const responses = [];
  
  for (const thought of thoughts) {
    // Build context including previous responses
    let contextPrompt = `<|im_start|>user\n${userInput}<|im_end|>\n`;
    
    // Add previous responses if any
    responses.forEach((resp, idx) => {
      if (idx < thoughts.length) {
        contextPrompt += `<|im_start|>knowledge\n${thoughts[idx]}<|im_end|>\n`;
      }
      contextPrompt += `<|im_start|>assistant\n${resp}<|im_end|>\n`;
    });
    
    // Add current thought
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
      responses.push(response);
      self.postMessage({ type: "enhanced_response", response });
    }
  }

  return responses;
};

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
  const splitter = new TextSplitterStream();
  const stream = tts.stream(splitter, {
    voice,
  });
  (async () => {
    for await (const { text, audio } of stream) {
      self.postMessage({ type: "output", text, result: audio });
    }
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
    splitter.push(immediateResponse);
    self.postMessage({ type: "immediate_response", response: immediateResponse });
  }

  // 3. Generate thoughts asynchronously and enhance responses
  try {
    const thoughts = await generateThoughts(messages);
    if (thoughts.length > 0) {
      const enhancedResponses = await processThoughts(thoughts, text);
      
      // Add enhanced responses to TTS stream
      for (const response of enhancedResponses) {
        splitter.push(" " + response);
      }
    }
  } catch (error) {
    console.warn("Failed to generate thoughts:", error);
  }

  // Finally, close the stream to signal that no more text will be added.
  splitter.close();
  isPlaying = false;
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
        const thoughts = await generateThoughts(messages);
        if (thoughts.length > 0) {
          const enhancedResponses = await processThoughts(thoughts, text);
          for (const response of enhancedResponses) {
            splitter.push(" " + response);
          }
        }
      } catch (error) {
        console.warn("Failed to generate thoughts:", error);
      }
      
      splitter.close();
    } else {
      // Just generate thoughts without TTS
      try {
        const thoughts = await generateThoughts(messages);
        if (thoughts.length > 0) {
          await processThoughts(thoughts, text);
        }
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