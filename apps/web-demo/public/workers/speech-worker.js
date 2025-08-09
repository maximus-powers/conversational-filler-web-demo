import {
  // VAD
  AutoModel,
  
  // Speech recognition
  Tensor,
  pipeline,
} from "@huggingface/transformers";

import { KokoroTTS, TextSplitterStream } from "kokoro-js";

// Audio constants
const INPUT_SAMPLE_RATE = 16000;
const SPEECH_THRESHOLD = 0.3;
const EXIT_THRESHOLD = 0.1;
const MIN_SILENCE_DURATION_MS = 400;
const MIN_SILENCE_DURATION_SAMPLES = (MIN_SILENCE_DURATION_MS * INPUT_SAMPLE_RATE) / 1000;
const SPEECH_PAD_MS = 80;
const SPEECH_PAD_SAMPLES = (SPEECH_PAD_MS * INPUT_SAMPLE_RATE) / 1000;
const MIN_SPEECH_DURATION_SAMPLES = (250 * INPUT_SAMPLE_RATE) / 1000; // 250 ms
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

// Load models
const silero_vad = await AutoModel.from_pretrained(
  "onnx-community/silero-vad",
  {
    config: { model_type: "custom" },
    dtype: "fp32",
  },
).catch((error) => {
  self.postMessage({ error });
  throw error;
});

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

const llm_model_id = "maximuspowers/smollm-convo-filler-onnx-official";
const llm = await pipeline("text-generation", llm_model_id, {
  dtype: "fp32", 
  device: "webgpu",
});

await llm("test", { max_new_tokens: 1 }); // Compile shaders

const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
let voice;
const tts = await KokoroTTS.from_pretrained(model_id, {
  dtype: "fp32",
  device: "webgpu",
});

let messages = [];
let prevBuffers = [];

// Don't send ready immediately - wait for init message to avoid race condition
let isInitialized = false;

// Global audio buffer to store incoming audio
const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;

// Initial state for VAD
const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);

// Whether we are in the process of adding audio to the buffer
let isRecording = false;
let isPlaying = false;

/**
 * Perform Voice Activity Detection (VAD)
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
 * Generate thoughts from the timeline API
 */
async function generateThoughts(conversationHistory) {
  try {
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
      console.warn('Failed to get thoughts from API');
      return [];
    }

    const reader = response.body?.getReader();
    if (!reader) return [];

    const decoder = new TextDecoder();
    let thoughtBuffer = '';
    const thoughts = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      thoughtBuffer += chunk;

      // Extract thoughts from buffer with [bt] and [et] markers
      const regex = /\[bt\](.*?)\[et\]/g;
      let match;
      while ((match = regex.exec(thoughtBuffer)) !== null) {
        const thought = match[1].trim();
        if (thought && !thoughts.includes(thought)) {
          thoughts.push(thought);
          self.postMessage({ type: "thought", thought, index: thoughts.length - 1 });
        }
      }

      if (thoughtBuffer.includes('[done]')) {
        break;
      }
    }

    return thoughts;
  } catch (error) {
    console.warn("Failed to generate thoughts:", error);
    return [];
  }
}

/**
 * Process thoughts and generate enhanced responses
 */
async function processThoughts(thoughts, userInput) {
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
      self.postMessage({ 
        type: "enhanced_response", 
        response,
        model: "SmolLM",
        content: response
      });
    }
  }

  return responses;
}

/**
 * Main speech-to-speech processing
 */
const speechToSpeech = async (buffer) => {
  isPlaying = true;

  // 1. Transcribe the audio
  const transcriptionResult = await transcriber(buffer);
  const text = transcriptionResult.text?.trim() || "";
  
  if (["", "[BLANK_AUDIO]"].includes(text)) {
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
      self.postMessage({ type: "audio_output", text, audio });
    }
  })();

  // 2. Generate immediate response using SmolLM
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
    self.postMessage({ 
      type: "immediate_response", 
      response: immediateResponse,
      model: "SmolLM",
      content: immediateResponse
    });

    splitter.push(immediateResponse);
  }

  // 3. Generate thoughts and enhanced responses
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
    console.warn("Failed to generate enhanced responses:", error);
  }

  // Finally, close the stream to signal that no more text will be added.
  splitter.close();

  isPlaying = false;
};

// Track samples after last speech
let postSpeechSamples = 0;
const resetAfterRecording = (offset = 0) => {
  self.postMessage({
    type: "status",
    status: "recording_end",
    message: "Processing...",
  });
  BUFFER.fill(0, offset);
  bufferPointer = offset;
  isRecording = false;
  postSpeechSamples = 0;
};

const dispatchForTranscriptionAndResetAudioBuffer = (overflow) => {
  const overflowLength = overflow?.length ?? 0;
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

  if (overflow) {
    BUFFER.set(overflow, 0);
  }
  resetAfterRecording(overflowLength);
};

function greet(text) {
  isPlaying = true;
  const splitter = new TextSplitterStream();
  const stream = tts.stream(splitter, { voice });
  (async () => {
    for await (const { text: chunkText, audio } of stream) {
      self.postMessage({ type: "audio_output", text: chunkText, audio });
    }
  })();
  splitter.push(text);
  splitter.close();
  messages.push({ role: "assistant", content: text });
}

// Message handler
self.onmessage = async (event) => {
  const { type, buffer } = event.data;

  if (type === "audio" && isPlaying) return;

  switch (type) {
    case "init":
      if (!isInitialized) {
        isInitialized = true;
        self.postMessage({
          type: "status",
          status: "ready",
          message: "Ready!",
          voices: tts.voices,
        });
      }
      return;
    case "start_recording":
      const name = tts.voices[voice ?? "af_heart"]?.name ?? "Assistant";
      greet(`Hey there, my name is ${name}! How can I help you today?`);
      return;
    case "end_recording":
      messages = [];
      return;
    case "set_voice":
      voice = event.data.voice;
      return;
    case "playback_ended":
      isPlaying = false;
      return;
    case "audio":
      break; // Continue to audio processing below
    default:
      return;
  }

  // Audio processing
  const wasRecording = isRecording;
  const isSpeech = await vad(buffer);

  if (!wasRecording && !isSpeech) {
    if (prevBuffers.length >= MAX_NUM_PREV_BUFFERS) {
      prevBuffers.shift();
    }
    prevBuffers.push(buffer);
    return;
  }

  const remaining = BUFFER.length - bufferPointer;
  if (buffer.length >= remaining) {
    BUFFER.set(buffer.subarray(0, remaining), bufferPointer);
    bufferPointer += remaining;
    const overflow = buffer.subarray(remaining);
    dispatchForTranscriptionAndResetAudioBuffer(overflow);
    return;
  } else {
    BUFFER.set(buffer, bufferPointer);
    bufferPointer += buffer.length;
  }

  if (isSpeech) {
    if (!isRecording) {
      self.postMessage({
        type: "status",
        status: "recording_start",
        message: "Listening...",
      });
    }
    isRecording = true;
    postSpeechSamples = 0;
    return;
  }

  postSpeechSamples += buffer.length;

  if (postSpeechSamples < MIN_SILENCE_DURATION_SAMPLES) {
    return;
  }

  if (bufferPointer < MIN_SPEECH_DURATION_SAMPLES) {
    resetAfterRecording();
    return;
  }

  dispatchForTranscriptionAndResetAudioBuffer();
};