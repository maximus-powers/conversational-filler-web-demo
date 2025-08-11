import { AutoModel, Tensor, pipeline } from "@huggingface/transformers";

import { KokoroTTS, TextSplitterStream } from "kokoro-js";
console.log('KokoroTTS imported:', typeof KokoroTTS, 'TextSplitterStream:', typeof TextSplitterStream);

// wrap everything in async IIFE to handle top-level await
(async () => {

// Audio constants (also defined in ../audio-constants.ts)
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

// init TTS
const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
let voice;
let tts;

try {
  console.log('Initializing TTS with model:', model_id);
  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator; 
  const ttsDevice = hasWebGPU ? "webgpu" : "wasm"; // fallback to wasm if webgpu doens't work
  
  tts = await KokoroTTS.from_pretrained(model_id, {
    dtype: "fp32",
    device: ttsDevice,
  });
  console.log('TTS initialized successfully');  
} catch (error) {
  console.error('Failed to initialize TTS:', error);
  self.postMessage({ type: "error", error: `TTS initialization failed: ${error.message}` });
}

// init VAD
console.log('Initializing VAD');
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
console.log('VAD initialized successfully');


// init STT
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

console.log('Initializing TTS.');
self.postMessage({ 
  type: "info", 
  message: "Loading Whisper TTS...",
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
await transcriber(new Float32Array(INPUT_SAMPLE_RATE));
console.log('TTS Initializaed.');
self.postMessage({ 
  type: "info", 
  message: "Whisper model loaded successfully"
});

console.log('Initializing SmolLM.');
const llm_model_id = "maximuspowers/smollm-convo-filler-onnx-official";
const llm = await pipeline("text-generation", llm_model_id, {
  dtype: "fp32", 
  device: "webgpu",
});
await llm("test", { max_new_tokens: 1 }); // Compile shaders
let messages = [];
if (!voice && tts.voices) {
  voice = Object.keys(tts.voices)[0] || "af_heart";
}
console.log('SmolLM initialized successfully.');
self.postMessage({
  type: "status",
  status: "ready",
  message: "Ready!",
  voices: tts.voices,
});

// Global audio buffer to store incoming audio
const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;

// Initial state for VAD
const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);



let isRecording = false;
let isPlaying = false;

// Silence token tracking
let silenceTimer = null;
let isGeneratingSilence = false;

async function vad(buffer) {
  const input = new Tensor("float32", buffer, [1, buffer.length]);
  const { stateN, output } = await silero_vad({ input, sr, state });
  state = stateN; 
  const isSpeech = output.data[0];
  return (
    isSpeech > SPEECH_THRESHOLD || (isRecording && isSpeech >= EXIT_THRESHOLD)
  );
}

function startSilenceTimer(splitter) {
  if (silenceTimer) {
    clearTimeout(silenceTimer);
  }
  
  silenceTimer = setTimeout(() => {
    if (!isGeneratingSilence && splitter) {
      isGeneratingSilence = true;
      splitter.push("<sil>");
      self.postMessage({ type: "silence_token", token: "<sil>" });
    }
  }, 1000); // 1 second delay
}

function clearSilenceTimer() {
  if (silenceTimer) {
    clearTimeout(silenceTimer);
    silenceTimer = null;
  }
  isGeneratingSilence = false;
}

const generateAndProcessThoughts = async (conversationHistory, userInput, immediateResponse, splitter) => {
  // Clear any existing silence timer since we're starting thought processing
  clearSilenceTimer();
  
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
    // Start silence timer if no thoughts available
    if (splitter) {
      startSilenceTimer(splitter);
    }
    return [];
  }

  const reader = response.body?.getReader();
  if (!reader) {
    // Start silence timer if no thoughts available
    if (splitter) {
      startSilenceTimer(splitter);
    }
    return [];
  }

  const decoder = new TextDecoder();
  let buffer = '';
  const thoughts = [];
  let thoughtIndex = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value, { stream: true });
    buffer += chunk;

    // extract thoughts from buffer with [bt] and [et] markers
    let startIndex = buffer.indexOf('[bt]');
    while (startIndex !== -1) {
      const endIndex = buffer.indexOf('[et]', startIndex);
      if (endIndex !== -1) {
        const thought = buffer.substring(startIndex + 4, endIndex).trim();
        if (thought && !thoughts.includes(thought)) {
          thoughts.push(thought);
          
          // Reset silence timer since we received a new thought
          clearSilenceTimer();
          
          self.postMessage({ type: "thought", thought, index: thoughtIndex++ });
          
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
            self.postMessage({ type: "enhanced_response", response });
            if (splitter) { // add to TTS if available
              splitter.push(" " + response);
            }
          }
        }
        // rm processed thought from buffer
        buffer = buffer.substring(endIndex + 4);
        startIndex = buffer.indexOf('[bt]');
      } else {
        break;
      }
    }

    if (buffer.includes('[done]')) {
      break;
    }
  }

  // Start silence timer after all thoughts have been processed
  if (splitter && thoughts.length === 0) {
    // No thoughts were generated, start silence timer
    startSilenceTimer(splitter);
  } else if (splitter) {
    // Thoughts were generated, start timer for potential silence after them
    startSilenceTimer(splitter);
  }

  return thoughts;
};

const speechToSpeech = async (buffer) => {
  isPlaying = true;
  
  // Clear any existing silence timer when starting new speech processing
  clearSilenceTimer();

  // transcribe
  const text = await transcriber(buffer).then(({ text }) => text.trim());
  if (["", "[BLANK_AUDIO]"].includes(text)) {
    isPlaying = false;
    return;
  }
  messages.push({ role: "user", content: text });
  self.postMessage({ type: "transcription", text });

  if (!tts) {
    console.error('TTS not initialized, cannot speak response');
    return;
  }
  
  const splitter = new TextSplitterStream();
  
  // create stream - note: don't pass voice if it's undefined
  const streamOptions = voice ? { voice } : {};
  const stream = tts.stream(splitter, streamOptions);
  console.log('TTS stream created successfully with options:', streamOptions);
  
  // start TTS processing
  (async () => {
    let chunkCount = 0;
    self.postMessage({ type: "tts_start", text: "Starting TTS" });
    
    try {
      for await (const chunk of stream) {
        chunkCount++;
        console.log(`TTS chunk ${chunkCount}:`, chunk);
        
        let audioData;
        const text = chunk.text || chunk.content || '';
        
        if (chunk.audio) {
          if (chunk.audio.audio && chunk.audio.audio instanceof Float32Array) {
            audioData = chunk.audio.audio;
          } else if (chunk.audio instanceof Float32Array) {
            audioData = chunk.audio;
          }
        }
                
        if (audioData && audioData.length > 0) {
          self.postMessage({ type: "output", text: text, result: audioData });
        }
      }
    } catch (error) {
      console.error('Error in TTS stream:', error);
    }
    self.postMessage({ type: "tts_end", text: "TTS complete" });
  })();

  // generate immediate response using smollm
  const simplePrompt = `User: ${text}\nAssistant:`;
  const immediateResult = await llm(simplePrompt, {
    max_new_tokens: 25,
    temperature: 1,
    do_sample: true,
    return_full_text: false,
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
    
    // push text to TTS
    try {
      splitter.push(immediateResponse);
    } catch (error) {
      console.error('Error pushing text to splitter:', error);
    }
    
    self.postMessage({ type: "immediate_response", response: immediateResponse });
  }

  // generate and process thoughts as they stream in
  try {
    await generateAndProcessThoughts(messages, text, immediateResponse, splitter);
  } catch (error) {
    console.warn("Failed to generate thoughts:", error);
  }

  splitter.close();
};

// track number of speech samples after the last speech segment
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
  const overflowLength = overflow?.length ?? 0;

  // send the audio buffer to the worker
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

  // set overflow (if present) and reset the rest of the audio buffer
  if (overflow) {
    BUFFER.set(overflow, 0);
  }
  resetAfterRecording(overflowLength);
};

// prev buffers FIFO queue
let prevBuffers = [];


async function processTextMode(text, enableTTS = false) {
  isPlaying = true;
  
  // Clear any existing silence timer when starting text processing
  clearSilenceTimer();
  
  messages.push({ role: "user", content: text });
  
  // generate immediate response
  const simplePrompt = `User: ${text}\nAssistant:`;
  const immediateResult = await llm(simplePrompt, {
    max_new_tokens: 25,
    temperature: 1,
    do_sample: true,
    return_full_text: false,
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
      
      try {
        await generateAndProcessThoughts(messages, text, immediateResponse, splitter);
      } catch (error) {
        console.warn("Failed to generate thoughts:", error);
      }
      
      splitter.close();
    } else {
      // generate thoughts without tts
      try {
        await generateAndProcessThoughts(messages, text, immediateResponse, null);
      } catch (error) {
        console.warn("Failed to generate thoughts:", error);
      }
    }
  }
  
  isPlaying = false;
}

// message handler
self.onmessage = async (event) => {
  const { type } = event.data;

  // refuse new audio while playing back
  if (type === "audio" && isPlaying) return;

  switch (type) {
    case "init":
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
      clearSilenceTimer(); // Clear silence timer when playback ends
      return;
      
    case "process_text":
      // for text mode
      const text = event.data.text;
      const enableTTS = event.data.enableTTS || false;
      if (text) {
        await processTextMode(text, enableTTS);
      }
      return;
      
    case "end_call":
      messages = [];
      clearSilenceTimer(); // Clear silence timer when ending call
      return;
  }

  // audio processing
  const buffer = event.data.buffer || event.data.audio;
  if (type !== "audio" || !buffer) return;
  
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
        duration: "until_next",
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

})().catch(error => {
  console.error('Worker initialization error:', error);
  self.postMessage({ error: error.message });
});