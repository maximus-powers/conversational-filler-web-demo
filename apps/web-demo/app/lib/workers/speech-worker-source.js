import { AutoTokenizer, AutoModelForCausalLM, AutoModel, Tensor, pipeline } from "@huggingface/transformers";

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
console.log('TTS Initialized.');
self.postMessage({ 
  type: "info", 
  message: "Whisper model loaded successfully"
});

console.log('Initializing SmolLM.');
let llm_model_id = "maximuspowers/smollm-convo-filler-onnx-official";

// pipeline seemed to have a bug with loading the custom tokenizer
let tokenizer = await AutoTokenizer.from_pretrained(llm_model_id, {
  dtype: "fp32",
  device: "webgpu",
});
let llm = await AutoModelForCausalLM.from_pretrained(llm_model_id, {
  dtype: "fp32", 
  device: "webgpu",
});

// needs to warm up with proper format and compile shaders (this is why we were getting such bad responses before)
const warmupPrompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>knowledge\n<|sil|><|im_end|>\n";
let warmupInput = tokenizer(warmupPrompt);
await llm.generate({
  ...warmupInput,
  max_new_tokens: 10,
  do_sample: false,
});

async function switchModel(newModelId) {
  console.log(`Switching to model: ${newModelId}`);
  self.postMessage({ 
    type: "info", 
    message: `Loading model: ${newModelId}...`,
    duration: "until_next"
  });
  
  llm_model_id = newModelId;
  
  // Load new tokenizer and model
  tokenizer = await AutoTokenizer.from_pretrained(llm_model_id, {
    dtype: "fp32",
    device: "webgpu",
  });
  llm = await AutoModelForCausalLM.from_pretrained(llm_model_id, {
    dtype: "fp32", 
    device: "webgpu",
  });

  // Warm up the new model
  warmupInput = tokenizer(warmupPrompt);
  await llm.generate({
    ...warmupInput,
    max_new_tokens: 10,
    do_sample: false,
  });
  
  self.postMessage({ 
    type: "info", 
    message: `Model ${newModelId} loaded successfully`
  });
  console.log(`Model switched to: ${newModelId}`);
}
let messages = [];
let thoughtProvider = "gemini"; // default to Gemini
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

const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;

const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);

let isRecording = false;
let isPlaying = false;

let silenceTimer = null;
let isGeneratingSilence = false;
let isProcessingThought = false;
let thoughtQueue = [];
let streamComplete = false;

// timeline tracking
let conversationStartTime = null;
let conversationTurnStartTime = null;
let inferenceStartTime = null;
let firstResponseTime = null;
let firstThoughtTime = null;
let sttStartTime = null;
let sttEndTime = null;
let ttsStartTime = null;
let ttsEndTime = null;

async function vad(buffer) {
  const input = new Tensor("float32", buffer, [1, buffer.length]);
  const { stateN, output } = await silero_vad({ input, sr, state });
  state = stateN; 
  const isSpeech = output.data[0];
  return (
    isSpeech > SPEECH_THRESHOLD || (isRecording && isSpeech >= EXIT_THRESHOLD)
  );
}

const processThought = async (thought, userInput, thoughtResponsePairs, splitter) => {
  const thoughtProcessingStartTime = Date.now();
  self.postMessage({ 
    type: "thought_processing_start", 
    thought,
    timestamp: thoughtProcessingStartTime,
    turnOffset: thoughtProcessingStartTime - conversationTurnStartTime,
    turnStartTime: conversationTurnStartTime
  });

  let contextPrompt = `<|im_start|>user\n${userInput}<|im_end|>\n`;
  
  for (const pair of thoughtResponsePairs) {
    contextPrompt += `<|im_start|>knowledge\n${pair.thought}<|im_end|>\n<|im_start|>assistant\n${pair.response}<|im_end|>\n`;
  }
  if (thought.length > 0) {
    contextPrompt += `<|im_start|>knowledge\n${thought}<|im_end|>\n`;
  }

  console.log("DEBUG: SmolLM Prompt: ", contextPrompt);
  const inputs = tokenizer(contextPrompt);
  const outputs = await llm.generate({
    ...inputs,
    max_new_tokens: 128,
    temperature: 1.0,
    do_sample: false,
    pad_token_id: tokenizer.pad_token_id,
    eos_token_id: tokenizer.eos_token_id,
  });
  console.log("DEBUG: Output tokens:", [...outputs.data]);
  const newTokens = Array.from(outputs.data.slice(inputs.input_ids.data.length)).map(t => Number(t));
  const generatedText = tokenizer.decode(newTokens, { skip_special_tokens: true });
  console.log("DEBUG: Generated text:", generatedText);

  response = generatedText
    .replace(/<\|im_start\|>/g, "")
    .replace(/<\|im_end\|>/g, "")
    .replace(/^assistant\s*/i, "")
    .trim()
    .split("\n")[0];

  if (response) {
    let messageType;
    if (thought === "<|sil|>") {
      messageType = "filler_response";
      if (!firstResponseTime && conversationStartTime) {
        firstResponseTime = Date.now();
        self.postMessage({ 
          type: "first_response", 
          timestamp: firstResponseTime,
          timeFromStart: firstResponseTime - conversationStartTime,
          turnOffset: firstResponseTime - conversationTurnStartTime,
          turnStartTime: conversationTurnStartTime
        });
      }
    } else {
      messageType = "enhanced_response";
    }
    
    self.postMessage({ type: messageType, response, thoughtProvider });
    if (splitter) { // add to TTS if available
      splitter.push(thought === "" ? response : " " + response);
    }
    
    const thoughtProcessingEndTime = Date.now();
    self.postMessage({ 
      type: "thought_processing_end", 
      thought,
      response,
      timestamp: thoughtProcessingEndTime,
      turnOffset: thoughtProcessingEndTime - conversationTurnStartTime,
      turnStartTime: conversationTurnStartTime,
      duration: thoughtProcessingEndTime - thoughtProcessingStartTime,
      startTime: thoughtProcessingStartTime
    });
  }
  
  return response;
};

const processThoughtQueue = async (userInput, thoughtResponsePairs, splitter) => {
  if (isProcessingThought || thoughtQueue.length === 0) return;
  
  isProcessingThought = true;
  let localThoughtIndex = 0;
  while (thoughtQueue.length > 0) {
    const thought = thoughtQueue.shift();
    clearSilenceTimer();
    
    if (!firstThoughtTime && conversationStartTime) {
      firstThoughtTime = Date.now();
      self.postMessage({ 
        type: "first_thought", 
        timestamp: firstThoughtTime,
        timeFromStart: firstThoughtTime - conversationStartTime,
        turnOffset: firstThoughtTime - conversationTurnStartTime,
        turnStartTime: conversationTurnStartTime
      });
    }
    
    self.postMessage({ type: "thought", thought, index: localThoughtIndex++, thoughtProvider });
    const thoughtResponse = await processThought(thought, userInput, thoughtResponsePairs, splitter);
    
    if (thoughtResponse) {
      thoughtResponsePairs.push({ thought: thought, response: thoughtResponse });
    }
  }
  isProcessingThought = false;
  
  if (!isGeneratingSilence && !streamComplete) {
    startSilenceTimer(userInput, thoughtResponsePairs, splitter);
  }
};

function startSilenceTimer(userInput, thoughtResponsePairs, splitter) {
  if (silenceTimer) {
    clearTimeout(silenceTimer);
  }
  silenceTimer = setTimeout(async () => {
    if (!isGeneratingSilence && !isProcessingThought) {
      isGeneratingSilence = true;
      self.postMessage({ type: "silence_token", token: "<|sil|>" });
      await processThought("<|sil|>", userInput, thoughtResponsePairs, splitter);
    }
  }, 1000);
}
function clearSilenceTimer() {
  if (silenceTimer) {
    clearTimeout(silenceTimer);
    silenceTimer = null;
  }
  isGeneratingSilence = false;
}

const processInput = async (input, isVoiceMode, enableTTS) => {
  isPlaying = true;
  clearSilenceTimer();
  
  thoughtQueue = [];
  isProcessingThought = false;
  streamComplete = false;
  
  conversationStartTime = Date.now();
  conversationTurnStartTime = conversationStartTime; 
  inferenceStartTime = null;
  firstResponseTime = null;
  firstThoughtTime = null;
  sttStartTime = null;
  sttEndTime = null;
  ttsStartTime = null;
  ttsEndTime = null;
  
  self.postMessage({ 
    type: "conversation_turn_start", 
    timestamp: conversationStartTime,
    turnStartTime: conversationTurnStartTime,
    turnOffset: 0
  });

  let userText = input;
  
  if (isVoiceMode) {
    sttStartTime = Date.now();
    self.postMessage({ 
      type: "transcription_start", 
      timestamp: sttStartTime,
      turnOffset: sttStartTime - conversationTurnStartTime
    });
    
    userText = await transcriber(input).then(({ text }) => text.trim());
    
    sttEndTime = Date.now();
    if (["", "[BLANK_AUDIO]"].includes(userText)) {
      isPlaying = false;
      return;
    }
    self.postMessage({ 
      type: "transcription", 
      text: userText, 
      timestamp: sttEndTime,
      duration: sttEndTime - sttStartTime,
      startTime: sttStartTime,
      turnOffset: sttStartTime - conversationTurnStartTime,
      turnStartTime: conversationTurnStartTime
    });
  }
  
  messages.push({ role: "user", content: userText });

  let splitter = null;
  if (enableTTS && tts) {
    splitter = new TextSplitterStream();
    const streamOptions = voice ? { voice } : {};
    const stream = tts.stream(splitter, streamOptions);
    
    (async () => {
      let chunkCount = 0;
      ttsStartTime = Date.now();
      self.postMessage({ 
        type: "tts_start", 
        text: "Starting TTS", 
        timestamp: ttsStartTime,
        turnOffset: ttsStartTime - conversationTurnStartTime,
        turnStartTime: conversationTurnStartTime
      });
      
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
      ttsEndTime = Date.now();
      self.postMessage({ 
        type: "tts_end", 
        text: "TTS complete", 
        timestamp: ttsEndTime,
        turnOffset: ttsEndTime - conversationTurnStartTime,
        turnStartTime: conversationTurnStartTime,
        duration: ttsEndTime - ttsStartTime,
        startTime: ttsStartTime
      });
    })();
  }

  let thoughtResponsePairs = [];
  streamComplete = false;
  
  inferenceStartTime = Date.now();
  self.postMessage({ 
    type: "inference_start", 
    timestamp: inferenceStartTime,
    turnOffset: inferenceStartTime - conversationTurnStartTime,
    turnStartTime: conversationTurnStartTime
  });
  
  let thoughtsPromise = null;
  let thoughtStartTime = null;
  
  try {
    clearSilenceTimer();
    
    if (thoughtProvider === 'none') {
      console.log('Using local-only mode with silence tokens');
      
      thoughtStartTime = Date.now();
      self.postMessage({ 
        type: "thought_generation_start", 
        timestamp: thoughtStartTime,
        turnOffset: thoughtStartTime - conversationTurnStartTime,
        turnStartTime: conversationTurnStartTime
      });
      
      thoughtsPromise = Promise.resolve().then(async () => {
        for (let i = 0; i < 4; i++) {
          const silTokenTime = Date.now();
          self.postMessage({ 
            type: "individual_thought_received", 
            thought: "<|sil|>",
            timestamp: silTokenTime,
            thoughtIndex: i,
            apiRequestStartTime: thoughtStartTime,
            duration: 0, // Immediate for local generation
            turnOffset: silTokenTime - conversationTurnStartTime,
            turnStartTime: conversationTurnStartTime
          });
          thoughtQueue.push("<|sil|>"); // queue 4 sil tokens for smollm to process without thoughts
        }
        streamComplete = true;
        
        const thoughtEndTime = Date.now();
        self.postMessage({ 
          type: "thought_generation_end", 
          timestamp: thoughtEndTime,
          duration: thoughtEndTime - thoughtStartTime,
          startTime: thoughtStartTime
        });
      });
    } else {
      thoughtStartTime = Date.now();
      self.postMessage({ 
        type: "thought_generation_start", 
        timestamp: thoughtStartTime,
        turnOffset: thoughtStartTime - conversationTurnStartTime,
        turnStartTime: conversationTurnStartTime
      });
      
      const thoughtsEndpoint = '/api/chat-thoughts-gemini';
      console.log(`Fetching thoughts from ${thoughtProvider} using ${thoughtsEndpoint}`);
      
      thoughtsPromise = fetch(thoughtsEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messages
        }),
      });
    }
    
    const immediateResponse = await processThought("<|sil|>", userText, [], splitter);
    if (immediateResponse) {
      thoughtResponsePairs.push({ thought: "<|sil|>", response: immediateResponse });
      messages.push({ role: "assistant", content: immediateResponse });
    }
    
    if (thoughtsPromise && thoughtProvider !== 'none') {
      const thoughtsResponse = await thoughtsPromise;
      
        // sil token handling
        if (!thoughtsResponse.ok) {
          console.warn(`Failed to get thoughts from ${thoughtProvider}`);
          startSilenceTimer(userText, thoughtResponsePairs, splitter);
        } else {
          const reader = thoughtsResponse.body?.getReader();
          if (!reader) {
            startSilenceTimer(userText, thoughtResponsePairs, splitter);
          } else {
            const decoder = new TextDecoder();
            let buffer = '';
            const thoughts = [];

            while (true) {
              const { done, value } = await reader.read();
              if (done) {
                streamComplete = true;
                break;
              }
              
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
                    
                    const thoughtReceivedTime = Date.now();
                    self.postMessage({ 
                      type: "individual_thought_received", 
                      thought,
                      timestamp: thoughtReceivedTime,
                      thoughtIndex: thoughts.length - 1,
                      apiRequestStartTime: thoughtStartTime,
                      duration: thoughtReceivedTime - thoughtStartTime,
                      turnOffset: thoughtReceivedTime - conversationTurnStartTime,
                      turnStartTime: conversationTurnStartTime
                    });
                    
                    thoughtQueue.push(thought);
                    processThoughtQueue(userText, thoughtResponsePairs, splitter);
                  }
                  // rm processed thought from buffer
                  buffer = buffer.substring(endIndex + 4);
                  startIndex = buffer.indexOf('[bt]');
                } else {
                  break;
                }
              }

              if (buffer.includes('[done]')) {
                streamComplete = true;
                clearSilenceTimer();
                break;
              }
            }

            await processThoughtQueue(userText, thoughtResponsePairs, splitter);
          }
        }
        
        const thoughtEndTime = Date.now();
        self.postMessage({ 
          type: "thought_generation_end", 
          timestamp: thoughtEndTime,
          duration: thoughtEndTime - thoughtStartTime,
          startTime: thoughtStartTime
        });
    } else if (thoughtsPromise && thoughtProvider === 'none') {
      await thoughtsPromise;
      await processThoughtQueue(userText, thoughtResponsePairs, splitter);
    }
  } catch (error) {
    console.warn("Failed to generate thoughts:", error);
  }
  
  const fullResponse = thoughtResponsePairs.map(pair => pair.response).join(" ");
  if (fullResponse !== immediateResponse) {
    messages[messages.length - 1].content = fullResponse;
  }
  
  if (splitter) {
    splitter.close();
  }
  
  isPlaying = false;
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
  processInput(paddedBuffer, true, true);

  // set overflow (if present) and reset the rest of the audio buffer
  if (overflow) {
    BUFFER.set(overflow, 0);
  }
  resetAfterRecording(overflowLength);
};

// prev buffers FIFO queue
let prevBuffers = [];



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

    case "set_thought_provider":
      thoughtProvider = event.data.provider;
      console.log(`Thought provider set to: ${thoughtProvider}`);
      return;

    case "set_model":
      const newModelId = event.data.modelId;
      console.log(`Switching to model: ${newModelId}`);
      switchModel(newModelId);
      return;
      
    case "playback_ended":
      isPlaying = false;
      clearSilenceTimer(); 
      return;
      
    case "process_text":
      // for text mode
      const text = event.data.text;
      const enableTTS = event.data.enableTTS || false;
      if (text) {
        await processInput(text, false, enableTTS);
      }
      return;
      
    case "end_call":
      messages = [];
      thoughtQueue = [];
      isProcessingThought = false;
      streamComplete = false;
      clearSilenceTimer();
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