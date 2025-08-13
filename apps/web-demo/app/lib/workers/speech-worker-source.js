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

const tokenizer = llm.tokenizer;
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

const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;

const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);

let isRecording = false;
let isPlaying = false;

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

const processThought = async (thought, userInput, thoughtResponsePairs, splitter) => {
  let contextPrompt = `<|im_start|>user\n${userInput}<|im_end|>\n`;
  
  // Add previous thought-response pairs if any
  for (const pair of thoughtResponsePairs) {
    contextPrompt += `<|im_start|>knowledge\n${pair.thought}<|im_end|>\n<|im_start|>assistant\n${pair.response}<|im_end|>\n`;
  }
  if (thought.length > 0) {
    contextPrompt += `<|im_start|>knowledge\n${thought}<|im_end|>\n`;
  }
  contextPrompt += `<|im_start|>assistant\n`;

  console.log("DEBUG: SmolLM Prompt: ", contextPrompt);
  const result = await llm(contextPrompt, {
    max_new_tokens: 128,
    temperature: 1,
    do_sample: false,
    return_full_text: false,
    pad_token_id: tokenizer.pad_token_id,
    eos_token_id: tokenizer.eos_token_id,
  });
  console.log("DEBUG: SmolLM Response: ", result);

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
    let messageType;
    if (thought === "<sil>") {
      messageType = "silence_response";
    } else if (thought === "") {
      messageType = "immediate_response";
    } else {
      messageType = "enhanced_response";
    }
    
    self.postMessage({ type: messageType, response });
    if (splitter) { // add to TTS if available
      splitter.push(thought === "" ? response : " " + response);
    }
  }
  
  return response;
};

function startSilenceTimer(userInput, thoughtResponsePairs, splitter) {
  if (silenceTimer) {
    clearTimeout(silenceTimer);
  }
  silenceTimer = setTimeout(async () => {
    if (!isGeneratingSilence) {
      isGeneratingSilence = true;
      self.postMessage({ type: "silence_token", token: "<sil>" });
      await processThought("<sil>", userInput, thoughtResponsePairs, splitter);
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

  let userText = input;
  
  if (isVoiceMode) {
    userText = await transcriber(input).then(({ text }) => text.trim());
    if (["", "[BLANK_AUDIO]"].includes(userText)) {
      isPlaying = false;
      return;
    }
    self.postMessage({ type: "transcription", text: userText });
  }
  
  messages.push({ role: "user", content: userText });

  let splitter = null;
  if (enableTTS && tts) {
    splitter = new TextSplitterStream();
    const streamOptions = voice ? { voice } : {};
    const stream = tts.stream(splitter, streamOptions);
    
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
  }

  let thoughtResponsePairs = [];
  
  const immediateResponse = await processThought("", userText, [], splitter);
  if (immediateResponse) {
    thoughtResponsePairs.push({ thought: "", response: immediateResponse });
    messages.push({ role: "assistant", content: immediateResponse });
    try {
      clearSilenceTimer();
      const thoughtsResponse = await fetch('/api/chat-thoughts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messages
        }),
      });

      // sil token handling
      if (!thoughtsResponse.ok) {
        console.warn('Failed to get thoughts from OpenAI');
        startSilenceTimer(userText, thoughtResponsePairs, splitter);
      } else {
        const reader = thoughtsResponse.body?.getReader();
        if (!reader) {
          startSilenceTimer(userText, thoughtResponsePairs, splitter);
        } else {
          const decoder = new TextDecoder();
          let buffer = '';
          const thoughts = [];
          let thoughtIndex = 0;
          let streamComplete = false;

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
                  
                  clearSilenceTimer();
                  
                  self.postMessage({ type: "thought", thought, index: thoughtIndex++ });
                  const thoughtResponse = await processThought(thought, userText, thoughtResponsePairs, splitter);
                  
                  if (thoughtResponse) {
                    thoughtResponsePairs.push({ thought: thought, response: thoughtResponse });
                  }
                  
                  // start sil timer once each thought is finished processing
                  if (!streamComplete) {
                    startSilenceTimer(userText, thoughtResponsePairs, splitter);
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
              streamComplete = true;
              break;
            }
          }

          if (thoughts.length === 0) {
            startSilenceTimer(userText, thoughtResponsePairs, splitter); // if no thoughts extracted
          }
        }
      }
    } catch (error) {
      console.warn("Failed to generate thoughts:", error);
    }
    
    const fullResponse = thoughtResponsePairs.map(pair => pair.response).join(" ");
    if (fullResponse !== immediateResponse) {
      messages[messages.length - 1].content = fullResponse;
    }
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
      
    case "playback_ended":
      isPlaying = false;
      clearSilenceTimer(); // Clear silence timer when playback ends
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