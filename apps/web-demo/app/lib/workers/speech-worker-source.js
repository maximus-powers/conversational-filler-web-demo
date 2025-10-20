import { AutoTokenizer, AutoModelForCausalLM, AutoModel, Tensor, pipeline } from "@huggingface/transformers";

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

// init TTS (ElevenLabs)
let voice = "21m00Tcm4TlvDq8ikWAM"; // Rachel voice
const availableVoices = {
  "21m00Tcm4TlvDq8ikWAM": { name: "Rachel (Female)" },
};

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
self.postMessage({ 
  type: "info", 
  message: "Whisper model loaded successfully"
});

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


let messages = [];
let thoughtProvider = "gemini";
let currentMessageId = null;

// pipeline config
let currentEnableThoughts = false;
let currentEnableSmolLM = true;
let currentEnableTTS = false;

self.postMessage({
  type: "status",
  status: "ready",
  message: "Ready!",
  voices: availableVoices,
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
let responseIndex = 0;
let ttsSynthesisId = 0;
let pendingSynthesisEvents = new Map();

// timeline tracking
let conversationStartTime = null;
let conversationTurnStartTime = null;
let inferenceStartTime = null;
let firstResponseTime = null;
let firstThoughtTime = null;
let sttStartTime = null;
let sttEndTime = null;

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
  for (const pair of thoughtResponsePairs) {
    contextPrompt += `<|im_start|>knowledge\n${pair.thought}<|im_end|>\n<|im_start|>assistant\n${pair.response}<|im_end|>\n`;
  }
  if (thought.length > 0) {
    contextPrompt += `<|im_start|>knowledge\n${thought}<|im_end|>\n`;
  }

  const thoughtProcessingStartTime = Date.now();
      self.postMessage({ 
      type: "smollm_submit", 
      thought,
      prompt: contextPrompt,
      timestamp: thoughtProcessingStartTime,
      turnOffset: thoughtProcessingStartTime - conversationTurnStartTime,
      turnStartTime: conversationTurnStartTime
    });

  const inputs = tokenizer(contextPrompt);
  const outputs = await llm.generate({
    ...inputs,
    max_new_tokens: 128,
    temperature: 1.0,
    do_sample: false,
    pad_token_id: tokenizer.pad_token_id,
    eos_token_id: tokenizer.eos_token_id,
  });
  const newTokens = Array.from(outputs.data.slice(inputs.input_ids.data.length)).map(t => Number(t));
  const generatedText = tokenizer.decode(newTokens, { skip_special_tokens: true });
  const rawResponse = tokenizer.decode(newTokens, { skip_special_tokens: false });
  
  response = generatedText
    .replace(/<\|im_start\|>/g, "")
    .replace(/<\|im_end\|>/g, "")
    .replace(/^assistant\s*/i, "")
    .trim()
    .split("\n")[0];

  if (response) {
    if (thought === "<|sil|>" && !firstResponseTime && conversationStartTime) {
      firstResponseTime = Date.now();
      self.postMessage({ 
        type: "immediate_response", 
        response,
        timestamp: firstResponseTime,
        timeFromStart: firstResponseTime - conversationStartTime,
        turnOffset: firstResponseTime - conversationTurnStartTime,
        turnStartTime: conversationTurnStartTime
      });
    }

    if (splitter) { // add to TTS queue if available
      const textToAdd = thought === "" ? response : " " + response;
      const currentSynthesisId = `synthesis_${++ttsSynthesisId}`;
      const synthesisStartTime = Date.now();

      self.postMessage({
        type: "tts_synthesis_start",
        text: textToAdd.trim(),
        responseIndex: responseIndex,
        synthesisId: currentSynthesisId,
        timestamp: synthesisStartTime,
        turnOffset: synthesisStartTime - conversationTurnStartTime
      });

      splitter.push({
        text: textToAdd.trim(),
        synthesisId: currentSynthesisId,
        responseIndex: responseIndex,
        startTime: synthesisStartTime
      });
    }
    
    responseIndex++;
    
    const thoughtProcessingEndTime = Date.now();
    self.postMessage({ 
      type: "smollm_response", 
      thought,
      response,
      rawResponse,
      fullPrompt: contextPrompt,
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

const waitForThoughtQueueComplete = async () => {
  while (thoughtQueue.length > 0 || isProcessingThought) {
    await new Promise(resolve => setTimeout(resolve, 10)); 
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

const processInput = async (input, enableSTT, enableThoughts, enableSmolLM, enableTTS) => {
  isPlaying = true;
  clearSilenceTimer();

  thoughtQueue = [];
  isProcessingThought = false;
  streamComplete = false;
  responseIndex = 0;
  ttsSynthesisId = 0;
  pendingSynthesisEvents.clear();
  currentMessageId = null;

  conversationStartTime = Date.now();
  conversationTurnStartTime = conversationStartTime;
  inferenceStartTime = null;
  firstResponseTime = null;
  firstThoughtTime = null;
  sttStartTime = null;
  sttEndTime = null;

  self.postMessage({
    type: "conversation_turn_start",
    timestamp: conversationStartTime,
    turnStartTime: conversationTurnStartTime,
    turnOffset: 0
  });

  let userText = input;

  if (enableSTT && typeof input !== 'string') {
    sttStartTime = Date.now();
    self.postMessage({
      type: "stt_start",
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
      type: "stt_end",
      text: userText,
      timestamp: sttEndTime,
      duration: sttEndTime - sttStartTime,
      startTime: sttStartTime,
      turnOffset: sttStartTime - conversationTurnStartTime,
      turnStartTime: conversationTurnStartTime
    });
  }
  
  messages.push({ role: "user", content: userText });

  // tts pipeline
  let splitter = null;
  let ttsStreamPromise = null;
  if (enableTTS) {
    splitter = [];
    ttsStreamPromise = (async () => {
      try {
        while (true) {
          if (splitter.length === 0) {
            await new Promise(resolve => setTimeout(resolve, 50));
            if (splitter.closed && splitter.length === 0) {
              break;
            }
            continue;
          }

          const chunk = splitter.shift();
          console.log(`Processing TTS chunk:`, chunk);

          try {
            const response = await fetch('/api/elevenlabs-tts', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                text: chunk.text,
                voice_id: voice
              })
            });

            if (!response.ok) {
              console.error('ElevenLabs API error:', response.statusText);
              continue;
            }

            const arrayBuffer = await response.arrayBuffer();
            const chunkEndTime = Date.now();

            self.postMessage({
              type: "tts_synthesis_end",
              text: chunk.text,
              responseIndex: chunk.responseIndex,
              synthesisId: chunk.synthesisId,
              timestamp: chunkEndTime,
              turnOffset: chunkEndTime - conversationTurnStartTime,
              duration: chunkEndTime - chunk.startTime,
              startTime: chunk.startTime
            });

            self.postMessage({
              type: "output_mp3",
              text: chunk.text,
              audioBuffer: arrayBuffer
            }, [arrayBuffer]);
          } catch (error) {
            console.error('Error processing TTS chunk:', error);
          }
        }
      } catch (error) {
        console.error('Error in TTS stream:', error);
      }
    })();
  }

  let thoughtResponsePairs = [];
  streamComplete = false;
  
  inferenceStartTime = Date.now();
  
  let thoughtsPromise = null;
  let thoughtStartTime = null;
  let immediateResponse = null;
  
  try {
    clearSilenceTimer();

    // thoughts pipeline
    if (enableThoughts) {
      thoughtStartTime = Date.now();
      self.postMessage({
        type: "thought_submit",
        timestamp: thoughtStartTime,
        turnOffset: thoughtStartTime - conversationTurnStartTime,
        turnStartTime: conversationTurnStartTime
      });
      const thoughtsEndpoint = '/api/chat-thoughts-gemini';
      thoughtsPromise = fetch(thoughtsEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messages
        }),
      });
    } else if (enableSmolLM) {
      thoughtsPromise = Promise.resolve().then(async () => {
        streamComplete = true;
        // queue 3 sil tokens if no thoughts
        for (let i = 0; i < 3; i++) {
          thoughtQueue.push("<|sil|>");
        }
      });
    }

    // smollm pipeline
    if (enableSmolLM) {
      immediateResponse = await processThought("<|sil|>", userText, [], splitter);
      if (immediateResponse) {
        thoughtResponsePairs.push({ thought: "<|sil|>", response: immediateResponse });
        messages.push({ role: "assistant", content: immediateResponse });
      }
    }
    if (thoughtsPromise && enableThoughts) {
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
            let firstTokenReceived = false;

            while (true) {
              const { done, value } = await reader.read();
              if (done) {
                streamComplete = true;
                break;
              }
              
              const chunk = decoder.decode(value, { stream: true });
              buffer += chunk;
              
              if (!firstTokenReceived && buffer.includes('[first_token]')) {
                self.postMessage({
                  type: "first_thought_token_received",
                  timestamp: Date.now(),
                  turnOffset: Date.now() - conversationTurnStartTime,
                  turnStartTime: conversationTurnStartTime
                });
                firstTokenReceived = true;
                buffer = buffer.replace('[first_token]', '');
              }

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
                      type: "thought_response", 
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
            await waitForThoughtQueueComplete();
          }
        }

      const thoughtEndTime = Date.now();
      self.postMessage({
        type: "thought_generation_end",
        timestamp: thoughtEndTime,
        duration: thoughtEndTime - thoughtStartTime,
        startTime: thoughtStartTime
      });
    } else if (thoughtsPromise && !enableThoughts && enableSmolLM) {
      await thoughtsPromise;
      await processThoughtQueue(userText, thoughtResponsePairs, splitter);
      await waitForThoughtQueueComplete();
    }
  } catch (error) {
    console.warn("Failed to generate thoughts:", error);
  }
  
  
  if (enableThoughts && !enableSmolLM) {
    // thoughts but no smollm, return thoughts directly
    await waitForThoughtQueueComplete();
    const thoughtsText = thoughtQueue.join(" ");
    const msgId = Date.now().toString();
    self.postMessage({
      type: "message",
      role: "assistant",
      content: thoughtsText,
      messageId: msgId
    });
    messages.push({ role: "assistant", content: thoughtsText });
  } else {
    await waitForThoughtQueueComplete();

    const fullResponse = thoughtResponsePairs.map(pair => pair.response).join(" ");
    if (fullResponse !== immediateResponse) {
      messages[messages.length - 1].content = fullResponse;
    }

    if (thoughtResponsePairs.length > 0 && currentMessageId) {
      self.postMessage({
        type: "thought_response_pairs",
        pairs: thoughtResponsePairs,
        messageId: currentMessageId
      });
    }
  }
  
  if (splitter) {
    await new Promise(resolve => setTimeout(resolve, 50));
    splitter.closed = true;
  }
  
  if (ttsStreamPromise) {
    await ttsStreamPromise;
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
  processInput(paddedBuffer, true, currentEnableThoughts, currentEnableSmolLM, currentEnableTTS);

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
        voices: availableVoices,
        message: "All models loaded"
      });
      return;
      
    case "set_voice":
      voice = event.data.voice;
      return;

    case "set_thought_provider":
      thoughtProvider = event.data.provider;
      currentEnableThoughts = (thoughtProvider !== "none");
      return;

    case "set_smollm_enabled":
      currentEnableSmolLM = event.data.enabled;
      return;

    case "set_tts_enabled":
      currentEnableTTS = event.data.enabled;
      return;

    case "set_stt_enabled":
      return; // managed by main thread now

    case "set_current_message_id":
      currentMessageId = event.data.messageId;
      return;

    case "playback_ended":
      isPlaying = false;
      clearSilenceTimer();
      return;
      
    case "process_text":
      // for text mode
      const text = event.data.text;
      const enableTTS = event.data.enableTTS || false;
      const enableThoughts = event.data.enableThoughts !== undefined ? event.data.enableThoughts : true;
      const enableSmolLM = event.data.enableSmolLM !== undefined ? event.data.enableSmolLM : true;

      currentEnableThoughts = enableThoughts;
      currentEnableSmolLM = enableSmolLM;
      currentEnableTTS = enableTTS;

      if (text) {
        await processInput(text, false, enableThoughts, enableSmolLM, enableTTS);
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