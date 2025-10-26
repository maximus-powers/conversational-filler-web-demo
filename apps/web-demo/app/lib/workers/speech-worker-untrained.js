import { AutoTokenizer, AutoModelForCausalLM, AutoModel, Tensor, pipeline } from "@huggingface/transformers";

// queue for messages if model not ready
let messageQueue = [];
let isWorkerInitialized = false;
self.onmessage = (event) => {
  if (!isWorkerInitialized) {
    messageQueue.push(event);
  }
};

// wrap everything in async IIFE to handle top-level await
(async () => {

// Audio constants (also defined in ../audio-constants.ts)
const INPUT_SAMPLE_RATE = 16000;
const INPUT_SAMPLE_RATE_MS = INPUT_SAMPLE_RATE / 1000;
const SPEECH_THRESHOLD = 0.3;
const EXIT_THRESHOLD = 0.1;
const MIN_SILENCE_DURATION_MS = 1200;
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
    device
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

let llm_model_id = null;
let tokenizer = null;
let llm = null;
const warmupPrompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";

async function loadLocalLM(modelId) {
  const targetModel = modelId || "HuggingFaceTB/SmolLM-360M-Instruct";

  self.postMessage({
    type: "info",
    message: `Loading LLM model: ${targetModel}...`,
    duration: "until_next"
  });

  llm_model_id = targetModel;

  tokenizer = await AutoTokenizer.from_pretrained(llm_model_id, {
    dtype: "fp32",
    device: "webgpu",
  });
  llm = await AutoModelForCausalLM.from_pretrained(llm_model_id, {
    dtype: "fp32",
    device: "webgpu",
  });

  const warmupInput = tokenizer(warmupPrompt);
  await llm.generate({
    ...warmupInput,
    max_new_tokens: 10,
    do_sample: false,
  });

  self.postMessage({
    type: "info",
    message: `LLM model ${llm_model_id} loaded successfully`
  });

  self.postMessage({
    type: "model_loaded",
    modelId: llm_model_id
  });
}

let messages = [];
let currentMessageId = null;
let currentPersona = "none";

// pipeline config
let currentEnableSmolLM = true;
let currentEnableTTS = false;

// Note: Don't send "ready" status here - wait for init message to load LLM model first

const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;

const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);

let isRecording = false;
let isPlaying = false;

let responseIndex = 0;
let ttsSynthesisId = 0;
let pendingSynthesisEvents = new Map();

// timeline tracking
let conversationStartTime = null;
let conversationTurnStartTime = null;
let inferenceStartTime = null;
let firstResponseTime = null;
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

const generateResponse = async (userInput, splitter) => {
  let contextPrompt = "<|im_start|>user\nYou are a helpful assistant, who responds in very brief responses, in sentences, not bullet points.<|im_end|>\n<|im_start|>assistant\nOkay, I understand.<|im_end|>\n";
  for (const msg of messages) {
    contextPrompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
  }
  contextPrompt += `<|im_start|>assistant\n`;

  const processingStartTime = Date.now();
  self.postMessage({
    type: "smollm_submit",
    prompt: contextPrompt,
    timestamp: processingStartTime,
    turnOffset: processingStartTime - conversationTurnStartTime,
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

  let response = generatedText
    .replace(/<\|im_start\|>/g, "")
    .replace(/<\|im_end\|>/g, "")
    .replace(/^assistant\s*/i, "")
    .trim();

  if (response) {
    if (!firstResponseTime && conversationStartTime) {
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

    if (splitter) {
      const currentSynthesisId = `synthesis_${++ttsSynthesisId}`;
      const synthesisStartTime = Date.now();

      self.postMessage({
        type: "tts_synthesis_start",
        text: response.trim(),
        responseIndex: responseIndex,
        synthesisId: currentSynthesisId,
        timestamp: synthesisStartTime,
        turnOffset: synthesisStartTime - conversationTurnStartTime
      });

      splitter.push({
        text: response.trim(),
        synthesisId: currentSynthesisId,
        responseIndex: responseIndex,
        startTime: synthesisStartTime
      });
    }

    responseIndex++;

    const processingEndTime = Date.now();
    self.postMessage({
      type: "smollm_response",
      response,
      rawResponse,
      fullPrompt: contextPrompt,
      timestamp: processingEndTime,
      turnOffset: processingEndTime - conversationTurnStartTime,
      turnStartTime: conversationTurnStartTime,
      duration: processingEndTime - processingStartTime,
      startTime: processingStartTime,
      isInitialResponse: true
    });
  }

  return response;
};

const processInput = async (input, enableSTT, enableSmolLM, enableTTS) => {
  isPlaying = true;

  responseIndex = 0;
  ttsSynthesisId = 0;
  pendingSynthesisEvents.clear();
  currentMessageId = null;

  conversationStartTime = Date.now();
  conversationTurnStartTime = conversationStartTime;
  inferenceStartTime = null;
  firstResponseTime = null;
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

  inferenceStartTime = Date.now();

  let generatedResponse = null;

  try {
    // Simple generation without thoughts
    if (enableSmolLM) {
      generatedResponse = await generateResponse(userText, splitter);
      if (generatedResponse) {
        messages.push({ role: "assistant", content: generatedResponse });
      }
    }
  } catch (error) {
    console.warn("Failed to generate response:", error);
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
  processInput(paddedBuffer, true, currentEnableSmolLM, currentEnableTTS);

  // set overflow (if present) and reset the rest of the audio buffer
  if (overflow) {
    BUFFER.set(overflow, 0);
  }
  resetAfterRecording(overflowLength);
};

// prev buffers FIFO queue
let prevBuffers = [];

const handleMessage = async (event) => {
  const { type } = event.data;
  // refuse new audio while playing back
  if (type === "audio" && isPlaying) return;

  switch (type) {
    case "init":
      const requestedModel = event.data.modelId;
      await loadLocalLM(requestedModel);

      self.postMessage({
        type: "status",
        status: "ready",
        voices: availableVoices,
        message: "All models loaded",
        modelId: llm_model_id
      });
      return;

    case "set_voice":
      voice = event.data.voice;
      return;

    case "set_thought_provider":
      // No-op for untrained worker - no thoughts support
      return;

    case "set_smollm_enabled":
      currentEnableSmolLM = event.data.enabled;
      return;

    case "set_tts_enabled":
      currentEnableTTS = event.data.enabled;
      return;

    case "set_persona":
      currentPersona = event.data.persona;
      return;

    case "set_stt_enabled":
      return; // managed by main thread now

    case "set_current_message_id":
      currentMessageId = event.data.messageId;
      return;

    case "playback_ended":
      isPlaying = false;
      return;

    case "process_text":
      if (!llm || !tokenizer) {
        console.error('LLM model not loaded yet');
        self.postMessage({
          type: "error",
          error: "LLM model not loaded. Wait for init to complete."
        });
        return;
      }

      // for text mode
      const text = event.data.text;
      const enableTTS = event.data.enableTTS || false;
      const enableSmolLM = event.data.enableSmolLM !== undefined ? event.data.enableSmolLM : true;

      currentEnableSmolLM = enableSmolLM;
      currentEnableTTS = enableTTS;

      if (text) {
        await processInput(text, false, enableSmolLM, enableTTS);
      }
      return;

    case "end_call":
      messages = [];
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

self.onmessage = handleMessage;
isWorkerInitialized = true;

for (const queuedEvent of messageQueue) {
  await handleMessage(queuedEvent);
}
messageQueue = [];

})().catch(error => {
  console.error('Worker initialization error:', error);
  self.postMessage({ error: error.message });
});
