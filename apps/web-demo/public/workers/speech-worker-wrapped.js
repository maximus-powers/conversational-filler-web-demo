// Wrapped worker to handle initialization properly
(async function initWorker() {
  try {
    self.postMessage({ type: "info", message: "Worker starting..." });
    
    // Import dependencies
    const { AutoModel, Tensor, pipeline } = await import("@huggingface/transformers");
    const { KokoroTTS, TextSplitterStream } = await import("kokoro-js");
    
    self.postMessage({ type: "info", message: "Dependencies loaded" });
    
    // Audio constants
    const INPUT_SAMPLE_RATE = 16000;
    const SPEECH_THRESHOLD = 0.3;
    const EXIT_THRESHOLD = 0.1;
    const MIN_SILENCE_DURATION_MS = 400;
    const MIN_SILENCE_DURATION_SAMPLES = (MIN_SILENCE_DURATION_MS * INPUT_SAMPLE_RATE) / 1000;
    const SPEECH_PAD_MS = 80;
    const SPEECH_PAD_SAMPLES = (SPEECH_PAD_MS * INPUT_SAMPLE_RATE) / 1000;
    const MIN_SPEECH_DURATION_SAMPLES = (250 * INPUT_SAMPLE_RATE) / 1000;
    const MAX_BUFFER_DURATION = 30;
    const NEW_BUFFER_SIZE = 512;
    const MAX_NUM_PREV_BUFFERS = Math.ceil(SPEECH_PAD_SAMPLES / NEW_BUFFER_SIZE);

    const device = "webgpu";
    
    self.postMessage({ type: "info", message: "Loading models..." });
    
    // Load VAD model
    const silero_vad = await AutoModel.from_pretrained(
      "onnx-community/silero-vad",
      {
        config: { model_type: "custom" },
        dtype: "fp32",
      },
    );
    
    self.postMessage({ type: "info", message: "VAD model loaded" });
    
    // Load ASR model
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
    );
    
    // Compile shaders
    await transcriber(new Float32Array(INPUT_SAMPLE_RATE));
    
    self.postMessage({ type: "info", message: "ASR model loaded" });
    
    // Load LLM model
    const llm_model_id = "maximuspowers/smollm-convo-filler-onnx-official";
    const llm = await pipeline("text-generation", llm_model_id, {
      dtype: "fp32", 
      device: "webgpu",
    });
    
    // Compile shaders
    await llm("test", { max_new_tokens: 1 });
    
    self.postMessage({ type: "info", message: "LLM model loaded" });
    
    // Load TTS model
    const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
    const tts = await KokoroTTS.from_pretrained(model_id, {
      dtype: "fp32",
      device: "webgpu",
    });
    
    self.postMessage({ type: "info", message: "TTS model loaded" });
    
    // State variables
    let isInitialized = false;
    let messages = [];
    let prevBuffers = [];
    let voice;
    
    // Global audio buffer to store incoming audio
    const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
    let bufferPointer = 0;
    
    // Initial state for VAD
    const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
    let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
    
    // Whether we are in the process of adding audio to the buffer
    let isRecording = false;
    let isPlaying = false;
    
    // Track samples after last speech
    let postSpeechSamples = 0;
    
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
     * Process text input (like speechToSpeech but skip ASR)
     */
    const processTextInput = async (text) => {
      isPlaying = true;
      
      // Generate immediate response using SmolLM with proper config and speak it immediately
      const immediatePrompt = `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
      
      const immediateResult = await llm(immediatePrompt, {
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
      
      let immediateResponse = "";
      if (Array.isArray(immediateResult) && immediateResult[0]?.generated_text) {
        immediateResponse = immediateResult[0].generated_text;
      } else if (immediateResult?.generated_text) {
        immediateResponse = immediateResult.generated_text;
      }
      
      immediateResponse = immediateResponse
        .replace(/<\|im_start\|>/g, "")
        .replace(/<\|im_end\|>/g, "")
        .replace(/^assistant\s*/i, "")
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
        
        // Speak immediately - don't wait for thoughts
        await speakResponse(immediateResponse);
      }
      
      // Generate thoughts and enhanced responses - speak each as it comes
      try {
        const thoughts = await generateThoughts(messages);
        if (thoughts.length > 0) {
          const enhancedResponses = await processThoughts(thoughts, text);
          
          // Speak each enhanced response immediately as it's generated
          for (const response of enhancedResponses) {
            await speakResponse(" " + response);
          }
        }
      } catch (error) {
        console.warn("Failed to generate enhanced responses:", error);
      }
      
      isPlaying = false;
    };

    /**
     * Speak a single response immediately using TTS
     */
    const speakResponse = async (text, responseType = 'response') => {
      const splitter = new TextSplitterStream();
      const stream = tts.stream(splitter, { voice });
      
      (async () => {
        for await (const { text: chunkText, audio } of stream) {
          self.postMessage({ type: "audio_output", text: chunkText, audio });
        }
      })();
      
      splitter.push(text);
      splitter.close();
    };

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
      
      // 2. Generate immediate response using SmolLM with proper config and speak it immediately
      const immediatePrompt = `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
      
      const immediateResult = await llm(immediatePrompt, {
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
      
      let immediateResponse = "";
      if (Array.isArray(immediateResult) && immediateResult[0]?.generated_text) {
        immediateResponse = immediateResult[0].generated_text;
      } else if (immediateResult?.generated_text) {
        immediateResponse = immediateResult.generated_text;
      }
      
      immediateResponse = immediateResponse
        .replace(/<\|im_start\|>/g, "")
        .replace(/<\|im_end\|>/g, "")
        .replace(/^assistant\s*/i, "")
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
        
        // Speak immediately - don't wait for thoughts
        await speakResponse(immediateResponse);
      }
      
      // 3. Generate thoughts and enhanced responses - speak each as it comes
      try {
        const thoughts = await generateThoughts(messages);
        if (thoughts.length > 0) {
          const enhancedResponses = await processThoughts(thoughts, text);
          
          // Speak each enhanced response immediately as it's generated
          for (const response of enhancedResponses) {
            await speakResponse(" " + response);
          }
        }
      } catch (error) {
        console.warn("Failed to generate enhanced responses:", error);
      }
      
      isPlaying = false;
    };
    
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
        // Reset playing state when TTS stream ends
        console.log('Greeting TTS stream ended, setting isPlaying = false');
        isPlaying = false;
      })();
      splitter.push(text);
      splitter.close();
      messages.push({ role: "assistant", content: text });
    }
    
    // Message handler - set up BEFORE signaling ready
    self.onmessage = async (event) => {
      const { type, buffer } = event.data;
      
      if (type === "audio") {
        console.log('Worker received audio buffer, length:', buffer?.length, 'isPlaying:', isPlaying);
        if (isPlaying) return;
      }
      
      switch (type) {
        case "init":
          if (!isInitialized) {
            isInitialized = true;
            self.postMessage({
              type: "status",
              status: "ready",
              message: "Ready!",
              voices: tts.voices || {},
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
          
        case "process_text":
          // Handle text input (for typed messages)
          const textInput = event.data.text;
          if (textInput) {
            console.log('Processing text input:', textInput);
            messages.push({ role: "user", content: textInput });
            
            // Send transcription event for timeline to show user input
            self.postMessage({ type: "transcription", text: textInput });
            
            // Process as if it was speech - skip ASR, go straight to LLM
            processTextInput(textInput);
          }
          return;
          
        case "playback_ended":
          isPlaying = false;
          return;
          
        case "audio":
          // Process audio data for VAD
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
          break;
          
        default:
          return;
      }
    };
    
    // Signal that worker is ready to receive messages
    self.postMessage({ type: "info", message: "Worker ready for messages" });
    
    // Auto-send ready status (don't wait for init)
    setTimeout(() => {
      if (!isInitialized) {
        isInitialized = true;
        self.postMessage({
          type: "status", 
          status: "ready",
          message: "Ready (auto)!",
          voices: tts.voices || {},
        });
      }
    }, 100);
    
  } catch (error) {
    self.postMessage({ 
      type: "error", 
      error: error.message || "Unknown error during initialization",
      stack: error.stack
    });
  }
})();