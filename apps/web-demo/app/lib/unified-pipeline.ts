import { INPUT_SAMPLE_RATE } from "./audio-constants";
import { EventTracker, TurnMetadata } from "./event-tracker";

export interface PipelineCallbacks {
  onMessageReceived?: (
    role: "user" | "assistant",
    content: string,
    messageId?: string,
  ) => void;
  onMessageUpdated?: (messageId: string, content: string) => void;
  onThoughtReceived?: (thought: string, index: number) => void;
  onThoughtResponsePairs?: (pairs: Array<{thought: string, response: string}>, messageId: string) => void;
  onTranscriptionReceived?: (text: string) => void;
  onStatusChange?: (status: string, message: string) => void;
  onEventData?: (eventData: any) => void;
  onConversationStart?: (startTime: number) => void;
}

export interface PipelineState {
  features: {
    enableSTT: boolean;
    enableThoughts: boolean;
    enableSmolLM: boolean;
    enableTTS: boolean;
  };
  isReady: boolean;
  isProcessing: boolean;
  isRecording: boolean;
  isPlaying: boolean;
  voices: Record<string, any>;
  currentMessageId: string | null;
}

export class UnifiedPipeline {
  private worker: Worker | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private worklet: AudioWorkletNode | null = null;
  private playbackNode: AudioWorkletNode | null = null;
  private callbacks: PipelineCallbacks;
  private state: PipelineState;
  private isWorkerReady = false;
  private eventTracker = new EventTracker();
  private hasStartedPlayback = false;

  constructor(callbacks: PipelineCallbacks, features?: PipelineState['features']) {
    this.callbacks = callbacks;
    this.state = {
      features: features || {
        enableSTT: false,
        enableThoughts: true,
        enableSmolLM: true,
        enableTTS: false,
      },
      isReady: false,
      isProcessing: false,
      isRecording: false,
      isPlaying: false,
      voices: {},
      currentMessageId: null,
    };
  }

  async initialize() {

    try {
      // force reload of worker
      const workerUrl = `/speech-worker-bundled.js?v=${Date.now()}`;
      this.worker = new Worker(workerUrl);
      this.setupWorkerListeners();

      if (this.state.features.enableSTT) {
        await this.setupAudioContexts();
      }
      this.worker.postMessage({ type: "init" });

      // wait for worker to be ready
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error("Worker initialization timed out"));
        }, 60000);
        const checkReady = () => {
          if (this.isWorkerReady) {
            clearTimeout(timeout);
            resolve();
          } else {
            setTimeout(checkReady, 100);
          }
        };
        checkReady();
      });

      const features = this.state.features;
      if (this.worker) {
        const provider = features.enableThoughts ? "gemini" : "none";
        this.worker.postMessage({ type: "set_thought_provider", provider });
        this.worker.postMessage({ type: "set_smollm_enabled", enabled: features.enableSmolLM });
        this.worker.postMessage({ type: "set_tts_enabled", enabled: features.enableTTS });
      }

      this.state.isReady = true;
    } catch (error) {
      console.error("Failed to initialize:", error);
      throw error;
    }
  }

  private setupWorkerListeners() {
    if (!this.worker) return;
    this.worker.onerror = (error) => {
      console.error("Worker error:", error);
    };

    this.worker.onmessage = ({ data }) => {
      if (data.error) {
        console.error("Worker error:", data.error);
        return;
      }
      switch (data.type) {
        case "info":
          console.log("Worker info:", data.message);
          break;
        case "status":
          this.handleStatusMessage(data);
          break;
        case "output":
          this.handleAudioOutput(data);
          break;
        case "output_mp3":
          this.handleMP3Output(data);
          break;
        case "stt_start":
          if (this.eventTracker.hasActiveTurn()) {
            this.eventTracker.addEvent("STTStart");
            this.callbacks.onEventData?.(this.eventTracker.getData());
          }
          break;
        case "stt_end":
          this.handleSTTEnd(data);
          break;
        case "conversation_turn_start":
          this.state.currentMessageId = null;
          this.hasStartedPlayback = false;
          this.startNewTurn();
          if (data.timestamp && this.callbacks.onConversationStart) {
            this.callbacks.onConversationStart(data.timestamp);
          }
          break;
        case "smollm_submit":
          this.handleSmolLMSubmit(data);
          break;
        case "smollm_response":
          this.handleSmolLMResponse(data);
          break;
        case "tts_synthesis_start":
          this.handleTTSSynthesisStart(data);
          break;
        case "tts_synthesis_end":
          this.handleTTSSynthesisEnd(data);
          break;
        case "thought_submit":
          if (this.eventTracker.hasActiveTurn()) {
            this.eventTracker.addEvent("ThoughtApiSubmit");
            this.callbacks.onEventData?.(this.eventTracker.getData());
          }
          break;
        case "first_thought_token_received":
          if (this.eventTracker.hasActiveTurn()) {
            this.eventTracker.addEvent("ThoughtApiFirstToken");
            this.callbacks.onEventData?.(this.eventTracker.getData());
          }
          break;
        case "thought_response":
          this.handleThoughtResponse(data);
          break;
        case "thought_response_pairs":
          this.handleThoughtResponsePairs(data);
          break;
      }
    };
  }

  private handleStatusMessage(data: any) {
    if (data.status === "ready") {
      this.isWorkerReady = true;
      this.state.voices = data.voices || {};
    } else if (data.status === "recording_start") {
      this.state.isRecording = true;
      if (this.eventTracker.hasActiveTurn()) {
        this.eventTracker.addEvent("VoiceDetectionStart");
        this.callbacks.onEventData?.(this.eventTracker.getData());
      }
    } else if (data.status === "recording_end") {
      this.state.isRecording = false;
      if (this.eventTracker.hasActiveTurn()) {
        this.eventTracker.addEvent("VoiceDetectionEnd");
        this.callbacks.onEventData?.(this.eventTracker.getData());
      }
    }
    this.callbacks.onStatusChange?.(data.status, data.message || "");
  }

  private handleAudioOutput(data: any) {
    if (!this.state.features.enableSTT || !data.result) return;
    const audioBuffer = data.result; // { type: 'output', text: string, result: Float32Array }
    if (this.playbackNode) {
      if (!this.hasStartedPlayback && this.eventTracker.hasActiveTurn()) {
        this.hasStartedPlayback = true;
        this.eventTracker.addEvent("AudioPlaybackStart");
        this.callbacks.onEventData?.(this.eventTracker.getData());
      }

      this.state.isPlaying = true;
      this.playbackNode.port.postMessage(audioBuffer);
    }
  }

  private async handleMP3Output(data: any) {
    if (!data.audioBuffer) return;
    if (!this.state.features.enableTTS) return;

    try {
      if (!this.audioContext) { // use main thread
        this.audioContext = new AudioContext({ sampleRate: 24000 });
      }

      const audioBuffer = await this.audioContext.decodeAudioData(data.audioBuffer);
      const float32Array = audioBuffer.getChannelData(0);

      if (!this.state.features.enableSTT) {
        // if no STT, play directly on main thread
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.start();
        if (!this.hasStartedPlayback && this.eventTracker.hasActiveTurn()) {
          this.hasStartedPlayback = true;
          this.eventTracker.addEvent("AudioPlaybackStart");
          this.callbacks.onEventData?.(this.eventTracker.getData());
        }
        source.onended = () => {
          if (this.worker) {
            this.worker.postMessage({ type: "playback_ended" });
          }
          if (this.eventTracker.hasActiveTurn()) {
            this.eventTracker.addEvent("AudioPlaybackEnd");
            this.callbacks.onEventData?.(this.eventTracker.getData());
          }
        };
      }

      // send playback to worklet
      else if (this.state.features.enableSTT && this.playbackNode) {
        if (!this.hasStartedPlayback && this.eventTracker.hasActiveTurn()) {
          this.hasStartedPlayback = true;
          this.eventTracker.addEvent("AudioPlaybackStart");
          this.callbacks.onEventData?.(this.eventTracker.getData());
        }
        this.state.isPlaying = true;
        this.playbackNode.port.postMessage(float32Array);
      }
      
    } catch (error) {
      console.error('Error decoding MP3:', error);
    }
  }

  private handleSTTEnd(data: any) {
    if (data.text) {
      this.callbacks.onMessageReceived?.(
        "user",
        data.text,
        `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
      );
      this.callbacks.onTranscriptionReceived?.(data.text);
    }

    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("STTEnd", { text: data.text });
      this.callbacks.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleSmolLMSubmit(data: any) {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("LocalLMSubmit", { prompt: data.prompt });
      this.callbacks.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleSmolLMResponse(data: any) {
    let shouldSendMessageId = false;

    if (data.isInitialResponse) {
      this.state.currentMessageId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
      this.callbacks.onMessageReceived?.(
        "assistant",
        data.response || data.content,
        this.state.currentMessageId,
      );
      shouldSendMessageId = true;
    } else {
      if (this.state.currentMessageId) {
        this.callbacks.onMessageUpdated?.(
          this.state.currentMessageId,
          data.response,
        );
      } else {
        this.state.currentMessageId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        this.callbacks.onMessageReceived?.(
          "assistant",
          data.response || data.content,
          this.state.currentMessageId,
        );
        shouldSendMessageId = true;
      }
    }

    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("LocalLMResponse", {
        response: data.rawResponse || data.response || data.content,
        prompt: data.fullPrompt
      });
      this.callbacks.onEventData?.(this.eventTracker.getData());
    }

    if (this.worker && this.state.currentMessageId && shouldSendMessageId) {
      this.worker.postMessage({
        type: "set_current_message_id",
        messageId: this.state.currentMessageId
      });
    }
  }


  private handleTTSSynthesisStart(data: any) {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("TTSSynthesisStart", {
        text: data.text,
        responseIndex: data.responseIndex,
        synthesisId: data.synthesisId
      });
      this.callbacks.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleTTSSynthesisEnd(data: any) {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("TTSSynthesisEnd", {
        text: data.text,
        responseIndex: data.responseIndex,
        synthesisId: data.synthesisId
      });
      this.callbacks.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleThoughtResponse(data: any): void {
    this.callbacks.onThoughtReceived?.(data.thought, data.index);
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("ThoughtParsed", { response: data.thought });
      this.callbacks.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleThoughtResponsePairs(data: any): void {
    if (data.pairs && data.messageId) {
      this.callbacks.onThoughtResponsePairs?.(data.pairs, data.messageId);
    }
  }

  private async setupAudioContexts(): Promise<void> {
    this.audioContext = new (window.AudioContext ||
      (window as any).webkitAudioContext)({
      sampleRate: 24000, // matches tts output sample rate
    });

    await this.audioContext.audioWorklet.addModule("/workers/play-worklet.js");
    this.playbackNode = new AudioWorkletNode(this.audioContext, "play-worklet");
    this.playbackNode.connect(this.audioContext.destination);

    // listen for playback ended and notify worker
    this.playbackNode.port.onmessage = (event) => {
      if (event.data.type === "playback_ended") {
        this.state.isPlaying = false;
        this.hasStartedPlayback = false; 
        
        if (this.eventTracker.hasActiveTurn()) {
          this.eventTracker.addEvent("AudioPlaybackEnd");
          this.callbacks.onEventData?.(this.eventTracker.getData());
        }
        
        if (this.worker) {
          this.worker.postMessage({ type: "playback_ended" });
        }
      }
    };

    await this.setupMicrophone();
  }

  private async setupMicrophone() {
    if (!this.audioContext) return;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: INPUT_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
        } as MediaTrackConstraints,
      });
      const source = this.audioContext.createMediaStreamSource(
        this.mediaStream,
      );

      // register VAD processor
      await this.audioContext.audioWorklet.addModule(
        "/workers/vad-processor.js",
      );
      this.worklet = new AudioWorkletNode(this.audioContext, "vad-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        processorOptions: { sampleRate: INPUT_SAMPLE_RATE },
      });

      source.connect(this.worklet);
      this.worklet.connect(this.audioContext.destination);

      let audioMessageCount = 0;
      this.worklet.port.onmessage = (event) => {
        audioMessageCount++;
        if (event.data.type === "audio" && this.worker) {
          this.worker.postMessage({
            type: "audio",
            buffer: event.data.audio,
          });
        }
      };
    } catch (error) {
      console.error("Failed to setup microphone:", error);
      throw error;
    }
  }

  async processText(text: string) {
    const config = this.state.features;

    if (!config.enableSmolLM && config.enableThoughts) {
      return this.processTextWithGeminiStandalone(text);
    }

    if (!this.worker || !this.isWorkerReady) {
      throw new Error("Pipeline not ready");
    }

    this.state.isProcessing = true;
    this.state.currentMessageId = null;

    if (!this.eventTracker.hasActiveTurn()) {
      this.startNewTurn();
    }

    const userMessageId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    this.callbacks.onMessageReceived?.("user", text, userMessageId);

    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("UserInputReceived", { text });
      this.callbacks.onEventData?.(this.eventTracker.getData());
    }

    this.worker.postMessage({
      type: "process_text",
      text: text.trim(),
      enableTTS: config.enableTTS,
      enableThoughts: config.enableThoughts,
      enableSmolLM: config.enableSmolLM,
    });
  }

  private async processTextWithGeminiStandalone(text: string) {
    this.state.isProcessing = true;
    this.state.currentMessageId = null;
    
    if (!this.eventTracker.hasActiveTurn()) {
      this.startNewTurn();
    }
    
    const userMessageId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    this.callbacks.onMessageReceived?.("user", text, userMessageId);
    
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("UserInputReceived", { text });
      this.callbacks.onEventData?.(this.eventTracker.getData());
    }

    try {
      const response = await fetch('/api/gemini-standalone', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [{ role: "user", content: text }]
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const assistantMessageId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
      let fullResponse = "";
      let firstToken = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = new TextDecoder().decode(value);
        
        if (chunk.includes('[first_token]')) {
          firstToken = true;
          this.callbacks.onStatusChange?.("processing_complete", "");
          continue;
        }
        
        if (chunk.includes('[done]')) {
          this.callbacks.onStatusChange?.("response_complete", "");
          this.state.isProcessing = false;
          break;
        }

        fullResponse += chunk;        
        if (firstToken) {
          this.callbacks.onMessageReceived?.("assistant", fullResponse, assistantMessageId);
          this.callbacks.onMessageUpdated?.(assistantMessageId, fullResponse);
        }
      }

    } catch (error) {
      console.error("Error in Gemini-standalone processing:", error);
      this.callbacks.onStatusChange?.("error", `Failed to process with Gemini: ${error}`);
      this.state.isProcessing = false;
    }
  }

  updateFeatures(features: Partial<PipelineState['features']>) {
    this.state.features = { ...this.state.features, ...features };
    if (this.worker) {
      if (features.enableThoughts !== undefined) {
        const provider = features.enableThoughts ? "gemini" : "none";
        this.worker.postMessage({ type: "set_thought_provider", provider });
      }
      if (features.enableSmolLM !== undefined) {
        this.worker.postMessage({ type: "set_smollm_enabled", enabled: features.enableSmolLM });
      }
      if (features.enableTTS !== undefined) {
        this.worker.postMessage({ type: "set_tts_enabled", enabled: features.enableTTS });
      }
      if (features.enableSTT !== undefined) {
        this.worker.postMessage({ type: "set_stt_enabled", enabled: features.enableSTT });
      }
    }
  }

  async toggleSTT(enable: boolean) {
    this.state.features.enableSTT = enable;
    if (enable && !this.audioContext) {
      await this.setupAudioContexts();
    } else if (!enable && this.audioContext) {
      this.disposeAudioContexts();
    }
  }

  private startNewTurn() {
    const metadata: TurnMetadata = {
      localModel: this.state.features.enableSmolLM ? "smollm-finetuned" : "none",
      thoughtModel: this.state.features.enableThoughts ? "gemini-flash-2.0" : "none",
      voiceMode: this.state.features.enableSTT
    };
    this.eventTracker.startNewTurn(metadata);
  }

  resetEventData() {
    this.eventTracker.reset();
  }

  private disposeAudioContexts() {
    if (this.state.isRecording && this.worker) {
      this.state.isRecording = false;
      this.worker.postMessage({ type: "stop_recording" });
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.worklet) {
      this.worklet.disconnect();
      this.worklet = null;
    }
    if (this.playbackNode) {
      this.playbackNode.disconnect();
      this.playbackNode = null;
    }
  }

  dispose() {
    this.disposeAudioContexts();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.isWorkerReady = false;
    this.state.isReady = false;
    this.state.isProcessing = false;
    this.state.isRecording = false;
    this.state.isPlaying = false;
  }
}
