import { INPUT_SAMPLE_RATE } from "./audio-constants";
import { EventTracker, TurnMetadata } from "./event-tracker";

export type InferenceMode = "text" | "voice";

export interface UnifiedPipelineConfig {
  onMessageReceived?: (
    role: "user" | "assistant",
    content: string,
    messageId?: string,
  ) => void;
  onMessageUpdated?: (messageId: string, content: string) => void;
  onThoughtReceived?: (thought: string, index: number) => void;
  onTranscriptionReceived?: (text: string) => void;
  onStatusChange?: (status: string, message: string) => void;
  onEventData?: (eventData: any) => void;
}

export interface UnifiedPipelineState {
  mode: InferenceMode;
  isReady: boolean;
  isProcessing: boolean;
  isRecording: boolean;
  isPlaying: boolean;
  voices: Record<string, any>;
  currentMessageId: string | null;
  thoughtProvider: "gemini" | "none";
  selectedModel: "maximuspowers/smollm-convo-filler-onnx-official" | "HuggingFaceTB/SmolLM-360M-Instruct" | "none";
}

export class UnifiedPipeline {
  private worker: Worker | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private worklet: AudioWorkletNode | null = null;
  private playbackNode: AudioWorkletNode | null = null;
  private config: UnifiedPipelineConfig;
  private state: UnifiedPipelineState;
  private isWorkerReady = false;
  private eventTracker = new EventTracker();

  constructor(config: UnifiedPipelineConfig) {
    this.config = config;
    this.state = {
      mode: "text",
      isReady: false,
      isProcessing: false,
      isRecording: false,
      isPlaying: false,
      voices: {},
      currentMessageId: null,
      thoughtProvider: "gemini",
      selectedModel: "maximuspowers/smollm-convo-filler-onnx-official",
    };
  }

  async initialize(mode: InferenceMode = "text") {
    this.state.mode = mode;

    try {
      this.worker = new Worker("/speech-worker-bundled.js");
      this.setupWorkerListeners();
      if (mode === "voice") {
        await this.setupAudioContexts();
      }
      this.worker.postMessage({ type: "init" });
      await this.waitForWorkerReady();
      this.state.isReady = true;
    } catch (error) {
      console.error("Failed to initialize:", error);
      throw error;
    }
  }

  private async waitForWorkerReady(): Promise<void> {
    return new Promise((resolve, reject) => {
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
        case "stt_start":
          this.handleSTTStart();
          break;
        case "stt_end":
          this.handleSTTEnd(data);
          break;
        case "conversation_turn_start":
          this.handleConversationTurnStart(data);
          break;
        case "smollm_submit":
          this.handleSmolLMSubmit(data);
          break;
        case "smollm_response":
          this.handleSmolLMResponse(data);
          break;
        case "tts_start":
          this.handleTTSStart();
          break;
        case "tts_end":
          this.handleTTSEnd();
          break;
        case "thought_submit":
          this.handleThoughtSubmit();
          break;
        case "first_thought_token_received":
          this.handleFirstThoughtToken();
          break;
        case "thought_response":
          this.handleThoughtResponse(data);
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
        this.config.onEventData?.(this.eventTracker.getData());
      }
    } else if (data.status === "recording_end") {
      this.state.isRecording = false;
      if (this.eventTracker.hasActiveTurn()) {
        this.eventTracker.addEvent("VoiceDetectionEnd");
        this.config.onEventData?.(this.eventTracker.getData());
      }
    }
    this.config.onStatusChange?.(data.status, data.message || "");
  }

  private handleTranscription(text: string) {
    if (!text) return;
    this.config.onMessageReceived?.(
      "user",
      text,
      `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
    );
    this.config.onTranscriptionReceived?.(text);
  }


  private handleAudioOutput(data: any) {
    if (this.state.mode !== "voice" || !data.result) return;
    const audioBuffer = data.result; // { type: 'output', text: string, result: Float32Array }
    if (this.playbackNode) {
      this.state.isPlaying = true;
      this.playbackNode.port.postMessage(audioBuffer);
    }
  }

  private handleSTTStart() {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("STTStart");
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleSTTEnd(data: any) {
    this.handleTranscription(data.text);
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("STTEnd", { text: data.text });
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleConversationTurnStart(data: any) {
    this.state.currentMessageId = null;
    this.startNewTurn();
  }

  private handleSmolLMSubmit(data: any) {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("LocalLMSubmit", { prompt: data.prompt });
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleSmolLMResponse(data: any) {
    if (data.isInitialResponse) {
      this.state.currentMessageId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
      this.config.onMessageReceived?.(
        "assistant",
        data.response || data.content,
        this.state.currentMessageId,
      );
    } else {
      if (this.state.currentMessageId) {
        this.config.onMessageUpdated?.(
          this.state.currentMessageId,
          data.response,
        );
      } else {
        this.state.currentMessageId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        this.config.onMessageReceived?.(
          "assistant",
          data.response || data.content,
          this.state.currentMessageId,
        );
      }
    }
    
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("LocalLMResponse", { response: data.response || data.content });
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleTTSStart() {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("TTSStart");
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleTTSEnd() {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("TTSEnd");
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleThoughtSubmit(): void {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("ThoughtApiSubmit");
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleFirstThoughtToken(): void {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("ThoughtApiFirstToken");
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }

  private handleThoughtResponse(data: any): void {
    this.config.onThoughtReceived?.(data.thought, data.index);
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("ThoughtParsed", { response: data.thought });
      this.config.onEventData?.(this.eventTracker.getData());
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
    if (this.state.selectedModel === "none") {
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
    this.config.onMessageReceived?.("user", text, userMessageId);
    
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("UserInputReceived", { text });
      this.config.onEventData?.(this.eventTracker.getData());
    }

    this.worker.postMessage({
      type: "process_text",
      text: text.trim(),
      enableTTS: this.state.mode === "voice",
    });
  }

  private async processTextWithGeminiStandalone(text: string) {
    this.state.isProcessing = true;
    this.state.currentMessageId = null;
    
    if (!this.eventTracker.hasActiveTurn()) {
      this.startNewTurn();
    }
    
    const userMessageId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    this.config.onMessageReceived?.("user", text, userMessageId);
    
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("UserInputReceived", { text });
      this.config.onEventData?.(this.eventTracker.getData());
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
          this.config.onStatusChange?.("processing_complete", "");
          continue;
        }
        
        if (chunk.includes('[done]')) {
          this.config.onStatusChange?.("response_complete", "");
          this.state.isProcessing = false;
          break;
        }

        fullResponse += chunk;        
        if (firstToken) {
          this.config.onMessageReceived?.("assistant", fullResponse, assistantMessageId);
          this.config.onMessageUpdated?.(assistantMessageId, fullResponse);
        }
      }

    } catch (error) {
      console.error("Error in Gemini-standalone processing:", error);
      this.config.onStatusChange?.("error", `Failed to process with Gemini: ${error}`);
      this.state.isProcessing = false;
    }
  }

  setVoice(voice: string) {
    if (this.worker) {
      this.worker.postMessage({ type: "set_voice", voice });
    }
  }

  setThoughtProvider(provider: "gemini" | "none") {
    this.state.thoughtProvider = provider;
    if (this.worker) {
      this.worker.postMessage({ type: "set_thought_provider", provider });
    }
  }

  setModel(modelId: "maximuspowers/smollm-convo-filler-onnx-official" | "HuggingFaceTB/SmolLM-360M-Instruct" | "none") {
    this.state.selectedModel = modelId;
    if (this.worker && modelId !== "none") {
      this.worker.postMessage({ type: "set_model", modelId });
    }
  }

  getVoices() {
    return this.state.voices;
  }

  private startNewTurn(): void {
    const metadata: TurnMetadata = {
      localModel: this.state.selectedModel === "maximuspowers/smollm-convo-filler-onnx-official" ? "smollm-finetuned" : "smollm-base",
      thoughtModel: this.state.thoughtProvider === "gemini" ? "gemini-flash-2.0" : "none",
      voiceMode: this.state.mode === "voice"
    };
    this.eventTracker.startNewTurn(metadata);
  }

  getEventData() {
    return this.eventTracker.getData();
  }

  resetEventData() {
    this.eventTracker.reset();
  }

  async switchMode(newMode: InferenceMode) {
    if (newMode === this.state.mode) return;
    this.dispose(); // clean up old mode
    await this.initialize(newMode);
  }

  dispose() {
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
