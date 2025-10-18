import { INPUT_SAMPLE_RATE } from "./audio-constants";
import { EventTracker, TurnMetadata } from "./event-tracker";

export interface PipelineConfig {
  enableSTT: boolean;
  enableThoughts: boolean;
  enableSmolLM: boolean;
  enableTTS: boolean;
}

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
  onConversationStart?: (startTime: number) => void;
}

export interface UnifiedPipelineState {
  pipelineConfig: PipelineConfig;
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
  private config: UnifiedPipelineConfig;
  private state: UnifiedPipelineState;
  private isWorkerReady = false;
  private eventTracker = new EventTracker();
  private hasStartedPlayback = false;

  constructor(config: UnifiedPipelineConfig, pipelineConfig?: PipelineConfig) {
    this.config = config;
    this.state = {
      pipelineConfig: pipelineConfig || {
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
      // Add cache busting parameter to force reload of worker
      const workerUrl = `/speech-worker-bundled.js?v=${Date.now()}`;
      this.worker = new Worker(workerUrl);
      this.setupWorkerListeners();

      // Only set up audio contexts if STT is enabled
      if (this.state.pipelineConfig.enableSTT) {
        await this.setupAudioContexts();
      }

      this.worker.postMessage({ type: "init" });
      await this.waitForWorkerReady();

      // Send initial pipeline configuration to worker
      const config = this.state.pipelineConfig;
      if (this.worker) {
        const provider = config.enableThoughts ? "gemini" : "none";
        this.worker.postMessage({ type: "set_thought_provider", provider });
        this.worker.postMessage({ type: "set_smollm_enabled", enabled: config.enableSmolLM });
        this.worker.postMessage({ type: "set_tts_enabled", enabled: config.enableTTS });
      }

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
        case "output_mp3":
          this.handleMP3Output(data);
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
        case "tts_synthesis_start":
          this.handleTTSSynthesisStart(data);
          break;
        case "tts_synthesis_end":
          this.handleTTSSynthesisEnd(data);
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
    if (!this.state.pipelineConfig.enableSTT || !data.result) return;
    const audioBuffer = data.result; // { type: 'output', text: string, result: Float32Array }
    if (this.playbackNode) {
      if (!this.hasStartedPlayback && this.eventTracker.hasActiveTurn()) {
        this.hasStartedPlayback = true;
        this.eventTracker.addEvent("AudioPlaybackStart");
        this.config.onEventData?.(this.eventTracker.getData());
      }

      this.state.isPlaying = true;
      this.playbackNode.port.postMessage(audioBuffer);
    }
  }

  private async handleMP3Output(data: any) {
    if (!data.audioBuffer) return;

    // Only play audio if TTS is enabled
    if (!this.state.pipelineConfig.enableTTS) return;

    try {
      // Decode MP3 on main thread
      if (!this.audioContext) {
        this.audioContext = new AudioContext({ sampleRate: 24000 });
      }

      const audioBuffer = await this.audioContext.decodeAudioData(data.audioBuffer);
      const float32Array = audioBuffer.getChannelData(0);

      // If STT is disabled (text mode), play directly through audio context
      if (!this.state.pipelineConfig.enableSTT) {
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.start();

        if (!this.hasStartedPlayback && this.eventTracker.hasActiveTurn()) {
          this.hasStartedPlayback = true;
          this.eventTracker.addEvent("AudioPlaybackStart");
          this.config.onEventData?.(this.eventTracker.getData());
        }

        source.onended = () => {
          // Notify worker that playback ended
          if (this.worker) {
            this.worker.postMessage({ type: "playback_ended" });
          }

          if (this.eventTracker.hasActiveTurn()) {
            this.eventTracker.addEvent("AudioPlaybackEnd");
            this.config.onEventData?.(this.eventTracker.getData());
          }
        };
      }
      // If STT enabled (voice mode), send to playback worklet
      else if (this.state.pipelineConfig.enableSTT && this.playbackNode) {
        if (!this.hasStartedPlayback && this.eventTracker.hasActiveTurn()) {
          this.hasStartedPlayback = true;
          this.eventTracker.addEvent("AudioPlaybackStart");
          this.config.onEventData?.(this.eventTracker.getData());
        }

        this.state.isPlaying = true;
        this.playbackNode.port.postMessage(float32Array);
      }
    } catch (error) {
      console.error('Error decoding MP3:', error);
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
    this.hasStartedPlayback = false;
    this.startNewTurn();
    
    if (data.timestamp && this.config.onConversationStart) {
      this.config.onConversationStart(data.timestamp);
    }
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
      this.eventTracker.addEvent("LocalLMResponse", { 
        response: data.rawResponse || data.response || data.content,
        prompt: data.fullPrompt 
      });
      this.config.onEventData?.(this.eventTracker.getData());
    }
  }


  private handleTTSSynthesisStart(data: any) {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("TTSSynthesisStart", {
        text: data.text,
        responseIndex: data.responseIndex,
        synthesisId: data.synthesisId
      });
      this.config.onEventData?.(this.eventTracker.getData());
    }
    
    console.log(`🔊 TTS Synthesis Start: "${data.text}" (Response #${data.responseIndex}, ID: ${data.synthesisId})`, {
      timestamp: new Date(data.timestamp).toISOString(),
      turnOffset: data.turnOffset
    });
  }

  private handleTTSSynthesisEnd(data: any) {
    if (this.eventTracker.hasActiveTurn()) {
      this.eventTracker.addEvent("TTSSynthesisEnd", {
        text: data.text,
        responseIndex: data.responseIndex,
        synthesisId: data.synthesisId
      });
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
        this.hasStartedPlayback = false; 
        
        if (this.eventTracker.hasActiveTurn()) {
          this.eventTracker.addEvent("AudioPlaybackEnd");
          this.config.onEventData?.(this.eventTracker.getData());
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
    const config = this.state.pipelineConfig;

    console.log('processText called with config:', {
      enableTTS: config.enableTTS,
      enableThoughts: config.enableThoughts,
      enableSmolLM: config.enableSmolLM
    });

    // If no SmolLM and no thoughts, use Gemini standalone
    if (!config.enableSmolLM && !config.enableThoughts) {
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

  updatePipelineConfig(config: Partial<PipelineConfig>) {
    this.state.pipelineConfig = { ...this.state.pipelineConfig, ...config };

    // Update worker with new config
    if (this.worker) {
      if (config.enableThoughts !== undefined) {
        const provider = config.enableThoughts ? "gemini" : "none";
        this.worker.postMessage({ type: "set_thought_provider", provider });
      }
      if (config.enableSmolLM !== undefined) {
        this.worker.postMessage({ type: "set_smollm_enabled", enabled: config.enableSmolLM });
      }
      if (config.enableTTS !== undefined) {
        this.worker.postMessage({ type: "set_tts_enabled", enabled: config.enableTTS });
      }
      if (config.enableSTT !== undefined) {
        this.worker.postMessage({ type: "set_stt_enabled", enabled: config.enableSTT });
      }
    }
  }

  async toggleSTT(enable: boolean) {
    this.state.pipelineConfig.enableSTT = enable;

    if (enable && !this.audioContext) {
      // Set up audio contexts if not already done
      await this.setupAudioContexts();
    } else if (!enable && this.audioContext) {
      // Clean up audio contexts
      this.disposeAudioContexts();
    }
  }

  toggleThoughts(enable: boolean) {
    this.updatePipelineConfig({ enableThoughts: enable });
  }

  toggleSmolLM(enable: boolean) {
    this.updatePipelineConfig({ enableSmolLM: enable });
  }

  toggleTTS(enable: boolean) {
    this.updatePipelineConfig({ enableTTS: enable });
  }

  getPipelineConfig(): PipelineConfig {
    return { ...this.state.pipelineConfig };
  }

  getVoices() {
    return this.state.voices;
  }

  private startNewTurn(): void {
    const metadata: TurnMetadata = {
      localModel: this.state.pipelineConfig.enableSmolLM ? "smollm-finetuned" : "none",
      thoughtModel: this.state.pipelineConfig.enableThoughts ? "gemini-flash-2.0" : "none",
      voiceMode: this.state.pipelineConfig.enableSTT
    };
    this.eventTracker.startNewTurn(metadata);
  }

  getEventData() {
    return this.eventTracker.getData();
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
