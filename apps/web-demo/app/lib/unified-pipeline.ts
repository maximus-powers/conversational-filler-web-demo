import { INPUT_SAMPLE_RATE } from './audio-constants';

export type AppMode = 'text' | 'voice';

export interface UnifiedPipelineConfig {
  onMessageReceived?: (role: 'user' | 'assistant', content: string, messageId?: string) => void;
  onMessageUpdated?: (messageId: string, content: string) => void;
  onThoughtReceived?: (thought: string, index: number) => void;
  onTranscriptionReceived?: (text: string) => void;
  onStatusChange?: (status: string, message: string) => void;
  onTimelineEvent?: (type: string, model: string, message: string, content?: string) => void;
}

export interface UnifiedPipelineState {
  mode: AppMode;
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
  private messageQueue: Array<{ type: string; data: any }> = [];
  private isWorkerReady = false;

  constructor(config: UnifiedPipelineConfig) {
    this.config = config;
    this.state = {
      mode: 'text',
      isReady: false,
      isProcessing: false,
      isRecording: false,
      isPlaying: false,
      voices: {},
      currentMessageId: null,
    };
  }

  async initialize(mode: AppMode = 'text'): Promise<void> {
    this.state.mode = mode;
    
    try {
      // Initialize worker as classic worker (bundled with esbuild)
      this.worker = new Worker('/speech-worker-bundled.js');

      // Setup worker listeners
      this.setupWorkerListeners();

      // Initialize audio contexts only for voice mode
      if (mode === 'voice') {
        await this.setupAudioContexts();
      }

      // Send init message
      this.worker.postMessage({ type: 'init' });

      // Wait for worker ready
      await this.waitForWorkerReady();

      this.state.isReady = true;
      console.log(`UnifiedPipeline initialized in ${mode} mode`);
    } catch (error) {
      console.error('Failed to initialize UnifiedPipeline:', error);
      throw error;
    }
  }

  private async waitForWorkerReady(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Increase timeout to 60 seconds for larger models
      const timeout = setTimeout(() => {
        reject(new Error('Worker initialization timeout - model loading may take longer for large models'));
      }, 120000);

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

  private setupWorkerListeners(): void {
    if (!this.worker) return;

    this.worker.onerror = (error) => {
      console.error('Worker error:', error);
      this.config.onTimelineEvent?.('error', 'Worker', 'Worker error', error.toString());
    };

    this.worker.onmessage = ({ data }) => {
      console.log('Worker message:', data.type, data);

      if (data.error) {
        console.error('Worker error:', data.error);
        this.config.onTimelineEvent?.('error', 'Worker', 'Processing error', data.error);
        return;
      }

      switch (data.type) {
        case 'info':
          console.log('Worker info:', data.message);
          break;

        case 'status':
          this.handleStatusMessage(data);
          break;

        case 'transcription':
          this.handleTranscription(data.text);
          break;

        case 'immediate_response':
          this.handleImmediateResponse(data);
          break;

        case 'enhanced_response':
          this.handleEnhancedResponse(data);
          break;

        case 'thought':
          this.config.onThoughtReceived?.(data.thought, data.index);
          this.config.onTimelineEvent?.('openai-thought', 'OpenAI', `Thought ${data.index + 1}`, data.thought);
          break;

        case 'output':
          this.handleAudioOutput(data);
          break;

        case 'tts_start':
          this.config.onTimelineEvent?.('tts-start', 'TTS', 'Speaking', data.text);
          break;

        case 'tts_end':
          this.config.onTimelineEvent?.('tts-end', 'TTS', 'Speech complete', data.text);
          break;
      }
    };
  }

  private handleStatusMessage(data: any): void {
    if (data.status === 'ready') {
      this.isWorkerReady = true;
      this.state.voices = data.voices || {};
      this.config.onTimelineEvent?.('model-ready', 'Pipeline', 'All models loaded', '');
    } else if (data.status === 'recording_start') {
      this.state.isRecording = true;
      this.config.onTimelineEvent?.('recording-start', 'VAD', 'Voice detected', '');
    } else if (data.status === 'recording_end') {
      this.state.isRecording = false;
      this.config.onTimelineEvent?.('recording-end', 'VAD', 'Processing speech', '');
    }
    this.config.onStatusChange?.(data.status, data.message || '');
  }

  private handleTranscription(text: string): void {
    if (!text || text === '[BLANK_AUDIO]') return;
    
    // Add user message
    const messageId = Date.now().toString();
    this.config.onMessageReceived?.('user', text, messageId);
    this.config.onTranscriptionReceived?.(text);
    this.config.onTimelineEvent?.('whisper-transcription', 'Whisper', 'Transcribed', text);
  }

  private handleImmediateResponse(data: any): void {
    // Create new assistant message
    const messageId = (Date.now() + 1).toString();
    this.state.currentMessageId = messageId;
    
    this.config.onMessageReceived?.('assistant', data.response || data.content, messageId);
    this.config.onTimelineEvent?.('smollm-response', 'SmolLM', 'Immediate response', data.response);
  }

  private handleEnhancedResponse(data: any): void {
    // Append to existing assistant message
    if (this.state.currentMessageId) {
      this.config.onMessageUpdated?.(this.state.currentMessageId, data.response);
      this.config.onTimelineEvent?.('smollm-enhanced', 'SmolLM', 'Enhanced response', data.response);
    }
  }

  private handleAudioOutput(data: any): void {
    if (this.state.mode !== 'voice' || !data.result) return;

    // The worker sends { type: 'output', text: string, result: Float32Array }
    const audioBuffer = data.result;
    console.log('Handling audio output, buffer length:', audioBuffer?.length, 'text:', data.text);
    
    if (audioBuffer instanceof Float32Array && this.playbackNode) {
      this.state.isPlaying = true;
      this.playbackNode.port.postMessage(audioBuffer);
    }
  }

  private async setupAudioContexts(): Promise<void> {
    // Use 24000 Hz to match TTS output sample rate
    // The browser will resample microphone input as needed
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
      sampleRate: 24000,
    });

    // Register play worklet for TTS output  
    await this.audioContext.audioWorklet.addModule('/workers/play-worklet.js');
    
    this.playbackNode = new AudioWorkletNode(this.audioContext, 'play-worklet');
    this.playbackNode.connect(this.audioContext.destination);
    
    // Listen for playback ended messages
    this.playbackNode.port.onmessage = (event) => {
      if (event.data.type === 'playback_ended') {
        this.state.isPlaying = false;
        // Notify worker that playback has ended
        if (this.worker) {
          this.worker.postMessage({ type: 'playback_ended' });
        }
      }
    };

    // Setup microphone for voice mode
    await this.setupMicrophone();
  }

  private async setupMicrophone(): Promise<void> {
    if (!this.audioContext) return;

    try {
      console.log('Requesting microphone permission...');
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: INPUT_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
        } as MediaTrackConstraints,
      });
      console.log('Microphone permission granted, stream:', this.mediaStream);

      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      
      // Register VAD processor
      await this.audioContext.audioWorklet.addModule('/workers/vad-processor.js');
      
      this.worklet = new AudioWorkletNode(this.audioContext, 'vad-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        processorOptions: { sampleRate: INPUT_SAMPLE_RATE },
      });

      // Connect audio pipeline
      source.connect(this.worklet);
      this.worklet.connect(this.audioContext.destination);

      // Handle audio from VAD
      let audioMessageCount = 0;
      this.worklet.port.onmessage = (event) => {
        audioMessageCount++;
        if (audioMessageCount % 100 === 0) {
          console.log('UnifiedPipeline received audio from VAD, count:', audioMessageCount, 'data:', event.data);
        }
        
        if (event.data.type === 'audio' && this.worker) {
          // Forward the audio buffer to the worker
          // The vad-processor sends { type: 'audio', audio: Float32Array }
          // We forward it as { type: 'audio', buffer: Float32Array } to match webgpu-demo
          this.worker.postMessage({
            type: 'audio',
            buffer: event.data.audio,
          });
        }
      };

      console.log('Microphone setup complete');
    } catch (error) {
      console.error('Failed to setup microphone:', error);
      throw error;
    }
  }

  async processText(text: string): Promise<void> {
    if (!this.worker || !this.isWorkerReady) {
      throw new Error('Pipeline not ready');
    }

    this.state.isProcessing = true;

    // Add user message immediately
    const userMessageId = Date.now().toString();
    this.config.onMessageReceived?.('user', text, userMessageId);
    this.config.onTimelineEvent?.('user-input', 'User', 'Text input', text);

    // Send to worker for processing
    this.worker.postMessage({
      type: 'process_text',
      text: text.trim(),
      enableTTS: this.state.mode === 'voice',
    });
  }

  async startRecording(): Promise<void> {
    if (this.state.mode !== 'voice' || !this.worker) {
      throw new Error('Voice mode not initialized');
    }

    this.state.isRecording = true;
    this.worker.postMessage({ type: 'start_recording' });
  }

  async stopRecording(): Promise<void> {
    if (!this.worker) return;
    
    this.state.isRecording = false;
    this.worker.postMessage({ type: 'stop_recording' });
  }

  setVoice(voice: string): void {
    if (this.worker) {
      this.worker.postMessage({ type: 'set_voice', voice });
    }
  }

  getVoices(): Record<string, any> {
    return this.state.voices;
  }

  getState(): UnifiedPipelineState {
    return { ...this.state };
  }

  async switchMode(newMode: AppMode): Promise<void> {
    if (newMode === this.state.mode) return;

    // Cleanup current mode
    this.dispose();

    // Reinitialize with new mode
    await this.initialize(newMode);
  }

  dispose(): void {
    // Stop recording if active
    if (this.state.isRecording) {
      this.stopRecording();
    }

    // Cleanup audio
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
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

    // Terminate worker
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