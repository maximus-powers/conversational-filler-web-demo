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
      this.worker = new Worker('/speech-worker-bundled.js');
      this.setupWorkerListeners();
      if (mode === 'voice') {
        await this.setupAudioContexts();
      }
      this.worker.postMessage({ type: 'init' });
      await this.waitForWorkerReady();
      this.state.isReady = true;
    } catch (error) {
      console.error('Failed to initialize:', error);
      throw error;
    }
  }

  private async waitForWorkerReady(): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Worker initialization timed out'));
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
      console.error('Worker error:', error);
      this.config.onTimelineEvent?.('error', 'Worker', 'Worker error', error.toString());
    };

    this.worker.onmessage = ({ data }) => {
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
          this.config.onTimelineEvent?.('thought', 'OpenAI', `Thought ${data.index + 1}`, data.thought);
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

  private handleStatusMessage(data: any) {
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

  private handleTranscription(text: string) {
    if (!text) return;
    this.config.onMessageReceived?.('user', text, Date.now().toString());
    this.config.onTranscriptionReceived?.(text);
    this.config.onTimelineEvent?.('transcription', 'Whisper', 'Transcribed', text);
  }

  private handleImmediateResponse(data: any) {
    this.state.currentMessageId = (Date.now() + 1).toString();
    this.config.onMessageReceived?.('assistant', data.response || data.content, (Date.now() + 1).toString());
    this.config.onTimelineEvent?.('smollm-response', 'SmolLM', 'Immediate response', data.response);
  }

  private handleEnhancedResponse(data: any) {
    if (this.state.currentMessageId) {
      this.config.onMessageUpdated?.(this.state.currentMessageId, data.response);
      this.config.onTimelineEvent?.('smollm-enhanced', 'SmolLM', 'Enhanced response', data.response);
    }
  }

  private handleAudioOutput(data: any): void {
    if (this.state.mode !== 'voice' || !data.result) return;
    const audioBuffer = data.result; // { type: 'output', text: string, result: Float32Array }
    console.log('Handling audio output, buffer length:', audioBuffer?.length, 'text:', data.text);
    if (this.playbackNode) {
      this.state.isPlaying = true;
      this.playbackNode.port.postMessage(audioBuffer);
    }
  }

  private async setupAudioContexts(): Promise<void> {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
      sampleRate: 24000, // matches tts output sample rate
    });

    await this.audioContext.audioWorklet.addModule('/workers/play-worklet.js');
    this.playbackNode = new AudioWorkletNode(this.audioContext, 'play-worklet');
    this.playbackNode.connect(this.audioContext.destination);
    
    // listen for playback ended and notify worker
    this.playbackNode.port.onmessage = (event) => {
      if (event.data.type === 'playback_ended') {
        this.state.isPlaying = false;
        if (this.worker) {
          this.worker.postMessage({ type: 'playback_ended' });
        }
      }
    };

    await this.setupMicrophone();
  }

  private async setupMicrophone(): Promise<void> {
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
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);

      // register VAD processor
      await this.audioContext.audioWorklet.addModule('/workers/vad-processor.js');
      this.worklet = new AudioWorkletNode(this.audioContext, 'vad-processor', {
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
        if (event.data.type === 'audio' && this.worker) {
          this.worker.postMessage({
            type: 'audio',
            buffer: event.data.audio,
          });
        }
      };
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
    const userMessageId = Date.now().toString();
    this.config.onMessageReceived?.('user', text, userMessageId);
    this.config.onTimelineEvent?.('user-input', 'User', 'Text input', text);

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

  setVoice(voice: string) {
    if (this.worker) {
      this.worker.postMessage({ type: 'set_voice', voice });
    }
  }

  getVoices(): Record<string, any> {
    return this.state.voices;
  }

  getState() {
    return { ...this.state };
  }

  async switchMode(newMode: AppMode): Promise<void> {
    if (newMode === this.state.mode) return;
    this.dispose(); // clean up old mode
    await this.initialize(newMode);
  }

  dispose(): void {
    if (this.state.isRecording) {
      this.stopRecording();
    }
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