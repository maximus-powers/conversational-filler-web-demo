import { INPUT_SAMPLE_RATE } from './audio-constants';

export interface SpeechProcessorConfig {
  onThoughtReceived?: (thought: string, index: number) => void;
  onTTSCompleted?: (text: string) => void;
  onTranscriptionReceived?: (text: string) => void;
  onImmediateResponse?: (response: string) => void;
  onEnhancedResponse?: (response: string) => void;
  onStatusChange?: (status: string, message: string) => void;
  onAudioOutput?: (audio: Float32Array) => void;
  enableTTS?: boolean;
}

export interface SpeechProcessorState {
  isRecording: boolean;
  isPlaying: boolean;
  isReady: boolean;
  voices: Record<string, any>;
}

export class SpeechProcessor {
  private worker: Worker | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private worklet: AudioWorkletNode | null = null;
  private playbackNode: AudioWorkletNode | null = null;
  private config: SpeechProcessorConfig;
  private state: SpeechProcessorState;
  
  constructor(config: SpeechProcessorConfig) {
    this.config = config;
    this.state = {
      isRecording: false,
      isPlaying: false,
      isReady: false,
      voices: {},
    };
  }

  async initialize(): Promise<void> {
    try {
      // Initialize web worker as module (needed for ES imports)
      this.worker = new Worker('/workers/speech-worker.js', { type: 'module' });
      this.setupWorkerListeners();
      
      // Initialize audio contexts
      await this.setupAudioContexts();
      
      // Send init message to trigger ready status (avoids race condition)
      this.worker.postMessage({ type: 'init' });
      
    } catch (error) {
      console.error('Failed to initialize SpeechProcessor:', error);
      throw error;
    }
  }

  private setupWorkerListeners(): void {
    if (!this.worker) return;

    this.worker.onmessage = ({ data }) => {
      if (data.error) {
        console.error('Worker error:', data.error);
        return;
      }

      switch (data.type) {
        case 'status':
          console.log('SpeechProcessor status:', data.status, data.message);
          if (data.status === 'ready') {
            this.state.isReady = true;
            this.state.voices = data.voices || {};
            console.log('SpeechProcessor ready! Voices:', Object.keys(this.state.voices));
          } else if (data.status === 'recording_start') {
            this.state.isRecording = true;
          } else if (data.status === 'recording_end') {
            this.state.isRecording = false;
          }
          this.config.onStatusChange?.(data.status, data.message || '');
          break;
          
        case 'thought':
          this.config.onThoughtReceived?.(data.thought, data.index);
          break;
          
        case 'transcription':
          this.config.onTranscriptionReceived?.(data.text);
          break;
          
        case 'immediate_response':
          this.config.onImmediateResponse?.(data.response);
          break;
          
        case 'enhanced_response':
          this.config.onEnhancedResponse?.(data.response);
          break;
          
        case 'audio_output':
          if (data.audio && this.config.enableTTS && !this.state.isPlaying) {
            // Direct audio streaming like webgpu-demo for better performance
            this.state.isPlaying = true;
            this.playbackNode?.port.postMessage(data.audio);
            this.config.onAudioOutput?.(data.audio);
          }
          break;
          
        case 'info':
          console.log('Worker info:', data.message);
          break;
      }
    };

    this.worker.onerror = (error) => {
      console.error('Worker error:', error);
    };
  }

  private async setupAudioContexts(): Promise<void> {
    // Input audio context for microphone
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
      sampleRate: INPUT_SAMPLE_RATE,
    });

    // Load VAD processor
    await this.audioContext.audioWorklet.addModule('/workers/vad-processor.js');
    
    // Load playback processor  
    await this.audioContext.audioWorklet.addModule('/workers/play-worklet.js');
  }

  async startRecording(): Promise<void> {
    console.log('StartRecording called. State:', {
      isReady: this.state.isReady,
      hasAudioContext: !!this.audioContext,
      hasWorker: !!this.worker
    });
    
    if (!this.state.isReady || !this.audioContext || !this.worker) {
      throw new Error('SpeechProcessor not ready');
    }

    try {
      // Get microphone stream
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          autoGainControl: true,
          noiseSuppression: true,
          sampleRate: INPUT_SAMPLE_RATE,
        },
      });

      // Create audio source and VAD worklet
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      
      this.worklet = new AudioWorkletNode(this.audioContext, 'vad-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
        channelCountMode: 'explicit',
        channelInterpretation: 'discrete',
      });

      source.connect(this.worklet);

      // Forward audio data to worker
      this.worklet.port.onmessage = (event) => {
        const { buffer } = event.data;
        this.worker?.postMessage({ type: 'audio', buffer });
      };

      // Setup playback node
      this.playbackNode = new AudioWorkletNode(
        this.audioContext,
        'buffered-audio-worklet-processor'
      );

      this.playbackNode.connect(this.audioContext.destination);

      this.playbackNode.port.onmessage = (event) => {
        if (event.data.type === 'playback_ended') {
          this.state.isPlaying = false;
          this.config.onTTSCompleted?.('Audio playback completed');
        }
      };

      this.worker?.postMessage({ type: 'start_recording' });

    } catch (error) {
      console.error('Failed to start recording:', error);
      throw error;
    }
  }

  async stopRecording(): Promise<void> {
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

    this.worker?.postMessage({ type: 'end_recording' });
    this.state.isRecording = false;
  }


  setVoice(voiceId: string): void {
    this.worker?.postMessage({ type: 'set_voice', voice: voiceId });
  }

  getState(): SpeechProcessorState {
    return { ...this.state };
  }

  getVoices(): Record<string, any> {
    return this.state.voices;
  }

  dispose(): void {
    this.stopRecording();
    
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}