import { INPUT_SAMPLE_RATE } from './audio-constants';

export interface SpeechProcessorConfig {
  onThoughtReceived?: (thought: string, index: number) => void;
  onTTSCompleted?: (text: string) => void;
  onTranscriptionReceived?: (text: string) => void;
  onImmediateResponse?: (response: string) => void;
  onEnhancedResponse?: (response: string) => void;
  onTTSStarted?: (text: string) => void;
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
      // Use bundled worker
      this.worker = new Worker('/speech-worker-bundled.js', { type: 'module' });
      
      // Create a promise that resolves when worker is ready
      const workerReady = new Promise<void>((resolve) => {
        const checkReady = () => {
          if (this.state.isReady) {
            resolve();
          } else {
            setTimeout(checkReady, 100);
          }
        };
        checkReady();
      });
      
      this.setupWorkerListeners();
      
      // Initialize audio contexts
      await this.setupAudioContexts();
      
      // Send init message to trigger ready status (avoids race condition)
      console.log('Sending init message to worker');
      this.worker.postMessage({ type: 'init' });
      
      // Wait for worker to be ready (with timeout)
      await Promise.race([
        workerReady,
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Worker initialization timeout')), 10000)
        )
      ]);
      
      console.log('SpeechProcessor initialization complete');
      
    } catch (error) {
      console.error('Failed to initialize SpeechProcessor:', error);
      throw error;
    }
  }

  private setupWorkerListeners(): void {
    if (!this.worker) return;

    this.worker.onerror = (error) => {
      console.error('Worker error event:', error);
    };

    this.worker.onmessage = ({ data }) => {
      console.log('Worker message received:', data.type, data);
      if (data.error) {
        console.error('Worker error:', data.error);
        return;
      }

      switch (data.type) {
        case 'info':
          console.log('Worker info:', data.message);
          break;
          
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
          // Trigger TTS start event for timeline
          this.config.onTTSStarted?.(data.response);
          break;
          
        case 'enhanced_response':
          this.config.onEnhancedResponse?.(data.response);
          // Trigger TTS start event for timeline
          this.config.onTTSStarted?.(data.response);
          break;
          
        case 'audio_output':
          if (data.audio && this.config.enableTTS) {
            console.log('Received TTS audio, accessing audio.audio for Float32Array');
            // WebGPU-demo pattern: access data.result.audio, but our worker sends data.audio.audio
            const audioBuffer = data.audio.audio || data.audio;
            console.log('Audio buffer type:', audioBuffer?.constructor?.name, 'length:', audioBuffer?.length);
            
            if (audioBuffer instanceof Float32Array) {
              // Direct audio streaming like webgpu-demo
              this.state.isPlaying = true;
              this.playbackNode?.port.postMessage(audioBuffer);
              this.config.onAudioOutput?.(audioBuffer);
              
              // Trigger TTS completion callback for timeline integration  
              if (data.text) {
                this.config.onTTSCompleted?.(data.text);
              }
            } else {
              console.warn('Expected Float32Array but got:', audioBuffer?.constructor?.name);
            }
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
      console.log('Requesting microphone access...');
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
      console.log('Microphone access granted, stream:', this.mediaStream);

      // Create audio source and VAD worklet
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      console.log('Created media stream source');
      
      this.worklet = new AudioWorkletNode(this.audioContext, 'vad-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
        channelCountMode: 'explicit',
        channelInterpretation: 'discrete',
      });
      console.log('Created VAD worklet node');

      source.connect(this.worklet);
      console.log('Connected source to worklet');

      // Ensure audio context is running
      if (this.audioContext.state !== 'running') {
        console.log('Audio context state:', this.audioContext.state, 'resuming...');
        await this.audioContext.resume();
        console.log('Audio context resumed, new state:', this.audioContext.state);
      }

      // Forward audio data to worker
      this.worklet.port.onmessage = (event) => {
        const { buffer } = event.data;
        console.log('VAD worklet received audio buffer, length:', buffer.length);
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
          this.config.onTTSCompleted?.('Audio playbook completed');
        }
      };

      console.log('Sending start_recording message to worker');
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

  // Process text input (for typed messages)
  async processText(text: string, abortSignal?: AbortSignal): Promise<void> {
    if (!this.state.isReady || !this.worker) {
      throw new Error('SpeechProcessor not ready');
    }
    
    console.log('Processing text input:', text);
    this.worker.postMessage({ 
      type: 'process_text', 
      text: text.trim(),
      enableTTS: this.config.enableTTS 
    });
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