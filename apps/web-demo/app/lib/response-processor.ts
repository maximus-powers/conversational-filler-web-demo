import { pipeline } from "@huggingface/transformers";
import PQueue from 'p-queue';

// We'll use PQueue for better queue management

interface ProcessorConfig {
  onThoughtReceived?: (thought: string, index: number) => void;
  enableTTS?: boolean;
}

interface ProcessorState {
  thoughtQueue: string[];
  responseHistory: string[];
  thoughtBuffer: string;
  processedContent: string;
  thoughtsToProcess: string[];
  isProcessing: boolean;
}

export class ResponseProcessor {
  private lmPipeline: any;
  private ttsPipeline: any | null = null;
  private enableTTS: boolean;
  private speakerEmbeddings = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin';
  private abortSignal: AbortSignal | null = null;
  private inferenceQueue: PQueue; // webgpu inference queue (resources can't be used by more than one TF pipeline at once)
  private audioQueue: PQueue; // audio playback queue because it was playing multiple at a time
  private currentOnUpdate: ((content: string) => void) | null = null;
  private onThoughtReceived?: (thought: string, index: number) => void;
  private state: ProcessorState;
  private currentInput: string = "";

  constructor(config: ProcessorConfig) {
    this.onThoughtReceived = config.onThoughtReceived;
    this.enableTTS = config.enableTTS || false;
    this.inferenceQueue = new PQueue({ concurrency: 1 });
    this.audioQueue = new PQueue({ concurrency: 1 });
    
    // init state
    this.state = {
      thoughtQueue: [],
      responseHistory: [],
      thoughtBuffer: "",
      processedContent: "",
      thoughtsToProcess: [],
      isProcessing: false
    };
  }

  async initialize(): Promise<void> {
    this.lmPipeline = await pipeline(
      "text-generation",
      "maximuspowers/smollm-convo-filler-onnx-official",
      {
        dtype: "fp32",
        device: "webgpu",
      },
    );

    if (this.enableTTS) {
      try {
        this.ttsPipeline = await pipeline(
          'text-to-speech', 
          'Stoned-Code/piper-en_US-glados-medium',
          { 
            dtype: 'q8',
            device: 'webgpu' 
          }
        );
        console.log("TTS pipeline ready");
      } catch (error) {
        console.warn("Failed to initialize TTS pipeline:", error);
        this.enableTTS = false;
      }
    }
  }

  private async playAudio(audioData: Float32Array, sampleRate: number): Promise<void> {
    return this.audioQueue.add(async () => {
      try {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const audioBuffer = audioContext.createBuffer(1, audioData.length, sampleRate);
        audioBuffer.copyToChannel(audioData, 0);
        
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        
        return new Promise<void>((resolve) => {
          source.onended = () => {
            audioContext.close();
            resolve();
          };
          source.start();
        });
      } catch (error) {
        console.warn("Failed to play audio:", error);
      }
    });
  }

  private async speakText(text: string): Promise<void> {
    if (!this.enableTTS || !this.ttsPipeline || !text.trim()) {
      return;
    }

    try {
      // queue tts synthesis to prevent webgpu conflicts
      const result = await this.inferenceQueue.add(() =>
        this.ttsPipeline(text, { 
          speaker_embeddings: this.speakerEmbeddings 
        })
      ) as any;
      
      if (result && result.audio && result.sampling_rate) {
        await this.playAudio(result.audio, result.sampling_rate);
      }
    } catch (error) {
      console.warn("Failed to synthesize speech:", error);
    }
  }

  async generate(currentInput: string, abortSignal: AbortSignal, onUpdate: (content: string) => void): Promise<void> {
    this.currentInput = currentInput;
    this.abortSignal = abortSignal;
    this.currentOnUpdate = onUpdate;
    
    // reset state for new generation
    this.state = {
      thoughtQueue: [],
      responseHistory: [],
      thoughtBuffer: "",
      processedContent: "",
      thoughtsToProcess: [],
      isProcessing: false
    };
    
    // first response doesn't wait for thoughts
    const immediatePrompt = `<|im_start|>user\n${currentInput}<|im_end|>\n<|im_start|>assistant\n`;
    try {
      const immediateResult = await this.inferenceQueue.add(() => 
        this.lmPipeline(immediatePrompt, {
          max_new_tokens: 50,
          temperature: 0.7,
          do_sample: true,
          return_full_text: false,
          repetition_penalty: 1.2,
          top_p: 0.9,
          top_k: 50,
          pad_token_id: 2,
          eos_token_id: 2,
        })
      ) as any;

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
        this.state.responseHistory.push(immediateResponse);
        this.state.processedContent = immediateResponse;
        onUpdate(this.state.processedContent);
        
        // tts
        this.speakText(immediateResponse);
      }
    } catch (error) {
      console.error("Immediate response generation failed:", error);
    }
  }

  private async processNextThought(): Promise<void> {
    if (this.state.isProcessing || this.state.thoughtsToProcess.length === 0) return;
    this.state.isProcessing = true;

    const thought = this.state.thoughtsToProcess.shift()!;

    // build context including prev responses
    let contextPrompt = `<|im_start|>user\n${this.currentInput}<|im_end|>\n`;
    this.state.responseHistory.forEach((resp: string, idx: number) => {
      if (idx < this.state.thoughtQueue.length) {
        contextPrompt += `<|im_start|>knowledge\n${this.state.thoughtQueue[idx]}<|im_end|>\n`;
      }
      contextPrompt += `<|im_start|>assistant\n${resp}<|im_end|>\n`;
    });
    // add current thought
    contextPrompt += `<|im_start|>knowledge\n${thought}<|im_end|>\n<|im_start|>assistant\n`;

    // run through local model
    try {
      const result = await this.inferenceQueue.add(() =>
        this.lmPipeline(contextPrompt, {
          max_new_tokens: 50,
          temperature: 0.7,
          do_sample: true,
          return_full_text: false,
          repetition_penalty: 1.2,
          top_p: 0.9,
          top_k: 50,
          pad_token_id: 2,
          eos_token_id: 2,
        })
      ) as any;
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
        this.state.responseHistory.push(response);
        this.state.processedContent += " " + response;
        if (this.currentOnUpdate) {
          this.currentOnUpdate(this.state.processedContent);
        }
        
        // tts
        this.speakText(response);
      }
    } catch (error) {
      console.error("Thought processing failed:", error);
    }
    this.state.isProcessing = false;

    // call recursively until queue empty
    if (this.state.thoughtsToProcess.length > 0) {
      this.processNextThought();
    }
  }

  processThoughtChunk(chunk: string): void {
    this.state.thoughtBuffer += chunk;

    // extract thoughts from buffer with [bt] and [et] markers
    let lastProcessedIndex = 0;
    const regex = /\[bt\](.*?)\[et\]/g;
    let match;

    while ((match = regex.exec(this.state.thoughtBuffer)) !== null) {
      const thought = match[1].trim();
      if (thought && !this.state.thoughtQueue.includes(thought)) {
        this.state.thoughtQueue.push(thought);
        this.state.thoughtsToProcess.push(thought);          
        if (this.onThoughtReceived) {
          this.onThoughtReceived(thought, this.state.thoughtQueue.length - 1);
        }
        // start processing if not already
        this.processNextThought();
        lastProcessedIndex = match.index + match[0].length;
      }
    }
    // remove processed part of buffer
    if (lastProcessedIndex > 0) {
      this.state.thoughtBuffer = this.state.thoughtBuffer.substring(lastProcessedIndex);
    }
  }
    
  async waitForCompletion(): Promise<void> {
    while (this.state.isProcessing || this.state.thoughtsToProcess.length > 0) {
      await new Promise(resolve => setTimeout(resolve, 0)); // timeout needed or page become unresponsive
    }
  }
    
  async enableTTSMode(): Promise<void> {
    if (this.enableTTS && this.ttsPipeline) {
      return; // already enabled
    }
    
    this.enableTTS = true;
    
    try {
      if (!this.ttsPipeline) {
        this.ttsPipeline = await this.inferenceQueue.add(() =>
          pipeline('text-to-speech', 'Xenova/speecht5_tts', { dtype: 'fp32' })
        ) as any;
        console.log("TTS pipeline ready");
      }
    } catch (error) {
      console.warn("Failed to initialize TTS pipeline:", error);
      this.enableTTS = false;
      throw error;
    }
  }
  
  disableTTSMode(): void {
    this.enableTTS = false;
  }
  
  isTTSEnabled(): boolean {
    return this.enableTTS && !!this.ttsPipeline;
  }

  getState(): ProcessorState {
    return this.state;
  }
}