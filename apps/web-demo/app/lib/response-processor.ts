import { pipeline } from "@huggingface/transformers";

interface ProcessorConfig {
  onThoughtReceived?: (thought: string, index: number) => void;
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
  private abortSignal: AbortSignal | null = null;
  private currentOnUpdate: ((content: string) => void) | null = null;
  private onThoughtReceived?: (thought: string, index: number) => void;
  private state: ProcessorState;
  private currentInput: string = "";

  constructor(config: ProcessorConfig) {
    this.onThoughtReceived = config.onThoughtReceived;
    
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
  }

  async generate(currentInput: string, abortSignal: AbortSignal, onUpdate: (content: string) => void): Promise<void> {
    this.currentInput = currentInput;
    this.abortSignal = abortSignal;
    this.currentOnUpdate = onUpdate;
    
    // Reset state for new generation
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
      const immediateResult = await this.lmPipeline(immediatePrompt, {
        max_new_tokens: 50,
        temperature: 0.7,
        do_sample: true,
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
        this.state.responseHistory.push(immediateResponse);
        this.state.processedContent = immediateResponse;
        onUpdate(this.state.processedContent);
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
      const result = await this.lmPipeline(contextPrompt, {
        max_new_tokens: 50,
        temperature: 0.7,
        do_sample: true,
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
        this.state.responseHistory.push(response);
        this.state.processedContent += " " + response;
        if (this.currentOnUpdate) {
          this.currentOnUpdate(this.state.processedContent);
        }
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
    
  getState(): ProcessorState {
    return this.state;
  }
}