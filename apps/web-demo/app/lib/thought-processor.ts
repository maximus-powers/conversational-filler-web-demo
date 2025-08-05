export async function ThoughtProcessor(config: {
  pipeline: any;
  currentInput: string;
  abortSignal: AbortSignal;
  onUpdate: (content: string) => void;
  onThoughtReceived?: (thought: string, index: number) => void;
}) {
  const { pipeline, currentInput, onUpdate, onThoughtReceived } = config;
  
  // init state
  const state: {
    thoughtQueue: string[];
    responseHistory: string[];
    thoughtBuffer: string;
    processedContent: string;
    thoughtsToProcess: string[];
    isProcessing: boolean;
  } = {
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
    const immediateResult = await pipeline(immediatePrompt, {
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
      state.responseHistory.push(immediateResponse);
      state.processedContent = immediateResponse;
      onUpdate(state.processedContent);
    }
  } catch (error) {
    console.error("Immediate response generation failed:", error);
  }

  // processes next thoughts from the queue
  const processNextThought = async () => {
    if (state.isProcessing || state.thoughtsToProcess.length === 0) return;
    state.isProcessing = true;

    const thought = state.thoughtsToProcess.shift()!;

    // build context including prev responses
    let contextPrompt = `<|im_start|>user\n${currentInput}<|im_end|>\n`;
    state.responseHistory.forEach((resp, idx) => {
      if (idx < state.thoughtQueue.length) {
        contextPrompt += `<|im_start|>knowledge\n${state.thoughtQueue[idx]}<|im_end|>\n`;
      }
      contextPrompt += `<|im_start|>assistant\n${resp}<|im_end|>\n`;
    });
    // add current thought
    contextPrompt += `<|im_start|>knowledge\n${thought}<|im_end|>\n<|im_start|>assistant\n`;

    // run through local model
    try {
      const result = await pipeline(contextPrompt, {
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
        state.responseHistory.push(response);
        state.processedContent += " " + response;
        onUpdate(state.processedContent);
      }
    } catch (error) {
      console.error("Thought processing failed:", error);
    }
    state.isProcessing = false;

    // call recurrsively until queue empty
    if (state.thoughtsToProcess.length > 0) {
      processNextThought();
    }
  };

  return {
    processThoughtChunk: (chunk: string) => {
      state.thoughtBuffer += chunk;

      // extract thoughts from buffer with [bt] and [et] markers
      let lastProcessedIndex = 0;
      const regex = /\[bt\](.*?)\[et\]/g;
      let match;

      while ((match = regex.exec(state.thoughtBuffer)) !== null) {
        const thought = match[1].trim();
        if (thought && !state.thoughtQueue.includes(thought)) {
          state.thoughtQueue.push(thought);
          state.thoughtsToProcess.push(thought);          
          if (onThoughtReceived) {
            onThoughtReceived(thought, state.thoughtQueue.length - 1);
          }
          // start processing if not already
          processNextThought();
          lastProcessedIndex = match.index + match[0].length;
        }
      }
      // remove processed part of buffer
      if (lastProcessedIndex > 0) {
        state.thoughtBuffer = state.thoughtBuffer.substring(lastProcessedIndex);
      }
    },
    
    waitForCompletion: async () => {
      while (state.isProcessing || state.thoughtsToProcess.length > 0) {
        await Promise.resolve();
      }
    },
    
    getState: () => state
  };
}