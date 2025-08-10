// Worker wrapper for Next.js compatibility
// This module provides a clean interface to the bundled worker

export function createUnifiedWorker(): Worker {
  // Use Next.js compatible pattern with new URL and import.meta.url
  // The bundled worker is in the public directory and contains all dependencies
  const worker = new Worker(
    new URL('../../../public/speech-worker-bundled.js', import.meta.url)
  );
  
  return worker;
}

export interface WorkerMessage {
  type: string;
  [key: string]: any;
}

export interface WorkerResponse {
  type: string;
  error?: string;
  [key: string]: any;
}

// Message types for type safety
export const MessageTypes = {
  // Initialization
  INIT: 'init',
  
  // Text processing
  PROCESS_TEXT: 'process_text',
  
  // Voice processing
  START_RECORDING: 'start_recording',
  STOP_RECORDING: 'stop_recording',
  AUDIO: 'audio',
  
  // Configuration
  SET_VOICE: 'set_voice',
  
  // Control
  END_CALL: 'end_call',
  PLAYBACK_ENDED: 'playback_ended',
} as const;

// Response types from worker
export const ResponseTypes = {
  // Status updates
  INFO: 'info',
  STATUS: 'status',
  ERROR: 'error',
  
  // Processing results
  TRANSCRIPTION: 'transcription',
  IMMEDIATE_RESPONSE: 'immediate_response',
  ENHANCED_RESPONSE: 'enhanced_response',
  THOUGHT: 'thought',
  
  // Audio
  AUDIO_OUTPUT: 'audio_output',
  TTS_START: 'tts_start',
  TTS_END: 'tts_end',
} as const;

// Helper to create typed messages
export function createMessage(type: string, data?: any): WorkerMessage {
  return { type, ...data };
}

// Helper to validate worker responses
export function isValidResponse(data: any): data is WorkerResponse {
  return data && typeof data.type === 'string';
}