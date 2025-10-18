"use client";

import { Mic, Brain, Speaker, MessageSquare } from "lucide-react";

export interface PipelineControlsProps {
  enableSTT: boolean;
  enableThoughts: boolean;
  enableSmolLM: boolean;
  enableTTS: boolean;
  onToggleSTT: (enabled: boolean) => void;
  onToggleThoughts: (enabled: boolean) => void;
  onToggleSmolLM: (enabled: boolean) => void;
  onToggleTTS: (enabled: boolean) => void;
  disabled?: boolean;
}

export function PipelineControls({
  enableSTT,
  enableThoughts,
  enableSmolLM,
  enableTTS,
  onToggleSTT,
  onToggleThoughts,
  onToggleSmolLM,
  onToggleTTS,
  disabled = false,
}: PipelineControlsProps) {
  return (
    <div className="flex items-center gap-3">
      {/* STT Toggle */}
      <button
        onClick={() => onToggleSTT(!enableSTT)}
        disabled={disabled}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          enableSTT
            ? "bg-purple-500 text-white hover:bg-purple-600"
            : "bg-muted text-muted-foreground hover:bg-muted/80"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
        title="Speech-to-Text"
      >
        <Mic className="h-3.5 w-3.5" />
        <span>STT</span>
      </button>

      {/* Thoughts Toggle */}
      <button
        onClick={() => onToggleThoughts(!enableThoughts)}
        disabled={disabled}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          enableThoughts
            ? "bg-green-500 text-white hover:bg-green-600"
            : "bg-muted text-muted-foreground hover:bg-muted/80"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
        title="Gemini Thoughts"
      >
        <Brain className="h-3.5 w-3.5" />
        <span>Thoughts</span>
      </button>

      {/* SmolLM Toggle */}
      <button
        onClick={() => onToggleSmolLM(!enableSmolLM)}
        disabled={disabled}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          enableSmolLM
            ? "bg-blue-500 text-white hover:bg-blue-600"
            : "bg-muted text-muted-foreground hover:bg-muted/80"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
        title="SmolLM Filler"
      >
        <MessageSquare className="h-3.5 w-3.5" />
        <span>SmolLM</span>
      </button>

      {/* TTS Toggle */}
      <button
        onClick={() => onToggleTTS(!enableTTS)}
        disabled={disabled}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          enableTTS
            ? "bg-emerald-500 text-white hover:bg-emerald-600"
            : "bg-muted text-muted-foreground hover:bg-muted/80"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
        title="Text-to-Speech"
      >
        <Speaker className="h-3.5 w-3.5" />
        <span>TTS</span>
      </button>
    </div>
  );
}
