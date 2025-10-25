"use client";

import { Mic, Brain, Speaker, MessageSquare, ChevronDown, User } from "lucide-react";
import { useState, useRef, useEffect } from "react";

export type STTMode = "disabled" | "local" | "api";
export type SmolLMMode = "convfill" | "untrained" | "none";

export interface PipelineControlsProps {
  sttMode: STTMode;
  enableThoughts: boolean;
  smolLMMode: SmolLMMode;
  enableTTS: boolean;
  persona: string;
  onSTTModeChange: (mode: STTMode) => void;
  onToggleThoughts: (enabled: boolean) => void;
  onSmolLMModeChange: (mode: SmolLMMode) => void;
  onToggleTTS: (enabled: boolean) => void;
  onPersonaChange: (persona: string) => void;
  disabled?: boolean;
}

const PERSONA_OPTIONS = [
  { value: "none", label: "None" },
  { value: "educator", label: "Educator" },
  { value: "customer_service", label: "Customer Service" },
  { value: "event_planner", label: "Event Planner" },
  { value: "medical", label: "Medical Professional" },
  { value: "coach", label: "Coach" },
];

export function PipelineControls({
  sttMode,
  enableThoughts,
  smolLMMode,
  enableTTS,
  persona,
  onSTTModeChange,
  onToggleThoughts,
  onSmolLMModeChange,
  onToggleTTS,
  onPersonaChange,
  disabled = false,
}: PipelineControlsProps) {
  const [isPersonaOpen, setIsPersonaOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsPersonaOpen(false);
      }
    };

    if (isPersonaOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isPersonaOpen]);

  const selectedPersona = PERSONA_OPTIONS.find(opt => opt.value === persona);

  return (
    <div className="flex items-center gap-3">
      {/* STT Toggle */}
      <button
        onClick={() => {
          if (sttMode === "disabled") {
            onSTTModeChange("local");
          } else if (sttMode === "local") {
            onSTTModeChange("api");
          } else {
            onSTTModeChange("disabled");
          }
        }}
        disabled={disabled}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          sttMode !== "disabled"
            ? "bg-purple-500 text-white hover:bg-purple-600"
            : "bg-muted text-muted-foreground hover:bg-muted/80"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
        title="Speech-to-Text"
      >
        <Mic className="h-3.5 w-3.5" />
        <span>{sttMode === "local" ? "STT (Local)" : sttMode === "api" ? "STT (API)" : "STT"}</span>
      </button>

      {/* Thoughts Toggle */}
      <button
        onClick={() => onToggleThoughts(!enableThoughts)}
        disabled={disabled || smolLMMode === "untrained"}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          enableThoughts
            ? "bg-green-500 text-white hover:bg-green-600"
            : "bg-muted text-muted-foreground hover:bg-muted/80"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
        title={smolLMMode === "untrained" ? "Thoughts not available in Untrained mode" : "Gemini Thoughts"}
      >
        <Brain className="h-3.5 w-3.5" />
        <span>Thoughts</span>
      </button>

      {/* SmolLM Toggle */}
      <button
        onClick={() => {
          if (smolLMMode === "none") {
            onSmolLMModeChange("convfill");
          } else if (smolLMMode === "convfill") {
            onSmolLMModeChange("untrained");
          } else {
            onSmolLMModeChange("none");
          }
        }}
        disabled={disabled}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          smolLMMode === "convfill"
            ? "bg-blue-500 text-white hover:bg-blue-600"
            : smolLMMode === "untrained"
            ? "bg-indigo-500 text-white hover:bg-indigo-600"
            : "bg-muted text-muted-foreground hover:bg-muted/80"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
        title={smolLMMode === "convfill" ? "ConvFill" : smolLMMode === "untrained" ? "Untrained SmolLM" : "SmolLM"}
      >
        <MessageSquare className="h-3.5 w-3.5" />
        <span>{smolLMMode === "convfill" ? "ConvFill" : smolLMMode === "untrained" ? "Untrained" : "SmolLM"}</span>
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

      {/* Persona Dropdown */}
      <div className="relative" ref={dropdownRef}>
        <button
          onClick={() => !disabled && setIsPersonaOpen(!isPersonaOpen)}
          disabled={disabled}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors bg-muted text-muted-foreground hover:bg-muted/80 disabled:opacity-50 disabled:cursor-not-allowed min-w-[140px] justify-between"
          title="Select Persona"
        >
          <div className="flex items-center gap-1.5">
            <User className="h-3.5 w-3.5" />
            <span>{selectedPersona?.label}</span>
          </div>
          <ChevronDown className={`h-3.5 w-3.5 transition-transform ${isPersonaOpen ? 'rotate-180' : ''}`} />
        </button>

        {isPersonaOpen && (
          <div className="absolute top-full mt-1 left-0 min-w-[180px] bg-card border border-border rounded-md shadow-lg z-50 py-1">
            {PERSONA_OPTIONS.map((option) => (
              <button
                key={option.value}
                onClick={() => {
                  onPersonaChange(option.value);
                  setIsPersonaOpen(false);
                }}
                className={`w-full text-left px-3 py-2 text-sm transition-colors hover:bg-muted ${
                  persona === option.value ? 'bg-muted/50 font-medium' : ''
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
