"use client";

import React from "react";
import { Bot, Brain, Volume2, Clock, Zap } from "lucide-react";

export interface TimelineEvent {
  id: string;
  timestamp: number;
  type: "smollm-response" | "openai-thought" | "smollm-enhanced" | "tts-start" | "tts-end" | "model-loading" | "model-ready";
  model: "SmolLM" | "OpenAI" | "TTS";
  message: string;
  content: string;
}

interface TimelineProps {
  events: TimelineEvent[];
  startTime: number | null;
}

export function Timeline({ events, startTime }: TimelineProps) {
  const getRelativeTime = (timestamp: number) => {
    if (!startTime) return "0ms";
    return `${timestamp - startTime}ms`;
  };

  const getEventIcon = (type: TimelineEvent["type"]) => {
    switch (type) {
      case "smollm-response":
        return <Bot className="h-3 w-3 text-blue-500" />;
      case "openai-thought":
        return <Brain className="h-3 w-3 text-green-500" />;
      case "smollm-enhanced":
        return <Bot className="h-3 w-3 text-blue-500" />;
      case "tts-start":
      case "tts-end":
        return <Volume2 className="h-3 w-3 text-orange-500" />;
      case "model-loading":
        return <Clock className="h-3 w-3 text-yellow-500 animate-spin" />;
      case "model-ready":
        return <Zap className="h-3 w-3 text-green-600" />;
      default:
        return <Clock className="h-3 w-3 text-gray-500" />;
    }
  };

  const getEventColor = (type: TimelineEvent["type"]) => {
    switch (type) {
      case "smollm-response":
        return "border-blue-500 bg-blue-50 dark:bg-blue-950";
      case "openai-thought":
        return "border-green-500 bg-green-50 dark:bg-green-950";
      case "smollm-enhanced":
        return "border-blue-500 bg-blue-50 dark:bg-blue-950";
      case "tts-start":
      case "tts-end":
        return "border-orange-500 bg-orange-50 dark:bg-orange-950";
      case "model-loading":
        return "border-yellow-500 bg-yellow-50 dark:bg-yellow-950";
      case "model-ready":
        return "border-green-600 bg-green-50 dark:bg-green-950";
      default:
        return "border-gray-500 bg-gray-50 dark:bg-gray-950";
    }
  };

  if (events.length === 0) {
    return (
      <div className="w-64 border-r bg-background/50 flex flex-col h-[600px]">
        <div className="p-4 border-b">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Timeline
          </h3>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="text-xs text-muted-foreground text-center">
            No events yet
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-64 border-r bg-background/50 flex flex-col h-[600px]">
      <div className="p-4 border-b">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Clock className="h-4 w-4" />
          Timeline
        </h3>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4">
        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-4 top-0 bottom-0 w-px bg-border" />
          
          {/* Events */}
          <div className="space-y-3">
          {events.map((event) => (
            <div key={event.id} className="relative flex items-start gap-3">
              {/* Icon circle */}
              <div className="flex-shrink-0 w-8 h-8 rounded-full border-2 bg-background flex items-center justify-center relative z-10">
                {getEventIcon(event.type)}
              </div>
              
              {/* Event content */}
              <div className={`flex-1 min-w-0 p-2 rounded border-l-2 ${getEventColor(event.type)}`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-medium text-muted-foreground">
                    {getRelativeTime(event.timestamp)}
                  </span>
                  <span className="text-xs font-mono text-muted-foreground">
                    {event.model}
                  </span>
                </div>
                
                <div className="text-xs text-white-foreground">
                  {event.message}
                </div>
                
                {event.content && (
                  <div className="text-xs text-foreground mt-1 bg-background/50 rounded px-1 py-0.5 truncate">
                    &ldquo;{event.content}&rdquo;
                  </div>
                )}
              </div>
            </div>
          ))}
          </div>
        </div>
      </div>
    </div>
  );
}