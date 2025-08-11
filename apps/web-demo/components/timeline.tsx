"use client";

import React, { useState } from "react";
import {
  Bot,
  Brain,
  Volume2,
  Clock,
  DownloadCloud,
  CheckCircle,
  Maximize2,
  Minimize2,
  Mic,
  MicOff,
} from "lucide-react";

export interface TimelineEvent {
  id: string;
  timestamp: number;
  type: string;
  model: string;
  message: string;
  content: string;
  fullContent?: string;
}

export function Timeline({
  events,
  conversationStartTime,
  mode,
}: {
  events: TimelineEvent[];
  conversationStartTime: number | null;
  mode?: "text" | "voice";
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const getRelativeTime = (timestamp: number) => {
    if (!conversationStartTime) return "0ms";
    return `${timestamp - conversationStartTime}ms`;
  };

  const getEventIcon = (type: TimelineEvent["type"]) => {
    switch (type) {
      case "smollm-response":
        return <Bot className="h-3 w-3 text-blue-500" />;
      case "thought":
        return <Brain className="h-3 w-3 text-green-500" />;
      case "smollm-enhanced":
        return <Bot className="h-3 w-3 text-blue-500" />;
      case "tts-start":
      case "tts-end":
        return <Volume2 className="h-3 w-3 text-orange-500" />;
      case "model-loading":
        return <DownloadCloud className="h-3 w-3 text-yellow-500" />;
      case "model-ready":
        return <CheckCircle className="h-3 w-3 text-green-600" />;
      case "transcription":
        return <Brain className="h-3 w-3 text-purple-500" />;
      case "recording-start":
        return <Mic className="h-3 w-3 text-red-500" />;
      case "recording-end":
        return <MicOff className="h-3 w-3 text-gray-500" />;
      case "mode-switch":
        return <Clock className="h-3 w-3 text-indigo-500" />;
      case "user-input":
        return <Brain className="h-3 w-3 text-cyan-500" />;
      case "error":
        return <Clock className="h-3 w-3 text-red-600" />;
      default:
        return <Clock className="h-3 w-3 text-gray-500" />;
    }
  };

  const getEventColor = (type: TimelineEvent["type"]) => {
    switch (type) {
      case "smollm-response":
        return "border-blue-500 bg-blue-50 dark:bg-blue-950";
      case "thought":
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
      case "transcription":
        return "border-purple-500 bg-purple-50 dark:bg-purple-950";
      case "recording-start":
        return "border-red-500 bg-red-50 dark:bg-red-950";
      case "recording-end":
        return "border-gray-500 bg-gray-50 dark:bg-gray-950";
      case "mode-switch":
        return "border-indigo-500 bg-indigo-50 dark:bg-indigo-950";
      case "user-input":
        return "border-cyan-500 bg-cyan-50 dark:bg-cyan-950";
      case "error":
        return "border-red-600 bg-red-50 dark:bg-red-950";
      default:
        return "border-gray-500 bg-gray-50 dark:bg-gray-950";
    }
  };

  if (events.length === 0) {
    return (
      <div
        className={`${isExpanded ? "absolute inset-0 z-50 bg-background border shadow-lg" : "w-64 border-r bg-background/50"} flex flex-col h-full`}
      >
        <div className="p-4 border-b">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Timeline
            </h3>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 hover:bg-muted rounded"
              title={isExpanded ? "Collapse timeline" : "Expand timeline"}
            >
              {isExpanded ? (
                <Minimize2 className="h-3 w-3" />
              ) : (
                <Maximize2 className="h-3 w-3" />
              )}
            </button>
          </div>
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
    <div
      className={`${isExpanded ? "absolute inset-0 z-50 bg-background border shadow-lg" : "w-64 border-r bg-background/50"} flex flex-col h-full`}
    >
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Timeline
          </h3>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 hover:bg-muted rounded"
            title={isExpanded ? "Collapse timeline" : "Expand timeline"}
          >
            {isExpanded ? (
              <Minimize2 className="h-3 w-3" />
            ) : (
              <Maximize2 className="h-3 w-3" />
            )}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="relative">
          <div
            className={`absolute left-4 top-0 bottom-0 w-px bg-border ${isExpanded ? "hidden" : ""}`}
          />

          <div className={`${isExpanded ? "space-y-6" : "space-y-3"}`}>
            {events.map((event) => (
              <div
                key={event.id}
                className={`relative flex items-start ${isExpanded ? "gap-4" : "gap-3"}`}
              >
                <div
                  className={`flex-shrink-0 ${isExpanded ? "w-10 h-10" : "w-8 h-8"} rounded-full border-2 bg-background flex items-center justify-center relative z-10`}
                >
                  {getEventIcon(event.type)}
                </div>
                <div
                  className={`flex-1 min-w-0 ${isExpanded ? "p-4" : "p-2"} rounded border-l-2 ${getEventColor(event.type)}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span
                      className={`${isExpanded ? "text-sm" : "text-xs"} font-medium text-muted-foreground`}
                    >
                      {getRelativeTime(event.timestamp)}
                    </span>
                    <span
                      className={`${isExpanded ? "text-sm" : "text-xs"} font-mono text-muted-foreground`}
                    >
                      {event.model}
                    </span>
                  </div>

                  <div
                    className={`${isExpanded ? "text-sm" : "text-xs"} text-white-foreground`}
                  >
                    {event.message}
                  </div>

                  {event.content && (
                    <div
                      className={`${isExpanded ? "text-sm" : "text-xs"} text-foreground mt-1 bg-background/50 rounded ${isExpanded ? "px-3 py-2" : "px-1 py-0.5"} ${isExpanded ? "whitespace-pre-wrap" : "truncate"}`}
                    >
                      &ldquo;
                      {isExpanded && event.fullContent
                        ? event.fullContent
                        : event.content}
                      &rdquo;
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
