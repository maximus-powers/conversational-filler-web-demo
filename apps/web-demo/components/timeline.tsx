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
  User,
  Play,
  Square,
} from "lucide-react";
import { EventData, TimelineEvent, EventName } from "../app/lib/event-tracker";

interface TimelineDisplayEvent {
  id: string;
  timestamp: number;
  type: EventName;
  message: string;
  content: string;
  fullContent?: string;
}

export function Timeline({
  eventData,
  conversationStartTime,
  mode,
}: {
  eventData: EventData | null;
  conversationStartTime: number | null;
  mode?: "text" | "voice";
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const getEventMessage = (eventName: EventName, event: TimelineEvent): string => {
    switch (eventName) {
      case "VoiceDetectionStart": return "Voice detected";
      case "VoiceDetectionEnd": return "Voice ended";
      case "STTStart": return "Transcribing speech";
      case "STTEnd": return "Speech transcribed";
      case "UserInputReceived": return "User input received";
      case "ThoughtApiSubmit": return "Requesting thoughts";
      case "ThoughtApiFirstToken": return "First thought token";
      case "LocalLMSubmit": return "Processing with model";
      case "LocalLMResponse": return "Model response";
      case "TTSSynthesisStart": return "Synthesizing text chunk";
      case "TTSSynthesisEnd": return "Text chunk synthesized";
      case "AudioPlaybackStart": return "Audio playback started";
      case "AudioPlaybackEnd": return "Audio playback ended";
      case "ThoughtParsed": return "Thought received";
      default: return eventName;
    }
  };
  
  const events: TimelineDisplayEvent[] = React.useMemo(() => {
    if (!eventData?.turns?.length) return [];
    
    const lastTurn = eventData.turns[eventData.turns.length - 1];
    const turnEvents: TimelineDisplayEvent[] = [];
    
    lastTurn.timeline.forEach((event, eventIndex) => {
      const timestamp = new Date(event.timestamp).getTime();
      
      let message = getEventMessage(event.eventName, event);
      let content = event.text || event.prompt || event.response || "";
      
      if (event.eventName === "TTSSynthesisStart" || event.eventName === "TTSSynthesisEnd") {
        if (typeof event.responseIndex === 'number') {
          message = `${message} (Response #${event.responseIndex})`;
        }
        if (event.synthesisId) {
          message = `${message} [${event.synthesisId}]`;
        }
      }
      
      turnEvents.push({
        id: `${eventData.turns.length - 1}-${eventIndex}`,
        timestamp,
        type: event.eventName,
        message,
        content,
        fullContent: content
      });
    });
    
    return turnEvents.sort((a, b) => a.timestamp - b.timestamp);
  }, [eventData]);
  
  const getRelativeTime = (timestamp: number) => {
    if (!eventData?.turns?.length) return "0ms";
    
    // Use the start time of the current turn instead of conversation start
    const lastTurn = eventData.turns[eventData.turns.length - 1];
    const turnStartTime = new Date(lastTurn.timeline[0]?.timestamp || 0).getTime();
    
    return `${timestamp - turnStartTime}ms`;
  };

  const getEventIcon = (type: EventName) => {
    switch (type) {
      case "VoiceDetectionStart":
        return <Mic className="h-3 w-3 text-red-500" />;
      case "VoiceDetectionEnd":
        return <MicOff className="h-3 w-3 text-gray-500" />;
      case "STTStart":
      case "STTEnd":
        return <Brain className="h-3 w-3 text-purple-500" />;
      case "UserInputReceived":
        return <User className="h-3 w-3 text-cyan-500" />;
      case "ThoughtApiSubmit":
      case "ThoughtApiFirstToken":
      case "ThoughtParsed":
        return <Brain className="h-3 w-3 text-green-500" />;
      case "LocalLMSubmit":
      case "LocalLMResponse":
        return <Bot className="h-3 w-3 text-blue-500" />;
      case "TTSSynthesisStart":
      case "TTSSynthesisEnd":
        return <Volume2 className="h-3 w-3 text-yellow-500" />;
      case "AudioPlaybackStart":
        return <Play className="h-3 w-3 text-orange-600" />;
      case "AudioPlaybackEnd":
        return <Square className="h-3 w-3 text-orange-600" />;
      default:
        return <Clock className="h-3 w-3 text-gray-500" />;
    }
  };

  const getEventColor = (type: EventName) => {
    switch (type) {
      case "VoiceDetectionStart":
        return "border-red-500 bg-red-50 dark:bg-red-950";
      case "VoiceDetectionEnd":
        return "border-gray-500 bg-gray-50 dark:bg-gray-950";
      case "STTStart":
      case "STTEnd":
        return "border-purple-500 bg-purple-50 dark:bg-purple-950";
      case "UserInputReceived":
        return "border-cyan-500 bg-cyan-50 dark:bg-cyan-950";
      case "ThoughtApiSubmit":
      case "ThoughtApiFirstToken":
      case "ThoughtParsed":
        return "border-green-500 bg-green-50 dark:bg-green-950";
      case "LocalLMSubmit":
      case "LocalLMResponse":
        return "border-blue-500 bg-blue-50 dark:bg-blue-950";
      case "TTSSynthesisStart":
      case "TTSSynthesisEnd":
        return "border-yellow-500 bg-yellow-50 dark:bg-yellow-950";
      case "AudioPlaybackStart":
      case "AudioPlaybackEnd":
        return "border-orange-600 bg-orange-100 dark:bg-orange-900";
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
                      {event.type}
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
