"use client";

import React from "react";
import { Clock, Zap, TrendingDown } from "lucide-react";
import { Tooltip } from "./tooltip";

export interface ConversationMetrics {
  timeToFirstResponse: number;
  timeToFirstThought: number;
  averageLatencyReduction: number;
  processTimeline: ProcessSegment[];
  thoughtTimeline: ThoughtSegment[];
}

export interface ProcessSegment {
  type: 'stt' | 'smollm' | 'tts';
  startTime: number;
  endTime: number;
  duration: number;
  label: string;
}

export interface ThoughtSegment {
  startTime: number;
  endTime: number;
  duration: number;
  index: number;
}

const getProcessColor = (type: ProcessSegment['type']) => {
  switch (type) {
    case 'stt':
      return 'bg-purple-500';
    case 'smollm':
      return 'bg-blue-500';
    case 'tts':
      return 'bg-orange-500';
    default:
      return 'bg-gray-500';
  }
};

export function StatsPanel({
  metrics,
  conversationStartTime,
}: {
  metrics: ConversationMetrics | null;
  conversationStartTime: number | null;
}) {
  if (!metrics || !conversationStartTime) {
    return (
      <div className="bg-card border-b px-6 py-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Waiting for conversation...</span>
          </div>
        </div>
      </div>
    );
  }

  // timeline scale
  const totalDuration = Math.max(
    ...metrics.processTimeline.map(seg => seg.endTime - conversationStartTime),
    ...metrics.thoughtTimeline.map(seg => seg.endTime - conversationStartTime)
  );

  const formatTime = (ms: number) => {
    if (ms < 1000) {
      return `${Math.round(ms)}ms`;
    }
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className="bg-card border-b px-6 py-4">
      {/* Key Metrics */}
      <div className="flex items-center gap-8 mb-4">
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-green-500" />
          <span className="text-sm font-medium">First Response:</span>
          <span className="text-sm font-mono text-green-600">
            {formatTime(metrics.timeToFirstResponse)}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-blue-500" />
          <span className="text-sm font-medium">First Thought:</span>
          <span className="text-sm font-mono text-blue-600">
            {formatTime(metrics.timeToFirstThought)}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <TrendingDown className="h-4 w-4 text-purple-500" />
          <span className="text-sm font-medium">Latency Reduction:</span>
          <span className="text-sm font-mono text-purple-600">
            {metrics.averageLatencyReduction > 0 
              ? `${formatTime(metrics.averageLatencyReduction)}`
              : 'N/A'
            }
          </span>
        </div>
      </div>

      {/* Timelines */}
      <div className="space-y-1">
        {/* Process Timeline */}
        <div className="relative h-6 bg-muted rounded overflow-hidden">
          {metrics.processTimeline.map((segment, index) => {
            const left = ((segment.startTime - conversationStartTime) / totalDuration) * 100;
            const width = (segment.duration / totalDuration) * 100;
            const startOffset = segment.startTime - conversationStartTime;
            const endOffset = segment.endTime - conversationStartTime;
            
            const tooltipContent = `${segment.label}
Duration: ${formatTime(segment.duration)}
Start: ${formatTime(startOffset)} from conversation start
End: ${formatTime(endOffset)} from conversation start
Type: ${segment.type.toUpperCase()}`;
            
            return (
              <Tooltip key={`${segment.type}-${index}`} content={tooltipContent} preserveChildPositioning={true}>
                <div
                  className={`absolute top-0 h-full ${getProcessColor(segment.type)} opacity-80 hover:opacity-100 cursor-help transition-opacity`}
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                  }}
                />
              </Tooltip>
            );
          })}
        </div>

        {/* Thought Timeline */}
        {metrics.thoughtTimeline.length > 0 && (
          <div className="relative h-4 bg-muted/50 rounded overflow-hidden">
            {metrics.thoughtTimeline.map((thought, index) => {
              const left = ((thought.startTime - conversationStartTime) / totalDuration) * 100;
              const width = (thought.duration / totalDuration) * 100;
              const startOffset = thought.startTime - conversationStartTime;
              const endOffset = thought.endTime - conversationStartTime;
              
              const tooltipContent = `Thought ${thought.index + 1}
Duration: ${formatTime(thought.duration)}
Start: ${formatTime(startOffset)} from conversation start
End: ${formatTime(endOffset)} from conversation start
API Latency: ${formatTime(thought.duration)}`;
              
              return (
                <Tooltip key={index} content={tooltipContent} preserveChildPositioning={true}>
                  <div
                    className="absolute top-0 h-full bg-green-500 opacity-60 hover:opacity-80 cursor-help transition-opacity"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                    }}
                  />
                </Tooltip>
              );
            })}
          </div>
        )}

        {/* Consolidated Color Key */}
        <div className="flex items-center gap-4 mt-2 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-purple-500 rounded-sm"></div>
            <span>STT</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-500 rounded-sm"></div>
            <span>SmolLM</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-orange-500 rounded-sm"></div>
            <span>TTS</span>
          </div>
          {metrics.thoughtTimeline.length > 0 && (
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-500 rounded-sm"></div>
              <span>Thoughts ({metrics.thoughtTimeline.length})</span>
            </div>
          )}
        </div>
      </div>

      {/* Timeline Scale */}
      <div className="mt-3 text-xs text-muted-foreground">
        <div className="flex justify-between">
          <span>0ms</span>
          <span>{formatTime(totalDuration)}</span>
        </div>
      </div>
    </div>
  );
}