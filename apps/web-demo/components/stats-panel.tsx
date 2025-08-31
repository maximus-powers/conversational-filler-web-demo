"use client";

import React from "react";
import { Clock, Zap, TrendingDown } from "lucide-react";
import { Tooltip } from "./tooltip";
import { EventData, EventName } from "../app/lib/event-tracker";

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
  eventData,
  conversationStartTime,
}: {
  eventData: EventData | null;
  conversationStartTime: number | null;
}) {
  const metrics = React.useMemo((): ConversationMetrics | null => {
    if (!eventData?.turns?.length || !conversationStartTime) return null;
    
    const allEvents = eventData.turns.flatMap(turn => 
      turn.timeline.map(event => ({
        ...event,
        timestamp: new Date(event.timestamp).getTime()
      }))
    ).sort((a, b) => a.timestamp - b.timestamp);
    
    // calc metrics
    const firstResponse = allEvents.find(e => e.eventName === "LocalLMResponse");
    const firstThought = allEvents.find(e => e.eventName === "ThoughtApiFirstToken");
    const timeToFirstResponse = firstResponse ? firstResponse.timestamp - conversationStartTime : 0;
    const timeToFirstThought = firstThought ? firstThought.timestamp - conversationStartTime : 0;
    
    const processTimeline: ProcessSegment[] = [];
    
    // STT segments
    const sttStarts = allEvents.filter(e => e.eventName === "STTStart");
    const sttEnds = allEvents.filter(e => e.eventName === "STTEnd");
    sttStarts.forEach((start) => {
      const end = sttEnds.find(endEvent => endEvent.timestamp > start.timestamp);
      if (end) {
        processTimeline.push({
          type: 'stt',
          startTime: start.timestamp,
          endTime: end.timestamp,
          duration: end.timestamp - start.timestamp,
          label: 'Speech-to-Text'
        });
        const endIndex = sttEnds.indexOf(end);
        if (endIndex > -1) {
          sttEnds.splice(endIndex, 1);
        }
      }
    });
    
    // smollm segments
    const lmStarts = allEvents.filter(e => e.eventName === "LocalLMSubmit");
    const lmEnds = allEvents.filter(e => e.eventName === "LocalLMResponse");
    const uniqueLmStarts = lmStarts.filter((start, index, arr) => {
      return index === 0 || start.timestamp !== arr[index - 1].timestamp;
    }); // dedupe
    const availableLmEnds = [...lmEnds];
    uniqueLmStarts.forEach((start) => {
      const endIndex = availableLmEnds.findIndex(endEvent => endEvent.timestamp > start.timestamp);
      if (endIndex !== -1) {
        const end = availableLmEnds[endIndex];
        processTimeline.push({
          type: 'smollm',
          startTime: start.timestamp,
          endTime: end.timestamp,
          duration: end.timestamp - start.timestamp,
          label: 'SmolLM Processing'
        });
        availableLmEnds.splice(endIndex, 1);
      }
    });
    
    // TTS segments
    const ttsStarts = allEvents.filter(e => e.eventName === "TTSStart");
    const ttsEnds = allEvents.filter(e => e.eventName === "TTSEnd");
    ttsStarts.forEach((start) => {
      const end = ttsEnds.find(endEvent => endEvent.timestamp > start.timestamp);
      if (end) {
        processTimeline.push({
          type: 'tts',
          startTime: start.timestamp,
          endTime: end.timestamp,
          duration: end.timestamp - start.timestamp,
          label: 'Text-to-Speech'
        });
        const endIndex = ttsEnds.indexOf(end);
        if (endIndex > -1) {
          ttsEnds.splice(endIndex, 1);
        }
      }
    });
    
    // thought segments
    const thoughtTimeline: ThoughtSegment[] = [];
    const thoughtSubmit = allEvents.find(e => e.eventName === "ThoughtApiSubmit");
    const thoughtFirstToken = allEvents.find(e => e.eventName === "ThoughtApiFirstToken");
    if (thoughtSubmit && thoughtFirstToken) {
      thoughtTimeline.push({
        startTime: thoughtSubmit.timestamp,
        endTime: thoughtFirstToken.timestamp,
        duration: thoughtFirstToken.timestamp - thoughtSubmit.timestamp,
        index: 0
      });
    }
    const thoughtsParsed = allEvents.filter(e => e.eventName === "ThoughtParsed");
    let lastThoughtTime = thoughtFirstToken?.timestamp || thoughtSubmit?.timestamp || conversationStartTime;
    thoughtsParsed.forEach((thought, i) => {
      if (lastThoughtTime && thought.timestamp > lastThoughtTime) {
        thoughtTimeline.push({
          startTime: lastThoughtTime,
          endTime: thought.timestamp,
          duration: thought.timestamp - lastThoughtTime,
          index: thoughtFirstToken ? i + 1 : i
        });
      }
      lastThoughtTime = thought.timestamp;
    });
    
    return {
      timeToFirstResponse,
      timeToFirstThought,
      averageLatencyReduction: timeToFirstThought > 0 ? Math.max(0, timeToFirstThought - timeToFirstResponse) : 0,
      processTimeline: processTimeline.sort((a, b) => a.startTime - b.startTime),
      thoughtTimeline: thoughtTimeline.sort((a, b) => a.startTime - b.startTime)
    };
  }, [eventData, conversationStartTime]);
  
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

      {/* Waterfall Trace */}
      <div className="space-y-0">
        <div className="text-xs font-medium text-muted-foreground mb-2">Processing Waterfall</div>
        
        {(() => {
          const allSegments = [
            ...metrics.processTimeline.map(s => ({...s, category: 'process'})),
            ...metrics.thoughtTimeline.map(t => ({
              type: 'thought',
              startTime: t.startTime,
              endTime: t.endTime,
              duration: t.duration,
              label: t.index === 0 ? 'API Latency' : `Thought ${t.index}`,
              category: 'thought',
              index: t.index
            }))
          ].sort((a, b) => a.startTime - b.startTime);

          return allSegments.map((segment, index) => {
            const left = ((segment.startTime - conversationStartTime) / totalDuration) * 100;
            const width = (segment.duration / totalDuration) * 100;
            const startOffset = segment.startTime - conversationStartTime;
            const endOffset = segment.endTime - conversationStartTime;
            
            let colorClass = '';
            if (segment.category === 'thought') {
              if (segment.label === 'API Latency') {
                colorClass = 'bg-red-500';
              } else {
                colorClass = 'bg-green-500';
              }
            } else {
              colorClass = getProcessColor(segment.type as ProcessSegment['type']);
            }
            
            const tooltipContent = `${segment.label}${segment.category === 'process' ? ` (${segment.type.toUpperCase()})` : ''}
Duration: ${formatTime(segment.duration)}`;
            
            return (
              <div key={`waterfall-${index}`} className="flex items-center gap-2">
                <span className="text-xs w-36 text-muted-foreground truncate" title={segment.label}>
                  {segment.label}
                </span>
                <div className="flex-1 relative h-3 bg-muted rounded overflow-hidden">
                  <Tooltip content={tooltipContent} preserveChildPositioning={true}>
                    <div
                      className={`absolute top-0 h-full ${colorClass} opacity-80 hover:opacity-100 cursor-help transition-opacity`}
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                      }}
                    />
                  </Tooltip>
                  {index > 0 && (
                    <div 
                      className="absolute top-1/2 h-0.5 bg-muted-foreground/30"
                      style={{
                        left: '0%',
                        width: `${left}%`,
                        transform: 'translateY(-50%)'
                      }}
                    />
                  )}
                </div>
              </div>
            );
          });
        })()}

        {/* Color Key */}
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