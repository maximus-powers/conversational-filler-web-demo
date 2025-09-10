"use client";

import React from "react";
import { Clock, Zap, TrendingDown } from "lucide-react";
import { Tooltip } from "./tooltip";
import { EventData, EventName } from "../app/lib/event-tracker";

export interface ConversationMetrics {
  timeToFirstResponse: number;
  timeToFirstThought: number;
  latencyReduction: number;
  processTimeline: ProcessSegment[];
  thoughtTimeline: ThoughtSegment[];
}

export interface ProcessSegment {
  type: 'stt' | 'smollm' | 'tts' | 'audio_playback';
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
      return 'bg-orange-300'; // Lighter orange for synthesis
    case 'audio_playback':
      return 'bg-orange-600'; // Darker orange for actual playback
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

    // only last turn gets displayed in waterfall
    const lastTurn = eventData.turns[eventData.turns.length - 1];
    const allEvents = lastTurn.timeline.map(event => ({
      ...event,
      timestamp: new Date(event.timestamp).getTime()
    })).sort((a, b) => a.timestamp - b.timestamp);
    
    // calc metrics
    const turnStartTime = allEvents.length > 0 ? allEvents[0].timestamp : conversationStartTime;
    const firstResponse = allEvents.find(e => e.eventName === "LocalLMResponse");
    const firstThought = allEvents.find(e => e.eventName === "ThoughtApiFirstToken");
    const timeToFirstResponse = firstResponse ? firstResponse.timestamp - turnStartTime : 0;
    const timeToFirstThought = firstThought ? firstThought.timestamp - turnStartTime : 0;
    
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
    
    // TTS synthesis segments (keep for debugging)
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
          label: 'TTS Synthesis'
        });
        const endIndex = ttsEnds.indexOf(end);
        if (endIndex > -1) {
          ttsEnds.splice(endIndex, 1);
        }
      }
    });
    
    // audio playing segment
    const playbackStarts = allEvents.filter(e => e.eventName === "AudioPlaybackStart");
    const playbackEnds = allEvents.filter(e => e.eventName === "AudioPlaybackEnd");
    playbackStarts.forEach((start) => {
      const end = playbackEnds.find(endEvent => endEvent.timestamp > start.timestamp);
      if (end) {
        processTimeline.push({
          type: 'audio_playback',
          startTime: start.timestamp,
          endTime: end.timestamp,
          duration: end.timestamp - start.timestamp,
          label: 'Audio Playback'
        });
        const endIndex = playbackEnds.indexOf(end);
        if (endIndex > -1) {
          playbackEnds.splice(endIndex, 1);
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
    
    const firstThoughtParsed = allEvents.find(e => e.eventName === "ThoughtParsed");
    const firstSmolLMEnd = allEvents.find(e => e.eventName === "LocalLMResponse");
    const latencyReduction = firstThoughtParsed && firstSmolLMEnd 
      ? Math.max(0, firstThoughtParsed.timestamp - firstSmolLMEnd.timestamp)
      : 0;

    return {
      timeToFirstResponse,
      timeToFirstThought,
      latencyReduction: latencyReduction,
      processTimeline: processTimeline.sort((a, b) => a.startTime - b.startTime),
      thoughtTimeline: thoughtTimeline.sort((a, b) => a.startTime - b.startTime)
    };
  }, [eventData, conversationStartTime]);
  
  if (!metrics || !conversationStartTime) {
    return null;
  }

  // timeline scale - use last turn's timeframe instead of full conversation
  const lastTurn = eventData?.turns[eventData.turns.length - 1];
  const lastTurnStartTime = lastTurn ? new Date(lastTurn.timeline[0]?.timestamp || 0).getTime() : conversationStartTime;
  const totalDuration = Math.max(
    ...metrics.processTimeline.map(seg => seg.endTime - lastTurnStartTime),
    ...metrics.thoughtTimeline.map(seg => seg.endTime - lastTurnStartTime)
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
            {metrics.latencyReduction > 0 
              ? `${formatTime(metrics.latencyReduction)}`
              : 'N/A'
            }
          </span>
        </div>
      </div>

      {/* Waterfall Trace */}
      <div className="space-y-0">
        <div className="flex items-center gap-2 mb-2">
          <div className="text-xs font-medium text-muted-foreground w-36">Process Trace</div>
          <div className="flex-1 relative text-xs text-muted-foreground">
            <div className="flex justify-between">
              <span>0ms</span>
              <span>{formatTime(totalDuration)}</span>
            </div>
            {/* Half-second increment marks */}
            {(() => {
              const marks = [];
              const durationMs = totalDuration;
              const halfSecondIntervals = Math.ceil(durationMs / 500);
              
              for (let i = 1; i < halfSecondIntervals; i++) {
                const intervalMs = i * 500; // Every 0.5 seconds
                if (intervalMs < durationMs) {
                  const position = (intervalMs / durationMs) * 100;
                  marks.push(
                    <div
                      key={i}
                      className="absolute top-0 flex flex-col items-center"
                      style={{ left: `${position}%`, transform: 'translateX(-50%)' }}
                    >
                      <span className="text-xs text-muted-foreground mt-0.5">{i/2}s</span>
                    </div>
                  );
                }
              }
              return marks;
            })()}
          </div>
        </div>
        
        {(() => {
          if (!eventData?.turns?.length) return null;
          const lastTurn = eventData.turns[eventData.turns.length - 1];
          const waterfallEvents = lastTurn.timeline.map(event => ({
            ...event,
            timestamp: new Date(event.timestamp).getTime()
          }));
          
          const turnStartTime = new Date(lastTurn.timeline[0]?.timestamp || 0).getTime();
          
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

          const firstSmolLMSegment = allSegments.find(s => s.category === 'process' && s.type === 'smollm');

          return allSegments.map((segment, index) => {
            const left = ((segment.startTime - turnStartTime) / totalDuration) * 100;
            const width = (segment.duration / totalDuration) * 100;
            
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
            
            let tooltipContent = `${segment.label}${segment.category === 'process' ? ` (${segment.type.toUpperCase()})` : ''} \nDuration: ${formatTime(segment.duration)}`;
            let relevantEvent = null;
            
            if (segment.category === 'process') {
              if (segment.type === 'smollm') {
                const submitEvent = waterfallEvents.find((e) => 
                  e.eventName === 'LocalLMSubmit' && Math.abs(e.timestamp - segment.startTime) < 50
                );
                const responseEvent = waterfallEvents.find((e) => 
                  e.eventName === 'LocalLMResponse' && Math.abs(e.timestamp - segment.endTime) < 50
                );
                relevantEvent = submitEvent || responseEvent;
              } else if (segment.type === 'stt') {
                relevantEvent = waterfallEvents.find((e) => 
                  (e.eventName === 'STTEnd') && Math.abs(e.timestamp - segment.endTime) < 50
                );
              } else if (segment.type === 'tts') {
                relevantEvent = waterfallEvents.find((e) => 
                  (e.eventName === 'TTSStart' || e.eventName === 'TTSEnd') && 
                  e.timestamp >= segment.startTime && e.timestamp <= segment.endTime
                );
              } else if (segment.type === 'audio_playback') {
                relevantEvent = waterfallEvents.find((e) => 
                  (e.eventName === 'AudioPlaybackStart' || e.eventName === 'AudioPlaybackEnd') && 
                  e.timestamp >= segment.startTime && e.timestamp <= segment.endTime
                );
              }
            } else {
              if (segment.label === 'API Latency') {
                relevantEvent = waterfallEvents.find((e) => e.eventName === 'ThoughtApiSubmit');
              } else {
                relevantEvent = waterfallEvents.find((e) => 
                  e.eventName === 'ThoughtParsed' && Math.abs(e.timestamp - segment.endTime) < 50
                );
              }
            }
            
            if (segment.category === 'process' && segment.type === 'smollm') {
              const submitEvent = waterfallEvents.find((e) => 
                e.eventName === 'LocalLMSubmit' && Math.abs(e.timestamp - segment.startTime) < 50
              );
              const responseEvent = waterfallEvents.find((e) => 
                e.eventName === 'LocalLMResponse' && Math.abs(e.timestamp - segment.endTime) < 50
              );
              
              if (submitEvent?.prompt) {
                tooltipContent += `\nPrompt: ${submitEvent.prompt}`;
              }
              if (responseEvent?.response) {
                tooltipContent += `\nResponse: ${responseEvent.response}`;
              }
            } else if (relevantEvent) {
              const content = relevantEvent.text || relevantEvent.prompt || relevantEvent.response;
              if (content) {
                tooltipContent += `\nContent: ${content}`;
              }
            }
            
            const isFirstSmolLM = segment === firstSmolLMSegment;
            
            return (
              <div key={`waterfall-${index}`} className="flex items-center gap-2">
                <span className="text-xs w-36 text-muted-foreground truncate" title={segment.label}>
                  {segment.label}
                </span>
                <div className="flex-1 relative h-3 rounded overflow-hidden">
                  {(() => {
                    const gridLines = [];
                    const durationMs = totalDuration;
                    const halfSecondIntervals = Math.ceil(durationMs / 500); 
                    
                    for (let i = 1; i < halfSecondIntervals; i++) {
                      const intervalMs = i * 500;
                      if (intervalMs < durationMs) {
                        const position = (intervalMs / durationMs) * 100;
                        gridLines.push(
                          <div
                            key={`row-grid-${index}-${i}`}
                            className="absolute top-0 bottom-0 w-px bg-muted-foreground opacity-20 pointer-events-none"
                            style={{ left: `${position}%` }}
                          />
                        );
                      }
                    }
                    return gridLines;
                  })()}
                  
                  <Tooltip content={tooltipContent} preserveChildPositioning={true}>
                    <div
                      className={`absolute top-0 h-full ${colorClass} opacity-80 hover:opacity-100 transition-opacity rounded`}
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                      }}
                    />
                  </Tooltip>
                  
                  {/* Latency reduction line */}
                  {isFirstSmolLM && firstSmolLMSegment && (() => {
                    const firstThoughtParsed = waterfallEvents.find(e => e.eventName === "ThoughtParsed");
                    if (!firstThoughtParsed) return null;
                    const reductionMs = firstThoughtParsed.timestamp - firstSmolLMSegment.endTime;
                    const lineWidth = ((firstThoughtParsed.timestamp - turnStartTime) / totalDuration) * 100 - (left + width);
                    
                    return (
                      <div
                        className="absolute top-1/2 flex items-center px-1"
                        style={{
                          left: `${left + width}%`,
                          width: `${lineWidth}%`,
                          transform: 'translateY(-50%)',
                          minWidth: '80px'
                        }}
                      >
                        <div className="flex-1 h-0.5 bg-muted relative flex items-center justify-center">
                          <span className="text-xs text-muted-foreground bg-background px-1 whitespace-nowrap">
                            Reduction - {Math.round(reductionMs)}ms
                          </span>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              </div>
            );
          });
        })()}
      </div>

    </div>
  );
}