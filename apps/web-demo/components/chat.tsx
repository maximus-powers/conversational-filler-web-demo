"use client";

import { Button } from "@convo-filler/ui/components/button";
import { useState, useRef, useEffect } from "react";
import { Bot, User, Loader2, Send, Mic, MicOff } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { UnifiedPipeline, AppMode } from "../app/lib/unified-pipeline";
import { Timeline, TimelineEvent } from "./timeline";
import { ModeSwitcher } from "./mode-switcher";
import { StatsPanel, ConversationMetrics, ProcessSegment, ThoughtSegment } from "./stats-panel";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  processedContent?: string;
  thoughts?: string[];
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelLoadingProgress, setModelLoadingProgress] = useState<string>("");
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
  const [conversationStartTime, setConversationStartTime] = useState<
    number | null
  >(null);
  const [mode, setMode] = useState<AppMode>("text");
  const [isListening, setIsListening] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState<string>("af_heart");
  const [availableVoices, setAvailableVoices] = useState<Record<string, any>>(
    {},
  );
  const [thoughtProvider, setThoughtProvider] = useState<"gemini" | "none">("gemini");
  const [selectedModel, setSelectedModel] = useState<"maximuspowers/smollm-convo-filler-onnx-official" | "HuggingFaceTB/SmolLM-360M-Instruct">("maximuspowers/smollm-convo-filler-onnx-official");
  const [conversationMetrics, setConversationMetrics] = useState<ConversationMetrics | null>(null);
  const [currentTurnStartTime, setCurrentTurnStartTime] = useState<number | null>(null);
  const pipelineRef = useRef<UnifiedPipeline | null>(null);
  const messagesRef = useRef<Map<string, Message>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentMetricsRef = useRef<{
    startTime: number | null;
    firstResponseTime: number | null;
    firstThoughtTime: number | null;
    processSegments: ProcessSegment[];
    thoughtSegments: ThoughtSegment[];
    thoughtGenerationStart: number | null;
    thoughtIndex: number;
    activeSmolLMSegments: Map<string, ProcessSegment>;
  }>({
    startTime: null,
    firstResponseTime: null,
    firstThoughtTime: null,
    processSegments: [],
    thoughtSegments: [],
    thoughtGenerationStart: null,
    thoughtIndex: 0,
    activeSmolLMSegments: new Map(),
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const updateMetricsDisplay = () => {
    const metrics = currentMetricsRef.current;
    if (metrics.startTime && metrics.firstResponseTime) {
      const timeToFirstResponse = metrics.firstResponseTime - metrics.startTime;
      const timeToFirstThought = metrics.firstThoughtTime ? metrics.firstThoughtTime - metrics.startTime : 0;
      
      const newMetrics = {
        timeToFirstResponse,
        timeToFirstThought,
        averageLatencyReduction: timeToFirstThought > 0 ? Math.max(0, timeToFirstThought - timeToFirstResponse) : 0,
        processTimeline: [...metrics.processSegments],
        thoughtTimeline: [...metrics.thoughtSegments]
      };
      
      setConversationMetrics(newMetrics);
    }
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addTimelineEvent = (
    type: TimelineEvent["type"],
    model: TimelineEvent["model"],
    message: string,
    content: string | any = "",
    eventTimestamp?: number,
    thoughtData?: any
  ) => {
    const contentStr = typeof content === 'string' ? content : '';
    const event: TimelineEvent = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: eventTimestamp || Date.now(),
      type,
      model,
      message,
      content: contentStr.slice(0, 50) + (contentStr.length > 50 ? "..." : ""),
      fullContent: contentStr,
      thoughtData: thoughtData
    };
    setTimelineEvents((prev) => [...prev, event]);
  };

  const clearTimeline = () => {
    setTimelineEvents([]);
    setConversationStartTime(null);
  };

  // init pipeline
  useEffect(() => {
    const initializePipeline = async () => {
      const initStartTime = Date.now();
      setConversationStartTime(initStartTime);

      setModelLoadingProgress("Loading models...");
      addTimelineEvent(
        "model-loading",
        "Pipeline",
        "Initializing chat pipeline",
        "",
      );

      pipelineRef.current = new UnifiedPipeline({
        onMessageReceived: (role, content, messageId) => {
          const message: Message = {
            id: messageId || Date.now().toString(),
            role,
            content,
            processedContent: content,
          };
          if (messageId) {
            messagesRef.current.set(messageId, message);
          }
          setMessages((prev) => {
            const existing = prev.find((m) => m.id === message.id);
            if (existing) {
              return prev.map((m) => (m.id === message.id ? message : m));
            }
            return [...prev, message];
          });
        },

        onMessageUpdated: (messageId, newContent) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id === messageId) {
                const currentContent = msg.processedContent || msg.content;
                return {
                  ...msg,
                  processedContent: currentContent + " " + newContent,
                };
              }
              return msg;
            }),
          );
        },

        onTranscriptionReceived: (text) => {
          addTimelineEvent(
            "transcription",
            "Whisper",
            "Transcribed speech",
            text,
          );
        },

        onStatusChange: (status, message) => {
          setModelLoadingProgress(message);
          if (status === "ready") {
            setModelLoading(false);
            setModelLoadingProgress("");
          } else if (status === "recording_start") {
            setIsListening(true);
          } else if (status === "recording_end") {
            setIsListening(false);
          }
        },

        onTimelineEvent: (type, model, message, content, eventTimestamp, thoughtData) => {
          addTimelineEvent(
            type as TimelineEvent["type"],
            model as TimelineEvent["model"],
            message,
            content || "",
            eventTimestamp,
            thoughtData
          );
          
          const now = Date.now();
          const metrics = currentMetricsRef.current;
          
          if (type === "thought_received" && thoughtData) {
            const thoughtIndex = thoughtData.thoughtIndex;
            const thoughtReceivedTime = thoughtData.turnStartTime + thoughtData.turnOffset;
            
            let segmentStartTime;
            let actualDuration;
            
            if (thoughtIndex === 0) {
              // first thought: submit api -> parsed thought 1
              segmentStartTime = thoughtData.apiRequestStartTime;
              actualDuration = thoughtReceivedTime - thoughtData.apiRequestStartTime;
            } else {
              const previousThought = metrics.thoughtSegments.find(t => t.index === thoughtIndex - 1);
              if (previousThought) {
                segmentStartTime = previousThought.endTime;
                actualDuration = thoughtReceivedTime - segmentStartTime;
                console.log(`Thought ${thoughtIndex}: Previous ended at ${previousThought.endTime}, current at ${thoughtReceivedTime}, duration: ${actualDuration}ms`);
              } else {
                segmentStartTime = thoughtData.apiRequestStartTime;
                actualDuration = thoughtReceivedTime - segmentStartTime; // subsequent thoughts: last parse thought time -> next parsed thought time
                console.log(`Thought ${thoughtIndex}: No previous thought found, using API start time`);
              }
            }
            
            if (actualDuration <= 0) {
              console.warn(`Thought ${thoughtIndex} has invalid duration: ${actualDuration}ms, setting to 10ms`);
              actualDuration = 10;
            }
            
            const newSegment = {
              startTime: segmentStartTime,
              endTime: thoughtReceivedTime,
              duration: actualDuration,
              index: thoughtIndex
            };

            metrics.thoughtSegments.push(newSegment);
            updateMetricsDisplay();
            return; 
          }
          
          
          switch (type) {
            case "conversation-turn":
              const eventData = content as any;
              const turnStartTime = eventData?.turnStartTime || now;
              metrics.startTime = turnStartTime;
              metrics.firstResponseTime = null;
              metrics.firstThoughtTime = null;
              metrics.processSegments = [];
              metrics.thoughtSegments = [];
              metrics.thoughtGenerationStart = null;
              metrics.thoughtIndex = 0;
              metrics.activeSmolLMSegments.clear();
              setCurrentTurnStartTime(turnStartTime);
              setConversationMetrics(null);
              break;
              
            case "transcription":
              if (content && (content as any).startTime && (content as any).duration && (content as any).turnStartTime) {
                const data = content as any;
                const turnRelativeStart = data.turnStartTime + data.turnOffset;
                metrics.processSegments.push({
                  type: 'stt',
                  startTime: turnRelativeStart,
                  endTime: turnRelativeStart + data.duration,
                  duration: data.duration,
                  label: 'Speech-to-Text'
                });
                updateMetricsDisplay();
              }
              break;
              
            case "first_response":
              if (content && (content as any).turnStartTime && (content as any).turnOffset) {
                const data = content as any;
                metrics.firstResponseTime = data.turnStartTime + data.turnOffset;
                
                const fillerResponseTime = data.turnStartTime + data.turnOffset;
                const fillerSegment = {
                  type: 'smollm' as const,
                  startTime: metrics.startTime || fillerResponseTime, 
                  endTime: fillerResponseTime,
                  duration: fillerResponseTime - (metrics.startTime || fillerResponseTime),
                  label: 'SmolLM: Immediate Response'
                };
                if (fillerSegment.duration > 0) {
                  metrics.processSegments.push(fillerSegment);
                }
              } else {
                metrics.firstResponseTime = now;
              }
              updateMetricsDisplay();
              break;
              
            case "first_thought":
              if (content && (content as any).turnStartTime && (content as any).turnOffset) {
                const data = content as any;
                metrics.firstThoughtTime = data.turnStartTime + data.turnOffset;
              } else {
                metrics.firstThoughtTime = now;
              }
              updateMetricsDisplay();
              break;
              
            case "tts-start":
              const existingTTS = metrics.processSegments.find(seg => seg.type === 'tts');
              if (!existingTTS && content && (content as any).turnStartTime && (content as any).turnOffset) {
                const data = content as any;
                const ttsStartTime = data.turnStartTime + data.turnOffset;
                metrics.processSegments.push({
                  type: 'tts',
                  startTime: ttsStartTime,
                  endTime: ttsStartTime,
                  duration: 0,
                  label: 'Text-to-Speech'
                });
                updateMetricsDisplay();
              }
              break;
              
            case "tts-end":
              const ttsSegment = metrics.processSegments.find(seg => seg.type === 'tts');
              if (ttsSegment && content && (content as any).duration && (content as any).turnStartTime && (content as any).turnOffset) {
                const data = content as any;
                ttsSegment.endTime = data.turnStartTime + data.turnOffset;
                ttsSegment.duration = data.duration;
                updateMetricsDisplay();
              }
              break;
              
            case "thought_generation_start":
              if (content && (content as any).turnStartTime && (content as any).turnOffset) {
                const data = content as any;
                metrics.thoughtGenerationStart = data.turnStartTime + data.turnOffset;
              } else {
                metrics.thoughtGenerationStart = now;
              }
              break;
              
            case "thought_generation_end":
              updateMetricsDisplay();
              break;
              
              
            case "thought_processing_start":
              if (content && (content as any).thought && (content as any).turnStartTime && (content as any).turnOffset) {
                const data = content as any;
                const processingStartTime = data.turnStartTime + data.turnOffset;
                const thoughtKey = data.thought + '_' + data.timestamp;
                const segment: ProcessSegment = {
                  type: 'smollm',
                  startTime: processingStartTime,
                  endTime: processingStartTime,
                  duration: 0,
                  label: `SmolLM: ${data.thought.slice(0, 20)}...`
                };
                metrics.activeSmolLMSegments.set(thoughtKey, segment);
                updateMetricsDisplay();
              }
              break;
              
            case "thought_processing_end":
              if (content && (content as any).thought && (content as any).startTime && (content as any).turnStartTime && (content as any).turnOffset) {
                const data = content as any;
                const thoughtKey = data.thought + '_' + data.startTime;
                const segment = metrics.activeSmolLMSegments.get(thoughtKey);
                if (segment) {
                  segment.endTime = data.turnStartTime + data.turnOffset;
                  segment.duration = data.duration;
                  metrics.processSegments.push(segment);
                  metrics.activeSmolLMSegments.delete(thoughtKey);
                  updateMetricsDisplay();
                }
              }
              break;
          }
        },
      });

      try {
        await pipelineRef.current.initialize(mode);
        const voices = pipelineRef.current.getVoices();
        setAvailableVoices(voices);

        const initEndTime = Date.now();
        const loadTime = ((initEndTime - initStartTime) / 1000).toFixed(2);
        addTimelineEvent(
          "model-ready",
          "Pipeline",
          `Models loaded in ${loadTime}s`,
          "",
        );
      } catch (error) {
        console.error("Failed to initialize pipeline:", error);
        setModelLoadingProgress("Failed to load models");
        addTimelineEvent(
          "error",
          "Pipeline",
          "Failed to initialize",
          error?.toString() || "",
        );
      }
    };

    initializePipeline();

    return () => {
      if (pipelineRef.current) {
        pipelineRef.current.dispose();
      }
    };
  }, []);

  const handleModeChange = async (newMode: AppMode) => {
    if (!pipelineRef.current || newMode === mode) return;

    setModelLoading(true);
    setModelLoadingProgress(`Switching to ${newMode} mode...`);

    try {
      await pipelineRef.current.switchMode(newMode);
      setMode(newMode);
      const voices = pipelineRef.current.getVoices();
      setAvailableVoices(voices);
      addTimelineEvent(
        "mode-switch",
        "Pipeline",
        `Switched to ${newMode} mode`,
        "",
      );
    } catch (error) {
      console.error("Failed to switch mode:", error);
      addTimelineEvent(
        "error",
        "Pipeline",
        "Failed to switch mode",
        error?.toString() || "",
      );
    } finally {
      setModelLoading(false);
      setModelLoadingProgress("");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || modelLoading || !pipelineRef.current)
      return;

    const currentInput = input;
    setInput("");
    setIsLoading(true);

    // clear timeline
    if (messages.length === 0) {
      setTimelineEvents([]);
      setConversationStartTime(Date.now());
    }

    try {
      await pipelineRef.current.processText(currentInput);
    } catch (error) {
      console.error("Text processing error:", error);
      addTimelineEvent(
        "error",
        "System",
        "Processing failed",
        error?.toString() || "",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    messagesRef.current.clear();
    clearTimeline();
    setIsLoading(false);
  };

  useEffect(() => {
    if (selectedVoice && pipelineRef.current) {
      pipelineRef.current.setVoice(selectedVoice);
    }
  }, [selectedVoice]);

  useEffect(() => {
    if (pipelineRef.current) {
      pipelineRef.current.setThoughtProvider(thoughtProvider);
    }
  }, [thoughtProvider]);

  useEffect(() => {
    if (selectedModel && pipelineRef.current) {
      pipelineRef.current.setModel(selectedModel);
    }
  }, [selectedModel]);

  return (
    <div className="flex h-full w-full overflow-hidden">
      <Timeline
        events={timelineEvents}
        conversationStartTime={conversationStartTime}
        mode={mode}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        <div className="bg-card border-b px-6 py-3 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ModeSwitcher
                currentMode={mode}
                onModeChange={handleModeChange}
                disabled={modelLoading || isLoading}
              />

              {mode === "voice" && Object.keys(availableVoices).length > 0 && (
                <select
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  className="text-sm px-2 py-1 border rounded-md bg-background"
                  disabled={modelLoading}
                >
                  {Object.entries(availableVoices).map(
                    ([id, voice]: [string, any]) => (
                      <option key={id} value={id}>
                        {voice.name || id}
                      </option>
                    ),
                  )}
                </select>
              )}

              <select
                value={thoughtProvider}
                onChange={(e) => setThoughtProvider(e.target.value as "gemini" | "none")}
                className="text-sm px-2 py-1 border rounded-md bg-background"
                disabled={modelLoading}
                title="Select thought provider"
              >
                <option value="gemini">Google (Gemini)</option>
                <option value="none">None</option>
              </select>

              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as "maximuspowers/smollm-convo-filler-onnx-official" | "HuggingFaceTB/SmolLM-360M-Instruct")}
                className="text-sm px-2 py-1 border rounded-md bg-background"
                disabled={modelLoading}
                title="Select SmolLM model"
              >
                <option value="maximuspowers/smollm-convo-filler-onnx-official">SmolLM Convo Filler</option>
                <option value="HuggingFaceTB/SmolLM-360M-Instruct">SmolLM 360M Instruct</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <Button
                onClick={clearChat}
                variant="outline"
                size="sm"
                disabled={messages.length === 0}
              >
                Clear Chat
              </Button>

              <ThemeToggle />
            </div>
          </div>

          {/* Status Bar */}
          <div className="mt-2 text-sm text-muted-foreground flex items-center gap-2">
            {modelLoading ? (
              modelLoadingProgress || "Loading models..."
            ) : mode === "text" ? (
              ""
            ) : (
              <>
                {isListening ? (
                  <div className="flex items-center gap-2">
                    <Mic className="h-4 w-4 text-green-500 animate-pulse" />
                    <span className="text-green-500">Listening...</span>
                  </div>
                ) : (
                  ""
                )}
              </>
            )}
          </div>
        </div>

        {/* Stats Panel */}
        <StatsPanel 
          metrics={conversationMetrics}
          conversationStartTime={currentTurnStartTime}
        />

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-3 bg-background">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
              <Bot className="h-16 w-16 mb-4 opacity-20" />
              <h2 className="text-xl font-medium mb-2">
                Welcome to the Conversational Filler
              </h2>
              <p className="text-sm max-w-md">
                {mode === "text"
                  ? "Type a message below to start chatting with SmolLM, enhanced by OpenAI for context."
                  : "Just start speaking! I'm listening and will respond with voice."}
              </p>
              {modelLoading && (
                <div className="mt-6">
                  <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2" />
                  <p className="text-xs">{modelLoadingProgress}</p>
                </div>
              )}
            </div>
          )}

          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`flex gap-3 max-w-[70%] ${message.role === "user" ? "flex-row-reverse" : ""}`}
                >
                  <div
                    className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    {message.role === "user" ? (
                      <User className="h-4 w-4" />
                    ) : (
                      <Bot className="h-4 w-4" />
                    )}
                  </div>
                  <div
                    className={`px-4 py-2 rounded-lg ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">
                      {message.processedContent || message.content}
                    </p>
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="flex gap-3 max-w-[70%]">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                    <Bot className="h-4 w-4" />
                  </div>
                  <div className="px-4 py-2 rounded-lg bg-muted">
                    <Loader2 className="h-4 w-4 animate-spin" />
                  </div>
                </div>
              </div>
            )}
          </div>

          <div ref={messagesEndRef} />
        </div>

        {/* Input Box */}
        <div className="border-t bg-card px-4 py-3 flex-shrink-0">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                modelLoading
                  ? "Waiting for models to load..."
                  : mode === "voice"
                    ? "Type a message or use voice recording..."
                    : "Type your message..."
              }
              className="flex-1 px-4 py-2 border rounded-lg bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isLoading || modelLoading}
            />
            <Button
              type="submit"
              disabled={!input.trim() || isLoading || modelLoading}
            >
              <Send className="h-4 w-4 mr-2" />
              Send
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}
