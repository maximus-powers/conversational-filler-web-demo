"use client";

import { Button } from "@convo-filler/ui/components/button";
import { useState, useRef, useEffect } from "react";
import { Bot, User, Loader2, Send, Mic, MicOff, MessageSquare } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { UnifiedPipeline, AppMode } from "../app/lib/unified-pipeline";
import { Timeline, TimelineEvent } from "./timeline";
import { ModeSwitcher } from "./mode-switcher";

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
  const [conversationStartTime, setConversationStartTime] = useState<number | null>(null);
  const [mode, setMode] = useState<AppMode>("text");
  const [isListening, setIsListening] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState<string>("af_heart");
  const [availableVoices, setAvailableVoices] = useState<Record<string, any>>({});
  const pipelineRef = useRef<UnifiedPipeline | null>(null);
  const messagesRef = useRef<Map<string, Message>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addTimelineEvent = (
    type: TimelineEvent["type"],
    model: TimelineEvent["model"],
    message: string,
    content: string = ""
  ) => {
    const event: TimelineEvent = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      type,
      model,
      message,
      content: content.slice(0, 50) + (content.length > 50 ? "..." : ""),
      fullContent: content,
    };
    setTimelineEvents(prev => [...prev, event]);
  };

  const clearTimeline = () => {
    setTimelineEvents([]);
    setConversationStartTime(null);
  };

  // Initialize pipeline
  useEffect(() => {
    const initializePipeline = async () => {
      const initStartTime = Date.now();
      setConversationStartTime(initStartTime);
      
      setModelLoadingProgress("Loading models...");
      addTimelineEvent("model-loading", "Pipeline", "Initializing unified pipeline", "");
      
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
          
          setMessages(prev => {
            const existing = prev.find(m => m.id === message.id);
            if (existing) {
              return prev.map(m => m.id === message.id ? message : m);
            }
            return [...prev, message];
          });
        },
        
        onMessageUpdated: (messageId, newContent) => {
          setMessages(prev => prev.map(msg => {
            if (msg.id === messageId) {
              const currentContent = msg.processedContent || msg.content;
              return {
                ...msg,
                processedContent: currentContent + " " + newContent
              };
            }
            return msg;
          }));
        },
        
        onThoughtReceived: (thought, index) => {
          addTimelineEvent("openai-thought", "OpenAI", `Thought ${index + 1}`, thought);
        },
        
        onTranscriptionReceived: (text) => {
          addTimelineEvent("whisper-transcription", "Whisper", "Transcribed speech", text);
        },
        
        onStatusChange: (status, message) => {
          setModelLoadingProgress(message);
          if (status === 'ready') {
            setModelLoading(false);
            setModelLoadingProgress("");
          } else if (status === 'recording_start') {
            setIsListening(true);
          } else if (status === 'recording_end') {
            setIsListening(false);
          }
        },
        
        onTimelineEvent: (type, model, message, content) => {
          addTimelineEvent(type as TimelineEvent["type"], model as TimelineEvent["model"], message, content || "");
        },
      });

      try {
        await pipelineRef.current.initialize(mode);
        const voices = pipelineRef.current.getVoices();
        setAvailableVoices(voices);
        
        const initEndTime = Date.now();
        const loadTime = ((initEndTime - initStartTime) / 1000).toFixed(2);
        addTimelineEvent("model-ready", "Pipeline", `Models loaded in ${loadTime}s`, "");
      } catch (error) {
        console.error("Failed to initialize pipeline:", error);
        setModelLoadingProgress("Failed to load models");
        addTimelineEvent("error", "Pipeline", "Failed to initialize", error?.toString() || "");
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
      addTimelineEvent("mode-switch", "System", `Switched to ${newMode} mode`, "");
    } catch (error) {
      console.error("Failed to switch mode:", error);
      addTimelineEvent("error", "System", "Failed to switch mode", error?.toString() || "");
    } finally {
      setModelLoading(false);
      setModelLoadingProgress("");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || modelLoading || !pipelineRef.current) return;
    
    const currentInput = input;
    setInput("");
    setIsLoading(true);
    
    // Clear timeline for new conversation
    if (messages.length === 0) {
      setTimelineEvents([]);
      setConversationStartTime(Date.now());
    }
    
    try {
      await pipelineRef.current.processText(currentInput);
    } catch (error) {
      console.error("Text processing error:", error);
      addTimelineEvent("error", "System", "Processing failed", error?.toString() || "");
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

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* Timeline - Left Side */}
      <Timeline 
        events={timelineEvents} 
        conversationStartTime={conversationStartTime}
        mode={mode}
      />
      
      {/* Main Chat Area - Right Side */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header - Fixed */}
        <div className="bg-card border-b px-6 py-3 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ModeSwitcher 
                currentMode={mode}
                onModeChange={handleModeChange}
                disabled={modelLoading || isLoading}
              />
              
              {mode === 'voice' && Object.keys(availableVoices).length > 0 && (
                <select
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  className="text-sm px-2 py-1 border rounded-md bg-background"
                  disabled={modelLoading}
                >
                  {Object.entries(availableVoices).map(([id, voice]: [string, any]) => (
                    <option key={id} value={id}>
                      {voice.name || id}
                    </option>
                  ))}
                </select>
              )}
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                onClick={clearChat}
                variant="outline"
                size="sm"
                disabled={messages.length === 0}
              >
                Clear
              </Button>
              
              <ThemeToggle />
            </div>
          </div>
          
          {/* Status Bar */}
          <div className="mt-2 text-sm text-muted-foreground flex items-center gap-2">
            {modelLoading ? (
              modelLoadingProgress || "Loading models..."
            ) : mode === 'text' ? (
              "Text mode: Type messages to chat with SmolLM enhanced by OpenAI thoughts"
            ) : (
              <>
                {isListening ? (
                  <div className="flex items-center gap-2">
                    <Mic className="h-4 w-4 text-red-500 animate-pulse" />
                    <span className="text-red-500">Listening...</span>
                  </div>
                ) : (
                  "Voice mode: Speak naturally and I'll respond with voice"
                )}
              </>
            )}
          </div>
        </div>

        {/* Messages Area - Scrollable */}
        <div className="flex-1 overflow-y-auto px-4 py-3 bg-background">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
              <Bot className="h-16 w-16 mb-4 opacity-20" />
              <h2 className="text-xl font-medium mb-2">Welcome to Conversational Filler</h2>
              <p className="text-sm max-w-md">
                {mode === 'text' 
                  ? "Type a message below to start chatting with SmolLM, enhanced by OpenAI's contextual thoughts."
                  : "Just start speaking! I'm always listening and will respond with voice."}
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
                <div className={`flex gap-3 max-w-[70%] ${message.role === "user" ? "flex-row-reverse" : ""}`}>
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                  }`}>
                    {message.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                  </div>
                  <div className={`px-4 py-2 rounded-lg ${
                    message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                  }`}>
                    <p className="text-sm whitespace-pre-wrap">
                      {message.processedContent || message.content}
                    </p>
                    {message.thoughts && message.thoughts.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-current opacity-50">
                        <p className="text-xs font-medium mb-1">OpenAI Thoughts:</p>
                        {message.thoughts.map((thought, idx) => (
                          <p key={idx} className="text-xs">• {thought}</p>
                        ))}
                      </div>
                    )}
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

        {/* Input Area - Fixed at Bottom */}
        <div className="border-t bg-card px-4 py-3 flex-shrink-0">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                modelLoading 
                  ? "Waiting for models to load..." 
                  : mode === 'voice'
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