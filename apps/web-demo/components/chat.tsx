"use client";

import { Button } from "@convo-filler/ui/components/button";
import { useState, useRef, useEffect } from "react";
import { Bot, User, Loader2, Send, Mic, MicOff, Eye, EyeOff } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { UnifiedPipeline, InferenceMode } from "../app/lib/unified-pipeline";
import { Timeline } from "./timeline";
import { InferenceModeSwitcher } from "./inference-mode-switcher";
import { StatsPanel } from "./stats-panel";
import { EventData } from "../app/lib/event-tracker";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  processedContent?: string;
  thoughts?: string[];
}

type ModelConfig = {
  localModel: string | null;
  thoughtModel: "gemini" | "none";
};

export function Chat({
  feedbackMode = false,
  config,
  voiceMode = false,
  disabled = false,
  onTurnComplete,
}: {
  feedbackMode?: boolean;
  config?: ModelConfig;
  voiceMode?: boolean;
  disabled?: boolean;
  onTurnComplete?: (prompts: Array<{
    prompt: string;
    thought: string | null;
    generatedResponse: string;
  }>, events?: EventData) => void;
} = {}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelLoadingProgress, setModelLoadingProgress] = useState<string>("");
  const [eventData, setEventData] = useState<EventData | null>(null);
  const [currentUserPrompt, setCurrentUserPrompt] = useState<string>("");
  const [currentTurnResponse, setCurrentTurnResponse] = useState<string>("");
  const [responseCompleteTimeout, setResponseCompleteTimeout] = useState<NodeJS.Timeout | null>(null);
  const [currentAssistantMessageId, setCurrentAssistantMessageId] = useState<string | null>(null);
  const [conversationStartTime, setConversationStartTime] = useState<
    number | null
  >(null);
  const [mode, setMode] = useState<InferenceMode>(voiceMode ? "voice" : "text");
  const [isListening, setIsListening] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState<string>("af_heart");
  const [availableVoices, setAvailableVoices] = useState<Record<string, any>>(
    {},
  );
  const [thoughtProvider, setThoughtProvider] = useState<"gemini" | "none">(
    config?.thoughtModel || "gemini"
  );
  const [selectedModel, setSelectedModel] = useState<"maximuspowers/smollm-convo-filler-onnx-official" | "HuggingFaceTB/SmolLM-360M-Instruct" | "none">(
    (config?.localModel as any) || "maximuspowers/smollm-convo-filler-onnx-official"
  );
  const [showTimeline, setShowTimeline] = useState(!feedbackMode);
  const [showStatsPanel, setShowStatsPanel] = useState(!feedbackMode);
  const pipelineRef = useRef<UnifiedPipeline | null>(null);
  const messagesRef = useRef<Map<string, Message>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const clearEventData = () => {
    setEventData(null);
    setConversationStartTime(null);
    if (pipelineRef.current) {
      pipelineRef.current.resetEventData();
    }
  };

  const handleResponseComplete = (messageId: string, fullResponse: string) => {    
    if (feedbackMode && onTurnComplete) {
      const turnData = [{
        prompt: currentUserPrompt || "User message",
        thought: null, // TODO: extract actual thought data
        generatedResponse: fullResponse,
      }];
      onTurnComplete(turnData, eventData || undefined);
    }
  };

  const scheduleResponseComplete = (messageId: string) => {
    if (responseCompleteTimeout) {
      clearTimeout(responseCompleteTimeout);
    }
    
    // 500ms after complete it triggers questionaire
    const timeout = setTimeout(() => {
      const message = messagesRef.current.get(messageId);
      if (message) {
        const fullResponse = message.processedContent || message.content;
        handleResponseComplete(messageId, fullResponse || "");
      }
    }, 500);
    
    setResponseCompleteTimeout(timeout);
  };

  // init pipeline
  useEffect(() => {
    const initializePipeline = async () => {
      const initStartTime = Date.now();
      setConversationStartTime(initStartTime);

      setModelLoadingProgress("Loading models...");

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

          if (role === "assistant") {
            setIsLoading(false);
            const msgId = messageId || Date.now().toString();
            setCurrentAssistantMessageId(msgId);
            if (feedbackMode) {
              setTimeout(() => {
                scheduleResponseComplete(msgId);
              }, 3000); // 3 seconds as fallback
            }
          }
        },

        onMessageUpdated: (messageId, newContent) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id === messageId) {
                const currentContent = msg.processedContent || msg.content;
                const updatedMessage = {
                  ...msg,
                  processedContent: currentContent + " " + newContent,
                };
                messagesRef.current.set(messageId, updatedMessage);
                return updatedMessage;
              }
              return msg;
            }),
          );
          if (feedbackMode && messageId === currentAssistantMessageId) {
            scheduleResponseComplete(messageId);
          }
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
          } else if (status === "response_complete" || status === "processing_complete" || status === "generation_complete") {
            if (feedbackMode && currentAssistantMessageId) {
              const message = messagesRef.current.get(currentAssistantMessageId);
              if (message) {
                const fullResponse = message.processedContent || message.content;
                handleResponseComplete(currentAssistantMessageId, fullResponse);
              }
            }
          }
        },

        onEventData: (data: EventData) => {
          setEventData(data);
        },
      });

      try {
        await pipelineRef.current.initialize(mode);
        const voices = pipelineRef.current.getVoices();
        setAvailableVoices(voices);

        const initEndTime = Date.now();
        const loadTime = ((initEndTime - initStartTime) / 1000).toFixed(2);
        console.log(`Models loaded in ${loadTime}s`);
      } catch (error) {
        console.error("Failed to initialize pipeline:", error);
        setModelLoadingProgress("Failed to load models - continuing in fallback mode");
        setModelLoading(false);
      }
    };

    initializePipeline();

    return () => {
      if (pipelineRef.current) {
        pipelineRef.current.dispose();
      }
    };
  }, []);

  const handleModeChange = async (newMode: InferenceMode) => {
    if (!pipelineRef.current || newMode === mode) return;

    setModelLoading(true);
    setModelLoadingProgress(`Switching to ${newMode} mode...`);

    try {
      await pipelineRef.current.switchMode(newMode);
      setMode(newMode);
      const voices = pipelineRef.current.getVoices();
      setAvailableVoices(voices);
      console.log(`Switched to ${newMode} mode`);
    } catch (error) {
      console.error("Failed to switch mode:", error);
    } finally {
      setModelLoading(false);
      setModelLoadingProgress("");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || modelLoading || disabled)
      return;

    const currentInput = input;
    setInput("");
    setIsLoading(true);
    
    if (feedbackMode) {
      setCurrentUserPrompt(currentInput);
    }

    // clear timeline
    if (messages.length === 0) {
      setConversationStartTime(Date.now());
    }

    try {
      if (pipelineRef.current) {
        await pipelineRef.current.processText(currentInput);
      } else {
        // todo: remove this

        const userMessage: Message = {
          id: Date.now().toString(),
          role: "user",
          content: currentInput,
          processedContent: currentInput,
        };
        setMessages(prev => [...prev, userMessage]);
        
        setTimeout(() => {
          const aiMessageId = (Date.now() + 1).toString();
          const aiMessage: Message = {
            id: aiMessageId,
            role: "assistant", 
            content: "This is a fallback response since the AI pipeline is not available.",
            processedContent: "This is a fallback response since the AI pipeline is not available.",
          };
          setMessages(prev => [...prev, aiMessage]);
          messagesRef.current.set(aiMessageId, aiMessage);
          
          setIsLoading(false);
          
          if (feedbackMode) {
            setCurrentAssistantMessageId(aiMessageId);
            scheduleResponseComplete(aiMessageId);
          }
        }, 1000);
        return;
      }
    } catch (error) {
      console.error("Text processing error:", error);
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    messagesRef.current.clear();
    clearEventData();
    setIsLoading(false);
    if (responseCompleteTimeout) {
      clearTimeout(responseCompleteTimeout);
      setResponseCompleteTimeout(null);
    }
    setCurrentAssistantMessageId(null);
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
      {showTimeline && (
        <Timeline
          eventData={eventData}
          conversationStartTime={conversationStartTime}
          mode={mode}
        />
      )}

      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        {!feedbackMode && (
          <div className="bg-card border-b px-6 py-2 flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <InferenceModeSwitcher
                  currentMode={mode}
                  onModeChange={handleModeChange}
                  disabled={modelLoading || isLoading}
                />

                {mode === "voice" && Object.keys(availableVoices).length > 0 && (
                  <select
                    value={selectedVoice}
                    onChange={(e) => setSelectedVoice(e.target.value)}
                    className="text-sm px-2 border rounded-md bg-background"
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
                  className="text-sm px-2 py-1 border rounded-md bg-background h-8"
                  disabled={modelLoading}
                  title="Select thought provider"
                >
                  <option value="gemini">Google (Gemini)</option>
                  <option value="none">None</option>
                </select>

                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value as "maximuspowers/smollm-convo-filler-onnx-official" | "HuggingFaceTB/SmolLM-360M-Instruct" | "none")}
                  className="text-sm px-2 py-1 border rounded-md bg-background h-8"
                  disabled={modelLoading}
                  title="Select SmolLM model"
                >
                  <option value="maximuspowers/smollm-convo-filler-onnx-official">SmolLM Convo Filler</option>
                  <option value="HuggingFaceTB/SmolLM-360M-Instruct">SmolLM 360M Instruct</option>
                  <option value="none">None (Gemini only)</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  onClick={() => setShowTimeline(!showTimeline)}
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-1"
                >
                  {showTimeline ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                  Timeline
                </Button>

                <Button
                  onClick={() => setShowStatsPanel(!showStatsPanel)}
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-1"
                >
                  {showStatsPanel ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                  Stats Panel
                </Button>

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
            {mode === "voice" && isListening && (
              <div className="mt-2 text-sm text-muted-foreground flex items-center gap-2">
                <div className="flex items-center gap-2">
                  <Mic className="h-4 w-4 text-green-500 animate-pulse" />
                  <span className="text-green-500">Listening...</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Stats Panel */}
        {showStatsPanel && (
          <StatsPanel 
            eventData={eventData}
            conversationStartTime={conversationStartTime}
          />
        )}

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
                  ? "Type a message below to start chatting with SmolLM, enhanced with context from Gemini."
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
              disabled={isLoading || modelLoading || disabled}
            />
            <Button
              type="submit"
              disabled={!input.trim() || isLoading || modelLoading || disabled}
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