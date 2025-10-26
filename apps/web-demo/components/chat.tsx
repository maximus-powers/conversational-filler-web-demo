"use client";

import { Button } from "@convo-filler/ui/components/button";
import { useState, useRef, useEffect } from "react";
import { Loader2, Send, Eye, EyeOff } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { UnifiedPipeline, UntrainedPipeline, PipelineState } from "../app/lib/unified-pipeline";
import { Timeline } from "./timeline";
import { PipelineControls } from "./pipeline-controls";
import { StatsPanel } from "./stats-panel";
import { MessageList } from "./message-list";
import { EventData } from "../app/lib/event-tracker";
import { saveConversation } from "../app/lib/utils/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  processedContent?: string;
  thoughts?: string[];
  thoughtResponsePairs?: Array<{thought: string, response: string}>;
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
  const [responseCompleteTimeout, setResponseCompleteTimeout] = useState<NodeJS.Timeout | null>(null);
  const [currentAssistantMessageId, setCurrentAssistantMessageId] = useState<string | null>(null);
  const [conversationStartTime, setConversationStartTime] = useState<
    number | null
  >(null);
  // pipeline config
  const [sttMode, setSTTMode] = useState<"disabled" | "local" | "api">(voiceMode ? "local" : "disabled");
  const [enableThoughts, setEnableThoughts] = useState(config?.thoughtModel !== "none");
  const [smolLMMode, setSmolLMMode] = useState<"convfill" | "untrained" | "none">(
    config?.localModel !== null ? "convfill" : "none"
  );
  const [enableTTS, setEnableTTS] = useState(false);
  const [persona, setPersona] = useState("none");
  const [isListening, setIsListening] = useState(false);

  const [conversationId] = useState<string>(() => crypto.randomUUID());
  const [showTimeline, setShowTimeline] = useState(false);
  const [showStatsPanel, setShowStatsPanel] = useState(!feedbackMode);
  const [showThoughts, setShowThoughts] = useState(false);
  const [showSideBySide, setShowSideBySide] = useState(false);
  const [comparisonMessages, setComparisonMessages] = useState<Message[]>([]);
  const [isComparisonLoading, setIsComparisonLoading] = useState(false);
  const [untrainedMessages, setUntrainedMessages] = useState<Message[]>([]);
  const [isUntrainedLoading, setIsUntrainedLoading] = useState(false);

  // Column widths for side-by-side mode (percentages)
  const [columnWidths, setColumnWidths] = useState([33.33, 33.33, 33.34]);
  const [isDragging, setIsDragging] = useState<number | null>(null);

  const pipelineRef = useRef<UnifiedPipeline | null>(null);
  const untrainedPipelineRef = useRef<UntrainedPipeline | null>(null);
  const messagesRef = useRef<Map<string, Message>>(new Map());
  const untrainedMessagesRef = useRef<Map<string, Message>>(new Map());
  const currentUserPromptRef = useRef<string>("");
  const eventDataRef = useRef<EventData | null>(null);
  const pendingGeminiInputRef = useRef<string | null>(null);
  const pendingUntrainedInputRef = useRef<string | null>(null);
  const comparisonMessagesRef = useRef<Message[]>([]);
  const pipelineCallbacksRef = useRef<any>(null);

  const clearEventData = () => {
    setEventData(null);
    setConversationStartTime(null);
    if (pipelineRef.current) {
      pipelineRef.current.resetEventData();
    }
  };

  // Handle column resizing for side-by-side mode
  const handleMouseDown = (index: number) => {
    setIsDragging(index);
  };

  useEffect(() => {
    if (isDragging === null) return;

    const handleMouseMove = (e: MouseEvent) => {
      const container = document.getElementById('side-by-side-container');
      if (!container) return;

      const containerRect = container.getBoundingClientRect();
      const mouseX = e.clientX - containerRect.left;
      const percentage = (mouseX / containerRect.width) * 100;

      setColumnWidths((prev) => {
        const newWidths = [...prev];
        const minWidth = 15;

        if (isDragging === 0) {
          // Resizing first divider (between Gemini and Untrained)
          const availableWidth = 100 - prev[2];
          const newFirstWidth = Math.max(minWidth, Math.min(availableWidth - minWidth, percentage));
          const newSecondWidth = availableWidth - newFirstWidth;

          newWidths[0] = newFirstWidth;
          newWidths[1] = newSecondWidth;
          newWidths[2] = prev[2]; // Keep ConvFill column fixed
        } else if (isDragging === 1) {
          // Resizing second divider (between Untrained and ConvFill)
          const availableWidth = 100 - prev[0];
          const secondColumnStart = prev[0];
          const relativePercentage = percentage - secondColumnStart;
          const newSecondWidth = Math.max(minWidth, Math.min(availableWidth - minWidth, relativePercentage));
          const newThirdWidth = availableWidth - newSecondWidth;

          newWidths[0] = prev[0]; // Keep Gemini column fixed
          newWidths[1] = newSecondWidth;
          newWidths[2] = newThirdWidth;
        }

        return newWidths;
      });
    };

    const handleMouseUp = () => {
      setIsDragging(null);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);


  const handleResponseComplete = (_messageId: string, fullResponse: string, userPrompt?: string) => {
    const promptToUse = userPrompt || currentUserPrompt;
    if (!feedbackMode && promptToUse) {
      saveConversation({
        conversationId,
        localModel: smolLMMode === "convfill" ? "maximuspowers/smollm-convo-filler-onnx-official" :
                    smolLMMode === "untrained" ? "HuggingFaceTB/SmolLM-360M-Instruct" : null,
        thoughtModel: enableThoughts ? "gemini" : "none",
        voiceMode: sttMode !== "disabled",
        userPrompt: promptToUse,
        aiResponse: fullResponse,
        eventData: eventDataRef.current || eventData,
        feedbackMode
      });
    }
    // wait until questionnaire is done in feedback mode
    if (feedbackMode && onTurnComplete) {
      const turnData = [{
        prompt: promptToUse || "User message",
        thought: null,
        generatedResponse: fullResponse,
      }];
      onTurnComplete(turnData, eventData || undefined);
    }
  };

  const scheduleResponseComplete = (messageId: string, userPrompt?: string) => {
    if (responseCompleteTimeout) {
      clearTimeout(responseCompleteTimeout);
    }
    const timeout = setTimeout(() => {
      const message = messagesRef.current.get(messageId);
      if (message) {
        const fullResponse = message.processedContent || message.content;
        handleResponseComplete(messageId, fullResponse || "", userPrompt);
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

      const enableSTT = sttMode !== "disabled";
      const pipelineFeatures: PipelineState['features'] = {
        enableSTT,
        sttMode: sttMode === "disabled" ? undefined : sttMode,
        enableThoughts,
        smolLMMode,
        enableTTS,
        persona,
      };

      const callbacks = {
        onMessageReceived: (role: "user" | "assistant", content: string, messageId?: string) => {
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
            setTimeout(() => {
              scheduleResponseComplete(msgId, currentUserPromptRef.current); // not sure if this is even working
            }, 3000); // 3 seconds as fallback
          }
        },

        onMessageUpdated: (messageId: string, newContent: string) => {
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
          if (messageId === currentAssistantMessageId) {
            scheduleResponseComplete(messageId, currentUserPromptRef.current);
          }
        },


        onStatusChange: (status: string, message: string) => {
          setModelLoadingProgress(message);
          if (status === "ready") {
            setModelLoading(false);
            setModelLoadingProgress("");
          } else if (status === "recording_start") {
            setIsListening(true);
          } else if (status === "recording_end") {
            setIsListening(false);
          } else if (status === "response_complete" || status === "processing_complete" || status === "generation_complete") {
            if (currentAssistantMessageId) {
              const message = messagesRef.current.get(currentAssistantMessageId);
              if (message) {
                const fullResponse = message.processedContent || message.content;
                handleResponseComplete(currentAssistantMessageId, fullResponse, currentUserPromptRef.current);
              }
            }
          }
        },

        onEventData: (data: EventData) => {
          setEventData(data);
          eventDataRef.current = data;
        },

        onConversationStart: (startTime: number) => {
          setConversationStartTime(startTime);
        },

        onThoughtResponsePairs: (pairs: Array<{thought: string, response: string}>, messageId: string) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id === messageId) {
                const updatedMessage = {
                  ...msg,
                  thoughtResponsePairs: pairs,
                };
                messagesRef.current.set(messageId, updatedMessage);
                return updatedMessage;
              }
              return msg;
            }),
          );

          // comparison waiting until after convfill done
          if (pendingGeminiInputRef.current) {
            fetchGeminiComparison(pendingGeminiInputRef.current);
            pendingGeminiInputRef.current = null;
          }

          // trigger untrained model after trained response completes
          if (pendingUntrainedInputRef.current) {
            processUntrainedModel(pendingUntrainedInputRef.current);
            pendingUntrainedInputRef.current = null;
          }
        },
      };

      pipelineCallbacksRef.current = callbacks;
      pipelineRef.current = new UnifiedPipeline(callbacks, pipelineFeatures);

      try {
        await pipelineRef.current.initialize();

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
      if (untrainedPipelineRef.current) {
        untrainedPipelineRef.current.dispose();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSTTModeChange = async (mode: "disabled" | "local" | "api") => {
    setSTTMode(mode);
    if (pipelineRef.current) {
      if (mode === "disabled") {
        await pipelineRef.current.toggleSTT(false);
      } else {
        await pipelineRef.current.toggleSTT(true, mode);
      }
    }
  };

  const handleToggleThoughts = (enabled: boolean) => {
    if (enabled && smolLMMode === "untrained") {
      return;
    }
    setEnableThoughts(enabled);
    if (pipelineRef.current) {
      pipelineRef.current.updateFeatures({ enableThoughts: enabled });
    }
  };

  const handleSmolLMModeChange = (mode: "convfill" | "untrained" | "none") => {
    setSmolLMMode(mode);
    if (mode === "untrained") {
      setEnableThoughts(false);
    }
    if (pipelineRef.current) {
      pipelineRef.current.updateFeatures({
        smolLMMode: mode,
        enableThoughts: mode === "untrained" ? false : enableThoughts
      });
    }
  };

  const handleToggleTTS = (enabled: boolean) => {
    setEnableTTS(enabled);
    if (pipelineRef.current) {
      pipelineRef.current.updateFeatures({ enableTTS: enabled });
    }
  };

  const handlePersonaChange = (newPersona: string) => {
    setPersona(newPersona);
    if (pipelineRef.current) {
      pipelineRef.current.updateFeatures({ persona: newPersona });
    }
  };

  const fetchGeminiComparison = async (userInput: string) => {
    setIsComparisonLoading(true);
    const userMessageId = `comparison-${Date.now()}`;
    const userMessage: Message = {
      id: userMessageId,
      role: "user",
      content: userInput,
      processedContent: userInput,
    };

    const updatedMessages = [...comparisonMessagesRef.current, userMessage]; // store comparison messages in ref too for consistency
    comparisonMessagesRef.current = updatedMessages;
    setComparisonMessages(updatedMessages);

    try {
      const personaParam = persona !== "none" ? `?persona=${persona}` : '';
      const response = await fetch(`/api/gemini-standalone${personaParam}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: updatedMessages.map(m => ({
            role: m.role,
            content: m.content
          }))
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const assistantMessageId = `comparison-${Date.now()}-assistant`;
      let fullResponse = "";
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        fullResponse += chunk;

        setComparisonMessages(prev => {
          const existing = prev.find(m => m.id === assistantMessageId);
          let updated;
          if (existing) {
            updated = prev.map(m => m.id === assistantMessageId ? { ...m, content: fullResponse, processedContent: fullResponse } : m);
          } else {
            updated = [...prev, {
              id: assistantMessageId,
              role: "assistant" as const,
              content: fullResponse,
              processedContent: fullResponse,
            }];
          }
          comparisonMessagesRef.current = updated;
          return updated;
        });
      }

      setIsComparisonLoading(false);
    } catch (error) {
      console.error("Gemini comparison error:", error);
      setIsComparisonLoading(false);
    }
  };

  const processUntrainedModel = async (userInput: string) => {
    setIsUntrainedLoading(true);

    // Initialize untrained pipeline if needed (lazily on first use)
    if (!untrainedPipelineRef.current && pipelineRef.current) {
      const sharedWorker = pipelineRef.current.getUntrainedWorker();
      if (!sharedWorker || !pipelineRef.current.isUntrainedWorkerInitialized()) {
        console.error("Untrained worker not ready");
        setIsUntrainedLoading(false);
        return;
      }

      untrainedPipelineRef.current = new UntrainedPipeline(sharedWorker, {
        onMessageReceived: (role: "user" | "assistant", content: string, messageId?: string) => {
          const message: Message = {
            id: messageId || `untrained-${Date.now()}`,
            role,
            content,
            processedContent: content,
          };

          if (messageId) {
            untrainedMessagesRef.current.set(messageId, message);
          }

          setUntrainedMessages((prev) => {
            const existing = prev.find((m) => m.id === message.id);
            return existing
              ? prev.map((m) => (m.id === message.id ? message : m))
              : [...prev, message];
          });

          if (role === "assistant") {
            setIsUntrainedLoading(false);
          }
        },
        onMessageUpdated: (messageId: string, content: string) => {
          const updatedMessage: Message = {
            id: messageId,
            role: "assistant",
            content,
            processedContent: content,
          };

          untrainedMessagesRef.current.set(messageId, updatedMessage);
          setUntrainedMessages((prev) =>
            prev.map((msg) => (msg.id === messageId ? updatedMessage : msg))
          );
        },
      });
    }

    try {
      if (untrainedPipelineRef.current) {
        await untrainedPipelineRef.current.processText(userInput);
      }
    } catch (error) {
      console.error("Untrained model processing error:", error);
      setIsUntrainedLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || modelLoading || disabled)
      return;

    const currentInput = input;
    setInput("");
    setIsLoading(true);

    setCurrentUserPrompt(currentInput);
    currentUserPromptRef.current = currentInput;

    if (showSideBySide) {
      pendingGeminiInputRef.current = currentInput;
      pendingUntrainedInputRef.current = currentInput;
    }

    try {
      if (smolLMMode === "untrained" && untrainedPipelineRef.current) {
        await untrainedPipelineRef.current.processText(currentInput);
      } else if (pipelineRef.current) {
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

          setCurrentAssistantMessageId(aiMessageId);
          scheduleResponseComplete(aiMessageId, currentUserPromptRef.current);
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
    setComparisonMessages([]);
    setUntrainedMessages([]);
    comparisonMessagesRef.current = [];
    messagesRef.current.clear();
    untrainedMessagesRef.current.clear();
    clearEventData();
    setIsLoading(false);
    setIsComparisonLoading(false);
    setIsUntrainedLoading(false);
    pendingGeminiInputRef.current = null;
    pendingUntrainedInputRef.current = null;
    if (responseCompleteTimeout) {
      clearTimeout(responseCompleteTimeout);
      setResponseCompleteTimeout(null);
    }
    setCurrentAssistantMessageId(null);

    if (pipelineRef.current) {
      pipelineRef.current.clearMessages();
    }
    if (untrainedPipelineRef.current) {
      untrainedPipelineRef.current.clearMessages();
    }
  };


  return (
    <div className="flex h-full w-full overflow-hidden">
      {showTimeline && (
        <Timeline
          eventData={eventData}
          conversationStartTime={conversationStartTime}
          mode={sttMode !== "disabled" ? "voice" : "text"}
        />
      )}

      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        {!feedbackMode && (
          <div className="bg-card border-b px-6 py-2 flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <PipelineControls
                  sttMode={sttMode}
                  enableThoughts={enableThoughts}
                  smolLMMode={smolLMMode}
                  enableTTS={enableTTS}
                  persona={persona}
                  onSTTModeChange={handleSTTModeChange}
                  onToggleThoughts={handleToggleThoughts}
                  onSmolLMModeChange={handleSmolLMModeChange}
                  onToggleTTS={handleToggleTTS}
                  onPersonaChange={handlePersonaChange}
                  disabled={modelLoading || isLoading}
                />

                {isListening && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="h-2 w-2 bg-red-500 rounded-full animate-pulse" />
                    <span>Listening...</span>
                  </div>
                )}
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
                  onClick={() => setShowThoughts(!showThoughts)}
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-1"
                >
                  {showThoughts ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                  Show Thoughts
                </Button>

                <Button
                  onClick={() => setShowSideBySide(!showSideBySide)}
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-1"
                >
                  {showSideBySide ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                  Side by Side
                </Button>

                <Button
                  onClick={clearChat}
                  size="sm"
                  disabled={messages.length === 0}
                  className="bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
                >
                  Reset
                </Button>

                <ThemeToggle />
              </div>
            </div>
          </div>
        )}

        {/* Stats Panel */}
        {showStatsPanel && eventData && conversationStartTime && (
          <StatsPanel
            eventData={eventData}
            conversationStartTime={conversationStartTime}
          />
        )}

        {/* Messages */}
        <div id="side-by-side-container" className="flex-1 flex overflow-hidden" style={{ position: 'relative' }}>
          {/* Gemini Side */}
          {showSideBySide && (
            <div
              className="flex flex-col border-r"
              style={{ width: `${columnWidths[0]}%` }}
            >
              <MessageList
                messages={comparisonMessages}
                isLoading={isComparisonLoading}
                showThoughts={false}
                title="Gemini"
                isEmpty={comparisonMessages.length === 0}
                emptyMessage="Comparison responses will appear here"
              />
            </div>
          )}

          {/* Resize Handle 1 */}
          {showSideBySide && (
            <div
              className="w-1 bg-border hover:bg-blue-500 cursor-col-resize flex-shrink-0 transition-colors"
              onMouseDown={() => handleMouseDown(0)}
              style={{ cursor: 'col-resize' }}
            />
          )}

          {/* Untrained SmolLM Side */}
          {showSideBySide && (
            <div
              className="flex flex-col border-r"
              style={{ width: `${columnWidths[1]}%` }}
            >
              <MessageList
                messages={untrainedMessages}
                isLoading={isUntrainedLoading}
                showThoughts={false}
                title="SmolLM (Untrained)"
                isEmpty={untrainedMessages.length === 0}
                emptyMessage="Untrained model responses will appear here"
              />
            </div>
          )}

          {/* Resize Handle 2 */}
          {showSideBySide && (
            <div
              className="w-1 bg-border hover:bg-blue-500 cursor-col-resize flex-shrink-0 transition-colors"
              onMouseDown={() => handleMouseDown(1)}
              style={{ cursor: 'col-resize' }}
            />
          )}

          {/* ConvFill Side */}
          <div
            className="flex flex-col"
            style={{ width: showSideBySide ? `${columnWidths[2]}%` : '100%' }}
          >
            <MessageList
              messages={messages}
              isLoading={isLoading}
              showThoughts={showThoughts}
              title={showSideBySide ? "ConvFill" : undefined}
              isEmpty={messages.length === 0}
              emptyMessage={sttMode !== "disabled"
                ? "Just start speaking! I'm listening and will respond."
                : "Type a message below to start chatting."}
              showWelcome={!showSideBySide}
              modelLoading={modelLoading}
              modelLoadingProgress={modelLoadingProgress}
            />
          </div>
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
                  : sttMode !== "disabled"
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