"use client";

import { Button } from "@convo-filler/ui/components/button";
import { useState, useRef, useEffect } from "react";
import { Bot, User, Loader2, Send, Volume2, VolumeX, Mic, MicOff } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import { ResponseProcessor } from "../app/lib/response-processor";
import { SpeechProcessor } from "../app/lib/speech-processor";
import { Timeline, TimelineEvent } from "./timeline";

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
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
  const [conversationStartTime, setConversationStartTime] = useState<number | null>(null);
  const [speechEnabled, setSpeechEnabled] = useState(false);
  const [speechLoading, setSpeechLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState<string>("af_heart");
  const [availableVoices, setAvailableVoices] = useState<Record<string, any>>({});
  const processorRef = useRef<ResponseProcessor | null>(null);
  const speechProcessorRef = useRef<SpeechProcessor | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

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

  // init response processor
  useEffect(() => {
    const initializeProcessor = async () => {
      const initStartTime = Date.now();
      setConversationStartTime(initStartTime);
      
      setModelLoadingProgress("Loading local model...");
      addTimelineEvent("model-loading", "SmolLM", "Starting model download", "");
      
      processorRef.current = new ResponseProcessor({
        onThoughtReceived: (thought: string, index: number) => {
          addTimelineEvent("openai-thought", "OpenAI", `Thought ${index + 1} Received`, thought);
        },
        onTTSCompleted: (text: string) => {
          addTimelineEvent("tts-end", "TTS", "TTS Processing Completed", text);
        }
      });
      
      await processorRef.current.initialize();
      console.log("Response processor ready");
      addTimelineEvent("model-ready", "SmolLM", "Model initialized and ready", "");
      
      setModelLoading(false);
      setModelLoadingProgress("");
      
      setTtsEnabled(processorRef.current.isTTSEnabled());
    };
    initializeProcessor();
    return () => {
      processorRef.current = null;
    };
  }, []);

  // init speech processor
  useEffect(() => {
    const initializeSpeechProcessor = async () => {
      setSpeechLoading(true);
      addTimelineEvent("model-loading", "Speech", "Loading speech models", "");
      
      speechProcessorRef.current = new SpeechProcessor({
        onThoughtReceived: (thought: string, index: number) => {
          addTimelineEvent("openai-thought", "OpenAI", `Speech Thought ${index + 1}`, thought);
        },
        onTranscriptionReceived: (text: string) => {
          addTimelineEvent("whisper-transcription", "Whisper", "Speech Transcribed", text);
        },
        onImmediateResponse: (response: string) => {
          addTimelineEvent("smollm-response", "SmolLM", "Immediate Speech Response", response);
        },
        onEnhancedResponse: (response: string) => {
          addTimelineEvent("smollm-enhanced", "SmolLM", "Enhanced Speech Response", response);
        },
        onStatusChange: (status: string, message: string) => {
          if (status === 'ready') {
            const voices = speechProcessorRef.current?.getVoices() || {};
            setAvailableVoices(voices);
            addTimelineEvent("model-ready", "Speech", "Speech models ready", "");
          } else if (status === 'recording_start') {
            setIsListening(true);
            addTimelineEvent("recording-start", "VAD", "Voice detected", "");
          } else if (status === 'recording_end') {
            setIsListening(false);
            addTimelineEvent("recording-end", "VAD", "Processing speech", "");
          }
        },
        onTTSCompleted: (text: string) => {
          addTimelineEvent("tts-end", "TTS", "Speech TTS Completed", text);
        },
        enableTTS: true,
      });

      try {
        await speechProcessorRef.current.initialize();
        setSpeechEnabled(true);
        setSpeechLoading(false);
        console.log("Speech processor ready");
      } catch (error) {
        console.error("Failed to initialize speech processor:", error);
        setSpeechLoading(false);
        addTimelineEvent("tts-end", "Speech", "Speech initialization failed", "");
      }
    };

    // Only initialize speech processor when explicitly requested
    // initializeSpeechProcessor();

    return () => {
      if (speechProcessorRef.current) {
        speechProcessorRef.current.dispose();
        speechProcessorRef.current = null;
      }
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || modelLoading || !processorRef.current)
      return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput("");
    setIsLoading(true);

    // clear timeline and reset start time for each new message
    setTimelineEvents([]);
    setConversationStartTime(Date.now());

    // message placeholder (we update with stream)
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      processedContent: "",
      thoughts: [],
    };
    setMessages((prev) => [...prev, assistantMessage]);

    // cancel prev req
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      // start gen
      await processorRef.current!.generate(
        currentInput, 
        abortControllerRef.current.signal,
        (processedContent) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, processedContent: processedContent }
                : msg,
            ),
          );
          
          const prevState = processorRef.current?.getState();
          if (prevState) {
            const currentResponse = prevState.responseHistory[prevState.responseHistory.length - 1];
            if (currentResponse && !processedContent.includes("Sorry, I encountered an error.")) {
              if (prevState.responseHistory.length === 1) {
                addTimelineEvent("smollm-response", "SmolLM", "Initial Response Emitted", currentResponse);
              } else {
                addTimelineEvent("smollm-enhanced", "SmolLM", `Enhanced Response ${prevState.responseHistory.length}`, currentResponse);
              }
            }
          }
        }
      );

      // call api for provider thoughts
      const response = await fetch("/api/chat-thoughts", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: messages.concat(userMessage)
        }),
        signal: abortControllerRef.current.signal,
      });
      if (!response.ok) {
        throw new Error(`API error, status: ${response.status}`);
      }

      const reader = response.body?.getReader(); // reads as stream to extract chunks before full generation finishes
      if (!reader) {
        throw new Error("No response body reader available");
      }
      const decoder = new TextDecoder();

      // process by chunk
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        const chunk = decoder.decode(value, { stream: true });
        processorRef.current!.processThoughtChunk(chunk);
        
        const currentState = processorRef.current!.getState();
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, thoughts: [...currentState.thoughtQueue] }
              : msg,
          ),
        );
      }
      await processorRef.current!.waitForCompletion();
      const finalState = processorRef.current!.getState();
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, thoughts: [...finalState.thoughtQueue] }
            : msg,
        ),
      );

    } catch (error) {
      console.error("Chat error:", error);
      if (error instanceof Error && error.name === "AbortError") {
        return;
      }
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: "Sorry, I encountered an error.",
                processedContent: "Sorry, I encountered an error.",
              }
            : msg,
        ),
      );

    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const clearChat = () => {
    setMessages([]);
    clearTimeline();
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  };

  const toggleTTS = async () => {
    if (!processorRef.current) return;
    
    if (ttsEnabled) {
      processorRef.current.disableTTSMode();
      setTtsEnabled(false);
    } else {
      setTtsLoading(true);
      addTimelineEvent("model-loading", "TTS", "Loading TTS model", "");
      try {
        await processorRef.current.enableTTSMode();
        setTtsEnabled(true);
        addTimelineEvent("model-ready", "TTS", "TTS Model Ready", "");
      } catch (error) {
        console.error("Failed to enable TTS:", error);
        addTimelineEvent("tts-end", "TTS", "TTS Failed to Load", "");
      } finally {
        setTtsLoading(false);
      }
    }
  };

  const toggleSpeech = async () => {
    if (!speechEnabled && !speechProcessorRef.current) {
      // Initialize speech processor
      setSpeechLoading(true);
      addTimelineEvent("model-loading", "Speech", "Loading speech models", "");
      
      speechProcessorRef.current = new SpeechProcessor({
        onThoughtReceived: (thought: string, index: number) => {
          addTimelineEvent("openai-thought", "OpenAI", `Speech Thought ${index + 1}`, thought);
        },
        onTranscriptionReceived: (text: string) => {
          addTimelineEvent("whisper-transcription", "Whisper", "Speech Transcribed", text);
          // Add transcribed text as a user message
          const userMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: text,
          };
          setMessages((prev) => [...prev, userMessage]);
        },
        onImmediateResponse: (response: string) => {
          addTimelineEvent("smollm-response", "SmolLM", "Immediate Speech Response", response);
          // Add immediate response as assistant message
          const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: response,
            processedContent: response,
          };
          setMessages((prev) => [...prev, assistantMessage]);
        },
        onEnhancedResponse: (response: string) => {
          addTimelineEvent("smollm-enhanced", "SmolLM", "Enhanced Speech Response", response);
          // Update the last assistant message with enhanced content
          setMessages((prev) => {
            const newMessages = [...prev];
            for (let i = newMessages.length - 1; i >= 0; i--) {
              if (newMessages[i].role === "assistant") {
                newMessages[i] = {
                  ...newMessages[i],
                  processedContent: (newMessages[i].processedContent || newMessages[i].content) + " " + response
                };
                break;
              }
            }
            return newMessages;
          });
        },
        onStatusChange: (status: string, message: string) => {
          if (status === 'ready') {
            const voices = speechProcessorRef.current?.getVoices() || {};
            setAvailableVoices(voices);
            addTimelineEvent("model-ready", "Speech", "Speech models ready", "");
          } else if (status === 'recording_start') {
            setIsListening(true);
            addTimelineEvent("recording-start", "VAD", "Voice detected", "");
          } else if (status === 'recording_end') {
            setIsListening(false);
            addTimelineEvent("recording-end", "VAD", "Processing speech", "");
          }
        },
        onTTSCompleted: (text: string) => {
          addTimelineEvent("tts-end", "TTS", "Speech TTS Completed", text);
        },
        enableTTS: true,
      });

      try {
        await speechProcessorRef.current.initialize();
        setSpeechEnabled(true);
        setSpeechLoading(false);
        console.log("Speech processor ready");
      } catch (error) {
        console.error("Failed to initialize speech processor:", error);
        addTimelineEvent("tts-end", "Speech", "Speech initialization failed", "");
        setSpeechLoading(false);
      }
    } else if (speechEnabled && speechProcessorRef.current && !isRecording) {
      // Start recording
      try {
        await speechProcessorRef.current.startRecording();
        setIsRecording(true);
      } catch (error) {
        console.error("Failed to start recording:", error);
      }
    } else if (speechEnabled && speechProcessorRef.current && isRecording) {
      // Stop recording
      try {
        await speechProcessorRef.current.stopRecording();
        setIsRecording(false);
        setIsListening(false);
      } catch (error) {
        console.error("Failed to stop recording:", error);
      }
    }
  };

  return (
    <div className="flex relative w-full">
      <Timeline events={timelineEvents} startTime={conversationStartTime} />
      <div className="flex flex-col flex-1 bg-background">
      {/* Header */}
      <div className="flex flex-col gap-2 p-4 border-b bg-muted/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" />
            <span className="font-semibold">Conversational Filler Demo</span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={toggleTTS}
              variant="outline"
              size="sm"
              disabled={modelLoading || ttsLoading}
              title={ttsEnabled ? "Disable Text-to-Speech" : "Enable Text-to-Speech"}
            >
              {ttsLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : ttsEnabled ? (
                <Volume2 className="h-4 w-4" />
              ) : (
                <VolumeX className="h-4 w-4" />
              )}
            </Button>
            <Button
              onClick={toggleSpeech}
              variant="outline"
              size="sm"
              disabled={modelLoading || speechLoading}
              title={
                speechLoading 
                  ? "Loading speech models..." 
                  : !speechEnabled 
                  ? "Enable Speech Mode" 
                  : isRecording 
                  ? "Stop Recording" 
                  : "Start Recording"
              }
              className={isListening ? "bg-red-100 border-red-300" : ""}
            >
              {speechLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : !speechEnabled ? (
                <MicOff className="h-4 w-4" />
              ) : isRecording ? (
                <Mic className={`h-4 w-4 ${isListening ? "text-red-500" : "text-green-500"}`} />
              ) : (
                <Mic className="h-4 w-4 text-gray-500" />
              )}
            </Button>
            <ThemeToggle />
            <Button
              onClick={clearChat}
              variant="outline"
              size="sm"
              disabled={isLoading}
            >
              Clear Chat
            </Button>
          </div>
        </div>

        <div className="text-xs text-muted-foreground">
          {modelLoading
            ? modelLoadingProgress || "Loading SmolLM..."
            : speechLoading
            ? "Loading speech models (VAD, Whisper, TTS)..."
            : ttsLoading
            ? "Loading TTS model..."
            : `Fine-tuned SmolLM runs in browser • OpenAI provides thoughts${ttsEnabled ? " • TTS enabled" : ""}${speechEnabled ? " • Speech ready" : ""}${isRecording ? " • Recording active" : ""}${isListening ? " • Listening..." : ""}`}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-muted-foreground py-8">
            <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">Start a conversation</p>
            <p className="text-sm">
              Browser-based SmolLM processes OpenAI thoughts locally.<br/>
              Enable speech mode for voice conversations with VAD, Whisper ASR & TTS.
            </p>
            {modelLoading && (
              <div className="mt-4">
                <Loader2 className="h-4 w-4 animate-spin mx-auto mb-2" />
                <p className="text-xs">{modelLoadingProgress}</p>
              </div>
            )}
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {message.role === "assistant" && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center">
                <Bot className="h-4 w-4" />
              </div>
            )}
            <div
              className={`flex-1 max-w-[80%] rounded-lg px-3 py-2 ${
                message.role === "user"
                  ? "bg-primary text-primary-foreground ml-12"
                  : "bg-muted mr-12"
              }`}
            >
              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                {message.role === "assistant" && message.processedContent
                  ? message.processedContent
                  : message.content}
                {message.role === "assistant" &&
                  isLoading &&
                  !message.processedContent && (
                    <span className="inline-block w-2 h-4 bg-foreground ml-1 animate-pulse" />
                  )}
              </p>
            </div>
            {message.role === "user" && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted text-muted-foreground flex items-center justify-center">
                <User className="h-4 w-4" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="border-t p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              modelLoading ? "Waiting for model to load..." : "Ask anything..."
            }
            className="flex-1 px-3 py-2 border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            disabled={isLoading || modelLoading}
          />
          <Button
            type="submit"
            disabled={!input.trim() || isLoading || modelLoading}
            size="sm"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
    </div>
  );
}
