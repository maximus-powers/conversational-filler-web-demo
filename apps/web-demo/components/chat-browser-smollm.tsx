"use client";

import { Button } from "@convo-filler/ui/components/button";
import { useState, useRef, useEffect, useCallback } from "react";
import { Bot, User, Loader2, Send } from "lucide-react";
import { pipeline } from "@huggingface/transformers";
import { ThemeToggle } from "./theme-toggle";
import { ThoughtProcessor } from "../app/lib/thought-processor";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  processedContent?: string;
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelLoadingProgress, setModelLoadingProgress] = useState<string>("");
  const pipelineRef = useRef<any>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // init smollm in browser with webgpu
  useEffect(() => {
    const initializeModel = async () => {
      setModelLoadingProgress("Loading local model...");
      pipelineRef.current = await pipeline(
        "text-generation",
        "maximuspowers/smollm-convo-filler-onnx-official",
        {
          dtype: "fp32",
          device: "webgpu",
        },
      );;
      console.log("Local LM pipeline ready");
      setModelLoading(false);
      setModelLoadingProgress("");
    };
    initializeModel();
    return () => {
      if (pipelineRef.current) {
        pipelineRef.current = null;
      }
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || modelLoading || !pipelineRef.current)
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

    // message placeholder (we update with stream)
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      processedContent: "",
    };
    setMessages((prev) => [...prev, assistantMessage]);

    // cancel prev req
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      // init thought processor
      const thoughtProcessor = await ThoughtProcessor({
        pipeline: pipelineRef.current,
        currentInput: currentInput,
        abortSignal: abortControllerRef.current.signal,
        onUpdate: (processedContent) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, processedContent: processedContent }
                : msg,
            ),
          );
        }, // updates placeholder directly after each thought chunk gets processed
      });

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
        thoughtProcessor.processThoughtChunk(chunk);
      }
      await thoughtProcessor.waitForCompletion();

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
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-[600px] w-full max-w-2xl mx-auto border rounded-lg bg-background">
      {/* Header */}
      <div className="flex flex-col gap-2 p-4 border-b bg-muted/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" />
            <span className="font-semibold">Conversational Filler Demo</span>
          </div>
          <div className="flex items-center gap-2">
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
            : "Fine-tuned SmolLM runs in browser • OpenAI provides thoughts"}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-muted-foreground py-8">
            <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">Start a conversation</p>
            <p className="text-sm">
              Browser-based SmolLM processes OpenAI thoughts locally.
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
  );
}
