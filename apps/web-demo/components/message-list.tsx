"use client";

import { Bot, User, Loader2 } from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  processedContent?: string;
  thoughts?: string[];
  thoughtResponsePairs?: Array<{thought: string, response: string}>;
}

export function MessageList({
  messages,
  isLoading,
  showThoughts,
  title,
  isEmpty = false,
  emptyMessage = "No messages yet",
  showWelcome = false,
  modelLoading = false,
  modelLoadingProgress = ""
}: {
  messages: Message[];
  isLoading: boolean;
  showThoughts: boolean;
  title?: string;
  isEmpty?: boolean;
  emptyMessage?: string;
  showWelcome?: boolean;
  modelLoading?: boolean;
  modelLoadingProgress?: string;
}) {
  const renderMessageWithThoughts = (message: Message) => {
    if (!showThoughts || !message.thoughtResponsePairs || message.thoughtResponsePairs.length === 0) {
      return message.processedContent || message.content;
    }

    const pairs = message.thoughtResponsePairs;
    const elements: React.ReactNode[] = [];

    pairs.forEach((pair, index) => {
      if (pair.thought === "<|sil|>") {
        elements.push(
          <span key={`response-${index}`}>
            {pair.response}
            {index < pairs.length - 1 ? " " : ""}
          </span>
        );
      } else {
        elements.push(
          <span
            key={`thought-${index}`}
            className="bg-muted-foreground text-black px-1 mr-1 rounded"
          >
            {pair.thought}
            <span key={`arrow-${index}`} className="text-black pl-1">→</span>
          </span>
        );

        elements.push(
          <span key={`response-${index}`}>
            {pair.response}
          </span>
        );
        if (index < pairs.length - 1) {
          elements.push(<span key={`space-${index}`}> </span>);
        }
      }
    });

    return <>{elements}</>;
  };

  return (
    <div className="flex-1 overflow-y-auto bg-background flex flex-col">
      {title && (
        <h3 className="text-sm font-semibold text-muted-foreground mb-3 sticky top-0 bg-background z-10 px-4 py-3">
          {title}
        </h3>
      )}

      <div className="px-4 py-3 flex-1 flex flex-col">
        {isEmpty && (
          <div className="flex flex-col items-center justify-center flex-1 text-center text-muted-foreground">
            <Bot className={`${showWelcome ? "h-16 w-16 mb-4" : "h-12 w-12 mb-3"} opacity-20`} />
            {showWelcome && (
              <h2 className="text-xl font-medium mb-2">
                Welcome to the Conversational Filler
              </h2>
            )}
            <p className="text-sm max-w-md">{emptyMessage}</p>
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
              className={`flex gap-3 max-w-[85%] ${message.role === "user" ? "flex-row-reverse" : ""}`}
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
                  {renderMessageWithThoughts(message)}
                </p>
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[85%]">
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
      </div>
    </div>
  );
}
