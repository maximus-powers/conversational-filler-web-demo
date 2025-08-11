"use client";
import { Chat } from "../components/chat";

export default function Page() {
  return (
    <div className="h-screen flex flex-col">
      <div className="flex-shrink-0 px-6 py-4 border-b bg-background">
        <div className="text-center space-y-2">
          <h1 className="text-2xl font-bold">Conversational Filler Demo</h1>
          <p className="text-sm text-muted-foreground">
            Browser-based SmolLM processes OpenAI thoughts locally for natural
            conversation flow
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        <Chat />
      </div>
    </div>
  );
}
