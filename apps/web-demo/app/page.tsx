"use client";
import { Chat } from "../components/chat-browser-smollm";

export default function Page() {
  return (
    <div className="min-h-svh p-4">
      <div className="flex flex-col items-center justify-center gap-8 py-8">
        <div className="text-center space-y-4">
          <h1 className="text-3xl font-bold">AI Conversation Filler</h1>
          <p className="text-gray-600 text-center max-w-lg">
            Browser-based SmolLM processes OpenAI thoughts locally for natural
            conversation flow.
          </p>
        </div>

        <Chat />
      </div>
    </div>
  );
}
