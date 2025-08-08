"use client";
import { Chat } from "../components/chat";

export default function Page() {
  return (
    <div className="min-h-svh p-4">
      <div className="flex flex-col gap-8 py-8">
        <div className="text-center space-y-4">
          <h1 className="text-3xl font-bold">Conversational Filler Demo</h1>
          <p className="text-gray-600 text-center">
            Browser-based SmolLM processes OpenAI thoughts locally for natural conversation flow. (we can put links to paper/code here)
          </p>
        </div>

        <div className="flex justify-center">
          <div className="flex max-w-6xl w-full h-[600px] border rounded-lg bg-background overflow-hidden">
            <Chat />
          </div>
        </div>
      </div>
    </div>
  );
}
