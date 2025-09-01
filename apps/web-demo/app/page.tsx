"use client";
import { useState } from "react";
import { Chat } from "../components/chat";
import { HumanFeedbackMode } from "../components/human-feedback-mode";
import { Button } from "@convo-filler/ui/components/button";

type Mode = "demo" | "feedback";

export default function Page() {
  const [mode, setMode] = useState<Mode>("demo");

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <div className="flex-shrink-0 px-6 py-4 border-b bg-background">
        <div className="text-center space-y-4">
          <div>
            <h1 className="text-2xl font-bold">Conversational Filler Demo</h1>
            <p className="text-sm text-muted-foreground">
              Browser-based SmolLM processes OpenAI thoughts locally for natural
              conversation flow
            </p>
          </div>
          
          {/* Mode Toggle Buttons */}
          <div className="flex gap-2 justify-center">
            <Button
              variant={mode === "demo" ? "default" : "outline"}
              onClick={() => setMode("demo")}
            >
              Chat Demo
            </Button>
            <Button
              variant={mode === "feedback" ? "default" : "outline"}
              onClick={() => setMode("feedback")}
            >
              Human Feedback Mode
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-hidden p-6">
        <div className="h-full max-w-7xl mx-auto bg-white rounded-lg shadow-lg">
          {mode === "demo" ? <Chat /> : <HumanFeedbackMode />}
        </div>
      </div>
    </div>
  );
}
