"use client";

import { useState } from "react";
import { Button } from "@convo-filler/ui/components/button";
import { Chat } from "./chat";
import { EventData } from "../app/lib/event-tracker";

type WorkflowState = "intro" | "chat" | "questionnaire" | "complete";

function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

type ModelConfig = {
  localModel: string | null;
  thoughtModel: "gemini" | "none";
};

type ConversationData = {
  conversationId: string;
  config: ModelConfig;
  voiceMode: boolean;
  prompts: Array<{
    prompt: string;
    thought: string | null;
    generatedResponse: string;
  }>;
  events?: EventData;
};

const MODEL_CONFIGS = [
  { localModel: "maximuspowers/smollm-convo-filler-onnx-official", thoughtModel: "gemini" as const },
  { localModel: "maximuspowers/smollm-convo-filler-onnx-official", thoughtModel: "none" as const },
  { localModel: "HuggingFaceTB/SmolLM-360M-Instruct", thoughtModel: "gemini" as const },
  { localModel: "HuggingFaceTB/SmolLM-360M-Instruct", thoughtModel: "none" as const },
  { localModel: null, thoughtModel: "gemini" as const },
];

interface HumanFeedbackModeProps {
  onShowQuestionnaire: (data: {
    originalResponse: string;
    originalPrompt: string;
    voiceMode: boolean;
    onSubmit: (data: any) => void;
  }) => void;
  questionnaireActive: boolean;
}

export function HumanFeedbackMode({ 
  onShowQuestionnaire,
  questionnaireActive 
}: HumanFeedbackModeProps) {
  const [workflowState, setWorkflowState] = useState<WorkflowState>("intro");
  const [conversationCount, setConversationCount] = useState(0);
  const [currentConversation, setCurrentConversation] = useState<ConversationData | null>(null);
  const [lastAiResponse, setLastAiResponse] = useState<string>("");

  const getRandomConfig = (): ModelConfig => {
    return MODEL_CONFIGS[Math.floor(Math.random() * MODEL_CONFIGS.length)];
  };

  const startWorkflow = () => {
    setWorkflowState("chat");
    setConversationCount(0);
    startNewConversation();
  };

  const startNewConversation = () => {
    const newCount = conversationCount + 1;
    
    if (newCount >= 5) {
      setWorkflowState("complete");
      return;
    }
    
    setConversationCount(newCount);
    const isVoiceMode = newCount >= 3; // last 2 conversations are voice mode
    const config = getRandomConfig();
    
    setCurrentConversation({
      conversationId: generateUUID(),
      config: config,
      voiceMode: isVoiceMode,
      prompts: [],
    });
    
    setWorkflowState("chat");
  };

  const handleTurnComplete = (prompts: ConversationData["prompts"], events?: EventData) => {
    if (currentConversation) {
      setCurrentConversation({
        ...currentConversation,
        prompts,
        events,
      });
      
      if (prompts.length > 0) {
        const latestPrompt = prompts[prompts.length - 1];
        setLastAiResponse(latestPrompt.generatedResponse);
        
        onShowQuestionnaire({
          originalResponse: latestPrompt.generatedResponse,
          originalPrompt: latestPrompt.prompt,
          voiceMode: currentConversation.voiceMode,
          onSubmit: handleQuestionnaireSubmit,
        });
      }
    }
  };

  const handleQuestionnaireSubmit = async (data: any) => {
    if (!currentConversation) return;

    const payload = {
      conversationId: currentConversation.conversationId,
      localModel: currentConversation.config.localModel,
      thoughtModel: currentConversation.config.thoughtModel,
      voiceMode: currentConversation.voiceMode,
      prompts: currentConversation.prompts,
      events: currentConversation.events,
      ...data,
    };

    console.log('Submitting feedback payload:', payload);

    // Save feedback data to database
    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        console.error('API Error:', errorData);
        throw new Error(`Failed to save feedback: ${errorData.error}`);
      }
      
      const result = await response.json();
      console.log('Feedback saved successfully:', result);

      setWorkflowState("chat");
    } catch (error) {
      console.error("Error saving feedback:", error);
    }
  };

  const resetWorkflow = () => {
    setWorkflowState("intro");
    setConversationCount(0);
    setCurrentConversation(null);
  };

  if (workflowState === "intro") {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <div className="max-w-2xl text-center space-y-6">
          <h2 className="text-3xl font-bold text-foreground">Human Feedback Data Collection</h2>
          <div className="space-y-4 text-left">
            <p className="text-foreground">Welcome to our research study! Here&apos;s what you&apos;ll be doing:</p>
            <ul className="list-disc list-inside space-y-2 text-foreground">
              <li>You&apos;ll have <strong>5 self-led conversations</strong> in our chat interface</li>
              <li>The first <strong>3 conversations</strong> will be in text mode</li>
              <li>The last <strong>2 conversations</strong> will be in voice mode</li>
              <li>Aim for each conversation to be <strong>2-3 turns</strong> long</li>
              <li>After each turn, you&apos;ll fill out a brief questionnaire</li>
              <li>We&apos;ll randomly select different architecture configurations for each conversation</li>
            </ul>
            <p className="text-foreground">Your feedback will be used to improve our conversational model, and for our study into the quality/speed of this novel architecture. Thank you for participating!</p>
          </div>
          <Button onClick={startWorkflow} size="lg">
            Start Data Collection
          </Button>
        </div>
      </div>
    );
  }

  if (workflowState === "complete") {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <div className="max-w-2xl text-center space-y-6">
          <h2 className="text-3xl font-bold text-foreground">Thank You!</h2>
          <p className="text-foreground">You&apos;ve completed 5 conversations. Your feedback has been recorded and will be used in our study comparing the various architectures you tested. Thanks!</p>
          <Button onClick={resetWorkflow} variant="outline">
            Start Over
          </Button>
        </div>
      </div>
    );
  }

  if (workflowState === "chat" && currentConversation) {
    return (
      <div className="h-full flex flex-col">
        {/* Progress Counter */}
        <div className="p-4 border-b bg-muted/10">
          <div className="flex justify-between items-center">
            <div className="text-sm text-muted-foreground">
              Conversation {conversationCount + 1} of 5 • {currentConversation.voiceMode ? "Voice Mode" : "Text Mode"}
            </div>
            <Button onClick={startNewConversation} variant="outline" size="sm">
              Start New Conversation
            </Button>
          </div>
        </div>

        {/* Chat Section */}
        <div className="h-full">
          <Chat
            key={currentConversation.conversationId}
            feedbackMode={true}
            config={currentConversation.config}
            voiceMode={currentConversation.voiceMode}
            onTurnComplete={handleTurnComplete}
            disabled={questionnaireActive}
          />
        </div>
      </div>
    );
  }

  return null;
}