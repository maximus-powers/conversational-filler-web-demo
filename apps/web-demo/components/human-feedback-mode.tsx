"use client";

import { useState } from "react";
import { Button } from "@convo-filler/ui/components/button";
import { Chat } from "./chat";

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
  abConfig: ModelConfig;
  voiceMode: boolean;
  prompts: Array<{
    prompt: string;
    thought: string | null;
    generatedResponse: string;
  }>;
};

const AB_CONFIG_PAIRS = [
  // text mode
  {
    voiceMode: false,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}, 
    alternative: { localModel: "HuggingFaceTB/SmolLM-360M-Instruct" as const, thoughtModel: "gemini" as const}
  },
  {
    voiceMode: false,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}, 
    alternative: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "none" as const}
  },
  {
    voiceMode: false,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "none" as const},
    alternative: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}
  },
  {
    voiceMode: false,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}, 
    alternative: { localModel: null, thoughtModel: "gemini" as const}
  },
  {
    voiceMode: false,
    original: { localModel: null, thoughtModel: "gemini" as const},
    alternative: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}
  },
  {
    voiceMode: false,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}, 
    alternative: { localModel: "HuggingFaceTB/SmolLM-360M-Instruct" as const, thoughtModel: "none" as const}
  },
  {
    voiceMode: false,
    original: { localModel: "HuggingFaceTB/SmolLM-360M-Instruct" as const, thoughtModel: "none" as const},
    alternative: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const} 
  },
  // voice mode
  {
    voiceMode: true,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}, 
    alternative: { localModel: "HuggingFaceTB/SmolLM-360M-Instruct" as const, thoughtModel: "gemini" as const}
  },
  {
    voiceMode: true,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}, 
    alternative: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "none" as const}
  },
  {
    voiceMode: true,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "none" as const},
    alternative: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}
  },
  {
    voiceMode: true,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}, 
    alternative: { localModel: null, thoughtModel: "gemini" as const}
  },
  {
    voiceMode: true,
    original: { localModel: null, thoughtModel: "gemini" as const},
    alternative: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}
  },
  {
    voiceMode: true,
    original: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}, 
    alternative: { localModel: "HuggingFaceTB/SmolLM-360M-Instruct" as const, thoughtModel: "none" as const}
  },
  {
    voiceMode: true,
    original: { localModel: "HuggingFaceTB/SmolLM-360M-Instruct" as const, thoughtModel: "none" as const},
    alternative: { localModel: "maximuspowers/smollm-convo-filler-onnx-official" as const, thoughtModel: "gemini" as const}
  },
];

interface HumanFeedbackModeProps {
  onShowQuestionnaire: (data: {
    originalResponse: string;
    alternativeResponse: string;
    originalPrompt: string;
    abConfig: ModelConfig;
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
  const [alternativeResponse, setAlternativeResponse] = useState<string>("");

  const getRandomConfigPair = (voiceMode: boolean) => {
    // Filter config pairs by voice mode
    const relevantPairs = AB_CONFIG_PAIRS.filter(pair => pair.voiceMode === voiceMode);
    return relevantPairs[Math.floor(Math.random() * relevantPairs.length)];
  };

  const generatePlaceholderAlternative = (originalResponse: string, abConfig: ModelConfig): string => {
    // TODO: Replace with actual model inference using abConfig
    // For now, just return failure message since we want real model differences
    return "Failed to generate alternative";
  };

  const startWorkflow = () => {
    setWorkflowState("chat");
    setConversationCount(0);
    startNewConversation();
  };

  const startNewConversation = () => {
    const isVoiceMode = conversationCount >= 3; // Last 2 conversations are voice mode
    const configPair = getRandomConfigPair(isVoiceMode);
    
    setCurrentConversation({
      conversationId: generateUUID(),
      config: configPair.original,
      abConfig: configPair.alternative,
      voiceMode: isVoiceMode,
      prompts: [],
    });
    
    setWorkflowState("chat");
  };

  const handleTurnComplete = (prompts: ConversationData["prompts"]) => {
    if (currentConversation) {
      setCurrentConversation({
        ...currentConversation,
        prompts,
      });
      
      // Capture the latest AI response for the questionnaire
      let altResponse = "";
      if (prompts.length > 0) {
        setLastAiResponse(prompts[prompts.length - 1].generatedResponse);
        
        // Generate alternative response (placeholder for now - should use alternative model config)
        const originalResponse = prompts[prompts.length - 1].generatedResponse;
        altResponse = generatePlaceholderAlternative(originalResponse, currentConversation.abConfig);
        setAlternativeResponse(altResponse);
      }
      
      // Show questionnaire using the parent callback
      onShowQuestionnaire({
        originalResponse: prompts.length > 0 ? prompts[prompts.length - 1].generatedResponse : "",
        alternativeResponse: altResponse,
        originalPrompt: prompts.length > 0 ? prompts[prompts.length - 1].prompt : "",
        abConfig: currentConversation.abConfig,
        voiceMode: currentConversation.voiceMode,
        onSubmit: handleQuestionnaireSubmit,
      });
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
      abConfig: currentConversation.abConfig,
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

      // Move to next conversation or complete
      const newCount = conversationCount + 1;
      setConversationCount(newCount);
      
      if (newCount >= 5) {
        setWorkflowState("complete");
      } else {
        // Wait a moment before starting new conversation
        setTimeout(() => {
          startNewConversation();
        }, 1000);
      }
    } catch (error) {
      console.error("Error saving feedback:", error);
      // Handle error appropriately
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
            <p className="text-foreground">Welcome to our research study! Here's what you'll be doing:</p>
            <ul className="list-disc list-inside space-y-2 text-foreground">
              <li>You'll have <strong>5 conversations</strong> with our AI system</li>
              <li>The first <strong>3 conversations</strong> will be in text mode</li>
              <li>The last <strong>2 conversations</strong> will be in voice mode</li>
              <li>Each conversation should be <strong>2-3 turns</strong> long</li>
              <li>After each turn, you'll fill out a brief questionnaire</li>
              <li>We'll randomly select different AI configurations for each conversation</li>
            </ul>
            <p className="text-foreground">Your responses will help us improve our conversational AI system. Thank you for participating!</p>
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
          <p className="text-foreground">You've completed all 5 conversations. Your feedback has been recorded and will help us improve our AI system.</p>
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
            key={currentConversation.conversationId} // Force remount on new conversation
            feedbackMode={true}
            config={currentConversation.config}
            voiceMode={currentConversation.voiceMode}
            onTurnComplete={handleTurnComplete}
            disabled={questionnaireActive} // Disable chat input when questionnaire is shown
          />
        </div>
      </div>
    );
  }

  return null;
}