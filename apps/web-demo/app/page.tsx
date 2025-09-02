"use client";
import { useState } from "react";
import { Chat } from "../components/chat";
import { HumanFeedbackMode } from "../components/human-feedback-mode";
import { Questionnaire } from "../components/questionnaire";
import { Button } from "@convo-filler/ui/components/button";

type Mode = "demo" | "feedback";

type ModelConfig = {
  localModel: string | null;
  thoughtModel: "gemini" | "none";
};

export default function Page() {
  const [mode, setMode] = useState<Mode>("demo");
  const [showQuestionnaire, setShowQuestionnaire] = useState(false);
  const [questionnaireData, setQuestionnaireData] = useState<{
    originalResponse: string;
    alternativeResponse: string;
    originalPrompt: string;
    abConfig: ModelConfig;
    voiceMode: boolean;
    onSubmit: (data: any) => void;
  } | null>(null);

  const handleShowQuestionnaire = (data: {
    originalResponse: string;
    alternativeResponse: string;
    originalPrompt: string;
    abConfig: ModelConfig;
    voiceMode: boolean;
    onSubmit: (data: any) => void;
  }) => {
    setQuestionnaireData(data);
    setShowQuestionnaire(true);
  };

  const handleQuestionnaireSubmit = (data: any) => {
    if (questionnaireData) {
      questionnaireData.onSubmit(data);
    }
    setShowQuestionnaire(false);
    setQuestionnaireData(null);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="flex-shrink-0 px-6 py-4 border-b bg-background">
        <div className="text-center space-y-4">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Conversational Filler Demo</h1>
            <p className="text-sm text-muted-foreground">
              Browser-based SmolLM processes Gemini thoughts locally for natural
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

      <div className="p-6 bg-muted/20">
        <div className="max-w-7xl mx-auto bg-background rounded-lg shadow-lg border" style={{height: '70vh'}}>
          {mode === "demo" ? (
            <Chat />
          ) : (
            <HumanFeedbackMode 
              onShowQuestionnaire={handleShowQuestionnaire}
              questionnaireActive={showQuestionnaire}
            />
          )}
        </div>
      </div>

      {/* Questionnaire during rlhf mode */}
      {showQuestionnaire && questionnaireData && (
        <div className="bg-background border-t">
          <div className="max-w-7xl mx-auto">
            <Questionnaire
              originalResponse={questionnaireData.originalResponse}
              alternativeResponse={questionnaireData.alternativeResponse}
              originalPrompt={questionnaireData.originalPrompt}
              abConfig={questionnaireData.abConfig}
              voiceMode={questionnaireData.voiceMode}
              onSubmit={handleQuestionnaireSubmit}
            />
          </div>
        </div>
      )}
    </div>
  );
}
