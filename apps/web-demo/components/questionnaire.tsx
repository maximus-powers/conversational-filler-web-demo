"use client";

import { useState, useEffect } from "react";
import { Button } from "@convo-filler/ui/components/button";

type ModelConfig = {
  localModel: string | null;
  thoughtModel: "gemini" | "none";
};

type QuestionnaireProps = {
  originalResponse: string;
  alternativeResponse: string;
  originalPrompt: string;
  abConfig: ModelConfig;
  voiceMode: boolean;
  onSubmit: (data: {
    abResult: boolean;
    answerQuality: string;
    speedPerception: string;
    rlfhResponse: string;
    miscNotes: string | null;
  }) => void;
};

export function Questionnaire({
  originalResponse,
  alternativeResponse,
  originalPrompt,
  abConfig,
  voiceMode,
  onSubmit,
}: QuestionnaireProps) {
  const [abResponse, setAbResponse] = useState<string>("");
  const [abResult, setAbResult] = useState<boolean | null>(null);
  const [answerQuality, setAnswerQuality] = useState("");
  const [speedPerception, setSpeedPerception] = useState("");
  const [rlfhResponse, setRlfhResponse] = useState(originalResponse);
  const [miscNotes, setMiscNotes] = useState("");
  const [isGeneratingAB, setIsGeneratingAB] = useState(true);

  // Set the alternative response that was passed in
  useEffect(() => {
    setAbResponse(alternativeResponse);
    setIsGeneratingAB(false);
  }, [alternativeResponse]);

  const handleSubmit = () => {
    if (abResult === null || !answerQuality || !speedPerception) {
      alert("Please fill out all required fields.");
      return;
    }

    onSubmit({
      abResult,
      answerQuality,
      speedPerception,
      rlfhResponse,
      miscNotes: miscNotes.trim() || null,
    });
  };

  const gradeOptions = ["A", "B", "C", "D", "F"];

  return (
    <div className="p-6 mt-10 mb-20 bg-muted/10 border rounded-lg">
      <h3 className="text-xl font-semibold mb-4 text-foreground text-center">Please evaluate the response:</h3>
      <hr className="border-t border-black-600 my-6" />

      {/* Ratings - Side by Side */}
      <div className="grid grid-cols-2 gap-8">
        {/* Quality Rating */}
        <div>
          <h4 className="font-medium mb-4 text-center text-foreground">Response Quality Rating</h4>
          <div className="flex justify-center items-center">
            <div className="flex justify-between items-center w-full relative max-w-xs">
              <div className="absolute top-1/2 left-0 right-0 h-0.5 border-t-2 border-dashed border-muted-foreground -translate-y-1/2 z-0"></div>
              {gradeOptions.map((grade, index) => {
                const colors = [
                  'bg-red-500', // F
                  'bg-red-400', // D  
                  'bg-yellow-500', // C
                  'bg-lime-400', // B
                  'bg-green-500', // A
                ];
                const isSelected = answerQuality === grade;
                return (
                  <button
                    key={grade}
                    onClick={() => setAnswerQuality(grade)}
                    className={`relative z-10 w-10 h-10 rounded-full border-2 font-semibold text-white transition-all duration-200 hover:scale-110 ${
                      isSelected 
                        ? `${colors[gradeOptions.length - 1 - index]} border-white shadow-lg scale-110` 
                        : `${colors[gradeOptions.length - 1 - index]} border-gray-800`
                    }`}
                  >
                    {grade}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Speed Rating */}
        <div>
          <h4 className="font-medium mb-4 text-center text-foreground">Response Speed Rating</h4>
          <div className="flex justify-center items-center">
            <div className="flex justify-between items-center w-full relative max-w-xs">
              {/* Background line */}
              <div className="absolute top-1/2 left-0 right-0 h-0.5 border-t-2 border-dashed border-muted-foreground -translate-y-1/2 z-0"></div>
              {gradeOptions.map((grade, index) => {
                const colors = [
                  'bg-red-500', // F
                  'bg-red-400', // D  
                  'bg-yellow-500', // C
                  'bg-lime-400', // B
                  'bg-green-500', // A
                ];
                const isSelected = speedPerception === grade;
                return (
                  <button
                    key={grade}
                    onClick={() => setSpeedPerception(grade)}
                    className={`relative z-10 w-10 h-10 rounded-full border-2 font-semibold text-white transition-all duration-200 hover:scale-110 ${
                      isSelected 
                        ? `${colors[gradeOptions.length - 1 - index]} border-white shadow-lg scale-110` 
                        : `${colors[gradeOptions.length - 1 - index]} border-gray-800`
                    }`}
                  >
                    {grade}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <hr className="border-t border-black-600 my-6" />


      <div className="space-y-6">
        {/* A-B Testing Section */}
        <div>
          <h4 className="font-medium mb-3 text-foreground">Which response do you prefer?</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div 
                className={`p-4 border-2 rounded-lg text-sm text-foreground cursor-pointer transition-all duration-200 hover:shadow-md ${
                  abResult === true 
                    ? 'bg-muted/30 shadow-md' 
                    : 'bg-background border-border hover:border-blue-200'
                }`}
                onClick={() => setAbResult(true)}
              >
                {originalResponse}
              </div>
              <h5 className="text-sm font-medium mt-2 text-foreground text-center">Original</h5>

            </div>
            <div>
              <div 
                className={`p-4 border-2 rounded-lg text-sm text-foreground cursor-pointer transition-all duration-200 hover:shadow-md ${
                  abResult === false 
                    ? 'bg-muted/30 shadow-md' 
                    : 'bg-background border-border hover:border-white'
                } ${isGeneratingAB ? 'cursor-wait' : ''}`}
                onClick={() => !isGeneratingAB && setAbResult(false)}
              >
                {isGeneratingAB ? (
                  <div className="flex items-center">
                    <div className="animate-spin mr-2">⏳</div>
                    Generating alternative response...
                  </div>
                ) : (
                  abResponse
                )}
              </div>
              <h5 className="text-sm font-medium mt-2 text-foreground text-center">Alternative</h5>

            </div>
          </div>
          
        </div>
        

        {/* Divider */}
        <hr className="border-t border-black-600 my-6" />

        {/* Response Editing */}
        <div>
          <h4 className="font-medium mb-2 text-foreground">Improve Response</h4>
          <p className="text-sm text-muted-foreground mb-2">
            Edit the response below to fix any grammar issues or improve it:
          </p>
          <textarea
            value={rlfhResponse}
            onChange={(e) => setRlfhResponse(e.target.value)}
            className="w-full p-3 border rounded-md text-sm bg-background text-foreground"
            rows={4}
          />
        </div>

        {/* Divider */}
        <hr className="border-t border-black-600 my-6" />

        {/* Miscellaneous Notes */}
        <div>
          <h4 className="font-medium mb-2 text-foreground">Additional Notes (Optional)</h4>
          <textarea
            value={miscNotes}
            onChange={(e) => setMiscNotes(e.target.value)}
            placeholder="Any additional comments or observations..."
            className="w-full p-3 border rounded-md text-sm bg-background text-foreground placeholder:text-muted-foreground"
            rows={3}
          />
        </div>

        {/* Divider */}
        <hr className="border-t border-black-600 my-6" />

        {/* Submit Button */}
        <div className="flex justify-center">
          <Button 
            onClick={handleSubmit}
            disabled={abResult === null || !answerQuality || !speedPerception || isGeneratingAB}
          >
            Submit Feedback
          </Button>
        </div>
      </div>
    </div>
  );
}