import { EventData } from '../event-tracker';

export const saveConversation = async ({
  conversationId,
  localModel,
  thoughtModel,
  voiceMode,
  userPrompt,
  aiResponse,
  eventData,
  questionnaireData,
  feedbackMode = false
}: {
  conversationId: string;
  localModel: string | null;
  thoughtModel: "gemini" | "none";
  voiceMode: boolean;
  userPrompt: string;
  aiResponse: string;
  eventData?: EventData | null;
  questionnaireData?: {
    answerQuality?: string;
    speedPerception?: string;
    rlfhResponse?: string;
    miscNotes?: string;
  };
  feedbackMode?: boolean;
}) => {
  try {
    // extract the full prompts from the events
    if (!eventData?.turns) {
      return [];
    }
    const fullPrompts = eventData.turns.flatMap(turn =>
      turn.timeline
      .filter(e => e.eventName === "LocalLMResponse" && e.prompt && e.response)
      .map(responseEvent => ({
        fullPrompt: responseEvent.prompt,
        assistantResponse: responseEvent.response,
      }))
    );

    const payload = {
      conversationId,
      localModel,
      thoughtModel,
      voiceMode,
      prompts: fullPrompts.length > 0 ? fullPrompts : [{
        fullPrompt: `<|im_start|>user\n${userPrompt}<|im_end|>\n`,
        assistantResponse: aiResponse,
      }],
      events: eventData,
      answerQuality: questionnaireData?.answerQuality || null,
      speedPerception: questionnaireData?.speedPerception || null,
      rlfhResponse: questionnaireData?.rlfhResponse || null,
      miscNotes: questionnaireData?.miscNotes || (feedbackMode ? null : 'Chat mode conversation'),
    };

    const response = await fetch("/api/save-turn-to-db", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      console.error('API Error:', errorData);
      throw new Error(`Failed to save conversation: ${errorData.error}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error("Error saving conversation:", error);
    throw error;
  }
};