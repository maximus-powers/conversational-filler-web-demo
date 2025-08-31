export type EventName = 
  | "VoiceDetectionStart"
  | "VoiceDetectionEnd"
  | "STTStart"
  | "STTEnd"
  | "UserInputReceived"
  | "ThoughtApiSubmit"
  | "ThoughtApiFirstToken"
  | "LocalLMSubmit"
  | "LocalLMResponse"
  | "TTSStart"
  | "TTSEnd"
  | "ThoughtParsed";

export interface TimelineEvent {
  eventName: EventName;
  timestamp: string;
  text?: string;
  prompt?: string;
  response?: string;
}

export interface TurnMetadata {
  localModel: "smollm-finetuned" | "smollm-base";
  thoughtModel: "gemini-flash-2.0" | "none";
  voiceMode: boolean;
}

export interface Turn {
  metadata: TurnMetadata;
  timeline: TimelineEvent[];
}

export interface EventData {
  turns: Turn[];
}

export class EventTracker {
  private data: EventData = { turns: [] };
  private currentTurn: Turn | null = null;

  startNewTurn(metadata: TurnMetadata): void {
    const newTurn: Turn = {
      metadata,
      timeline: []
    };
    this.data.turns.push(newTurn);
    this.currentTurn = newTurn;
  }

  addEvent(eventName: EventName, additionalData?: { text?: string; prompt?: string; response?: string }): void {
    if (!this.currentTurn) {
      throw new Error("No active turn. Call startNewTurn() first.");
    }
    const event: TimelineEvent = {
      eventName,
      timestamp: new Date().toISOString(),
      ...additionalData
    };
    this.currentTurn.timeline.push(event);
  }

  getData(): EventData {
    return { ...this.data, turns: [...this.data.turns] };
  }

  getCurrentTurn(): Turn | null {
    return this.currentTurn ? { ...this.currentTurn, timeline: [...this.currentTurn.timeline] } : null;
  }

  reset(): void {
    this.data = { turns: [] };
    this.currentTurn = null;
  }

  getCurrentTurnEvents(): TimelineEvent[] {
    return this.currentTurn ? [...this.currentTurn.timeline] : [];
  }

  hasActiveTurn(): boolean {
    return this.currentTurn !== null;
  }
}