"use client";

import { Button } from "@convo-filler/ui/components/button";
import { MessageSquare, Mic } from "lucide-react";
import { AppMode } from "../app/lib/unified-pipeline";

export function ModeSwitcher({
  currentMode,
  onModeChange,
  disabled,
}: {
  currentMode: AppMode;
  onModeChange: (mode: AppMode) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-1 p-1 bg-muted rounded-lg">
      <Button
        variant={currentMode === "text" ? "default" : "ghost"}
        size="sm"
        onClick={() => onModeChange("text")}
        disabled={disabled}
        className="h-7 px-2"
      >
        <MessageSquare className="h-3 w-3 mr-1" />
        <span className="text-xs font-medium">Text</span>
      </Button>
      <Button
        variant={currentMode === "voice" ? "default" : "ghost"}
        size="sm"
        onClick={() => onModeChange("voice")}
        disabled={disabled}
        className="h-7 px-2"
      >
        <Mic className="h-3 w-3 mr-1" />
        <span className="text-xs font-medium">Voice</span>
      </Button>
    </div>
  );
}
