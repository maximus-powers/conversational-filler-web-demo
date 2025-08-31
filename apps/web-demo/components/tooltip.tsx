"use client";

import React, { useState, useRef } from "react";

interface TooltipProps {
  content: string | React.ReactNode;
  children: React.ReactNode;
  className?: string;
  preserveChildPositioning?: boolean;
}

export function Tooltip({ content, children, className = "", preserveChildPositioning = false }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseEnter = (e: React.MouseEvent) => {
    setPosition({
      x: e.clientX,
      y: e.clientY - 10,
    });
    setIsVisible(true);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isVisible) {
      setPosition({
        x: e.clientX,
        y: e.clientY - 10,
      });
    }
  };

  const handleMouseLeave = () => {
    setIsVisible(false);
  };

  return (
    <>
      <div
        ref={containerRef}
        onMouseEnter={handleMouseEnter}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className={`${preserveChildPositioning ? '' : 'relative cursor-pointer'} ${className}`}
      >
        {children}
      </div>
      
      {isVisible && (
        <div
          className="fixed z-50 px-3 py-2 text-xs bg-muted text-foreground rounded-lg shadow-lg pointer-events-none whitespace-pre-line max-w-xs border"
          style={{
            left: position.x,
            top: position.y,
            transform: 'translate(-50%, -100%)',
          }}
        >
          {content}
          {/* Arrow */}
          <div
            className="absolute top-full left-1/2 transform -translate-x-1/2"
            style={{
              width: 0,
              height: 0,
              borderLeft: '4px solid transparent',
              borderRight: '4px solid transparent',
              borderTop: '4px solid hsl(var(--muted))',
            }}
          />
        </div>
      )}
    </>
  );
}