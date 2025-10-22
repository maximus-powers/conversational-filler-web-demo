import { streamText } from "ai";
import { google } from "@ai-sdk/google";

const PERSONAS: Record<string, string> = {
  educator: "You are an educator who explains concepts clearly and encourages learning.",
  customer_service: "You are a customer service agent focused on solving problems and being helpful.",
  event_planner: "You are an event planner who thinks about logistics, timing, and coordinating details.",
  medical: "You are a medical professional who provides thoughtful, empathetic health guidance.",
  coach: "You are a coach and advice counselor who motivates and provides supportive guidance.",
};

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const url = new URL(req.url);
    const personaKey = url.searchParams.get("persona");
    const personaDescription = personaKey && PERSONAS[personaKey] ? PERSONAS[personaKey] : null;

    if (!messages || messages.length === 0) {
      return new Response(
        JSON.stringify({
          error: "No messages provided",
        }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    const formattedMessages = messages.map((msg: any) => ({
      role: msg.role,
      content: msg.content
    }));

    const result = await streamText({
      model: google("gemini-2.5-flash"),
      messages: [
        {
          role: "system",
          content: `${personaDescription ? personaDescription + "\n\n" : ""}You are a helpful, engaging conversational AI assistant. You provide thoughtful, informative responses while maintaining a natural conversational tone and very concise. You aim to be helpful, accurate, and engaging in your responses.`
        },
        ...formattedMessages
      ],
      temperature: 1,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        for await (const chunk of result.textStream) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    });

    return new Response(stream, {
      status: 200,
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Transfer-Encoding": "chunked",
      },
    });

  } catch (error) {
    console.error("Gemini standalone error:", error);
    return new Response(
      JSON.stringify({
        error: "Failed to process request",
        details: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}