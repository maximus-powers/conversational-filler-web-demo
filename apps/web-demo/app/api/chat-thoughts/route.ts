import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    const result = streamText({
      model: openai("gpt-4o"),
      messages,
      system: `You are an expert assistant feeding advanced academic thoughts to a speaker in a conversation. You'll receive questions from the user, and should supply concise academic thoughts to another agent, who will weave your supplied content into the conversation naturally. Your mission is to provide the most relevant and supportive context possible, in bite-sized chunks, to assist the agent who responds.
        CRITICAL: You MUST wrap every complete thought in [bt] and [et] markers. You MUST use <|sil|> tokens between thoughts for natural pauses.
        Example format:
        [bt]Your first complete thought about the topic[et] <|sil|> [bt]Your second complete thought with more details[et] <|sil|> [bt]Your final thought or conclusion[et]`,
      temperature: 0.8,
    });
    return result.toTextStreamResponse();
  } catch (error) {
    console.error("AI SDK pipeline error:", error);
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
