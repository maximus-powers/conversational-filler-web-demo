import { streamText } from "ai";
import { google } from "@ai-sdk/google";

// endpoint for streaming gemini without local model
export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    
    const formattedMessages = messages.map((msg: any) => ({
      role: msg.role,
      content: msg.content
    }));

    const result = await streamText({
      model: google("gemini-2.5-flash"),
      messages: [
        {
          role: "system",
          content: `You are a helpful, engaging conversational AI assistant. You provide thoughtful, informative responses while maintaining a natural conversational tone. You aim to be helpful, accurate, and engaging in your responses.

Key guidelines:
- Be conversational and natural
- Provide helpful and accurate information
- Ask follow-up questions when appropriate to keep the conversation engaging
- Be concise in your responses
- Show understanding of the user's context and needs
- Maintain a friendly and professional tone`
        },
        ...formattedMessages
      ],
      temperature: 0.7,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        let firstTokenSent = false;
        for await (const chunk of result.textStream) {
          if (!firstTokenSent) {
            controller.enqueue(encoder.encode("[first_token]"));
            firstTokenSent = true;
          }
          controller.enqueue(encoder.encode(chunk));
        }
        controller.enqueue(encoder.encode("[done]"));
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