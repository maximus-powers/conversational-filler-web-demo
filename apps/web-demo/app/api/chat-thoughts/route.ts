import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const conversationLines = [];
    for (const msg of messages) {
      if (msg.role === "user") {
        conversationLines.push(`User: ${msg.content}`);
      } else if (msg.role === "assistant") {
        const responseText = msg.processedContent || msg.content;
        if (responseText && responseText.trim()) {
          const cleanedResponse = responseText.trim() + " ";
          conversationLines.push(`Responder: ${cleanedResponse}`);
        }
      }
    }

    const conversationText = conversationLines.join("\n");

    const result = await streamText({
      model: openai("gpt-4o"),
      messages: [
        {
          role: "system",
          content: `Your job is to take the previous turns of the conversation and respond with distinct thoughts that could answer, separated by [bt], begin thought, and [et], end thought. The thoughts should be as short as possible while preserving meaning. Output only the spans of [bt] and [et].
First think of a good response, then summarize it. Be concise. Be proactive sometimes. Stay on topic.

Your distinct thoughts should be as if they were human thoughts, short, not full sentences but conveying the point of how you would continue an engaging conversation.
When you are done with all the thoughts, output the [done] token. They are NOT your internal thoughts, but rather the content of ONLY what you will say.

DO NOT exceed three thoughts.

Thought rules:
        * Thoughts should be hints about meaningful information
        * Questions that continue the conversation are meaningful
        * Advice can be meaningful
        * Giving recommendations when the user asks is meaningful
        * Explaining a concept can be meaningful
        * Demonstrate understanding

Do not have thoughts that:
        * Contain empathetic phrases
        * Paraphrase user words
        * Fill with useless words

Example Conversation:
User: Hey there, I just went to the park the other day and it was so nice!
Responder: Wow nice! The weather is getting nicer these days isn't it? What did you do there?
User: I was walking Buster and I took a few nice photos with the cherry blossoms, have you been?

Your thoughts for responding: Oh very nice! Can I see photos of the cherry blossoms? I personally haven't been to see them yet. I really hope I don't miss them!
Your response: [bt]Can I see photos?[et][bt]I haven't been yet.[et]Hope I don't miss.[et][done]

Here is the conversation:
${conversationText}`,
        },
        {
          role: "user",
          content: "Generate thoughts for this conversation.",
        },
      ],
      temperature: 1,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        let buffer = "";
        const thoughts: string[] = [];

        for await (const chunk of result.textStream) {
          buffer += chunk;

          // parse thoughts
          let startIndex = buffer.indexOf("[bt]");
          while (startIndex !== -1) {
            const endIndex = buffer.indexOf("[et]", startIndex);
            if (endIndex !== -1) {
              const thought = buffer.substring(startIndex + 4, endIndex).trim();
              if (thought && !thoughts.includes(thought)) {
                thoughts.push(thought);
                controller.enqueue(encoder.encode(`[bt]${thought}[et]`));
              }
              buffer = buffer.substring(endIndex + 4);
              startIndex = buffer.indexOf("[bt]");
            } else {
              // no complete thought yet
              break;
            }
          }

          if (buffer.includes("[done]")) {
            controller.enqueue(encoder.encode("[done]"));
            break;
          }
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
    console.error("AI SDK pipeline error:", error);
    return new Response(
      JSON.stringify({
        error: "Failed to process request",
        details: error instanceof Error ? error.message : "Unknown error",
        thoughts: [],
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
