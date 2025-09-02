import { NextRequest, NextResponse } from 'next/server';
import { createDbClient, feedbackData } from '@convo-filler/database';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const {
      conversationId,
      localModel,
      thoughtModel,
      voiceMode,
      prompts,
      abConfig,
      abResult,
      answerQuality,
      speedPerception,
      rlfhResponse,
      miscNotes,
    } = body;

    if (
      !conversationId ||
      thoughtModel === undefined ||
      voiceMode === undefined ||
      !prompts ||
      !abConfig ||
      abResult === undefined ||
      !answerQuality ||
      !speedPerception ||
      !rlfhResponse
    ) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    if (!process.env.DATABASE_URL) {
      return NextResponse.json(
        { error: 'Database not configured' },
        { status: 500 }
      );
    }

    const db = createDbClient(process.env.DATABASE_URL);
    const result = await db.insert(feedbackData).values({
      conversationId,
      localModel,
      thoughtModel,
      voiceMode,
      prompts,
      abConfig,
      abResult,
      answerQuality,
      speedPerception,
      rlfhResponse,
      miscNotes,
    }).returning();

    return NextResponse.json({ success: true, id: result[0].id });
  } catch (error) {
    console.error('Error saving feedback:', error);
    return NextResponse.json(
      { error: 'Failed to save feedback' },
      { status: 500 }
    );
  }
}