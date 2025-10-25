import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const audioFile = formData.get('file');

    if (!audioFile) {
      return NextResponse.json({ error: 'Audio file is required' }, { status: 400 });
    }

    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: 'ElevenLabs API key not configured' }, { status: 500 });
    }

    const elevenlabsFormData = new FormData();
    elevenlabsFormData.append('file', audioFile);
    elevenlabsFormData.append('model_id', 'scribe_v1');

    const response = await fetch(
      'https://api.elevenlabs.io/v1/speech-to-text',
      {
        method: 'POST',
        headers: {
          'xi-api-key': apiKey,
          'Accept': 'application/json',
        },
        body: elevenlabsFormData,
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error('ElevenLabs STT API error:', response.status, errorText);
      return NextResponse.json(
        { error: 'Failed to transcribe audio', details: errorText },
        { status: response.status }
      );
    }
    const result = await response.json();

    return NextResponse.json({
      transcript: result.text || '',
      language: result.language_code || 'en',
      words: result.words || [],
    });
  } catch (error) {
    console.error('Error in ElevenLabs STT:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: String(error) },
      { status: 500 }
    );
  }
}
