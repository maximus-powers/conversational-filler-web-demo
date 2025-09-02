import { pgTable, uuid, text, json, boolean, timestamp } from 'drizzle-orm/pg-core';

export const feedbackData = pgTable('feedback_data', {
  id: uuid('id').primaryKey().defaultRandom(),
  conversationId: uuid('conversation_id').notNull(),
  localModel: text('local_model'), // nullable - can be null if not used
  thoughtModel: text('thought_model'), // 'gemini' or 'none'
  voiceMode: boolean('voice_mode').notNull(),
  prompts: json('prompts').notNull(), // array of {prompt, thought, generatedResponse}
  events: json('events'), // nullable - timeline events from the turn
  answerQuality: text('answer_quality').notNull(), // A, B, C, D, F
  speedPerception: text('speed_perception').notNull(), // A, B, C, D, F
  rlfhResponse: text('rlfh_response').notNull(), // user-edited response
  miscNotes: text('misc_notes'), // nullable
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export type FeedbackData = typeof feedbackData.$inferInsert;
export type FeedbackDataSelect = typeof feedbackData.$inferSelect;