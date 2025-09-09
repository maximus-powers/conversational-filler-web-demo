import { pgTable, uuid, text, json, boolean, timestamp } from 'drizzle-orm/pg-core';

export const feedbackData = pgTable('feedback_data', {
  id: uuid('id').primaryKey().defaultRandom(),
  conversationId: uuid('conversation_id').notNull(),
  localModel: text('local_model'),
  thoughtModel: text('thought_model'), // 'gemini' or 'none'
  voiceMode: boolean('voice_mode').notNull(),
  prompts: json('prompts').notNull(), // array of {prompt, thought, generatedResponse}
  events: json('events'),
  answerQuality: text('answer_quality'), // A, B, C, D, F (null for chat demo)
  speedPerception: text('speed_perception'), // A, B, C, D, F (null for chat demo)
  rlfhResponse: text('rlfh_response'), // user-edited response (null for chat demo)
  miscNotes: text('misc_notes'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

export type FeedbackData = typeof feedbackData.$inferInsert;
export type FeedbackDataSelect = typeof feedbackData.$inferSelect;