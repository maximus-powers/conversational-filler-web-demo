CREATE TABLE "feedback_data" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"conversation_id" uuid NOT NULL,
	"local_model" text,
	"thought_model" text,
	"voice_mode" boolean NOT NULL,
	"prompts" json NOT NULL,
	"ab_config" json NOT NULL,
	"ab_result" boolean NOT NULL,
	"answer_quality" text NOT NULL,
	"speed_perception" text NOT NULL,
	"rlfh_response" text NOT NULL,
	"misc_notes" text,
	"created_at" timestamp DEFAULT now() NOT NULL
);
