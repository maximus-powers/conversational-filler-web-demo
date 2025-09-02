-- Remove A/B testing fields and add events field
ALTER TABLE "feedback_data" DROP COLUMN "ab_config";
ALTER TABLE "feedback_data" DROP COLUMN "ab_result";
ALTER TABLE "feedback_data" ADD COLUMN "events" json;