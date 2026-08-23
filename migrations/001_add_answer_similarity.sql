ALTER TABLE ragas_evaluations
ADD COLUMN IF NOT EXISTS answer_similarity NUMERIC;
