# Known-Bad Conversations

Test fixtures that deliberately trigger metric failures, used to validate that the evaluation framework correctly detects problems.

Run with:
```bash
python evaluate.py --check
```

## Files

| File | Primary Target | Expected Secondary Failures |
|------|---------------|---------------------------|
| `fail_response_accuracy_hallucination.json` | Response Accuracy (hallucination) | Chunk Alignment (actual has extra health retreat chunk not in expected) |
| `fail_response_accuracy_contradiction.json` | Response Accuracy (contradiction) | None (all chunk facts covered, one detail contradicted: "fragile and prone to chipping" vs "Indestructible") |
| `fail_response_completeness.json` | Response Completeness | None (one accurate claim, just incomplete) |
| `fail_factual_consistency.json` | Response Accuracy (contradiction) | None (uses correct terminology, only the number is wrong) |
| `fail_answer_relevance.json` | Answer Relevance | None (accurate and complete per chunks, but answers plan features instead of the cost question asked) |
| `fail_chunk_count.json` | ChunkCountMetric | None (correct answer, just too many chunks retrieved) |
| `fail_chunk_deduplication.json` | ChunkDeduplicationMetric | None |
| `fail_chunk_alignment.json` | Chunk Alignment | Faithfulness (response grounded in expected chunks; actual chunks are different health content) |
| `fail_contextual_recall.json` | ContextualRecall | Chunk Alignment (actual partial chunk ≠ full expected chunk) |
| `fail_contextual_precision.json` | ContextualPrecision | None (all chunks are health-related so Contextual Relevancy passes; best chunk is ranked last so Precision fails) |
| `fail_contextual_relevancy.json` | ContextualRelevancy | ContextualRecall (actual chunks don't cover expected output) |
| `fail_faithfulness.json` | FaithfulnessMetric | None (Stage 1 passes — response matches expected chunks) |
| `hr_benefits_test_fail.json` | Response Accuracy, Response Completeness | None (no actual_rag_content so Stage 2 does not run; response fabricates generic HR benefits and completely ignores the FantaCo-specific chunks) |
