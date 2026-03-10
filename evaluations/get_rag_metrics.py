"""
DeepEval metrics configuration for RAG conversation assessment.

This module defines evaluation metrics specifically designed for RAG
(Retrieval-Augmented Generation) conversations, focusing on retrieval
quality and response accuracy.
"""

from typing import Any, List, Optional

from deepeval.metrics import (
    ConversationalGEval,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, TurnParams


class ChunkCountMetric(BaseMetric):
    """
    Custom metric to validate that the number of retrieved chunks does not exceed the limit.

    This ensures the RAG system respects the configured chunk limit.
    """

    def __init__(self, max_chunks: int, threshold: float = 1.0):
        """
        Initialize the ChunkCountMetric.

        Args:
            max_chunks: Maximum allowed number of chunks
            threshold: Success threshold (default 1.0, must be perfect)
        """
        self.max_chunks = max_chunks
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    @property
    def __name__(self) -> str:
        return "Chunk Count Limit"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """
        Measure if the chunk count is within the limit.

        Args:
            test_case: The test case containing additional_metadata with original_chunk_count

        Returns:
            float: 1.0 if within limit, 0.0 if exceeds limit
        """
        # Get the original chunk count before limiting (stored in metadata)
        actual_count = 0
        if hasattr(test_case, "additional_metadata") and test_case.additional_metadata:
            actual_count = test_case.additional_metadata.get("original_chunk_count", 0)

        # Fallback to retrieval_context length if metadata not available
        if actual_count == 0:
            actual_count = (
                len(test_case.retrieval_context) if test_case.retrieval_context else 0
            )

        if actual_count <= self.max_chunks:
            self.score = 1.0
            self.success = True
            self.reason = f"Retrieved {actual_count} chunks, which is within the limit of {self.max_chunks}"
        else:
            self.score = 0.0
            self.success = False
            self.reason = f"Retrieved {actual_count} chunks, which exceeds the limit of {self.max_chunks}"

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Async version of measure (just calls synchronous version)."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Check if the metric passed."""
        return self.success


class ChunkDeduplicationMetric(BaseMetric):
    """
    Custom metric to detect near-duplicate chunks in retrieval results.

    Uses word-level Jaccard similarity to identify chunks that contain
    substantially overlapping content, which indicates redundant retrieval.
    """

    def __init__(self, similarity_threshold: float = 0.8, threshold: float = 1.0):
        """
        Initialize the ChunkDeduplicationMetric.

        Args:
            similarity_threshold: Jaccard similarity above which two chunks
                                  are considered duplicates (default 0.8)
            threshold: Success threshold for the metric score (default 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    @property
    def __name__(self) -> str:
        return "Chunk Deduplication"

    @staticmethod
    def _tokenize(text: str) -> set:
        """Tokenize text into a set of lowercased words."""
        return set(text.lower().split())

    @staticmethod
    def _jaccard_similarity(set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        union = set1 | set2
        return len(set1 & set2) / len(union) if union else 0.0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """
        Measure chunk deduplication quality.

        Compares all pairs of retrieved chunks using Jaccard similarity.
        Score is 1.0 if no duplicates are found, decreasing with more duplicates.

        Args:
            test_case: The test case containing retrieval_context chunks

        Returns:
            float: Score from 0.0 to 1.0 (1.0 = no duplicates)
        """
        chunks = test_case.retrieval_context or []

        if len(chunks) <= 1:
            self.score = 1.0
            self.success = True
            self.reason = f"Only {len(chunks)} chunk(s), no duplicates possible"
            return self.score

        tokenized = [self._tokenize(chunk) for chunk in chunks]

        duplicate_pairs = []
        for i in range(len(tokenized)):
            for j in range(i + 1, len(tokenized)):
                sim = self._jaccard_similarity(tokenized[i], tokenized[j])
                if sim >= self.similarity_threshold:
                    duplicate_pairs.append((i + 1, j + 1, sim))

        if not duplicate_pairs:
            self.score = 1.0
            self.success = True
            self.reason = f"No duplicate chunks found among {len(chunks)} chunks"
        else:
            total_pairs = len(chunks) * (len(chunks) - 1) // 2
            self.score = 1.0 - (len(duplicate_pairs) / total_pairs)
            self.success = self.score >= self.threshold
            pair_desc = [
                f"chunks {i} and {j} ({sim:.0%} similar)"
                for i, j, sim in duplicate_pairs[:5]
            ]
            desc = "; ".join(pair_desc)
            if len(duplicate_pairs) > 5:
                desc += f" ... and {len(duplicate_pairs) - 5} more"
            self.reason = f"Found {len(duplicate_pairs)} duplicate pair(s) among {len(chunks)} chunks: {desc}"

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Async version of measure (just calls synchronous version)."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Check if the metric passed."""
        return self.success


def get_rag_metrics(
    custom_model: Optional[DeepEvalBaseLLM] = None,
) -> List[Any]:
    """
    Create evaluation metrics for RAG conversation assessment.

    This function defines conversational metrics specifically designed
    to evaluate RAG system performance, including retrieval quality
    and response accuracy.

    Args:
        custom_model: Optional custom LLM model instance to use for evaluations.
                     If None, uses the default DeepEval model.

    Returns:
        List[ConversationalGEval]: List of evaluation metrics including:
            - Retrieval Relevance: Measures if retrieved content is relevant to the query
            - Response Accuracy: Evaluates if the response accurately uses retrieved content
            - Completeness: Assesses if the response fully answers the question
            - Factual Consistency: Checks if response facts match the retrieved content
    """
    metrics = []

    # Metric 2: Response Accuracy
    # Evaluates if the assistant's response accurately reflects the retrieved content,
    # including catching both hallucinated claims and direct contradictions.
    metrics.append(
        ConversationalGEval(
            name="Response Accuracy",
            criteria="Response Accuracy - Every factual claim in the assistant's response must be verifiable against the retrieval_context and must not contradict it. This metric does NOT evaluate whether the response answers the user's question.",
            evaluation_params=[
                TurnParams.CONTENT,
                TurnParams.ROLE,
                TurnParams.RETRIEVAL_CONTEXT,
            ],
            evaluation_steps=[
                "IMPORTANT: This metric evaluates ONLY whether the claims in the response are accurate relative to the retrieval_context. Do NOT evaluate whether the response answers the user's question - that is the sole concern of the Answer Relevance metric.",
                "Review the retrieval_context provided with the user's turn - this is the source material",
                "If retrieval_context is empty or not provided, the assistant should NOT fabricate an answer - acknowledging that the information is not available is accurate and should score highly",
                "If retrieval_context is present, identify all factual claims made in the assistant's response",
                "For each claim, check whether it is supported by the retrieval_context",
                "Also check for direct contradictions: if the response states a different number, name, or fact than what appears in the retrieval_context (e.g. '10 units' vs '365 units'), that claim is inaccurate and must be penalised severely",
                "A single contradicted or hallucinated fact is sufficient to score this metric low - do not average it against other correct claims",
                "Score highly only if all claims in the response are supported by and consistent with the retrieval_context",
            ],
            threshold=0.7,
            model=custom_model,
        )
    )

    # Metric 3: Completeness
    # Assesses whether the response covers the key information present in the retrieved chunks
    metrics.append(
        ConversationalGEval(
            name="Response Completeness",
            criteria="Context Coverage - The assistant's response should cover the key facts present in the retrieval_context. Evaluate this independently of whether the response answers the user's question.",
            evaluation_params=[
                TurnParams.CONTENT,
                TurnParams.ROLE,
                TurnParams.RETRIEVAL_CONTEXT,
            ],
            evaluation_steps=[
                "IMPORTANT: Ignore the user's question entirely for this metric. Do not consider whether the response answers the question.",
                "If retrieval_context is empty or not provided, score highly if the response states that information is not available",
                "If retrieval_context is present, list the key facts and topics it contains",
                "Check whether the assistant's response mentions those key facts",
                "Score based solely on how many of the retrieval_context's key facts appear in the response - a response that covers all key facts from the context scores high, a response that omits most of them scores low",
                "Do NOT reduce the score because the response fails to acknowledge gaps or does not address the user's question - that is measured by Answer Relevance",
            ],
            threshold=0.6,
            model=custom_model,
        )
    )

    # Metric 5: Answer Relevance
    # Evaluates if the response directly answers what was asked
    # NOTE: Does not use RETRIEVAL_CONTEXT - only checks if answer addresses question
    metrics.append(
        ConversationalGEval(
            name="Answer Relevance",
            criteria="Answer Relevance - The response should directly address the user's question without going off-topic",
            evaluation_params=[TurnParams.CONTENT, TurnParams.ROLE],
            evaluation_steps=[
                "Identify the SPECIFIC question or request from the user - not just the general topic but the precise information being sought (e.g. 'what benefits are included', 'how much does it cost', 'how do I enroll')",
                "Check whether the assistant's response attempts to address that specific question",
                "IMPORTANT: Do NOT penalise a response for being incomplete - a response that partially answers the specific question is still relevant. Completeness is measured by a separate metric.",
                "IMPORTANT: Judge relevance based on whether the response addresses the correct TOPIC, not on whether the content seems realistic. Whimsical or fictional content used to address the right specific question IS relevant.",
                "A response that states the requested information is not available IS relevant - it directly addresses the question.",
                "Score LOW if the response answers a DIFFERENT specific question than what was asked, even if both questions are about the same general topic (e.g. question asks about cost/eligibility, response describes plan features instead - that is NOT relevant).",
            ],
            threshold=0.7,
            model=custom_model,
        )
    )

    return metrics


def get_retrieval_metrics(
    custom_model: Optional[DeepEvalBaseLLM] = None,
    max_chunks: int = 10,
) -> List[Any]:
    """
    Create evaluation metrics for RAG retrieval quality assessment.

    These metrics evaluate the quality of retrieved chunks compared to expected chunks.
    They are designed to work with limited chunk counts to avoid context length issues.

    Args:
        custom_model: Optional custom LLM model instance to use for evaluations.
                     If None, uses the default DeepEval model.
        max_chunks: Maximum allowed number of chunks in retrieval results.
                   Used by ChunkCountMetric to validate chunk limit compliance.

    Returns:
        List of retrieval quality metrics including:
            - Chunk Count Limit: Validates chunk count does not exceed limit
            - Chunk Deduplication: Detects near-duplicate chunks in retrieval results
            - Chunk Alignment: Compares actual vs expected chunks for content coverage and order
            - Contextual Recall: Measures if actual chunks contain all expected information
            - Contextual Precision: Measures if relevant chunks are ranked higher
            - Contextual Relevancy: Measures relevance of each chunk to the input
            - Faithfulness: Measures if response claims are supported by retrieved context
    """
    metrics = []

    # Chunk Count Limit - Validates that retrieved chunks don't exceed the configured limit
    metrics.append(
        ChunkCountMetric(
            max_chunks=max_chunks,
            threshold=1.0,  # Must be perfect - either passes or fails
        )
    )

    # Chunk Deduplication - Detects near-duplicate chunks in retrieval results
    metrics.append(
        ChunkDeduplicationMetric(
            similarity_threshold=0.8,  # 80% word overlap = duplicate
            threshold=1.0,  # Must be perfect - no duplicates allowed
        )
    )

    # Chunk Alignment - Compares actual vs expected chunks for content coverage and order.
    metrics.append(
        GEval(
            name="Chunk Alignment",
            criteria="Chunk Alignment - The actual retrieved chunks (retrieval_context) should cover the same content as the expected chunks (context) and in a similar order",
            evaluation_params=[
                LLMTestCaseParams.CONTEXT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            evaluation_steps=[
                "IMPORTANT: The retrieval_context may contain duplicate chunks. Before evaluating, mentally deduplicate the retrieval_context - identify only the unique chunks and ignore any repeated copies. Score based solely on those unique chunks.",
                "The 'context' contains the expected/ideal content. These may be large document sections.",
                "The 'retrieval_context' contains the chunks that were actually retrieved (after deduplication in your mind). These may be smaller or differently structured than the expected chunks - this is normal and should NOT be penalized.",
                "For each expected chunk in 'context', identify the key facts, topics, and information points it contains.",
                "Check whether those key facts and information points appear anywhere across the retrieval_context. The information may be spread across multiple actual chunks with different boundaries - what matters is whether the content is present, not whether the chunk structure matches.",
                "Assess whether the order of the key topics across the retrieval_context roughly aligns with the order in the expected chunks.",
                "Penalize missing content (expected information not found anywhere in actual chunks) more heavily than extra content.",
                "Do NOT penalize for different chunk boundaries, chunk sizes, or chunk structure.",
                "Do NOT penalize for duplicated chunks - deduplication is handled by a separate metric. Treat the retrieval_context as if each unique chunk appears only once.",
                "Do NOT consider whether the chunks are relevant to any query - only compare actual vs expected chunks.",
            ],
            threshold=0.7,
            model=custom_model,
        )
    )

    # Contextual Recall - Measures if actual chunks contain all information from expected chunks
    # Not working with current judge LLMs, but leave in for possible inclusition later
    # metrics.append(
    #    ContextualRecallMetric(
    #        threshold=0.7,
    #        model=custom_model,
    #        include_reason=True,
    #    )
    # )

    # Contextual Precision - Measures if relevant chunks are ranked higher than irrelevant ones
    metrics.append(
        ContextualPrecisionMetric(
            threshold=0.7,
            model=custom_model,
            include_reason=True,
        )
    )

    # Contextual Relevancy - Measures relevance of each chunk to the input
    metrics.append(
        ContextualRelevancyMetric(
            threshold=0.7,
            model=custom_model,
            include_reason=True,
        )
    )

    # Faithfulness - Measures if claims in the response are supported by retrieval context
    metrics.append(
        FaithfulnessMetric(
            threshold=0.7,
            model=custom_model,
            include_reason=True,
        )
    )

    return metrics
