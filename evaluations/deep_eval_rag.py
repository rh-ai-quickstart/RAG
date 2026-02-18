#!/usr/bin/env python3
"""
DeepEval-based RAG conversation evaluator.

This script evaluates RAG conversation results using LLM-as-a-judge metrics
to assess retrieval quality and response accuracy.

Adapted from it-self-service-agent/evaluations/deep_eval.py
"""

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from deepeval.evaluate import AsyncConfig, DisplayConfig, evaluate
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from deepeval.test_run import global_test_run_manager
from get_rag_metrics import get_rag_metrics, get_retrieval_metrics
from helpers.custom_llm import CustomLLM, get_api_configuration
from helpers.token_counter import get_token_totals, reset_token_totals

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _configure_debug_logging() -> None:
    """Enable DEBUG-level logging for openai and httpx to show retries and timeouts."""
    logging.getLogger("openai").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("helpers.custom_llm").setLevel(logging.DEBUG)
    logger.info(
        "Debug logging enabled: openai, httpx, and custom_llm loggers set to DEBUG"
    )


if "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE" not in os.environ:
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"


def _no_op_wrap_up_test_run(*args: Any, **kwargs: Any) -> None:
    """
    No-operation function to override DeepEval's default wrap_up_test_run behavior.

    This prevents DeepEval from attempting to connect to online services.
    """
    pass


# Override DeepEval's default behavior
global_test_run_manager.wrap_up_test_run = _no_op_wrap_up_test_run  # type: ignore[method-assign]


def _convert_to_turns(conversation_data: List[Dict[str, Any]]) -> List[Turn]:
    """
    Convert conversation data from role/content format to DeepEval Turn objects.

    Args:
        conversation_data: List of dictionaries with 'role' and 'content' keys

    Returns:
        List[Turn]: List of DeepEval Turn objects ready for evaluation
    """
    turns = []
    for turn_data in conversation_data:
        role = turn_data.get("role", "")
        content = turn_data.get("content", "")

        # Skip empty content
        if not content.strip():
            continue

        # Ensure role is valid
        if role not in ["user", "assistant"]:
            role = "user"  # Default to user if role is invalid

        # Set retrieval_context on user turns from expected_rag_content so
        # ConversationalGEval metrics can verify the response against the chunks
        retrieval_context = None
        if role == "user":
            expected_rag = turn_data.get("expected_rag_content")
            if expected_rag:
                chunks = expected_rag.get("chunks", [])
                if chunks:
                    retrieval_context = chunks

        turns.append(
            Turn(role=role, content=content, retrieval_context=retrieval_context)
        )  # type: ignore[arg-type]

    return turns


def _print_metric_results(test_results):
    """
    Print metric results per test case.

    Args:
        test_results: Test results from evaluate() call (has test_results attribute)
    """
    if not hasattr(test_results, "test_results"):
        print("  No results available")
        return

    for test_result in test_results.test_results:
        test_name = test_result.name if hasattr(test_result, "name") else "unknown"
        print(f"\n  [{test_name}]")

        for metric_data in test_result.metrics_data:
            metric_name = (
                metric_data.name if hasattr(metric_data, "name") else "Unknown"
            )
            score = metric_data.score if hasattr(metric_data, "score") else None
            success = metric_data.success if hasattr(metric_data, "success") else None
            threshold = (
                metric_data.threshold if hasattr(metric_data, "threshold") else None
            )
            reason = metric_data.reason if hasattr(metric_data, "reason") else None
            error = metric_data.error if hasattr(metric_data, "error") else None

            status = "✅" if success else "❌"
            print(f"    {status} {metric_name}", end="")
            if score is not None:
                print(f" (score: {score:.2f}", end="")
                if threshold is not None:
                    print(f", threshold: {threshold}", end="")
                print(")", end="")
            print()

            if error:
                print(f"       ERROR: {error}")
            elif reason and not success:
                if len(reason) > 200:
                    reason = reason[:200] + "..."
                print(f"       Reason: {reason}")


def _print_evaluation_summary(
    conv_results, retrieval_results, stage2_error: Optional[str] = None
):
    """
    Print a comprehensive summary of evaluation results.

    Shows each test with consolidated pass/fail and indented failures/warnings.

    Returns:
        tuple[bool, bool]: (any_passed, any_failed) — whether at least one test
        passed and whether at least one test failed.
    """
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}\n")

    # ANSI color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # Group results by base test name
    test_groups = defaultdict(lambda: {"conversational": None, "retrieval": []})

    # Collect conversational results
    if conv_results and hasattr(conv_results, "test_results"):
        for test_result in conv_results.test_results:
            test_name = test_result.name if hasattr(test_result, "name") else "unknown"
            test_groups[test_name]["conversational"] = test_result

    # Collect retrieval results (group by base name, removing _turn_N suffix)
    if retrieval_results and hasattr(retrieval_results, "test_results"):
        for test_result in retrieval_results.test_results:
            test_name = test_result.name if hasattr(test_result, "name") else "unknown"
            # Extract base name (remove _turn_N suffix)
            base_name = re.sub(r"_turn_\d+$", "", test_name)
            test_groups[base_name]["retrieval"].append(test_result)

    any_passed = False
    any_failed = False

    # Print results for each test
    for base_test_name in sorted(test_groups.keys()):
        group = test_groups[base_test_name]
        conv_result = group["conversational"]
        retrieval_results_list = group["retrieval"]

        # Collect all failures for this test
        all_failures = []

        # Check conversational metrics
        if conv_result:
            for metric_data in conv_result.metrics_data:
                success = (
                    metric_data.success if hasattr(metric_data, "success") else False
                )

                if not success:
                    all_failures.append(("Conversational", metric_data))

        # Check retrieval metrics
        for retrieval_result in retrieval_results_list:
            for metric_data in retrieval_result.metrics_data:
                success = (
                    metric_data.success if hasattr(metric_data, "success") else False
                )

                if not success:
                    all_failures.append(("Retrieval", metric_data))

        # If Stage 2 failed to run entirely, mark this test as failed
        stage2_failed_to_run = stage2_error and not retrieval_results_list

        # Determine overall status
        overall_pass = len(all_failures) == 0 and not stage2_failed_to_run
        if overall_pass:
            any_passed = True
        else:
            any_failed = True
        status = f"{GREEN}✓ PASS{RESET}" if overall_pass else f"{RED}✗ FAIL{RESET}"
        print(f"{status} {base_test_name}")

        # Print failures indented
        if stage2_failed_to_run:
            print("  FAILURES:")
            print(f"    • [Retrieval] Stage 2 failed to run: {stage2_error}")
            print()
        elif all_failures:
            print("  FAILURES:")
            for eval_type, metric_data in all_failures:
                metric_name = (
                    metric_data.name if hasattr(metric_data, "name") else "Unknown"
                )
                score = metric_data.score if hasattr(metric_data, "score") else None
                threshold = (
                    metric_data.threshold if hasattr(metric_data, "threshold") else None
                )
                reason = (
                    metric_data.reason
                    if hasattr(metric_data, "reason")
                    else "No reason provided"
                )

                print(f"    • [{eval_type}] {metric_name}", end="")
                if score is not None and threshold is not None:
                    print(f" (score: {score:.2f}, threshold: {threshold})", end="")
                print()

                # Print reason with indentation
                if reason:
                    if len(reason) > 120:
                        reason = reason[:120] + "..."
                    wrapped = textwrap.wrap(reason, width=70)
                    for line in wrapped:
                        print(f"      {line}")

            # Add spacing after failures
            print()

    print(f"{'=' * 80}\n")
    return any_passed, any_failed


def _build_context_from_expected_rag(
    expected_rag_content: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Build context strings from expected RAG content.

    Args:
        expected_rag_content: Dictionary containing expected RAG chunks

    Returns:
        List of context strings with full expected retrieval content
    """
    if not expected_rag_content:
        return []

    context = []
    chunks = expected_rag_content.get("chunks", [])

    if chunks:
        context.append("=" * 80)
        context.append("GROUND TRUTH CONTEXT - RETRIEVED DOCUMENT CONTENT")
        context.append("=" * 80)
        context.append(
            "IMPORTANT: The following chunks are the EXACT, ACTUAL content from the source"
        )
        context.append(
            "document. Regardless of how unusual, whimsical, or unrealistic this content may"
        )
        context.append(
            "appear, it IS the real ground truth content that was retrieved by the RAG system."
        )
        context.append(
            "You MUST evaluate the assistant's response based SOLELY on whether it accurately"
        )
        context.append(
            "reflects THIS content, not on whether the content seems realistic or plausible."
        )
        context.append(
            "Do NOT judge the content as fictional simply because it contains unusual elements."
        )
        context.append("")
        context.append(f"Number of chunks: {len(chunks)}")
        context.append("")
        for i, chunk in enumerate(chunks, 1):
            context.append(f"--- Chunk {i} of {len(chunks)} ---")
            context.append(chunk)
            context.append("")
        context.append("=" * 80)
        context.append("END OF GROUND TRUTH CONTEXT")
        context.append("=" * 80)

    return context


def _evaluate_rag_conversations(
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    results_dir: str = "results/conversation_results",
    output_dir: str = "results/deep_eval_results",
    max_limited_chunks: int = 10,
    max_tokens: Optional[int] = None,
    sequential: bool = False,
    max_concurrent: int = 4,
    max_concurrent_calls: int = 16,
    stage: str = "both",
    expect_failures: bool = False,
) -> int:
    """
    Main evaluation function for RAG conversations.

    Args:
        api_endpoint: Optional custom OpenAI-compatible API endpoint URL
        api_key: API key for the LLM service
        results_dir: Path to directory containing conversation JSON files
        output_dir: Path to directory where evaluation results will be saved
        max_limited_chunks: Maximum number of chunks for limited-chunk metrics
        max_tokens: Maximum number of tokens for LLM responses
        sequential: If True, run evaluations sequentially instead of in parallel
        max_concurrent: Maximum number of concurrent evaluations when running async (default: 4)
        stage: Which evaluation stages to run: "1" (conversational), "2" (retrieval), or "both"

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Get API configuration
    api_key_found, current_endpoint, model_name = get_api_configuration(
        api_endpoint, api_key
    )

    if not api_key_found:
        logger.error("No API key configured. Cannot proceed with evaluation.")
        logger.error("Set OPENAI_API_KEY environment variable or pass --api-key")
        return 1

    # Reset token counters for this run
    reset_token_totals()

    # Create custom models for each evaluation stage
    # Part 1 (Conversational): uses instructor for structured output
    # Part 2 (Retrieval): uses JSON mode (more compatible with vLLM tool-calling limitations)
    # Toggle use_instructor per stage to switch between modes
    custom_model_part1 = CustomLLM(
        api_key=api_key_found,
        base_url=current_endpoint or "",
        model_name=model_name,
        use_instructor=True,
        max_tokens=max_tokens,
        max_concurrent_calls=max_concurrent_calls,
    )
    custom_model_part2 = CustomLLM(
        api_key=api_key_found,
        base_url=current_endpoint or "",
        model_name=model_name,
        use_instructor=True,
        max_tokens=max_tokens,
        max_concurrent_calls=max_concurrent_calls,
    )

    if not os.path.exists(results_dir):
        logger.error(f"Results directory {results_dir} does not exist")
        return 1

    # Clean up any existing results before running
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            file_path_to_remove = os.path.join(output_dir, f)
            if os.path.isfile(file_path_to_remove):
                os.remove(file_path_to_remove)
    os.makedirs(output_dir, exist_ok=True)

    # Discover all conversation files
    json_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))

    if not json_files:
        logger.warning(f"No JSON files found in {results_dir}")
        return 0

    print(f"\n{'=' * 60}")
    print(f"Found {len(json_files)} conversation files to evaluate")
    print(f"{'=' * 60}\n")

    # Get RAG evaluation metrics
    metrics = get_rag_metrics(custom_model_part1)

    # Process each conversation file
    all_test_cases = []
    all_limited_chunk_test_cases = []  # For RAG retrieval metrics (limited to top N chunks)
    retrieval_test_metadata = []  # Track which file/turn each retrieval test belongs to
    failed_files = []

    for filename in json_files:
        file_path = os.path.join(results_dir, filename)

        try:
            print(f"Processing {filename}...")

            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)

            # Validate file format
            if not isinstance(file_data, dict) or "conversation" not in file_data:
                logger.error(
                    f"Invalid format in {filename} - expected object with 'conversation'"
                )
                failed_files.append(filename)
                continue

            conversation_data = file_data["conversation"]
            config = file_data.get("config", {})

            # Convert to turns
            turns = _convert_to_turns(conversation_data)

            if len(turns) < 2:
                logger.warning(
                    f"Skipping {filename} - insufficient turns ({len(turns)})"
                )
                continue

            # Build scenario from expected RAG content (if present)
            scenario_parts = [f"RAG conversation from {filename}"]

            # Add config information
            mode = config.get("mode", "unknown")
            vector_dbs = config.get("vector_dbs", [])
            scenario_parts.append(f"Mode: {mode}")
            if vector_dbs:
                scenario_parts.append(f"Vector DBs: {', '.join(vector_dbs)}")

            # Extract expected RAG content from user messages for conversational evaluation
            # (actual RAG content is only used in retrieval-specific metrics)
            for turn_data in conversation_data:
                if turn_data.get("role") == "user":
                    expected_rag = turn_data.get("expected_rag_content")
                    if expected_rag:
                        rag_context = _build_context_from_expected_rag(expected_rag)
                        scenario_parts.extend(rag_context)

            # Join scenario parts into a single string (scenario field expects string, not list)
            scenario = "\n".join(scenario_parts)

            # Create conversational test case
            test_case = ConversationalTestCase(
                name=filename,
                turns=turns,
                additional_metadata={"filename": filename, "config": config},
                scenario=scenario,
            )

            all_test_cases.append(test_case)
            print(f"  ✓ Loaded {len(turns)} turns for conversational evaluation")

            # ALSO create retrieval test cases for turns with expected_output and actual RAG content
            turn_num = 0
            for idx, turn_data in enumerate(conversation_data):
                if turn_data.get("role") == "user":
                    turn_num += 1
                    expected_output = turn_data.get("expected_output")
                    expected_rag = turn_data.get("expected_rag_content")
                    actual_rag = turn_data.get("actual_rag_content")

                    # Find the next assistant response (actual_output) for this user turn
                    actual_output = ""
                    for j in range(idx + 1, len(conversation_data)):
                        if conversation_data[j].get("role") == "assistant":
                            actual_output = conversation_data[j].get("content", "")
                            break

                    if expected_output and expected_rag:
                        # Create LLMTestCase for retrieval evaluation
                        # Use empty list for actual_chunks if nothing was retrieved (should fail metrics)
                        actual_chunks = (
                            actual_rag.get("chunks", []) if actual_rag else []
                        )
                        expected_chunks = expected_rag.get("chunks", [])

                        # Limit chunks to avoid context length issues
                        if len(actual_chunks) > max_limited_chunks:
                            print(
                                f"  ⚠️  Turn {turn_num}: {len(actual_chunks)} chunks retrieved, limiting to top {max_limited_chunks} for evaluation"
                            )
                            limited_chunks = actual_chunks[:max_limited_chunks]
                        else:
                            limited_chunks = actual_chunks

                        limited_chunk_test = LLMTestCase(
                            name=f"{filename}_turn_{turn_num}",
                            input=turn_data.get("content", ""),
                            actual_output=actual_output,  # Assistant's actual response (used by Faithfulness/Hallucination)
                            expected_output=expected_output,
                            context=expected_chunks,  # Ground truth chunks (what should be retrieved)
                            retrieval_context=limited_chunks,  # Limited chunks to avoid token limits
                            additional_metadata={
                                "original_chunk_count": len(
                                    actual_chunks
                                ),  # Store original count for ChunkCountMetric
                            },
                        )
                        all_limited_chunk_test_cases.append(limited_chunk_test)

                        retrieval_test_metadata.append(
                            {
                                "filename": filename,
                                "turn": turn_num,
                                "user_query": turn_data.get("content", "")[:100]
                                + "...",
                            }
                        )

                        if not actual_chunks:
                            print(
                                f"  ⚠️  Created retrieval test case for turn {turn_num} - NO ACTUAL CHUNKS RETRIEVED"
                            )
                        else:
                            print(
                                f"  ✓ Created retrieval test case for turn {turn_num} ({len(actual_chunks)} total chunks, using {len(limited_chunks)} for evaluation)"
                            )

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            failed_files.append(filename)
            continue

    if not all_test_cases:
        logger.error("No valid test cases to evaluate")
        return 1

    print(f"\n{'=' * 60}")
    print(f"Evaluating {len(all_test_cases)} conversations...")
    print(f"{'=' * 60}\n")

    # Run conversational evaluation
    try:
        async_config = AsyncConfig(
            run_async=not sequential, max_concurrent=max_concurrent
        )
        conv_results = None
        limited_chunk_results = None
        stage2_error = None

        if stage in ("1", "both"):
            print("=" * 80)
            print("PART 1: Conversational Evaluation (Response Quality)")
            print("=" * 80)
            conv_results = evaluate(
                test_cases=all_test_cases,
                metrics=metrics,
                async_config=async_config,
                display_config=DisplayConfig(show_indicator=True, print_results=False),
            )

            # Print clean results without verbose content
            print("\nResults:")
            _print_metric_results(conv_results)

        # Run retrieval evaluation with limited chunks
        if stage in ("2", "both") and all_limited_chunk_test_cases:
            print(f"\n{'=' * 80}")
            print(
                f"PART 2: RAG Retrieval Evaluation (Limited Chunks) - {len(all_limited_chunk_test_cases)} turns"
            )
            print(f"{'=' * 80}\n")

            retrieval_metrics = get_retrieval_metrics(
                custom_model_part2, max_chunks=max_limited_chunks
            )

            try:
                limited_chunk_results = evaluate(
                    test_cases=all_limited_chunk_test_cases,
                    metrics=retrieval_metrics,
                    async_config=async_config,
                    display_config=DisplayConfig(
                        show_indicator=True, print_results=False
                    ),
                )
                print("\nRetrieval Results (Limited Chunks):")
                _print_metric_results(limited_chunk_results)
            except Exception as e:
                stage2_error = str(e)
                logger.error(f"Stage 2 evaluation failed: {e}")
                logger.error(
                    "This is often caused by the LLM truncating its response (max_tokens too low). Try --max-tokens 8192."
                )
                logger.warning("Continuing with Stage 1 results only.")

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")

        # Extract conversational results data
        results_data = {
            "timestamp": timestamp,
            "model": model_name,
            "total_conversations": len(all_test_cases),
            "total_retrieval_tests": len(all_limited_chunk_test_cases),
            "failed_files": failed_files,
            "conversational_results": [],
            "retrieval_results": [],
        }

        # Extract conversational evaluation results from the test run
        # After evaluate() runs, results are stored in test run manager
        if conv_results is not None and hasattr(conv_results, "test_results"):
            for test_result in conv_results.test_results:
                result_entry = {
                    "filename": test_result.name
                    if hasattr(test_result, "name")
                    else "unknown",
                    "metrics": [],
                }

                for metric_data in test_result.metrics_data:
                    result_entry["metrics"].append(
                        {
                            "name": metric_data.name
                            if hasattr(metric_data, "name")
                            else "unknown",
                            "score": metric_data.score
                            if hasattr(metric_data, "score")
                            else None,
                            "success": metric_data.success
                            if hasattr(metric_data, "success")
                            else None,
                            "reason": metric_data.reason
                            if hasattr(metric_data, "reason")
                            else None,
                        }
                    )

                results_data["conversational_results"].append(result_entry)
        elif conv_results is not None:
            # Fallback: try to extract from test cases
            for i, test_case in enumerate(all_test_cases):
                test_result = {
                    "filename": test_case.additional_metadata.get("filename"),
                    "metrics": [],
                }

                for metric in metrics:
                    metric_name = None
                    if hasattr(metric, "name"):
                        metric_name = metric.name
                    elif hasattr(metric, "__name__"):
                        metric_name = metric.__name__
                    else:
                        metric_name = metric.__class__.__name__

                    test_result["metrics"].append(
                        {
                            "name": metric_name,
                            "score": metric.score if hasattr(metric, "score") else None,
                            "success": metric.success
                            if hasattr(metric, "success")
                            else None,
                            "reason": metric.reason
                            if hasattr(metric, "reason")
                            else None,
                        }
                    )

                results_data["conversational_results"].append(test_result)

        # Extract retrieval evaluation results
        if limited_chunk_results and hasattr(limited_chunk_results, "test_results"):
            for i, test_result in enumerate(limited_chunk_results.test_results):
                result_entry = {
                    "filename": retrieval_test_metadata[i]["filename"]
                    if i < len(retrieval_test_metadata)
                    else "unknown",
                    "turn": retrieval_test_metadata[i]["turn"]
                    if i < len(retrieval_test_metadata)
                    else 0,
                    "user_query": retrieval_test_metadata[i]["user_query"]
                    if i < len(retrieval_test_metadata)
                    else "",
                    "metrics": [],
                }

                for metric_data in test_result.metrics_data:
                    result_entry["metrics"].append(
                        {
                            "name": metric_data.name
                            if hasattr(metric_data, "name")
                            else "unknown",
                            "score": metric_data.score
                            if hasattr(metric_data, "score")
                            else None,
                            "success": metric_data.success
                            if hasattr(metric_data, "success")
                            else None,
                            "reason": metric_data.reason
                            if hasattr(metric_data, "reason")
                            else None,
                        }
                    )

                results_data["retrieval_results"].append(result_entry)

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        # Print comprehensive summary
        any_passed, any_failed = _print_evaluation_summary(
            conv_results, limited_chunk_results, stage2_error=stage2_error
        )

        # Print token usage summary
        input_tokens, output_tokens, total_requests = get_token_totals()
        total_tokens = input_tokens + output_tokens

        print(f"\n{'=' * 80}")
        print("Evaluation Complete!")
        print(f"{'=' * 80}")
        print(f"Conversational evaluations: {len(all_test_cases)}")
        print(f"Retrieval evaluations: {len(all_limited_chunk_test_cases)}")
        print(f"Results saved to: {output_file}")
        print("\nToken Usage:")
        print(f"  LLM API requests: {total_requests}")
        print(f"  Input tokens:     {input_tokens:,}")
        print(f"  Output tokens:    {output_tokens:,}")
        print(f"  Total tokens:     {total_tokens:,}")
        print(f"{'=' * 80}\n")

        if expect_failures:
            # --check mode: bad conversations should all fail; a passing result means
            # the metrics didn't catch a known-bad conversation.
            return 1 if any_passed else 0
        else:
            # Normal mode: good conversations should pass; a failing result means
            # something went wrong with the RAG system or the evaluation.
            return 1 if any_failed else 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for the RAG evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG conversations using LLM-as-a-judge metrics"
    )
    parser.add_argument(
        "--api-endpoint",
        help="Custom OpenAI-compatible API endpoint (e.g., http://localhost:8321/v1)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the LLM service (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--results-dir",
        default="results/conversation_results",
        help="Directory containing conversation result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/deep_eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--max-limited-chunks",
        type=int,
        default=10,
        help="Maximum number of chunks for limited-chunk metrics (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens for LLM responses. If not set, the server default is used.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run evaluations sequentially instead of in parallel to reduce load",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent evaluations when running async (default: 4)",
    )
    parser.add_argument(
        "--max-concurrent-calls",
        type=int,
        default=16,
        help="Maximum number of concurrent API calls to the LLM (default: 16). "
        "DeepEval fires ~15 calls per test case simultaneously across all metrics; "
        "this semaphore limits total in-flight requests to avoid server overload.",
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Which evaluation stages to run: '1' (conversational), '2' (retrieval), or 'both' (default)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging for openai/httpx to show retries, timeouts, and request details",
    )
    parser.add_argument(
        "--expect-failures",
        action="store_true",
        help="Invert exit code logic: return non-zero if any evaluation PASSES "
        "(used by evaluate.py --check to verify bad conversations are caught by the metrics)",
    )

    args = parser.parse_args()

    if args.debug:
        _configure_debug_logging()

    return _evaluate_rag_conversations(
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        max_limited_chunks=args.max_limited_chunks,
        max_tokens=args.max_tokens,
        sequential=args.sequential,
        max_concurrent=args.max_concurrent,
        max_concurrent_calls=args.max_concurrent_calls,
        stage=args.stage,
        expect_failures=args.expect_failures,
    )


if __name__ == "__main__":
    sys.exit(main())
