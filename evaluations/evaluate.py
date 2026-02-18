#!/usr/bin/env python3
"""
Evaluation wrapper script.

Runs test_conversations_ui.py (via pytest) to generate conversation results,
then runs deep_eval_rag.py to evaluate them.

Usage:
    python evaluate.py [options]
    python evaluate.py --check [deep_eval options]

Options:
    --check         Skip conversation generation; run deep_eval_rag.py against
                    bad-conversations instead of generated results.

    All other arguments are passed through to the appropriate script:
      - deep_eval_rag.py args: --api-endpoint, --api-key, --results-dir, --output-dir,
                               --max-limited-chunks, --max-tokens, --sequential, --stage
      - All remaining args are forwarded to pytest (e.g. --base-url, -k, -v, etc.)
"""

import subprocess
import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).parent
GENERATED_RESULTS_DIR = EVALUATIONS_DIR / "results" / "conversation_results"
KNOWN_BAD_DIR = EVALUATIONS_DIR / "bad-conversations"

# Arguments belonging to deep_eval_rag.py
DEEP_EVAL_ARGS = {
    "--api-endpoint",
    "--api-key",
    "--results-dir",
    "--output-dir",
    "--max-limited-chunks",
    "--max-tokens",
    "--sequential",
    "--stage",
    "--expect-failures",
}


def split_args(argv):
    """
    Split argv into deep_eval args and pytest args.
    Handles both --flag value and --flag=value forms.
    --sequential is a store_true flag (no value).
    """
    deep_eval = []
    pytest_args = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        # Check if this arg (or its --flag= prefix) is a deep_eval arg
        flag = arg.split("=")[0]
        if flag in DEEP_EVAL_ARGS:
            deep_eval.append(arg)
            # If the flag has no = and is not a boolean flag, consume the next token as value
            BOOLEAN_FLAGS = {"--sequential", "--expect-failures"}
            if "=" not in arg and flag not in BOOLEAN_FLAGS:
                i += 1
                if i < len(argv):
                    deep_eval.append(argv[i])
        else:
            pytest_args.append(arg)
        i += 1
    return deep_eval, pytest_args


def run(cmd, description):
    """Run a command, printing it first. Returns the exit code."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Running: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    # Pull out --check before splitting the rest
    check_mode = "--check" in sys.argv
    remaining = [a for a in sys.argv[1:] if a != "--check"]

    deep_eval_extra, pytest_extra = split_args(remaining)

    if check_mode:
        # --check mode: skip conversation generation, evaluate bad-conversations
        if not KNOWN_BAD_DIR.exists():
            print(f"ERROR: bad-conversations directory not found: {KNOWN_BAD_DIR}")
            sys.exit(1)

        # Override --results-dir unless the caller already specified it
        if "--results-dir" not in " ".join(deep_eval_extra):
            deep_eval_extra += ["--results-dir", str(KNOWN_BAD_DIR)]

        # Invert exit code: return non-zero if any bad conversation passes
        if "--expect-failures" not in deep_eval_extra:
            deep_eval_extra += ["--expect-failures"]

        cmd = [
            sys.executable,
            str(EVALUATIONS_DIR / "deep_eval_rag.py"),
        ] + deep_eval_extra
        rc = run(cmd, "STEP 1/1: Evaluating bad-conversations")
        sys.exit(rc)

    else:
        # Normal mode: generate conversations then evaluate

        # Step 1: run pytest to generate conversations
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(EVALUATIONS_DIR / "test_conversations_ui.py"),
            "-v",
            "-s",
        ] + pytest_extra
        rc = run(
            pytest_cmd, "STEP 1/2: Generating conversations (test_conversations_ui.py)"
        )
        if rc != 0:
            print(
                f"\nConversation generation failed (exit code {rc}). Aborting evaluation."
            )
            sys.exit(rc)

        # Step 2: run deep_eval_rag.py on the generated results
        deep_eval_cmd = [
            sys.executable,
            str(EVALUATIONS_DIR / "deep_eval_rag.py"),
        ] + deep_eval_extra
        rc = run(deep_eval_cmd, "STEP 2/2: Evaluating conversations (deep_eval_rag.py)")
        sys.exit(rc)


if __name__ == "__main__":
    main()
