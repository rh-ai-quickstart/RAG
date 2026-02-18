#!/usr/bin/env python3
"""
Simple token counter for tracking LLM API usage.
Simplified version adapted from it-self-service-agent.
"""

import threading
from typing import Any, Optional


_lock = threading.Lock()
_total_input_tokens = 0
_total_output_tokens = 0
_total_requests = 0


def count_tokens_from_response(
    response: Any, model: Optional[str] = None, context: Optional[str] = None
) -> tuple[int, int]:
    """
    Extract and count tokens from an OpenAI-style response object.

    Accumulates totals that can be retrieved with get_token_totals().

    Args:
        response: OpenAI API response object
        model: Model name (optional, for logging)
        context: Context string (optional, for logging)

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    global _total_input_tokens, _total_output_tokens, _total_requests
    try:
        if hasattr(response, "usage") and response.usage is not None:
            try:
                usage = response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                with _lock:
                    _total_input_tokens += input_tokens
                    _total_output_tokens += output_tokens
                    _total_requests += 1
                return input_tokens, output_tokens
            except AttributeError:
                return 0, 0
        else:
            return 0, 0
    except Exception:
        return 0, 0


def get_token_totals() -> tuple[int, int, int]:
    """
    Return accumulated token totals.

    Returns:
        Tuple of (total_input_tokens, total_output_tokens, total_requests)
    """
    with _lock:
        return _total_input_tokens, _total_output_tokens, _total_requests


def reset_token_totals() -> None:
    """Reset accumulated token totals to zero."""
    global _total_input_tokens, _total_output_tokens, _total_requests
    with _lock:
        _total_input_tokens = 0
        _total_output_tokens = 0
        _total_requests = 0
