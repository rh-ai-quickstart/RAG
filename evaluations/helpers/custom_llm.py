#!/usr/bin/env python3
"""
Custom LLM wrapper for DeepEval evaluations using OpenAI-compatible endpoints.
Adapted from it-self-service-agent/evaluations/helpers/custom_llm.py
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple, Type, Union, cast

import instructor
import openai
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from .token_counter import count_tokens_from_response

logger = logging.getLogger(__name__)


def get_api_configuration(
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Get API configuration from parameters or environment variables.

    Supports both it-self-service-agent style variables (LLM_API_TOKEN, LLM_URL, LLM/LLM_ID)
    and standard OpenAI variables (OPENAI_API_KEY, OPENAI_API_BASE).

    Args:
        api_endpoint: Optional custom API endpoint URL
        api_key: Optional API key

    Returns:
        Tuple of (api_key, endpoint, model_name)
    """
    # Get API key from parameter or environment
    # Priority: parameter > LLM_API_TOKEN > OPENAI_API_KEY
    final_api_key = api_key or os.getenv("LLM_API_TOKEN") or os.getenv("OPENAI_API_KEY")

    # Get endpoint from parameter or environment
    # Priority: parameter > LLM_URL > OPENAI_API_BASE
    final_endpoint = (
        api_endpoint or os.getenv("LLM_URL") or os.getenv("OPENAI_API_BASE")
    )

    # Get model name from environment
    # Priority: LLM_ID > LLM > default
    model_name = os.getenv("LLM_ID") or os.getenv("LLM") or "gpt-4"

    if not final_api_key:
        logger.warning(
            "No API key found in parameters or LLM_API_TOKEN/OPENAI_API_KEY environment variables"
        )

    return final_api_key, final_endpoint, model_name


def _log_response(response: Any, elapsed: float, mode: str) -> None:
    """Log details from an API response including finish reason and token usage."""
    try:
        choice = response.choices[0] if response.choices else None
        finish_reason = choice.finish_reason if choice else "unknown"
        usage = response.usage

        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        logger.info(
            f"API call completed ({mode}): {elapsed:.2f}s, "
            f"finish_reason={finish_reason}, "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}"
        )

        if finish_reason == "length":
            print(
                f"WARNING: Response truncated (finish_reason=length): "
                f"output used {output_tokens} tokens - consider increasing --max-tokens"
            )
            logger.warning(
                f"Response truncated (finish_reason=length): "
                f"output used {output_tokens} tokens - response may be incomplete"
            )
    except Exception as e:
        logger.debug(f"Could not log response details: {e}")


class CustomLLM(DeepEvalBaseLLM):
    """Custom LLM class for using non-OpenAI endpoints with deepeval.

    Supports two modes:
    - instructor mode (use_instructor=True): Uses the instructor library for structured
      output with tool/function calling. More reliable schema validation but requires
      server support for tool calling.
    - JSON mode (use_instructor=False): Uses the OpenAI JSON response format. More
      compatible with vLLM and other inference servers. DeepEval handles JSON parsing.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: Optional[str] = None,
        use_instructor: bool = True,
        max_tokens: Optional[int] = None,
        max_concurrent_calls: int = 4,
    ):
        """
        Initialize the CustomLLM with API credentials and configuration.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the LLM API endpoint
            model_name: Optional model name
            use_instructor: If True, use instructor for structured output.
                           If False, use JSON mode and let DeepEval parse the response.
            max_tokens: Maximum number of tokens for LLM responses. If None, the server default is used.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name or os.getenv("LLM_ID") or "gpt-4"
        self.use_instructor = use_instructor
        self.max_tokens = max_tokens

        # Semaphore to cap total concurrent async API calls across all metrics and test
        # cases.
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)

        self.client = openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=600.0, max_retries=2
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=600.0, max_retries=2
        )

        # Create instructor clients once and reuse them (only if needed).
        # Mode.JSON uses response_format={"type":"json_object"} and validates the
        # response client-side with Pydantic, retrying on schema mismatch.
        if use_instructor:
            self.instructor_client = instructor.from_openai(
                self.client, mode=instructor.Mode.JSON
            )
            self.async_instructor_client = instructor.from_openai(
                self.async_client, mode=instructor.Mode.JSON
            )

        mode_label = "instructor/JSON" if use_instructor else "JSON"
        logger.debug(
            f"CustomLLM initialized: mode={mode_label}, "
            f"max_tokens={max_tokens}, timeout=600s, max_retries=2, "
            f"max_concurrent_calls={max_concurrent_calls}"
        )

    def load_model(self) -> Any:
        """
        Load and return the OpenAI client instance.

        Returns:
            OpenAI client configured with custom endpoint and API key
        """
        return self.client

    def generate(  # type: ignore[override]
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        """
        Generate a response to the given prompt using the custom LLM.

        Args:
            prompt: The input prompt to generate a response for
            schema: Optional Pydantic BaseModel class for structured output

        Returns:
            Pydantic model instance if schema provided and using instructor,
            otherwise string response

        Raises:
            Exception: If the API call fails or returns an error
        """
        prompt_words = len(prompt.split())
        try:
            # Instructor mode: use tool/function calling for structured output
            if schema is not None and self.use_instructor:
                logger.debug(
                    f"Generating (sync/instructor): schema={schema.__name__}, "
                    f"prompt_words={prompt_words}"
                )
                start_time = time.time()
                resp, completion = (
                    self.instructor_client.chat.completions.create_with_completion(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        response_model=schema,
                        temperature=0.1,
                        **(
                            {"max_tokens": self.max_tokens}
                            if self.max_tokens is not None
                            else {}
                        ),
                        max_retries=3,
                    )
                )
                elapsed = time.time() - start_time

                _log_response(completion, elapsed, "sync/instructor")
                count_tokens_from_response(
                    completion, self.model_name, "custom_llm_evaluation"
                )

                return cast(BaseModel, resp)

            # JSON mode: return raw string, let DeepEval parse it
            use_json = schema is not None or any(
                keyword in prompt.lower() for keyword in ["json", "schema", "format"]
            )

            api_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            if self.max_tokens is not None:
                api_kwargs["max_tokens"] = self.max_tokens

            if use_json:
                api_kwargs["response_format"] = {"type": "json_object"}

            logger.info(
                f"API call starting (sync/JSON): json_mode={use_json}, "
                f"prompt_words={prompt_words}, schema={schema.__name__ if schema else None}"
            )
            logger.debug(
                f"Prompt (sync/JSON, schema={schema.__name__ if schema else None}):\n{prompt}"
            )
            start_time = time.time()
            response = self.client.chat.completions.create(**api_kwargs)
            elapsed = time.time() - start_time

            _log_response(response, elapsed, "sync/JSON")
            count_tokens_from_response(
                response, self.model_name, "custom_llm_evaluation"
            )

            content = response.choices[0].message.content
            result = str(content) if content is not None else ""
            logger.debug(
                f"Response (sync/JSON, schema={schema.__name__ if schema else None}):\n{result}"
            )
            return result
        except openai.APITimeoutError as e:
            logger.error(f"API timeout (sync): {e}")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"API connection error (sync): {type(e).__name__}: {e}")
            raise
        except openai.APIStatusError as e:
            logger.error(
                f"API status error (sync): status={e.status_code}, "
                f"type={type(e).__name__}, message={e.message}"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error (sync): {type(e).__name__}: {e}")
            raise

    async def a_generate(  # type: ignore[override]
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        """
        Asynchronously generate a response to the given prompt using the custom LLM.

        Args:
            prompt: The input prompt to generate a response for
            schema: Optional Pydantic BaseModel class for structured output

        Returns:
            Pydantic model instance if schema provided and using instructor,
            otherwise string response

        Raises:
            Exception: If the API call fails or returns an error
        """
        prompt_words = len(prompt.split())
        async with self._semaphore:
            try:
                # Instructor mode: use tool/function calling for structured output
                if schema is not None and self.use_instructor:
                    logger.debug(
                        f"Generating (async/instructor): schema={schema.__name__}, "
                        f"prompt_words={prompt_words}"
                    )

                    start_time = time.time()
                    (
                        resp,
                        completion,
                    ) = await self.async_instructor_client.chat.completions.create_with_completion(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        response_model=schema,
                        temperature=0.1,
                        **(
                            {"max_tokens": self.max_tokens}
                            if self.max_tokens is not None
                            else {}
                        ),
                        max_retries=3,
                    )
                    elapsed = time.time() - start_time

                    _log_response(completion, elapsed, "async/instructor")
                    count_tokens_from_response(
                        completion, self.model_name, "custom_llm_evaluation_async"
                    )

                    return resp

                # JSON mode: return raw string, let DeepEval parse it
                use_json = schema is not None or any(
                    keyword in prompt.lower()
                    for keyword in ["json", "schema", "format"]
                )

                api_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                }
                if self.max_tokens is not None:
                    api_kwargs["max_tokens"] = self.max_tokens

                if use_json:
                    api_kwargs["response_format"] = {"type": "json_object"}

                logger.info(
                    f"API call starting (async/JSON): json_mode={use_json}, "
                    f"prompt_words={prompt_words}, schema={schema.__name__ if schema else None}"
                )
                logger.debug(
                    f"Prompt (async/JSON, schema={schema.__name__ if schema else None}):\n{prompt}"
                )
                start_time = time.time()
                response = await self.async_client.chat.completions.create(**api_kwargs)
                elapsed = time.time() - start_time

                _log_response(response, elapsed, "async/JSON")
                count_tokens_from_response(
                    response, self.model_name, "custom_llm_evaluation_async"
                )

                content = response.choices[0].message.content
                result = str(content) if content is not None else ""
                logger.debug(
                    f"Response (async/JSON, schema={schema.__name__ if schema else None}):\n{result}"
                )
                return result

            except openai.APITimeoutError as e:
                logger.error(f"API timeout (async): {e}")
                raise
            except openai.APIConnectionError as e:
                logger.error(f"API connection error (async): {type(e).__name__}: {e}")
                raise
            except openai.APIStatusError as e:
                logger.error(
                    f"API status error (async): status={e.status_code}, "
                    f"type={type(e).__name__}, message={e.message}"
                )
                raise
            except Exception as e:
                logger.error(f"Unexpected error (async): {type(e).__name__}: {e}")
                raise

    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name
