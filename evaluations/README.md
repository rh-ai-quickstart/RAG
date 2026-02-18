# RAG Evaluations

Two-step process: first run UI tests to collect conversations, then run DeepEval to score them.

## Setup

First deploy the RAG quickstart

Second change into the evaluations directory and setup

```bash
cd evaluations
uv sync
```

Playwright Chromium is auto-installed to `evaluations/bin/` on first run.

## Running both steps together

Set environment variables to configure the evaluator LLM, then run `evaluate.py`:

```bash
export LLM_URL=http://localhost:8321/v1
export LLM_API_TOKEN=dummy
export LLM_ID=llama-4-scout-17b-16e-w4a16

uv run python evaluate.py
```

## Running steps individually

**Step 1 — Collect conversations** (Playwright drives the RAG UI, saves responses + retrieved chunks to `results/conversation_results/`):

```bash
# Run all conversation tests
uv run pytest test_conversations_ui.py

# Run a category of tests (subdirectory under conversations/)
uv run pytest test_conversations_ui.py --subdir=hr
uv run pytest test_conversations_ui.py --subdir=legal

# Run a specific test
uv run pytest test_conversations_ui.py -k "hr_benefits"

# Debug mode (visible browser, slow)
uv run pytest test_conversations_ui.py -v --headed --slowmo=1000
```

**Step 2 — Score with DeepEval** (reads Step 1 output, evaluates with an LLM-as-judge):

```bash
uv run python deep_eval_rag.py
```

**Key options for Step 2:**

| Flag | Default | Description |
|---|---|---|
| `--api-endpoint` | `$LLM_URL` | OpenAI-compatible endpoint |
| `--api-key` | `$LLM_API_TOKEN` | API key |
| `--stage` | `both` | `1` (conversational only), `2` (retrieval only), `both` |
| `--max-concurrent` | `4` | Test cases evaluated simultaneously |
| `--max-concurrent-calls` | `16` | Max concurrent LLM API calls |
| `--debug` | off | Verbose HTTP/retry logging |

Results are saved to `results/deep_eval_results/evaluation_results_<timestamp>.json`.

## Validating the metrics

The `bad-conversations/` directory contains conversations with known incorrect responses. Use these to verify that the metrics are correctly identifying bad outputs without needing to run UI tests:

```bash
uv run python evaluate.py --check
```

## Metrics

**Stage 1 — Conversational** (per conversation):
These metrics validate the answer from the agent against "idealized" retrieval
chunks from the source documents that we've defined as part of the test case.
It tests how well the agent generated the answer from the chunks it received. The
agent may do a good job despite retrieval being poor.
- **Response Accuracy** — claims in response are supported by retrieved context
- **Response Completeness** — response covers key facts from retrieved context
- **Answer Relevance** — response addresses the actual question asked

**Stage 2 — Retrieval** (per turn with `expected_rag_content`):
These metrics validate how good the RAG retrieval was. To keep runtime/LLM capability
needed to a resonable level, we truncate to the first 10 actual RAG chunks returned
for these checks. The chunk count metric failing will show you when there were
more than number of chunks to we we truncated.
- **Chunk Count / Deduplication** — basic retrieval hygiene checks
- **Chunk Alignment** — actual chunks cover same content as expected chunks
- **Contextual Precision** — relevant chunks ranked above irrelevant ones
- **Contextual Relevancy** — each retrieved chunk is relevant to the query
- **Faithfulness** — response claims are grounded in retrieved context

## Directory structure

```
evaluations/
├── conversations/               # Test definitions (JSON)
├── results/
│   ├── conversation_results/    # Output from Step 1
│   └── deep_eval_results/       # Output from Step 2
├── evaluate.py                  # Runs Step 1 + Step 2 together
├── test_conversations_ui.py     # Step 1 — Playwright runner
├── deep_eval_rag.py             # Step 2 — DeepEval scorer
├── get_rag_metrics.py           # Metric definitions
└── helpers/
    ├── custom_llm.py            # OpenAI-compatible LLM wrapper
    ├── endpoint.py              # RAG UI endpoint detection
    └── token_counter.py         # Token usage tracking
```
