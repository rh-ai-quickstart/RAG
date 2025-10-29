# E2E Tests

The end-to-end tests have been reorganized:

## New Structure

### LlamaStack API Integration Tests
**Location:** `tests/integration/llamastack/`

Tests that exercise the LlamaStack API directly (formerly in this directory):
- `test_rag_with_vectordb.py` - RAG functionality with vector databases
- `test_user_workflow.py` - Complete user workflow testing

### UI E2E Tests
**Location:** `tests/e2e_ui/`

Playwright-based tests that interact with the Streamlit UI through a browser:
- `test_chat_ui.py` - Chat interface testing
- `test_upload_ui.py` - Document upload testing

## Migration Note

The tests previously in `tests/e2e/` have been moved to `tests/integration/llamastack/` to better reflect their purpose. They test the LlamaStack API integration, not the full application UI.

True end-to-end tests that exercise the UI are now in `tests/e2e_ui/` using Playwright.
