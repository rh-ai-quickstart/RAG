# End-to-End UI Tests with Playwright

This directory contains end-to-end UI tests using Playwright to test the Streamlit application through an actual browser.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers:
```bash
playwright install chromium
```

## Running Tests

### Run all UI tests:
```bash
pytest tests/e2e_ui/ -v
```

### Run specific test file:
```bash
pytest tests/e2e_ui/test_chat_ui.py -v
```

### Run with headed browser (see the browser):
```bash
pytest tests/e2e_ui/ -v --headed
```

### Run with specific browser:
```bash
pytest tests/e2e_ui/ -v --browser firefox
```

### Run with slowmo (slow down actions for debugging):
```bash
pytest tests/e2e_ui/ -v --headed --slowmo 1000
```

## Environment Variables

- `RAG_UI_ENDPOINT`: URL of the RAG UI (default: `http://localhost:8501`)
- `LLAMA_STACK_ENDPOINT`: URL of Llama Stack backend (default: `http://localhost:8321`)

## Test Structure

- `test_chat_ui.py`: Tests for chat/playground functionality
  - Basic UI loading
  - Direct mode chat
  - Agent mode chat
  - Configuration options
  - RAG configuration
  - Response display

- `test_upload_ui.py`: Tests for document upload functionality
  - File uploader
  - Vector DB creation
  - Upload validation
  - Success messaging

## Notes

- Many tests are marked with `@pytest.mark.skip` because they require:
  - Live models for inference
  - File system access for uploads
  - Complete backend setup
  
- These can be enabled when running in a full integration environment

- The tests use Playwright's `expect` assertions which include automatic waiting and retries

