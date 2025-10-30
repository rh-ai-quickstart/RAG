# RAG Application Test Suite

This directory contains a comprehensive test suite for the RAG (Retrieval-Augmented Generation) application, covering multiple testing layers from unit tests to end-to-end UI tests.

## Test Structure

```
tests/
├── unit/                           # Unit tests for individual functions
│   ├── test_chat.py               # Chat module unit tests
│   ├── test_upload.py             # Upload module unit tests
│   └── requirements.txt
├── integration/                    # Integration tests
│   ├── test_chat_integration.py   # Chat functionality integration tests
│   ├── test_upload_integration.py # Upload functionality integration tests
│   ├── llamastack/                # LlamaStack API integration tests
│   │   ├── test_user_workflow.py
│   │   ├── test_rag_with_vectordb.py
│   │   └── requirements.txt
│   └── requirements.txt
├── e2e_ui/                        # End-to-end UI tests with Playwright
│   ├── test_chat_ui.py           # Chat UI E2E tests
│   ├── test_upload_ui.py         # Upload UI E2E tests
│   ├── conftest.py               # Playwright test configuration
│   └── requirements.txt
└── e2e/                          # Legacy directory (see README)
```

## Test Layers

### 1. Unit Tests (`tests/unit/`)

**Purpose:** Test individual functions and components in isolation

**Coverage:**
- `test_chat.py`: 
  - Sampling strategy configuration
  - Message formatting (with/without RAG context)
  - Agent type configurations
  - System prompt handling
  - Tool group selection and configuration
  
- `test_upload.py`:
  - Vector DB configuration
  - Document processing
  - RAGDocument creation
  - Provider detection
  - Upload validation

**Running:**
```bash
pip install -r tests/unit/requirements.txt
pytest tests/unit/ -v
```

**Key Features:**
- Fast execution (~seconds)
- No external dependencies
- Mocked Streamlit and LlamaStack clients
- Ideal for TDD and quick feedback

### 2. Integration Tests (`tests/integration/`)

**Purpose:** Test Streamlit app components programmatically without UI

**Coverage:**

#### Streamlit App Integration Tests:
- `test_chat_integration.py`:
  - Direct mode RAG query construction
  - Sampling parameters configuration
  - Agent session creation
  - Direct mode with/without RAG
  - Agent mode tool configuration
  - Message history management
  - Shield configuration

- `test_upload_integration.py`:
  - Single and multiple file uploads
  - Vector DB registration workflow
  - Document insertion with chunking
  - Provider detection
  - Data URL conversion

#### LlamaStack API Integration Tests (`llamastack/`):
- `test_user_workflow.py`:
  - Complete user workflow simulation
  - Service connectivity
  - Model availability checks
  - Chat completion with MaaS
  - RAG query with vector DB

- `test_rag_with_vectordb.py`:
  - Vector DB creation and population
  - Document insertion
  - RAG retrieval testing

**Running:**
```bash
# Streamlit app integration tests
pip install -r tests/integration/requirements.txt
pytest tests/integration/test_*.py -v

# LlamaStack integration tests (requires running services)
pip install -r tests/integration/llamastack/requirements.txt
export LLAMA_STACK_ENDPOINT=http://localhost:8321
export RAG_UI_ENDPOINT=http://localhost:8501
pytest tests/integration/llamastack/ -v
```

**Key Features:**
- Tests actual code paths
- Mocked external dependencies
- No browser required
- Medium execution time (~minutes)

### 3. UI E2E Tests (`tests/e2e_ui/`)

**Purpose:** Test the actual user interface with browser automation

**Coverage:**
- `test_chat_ui.py`:
  - Page loading and rendering
  - Sidebar configuration visibility
  - Direct mode selection and usage
  - Agent mode selection and features
  - Temperature, max tokens, system prompt controls
  - Clear chat functionality
  - RAG configuration UI
  - Tool debug toggle
  - Responsive design (mobile, tablet)

- `test_upload_ui.py`:
  - File uploader component
  - Vector DB naming
  - Upload validation
  - Success messaging
  - Error handling
  - Keyboard navigation
  - Accessibility

**Running:**
```bash
pip install -r tests/e2e_ui/requirements.txt
playwright install chromium

# Start the application first
export RAG_UI_ENDPOINT=http://localhost:8501
export LLAMA_STACK_ENDPOINT=http://localhost:8321

# Run tests
pytest tests/e2e_ui/ -v

# Run with visible browser for debugging
pytest tests/e2e_ui/ -v --headed

# Run with slowmo for better observation
pytest tests/e2e_ui/ -v --headed --slowmo 1000
```

**Key Features:**
- Real browser interaction
- Visual regression potential
- Tests actual user experience
- Slower execution (~minutes to hours)
- Screenshots on failure

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/e2e-tests.yaml`) runs all test types:

### Test Workflow

1. **Unit Tests** (`unit-tests` job)
   - Runs first, fastest feedback
   - No external dependencies
   - Generates code coverage reports

2. **Integration Tests** (`integration-tests` job)
   - Runs after unit tests pass
   - Tests Streamlit components programmatically
   - Needs: unit-tests

3. **LlamaStack Integration Tests** (`llamastack-integration-tests` job)
   - Deploys full stack on Kind cluster
   - Tests with MaaS inference
   - Validates complete RAG workflow
   - Needs: unit-tests

4. **UI E2E Tests** (`ui-e2e-tests` job)
   - Deploys full stack on separate Kind cluster
   - Runs Playwright browser tests
   - Tests actual UI interactions
   - Needs: unit-tests, integration-tests

### Workflow Triggers

Tests run on:
- Pull requests to `main` branch
- Pushes to `main` branch
- Manual workflow dispatch
- Changes to `frontend/`, `deploy/helm/`, `tests/`, or workflow file

## Test Coverage Summary

| Test Type | Component | What's Tested |
|-----------|-----------|---------------|
| **Unit** | `chat.py` | Sampling strategy, message formatting, agent types |
| **Unit** | `upload.py` | Vector DB config, document processing, validation |
| **Integration** | Chat | Direct/agent modes, RAG queries, tool configuration |
| **Integration** | Upload | File uploads, DB creation, provider detection |
| **Integration** | LlamaStack API | Complete workflows, MaaS integration, RAG retrieval |
| **E2E UI** | Chat UI | User interactions, configuration changes, responsiveness |
| **E2E UI** | Upload UI | File selection, DB naming, upload workflow |

## Running All Tests Locally

### Quick Start

```bash
# 1. Install all dependencies
pip install -r tests/unit/requirements.txt
pip install -r tests/integration/requirements.txt
pip install -r tests/e2e_ui/requirements.txt
playwright install chromium

# 2. Run unit tests (fast, no dependencies)
pytest tests/unit/ -v

# 3. Run integration tests (requires mocks)
pytest tests/integration/test_*.py -v

# 4. Start the application for E2E tests
# (In separate terminals or use docker-compose)
export LLAMA_STACK_ENDPOINT=http://localhost:8321
export RAG_UI_ENDPOINT=http://localhost:8501

# 5. Run LlamaStack integration tests
pytest tests/integration/llamastack/ -v

# 6. Run UI E2E tests
pytest tests/e2e_ui/ -v
```

### Full Test Suite

```bash
# Run everything (except UI tests that need running app)
pytest tests/unit/ tests/integration/test_*.py -v
```

## Test Development Guidelines

### Unit Tests
- Mock all external dependencies
- Test one function/method per test
- Use descriptive test names
- Keep tests fast (<1s each)

### Integration Tests
- Mock external services (LlamaStack API calls)
- Test component interactions
- Verify data flows correctly
- Use fixtures for common setups

### UI E2E Tests
- Test user workflows, not implementation
- Use stable selectors (text, labels)
- Add waits for dynamic content
- Capture screenshots on failure
- Mark slow/flaky tests appropriately

## Common Issues and Solutions

### Unit Tests Failing
- **Issue:** Import errors
- **Solution:** Check that frontend directory is in Python path

### Integration Tests Failing
- **Issue:** Mock not configured correctly
- **Solution:** Review mock setup in fixtures

### E2E Tests Failing
- **Issue:** Services not ready
- **Solution:** Increase timeout or add explicit waits

### Playwright Issues
- **Issue:** Browser not installed
- **Solution:** Run `playwright install chromium`

## Contributing

When adding new features to the application:

1. **Add unit tests** for new functions
2. **Add integration tests** for new workflows
3. **Add UI E2E tests** for new user-facing features
4. **Update this README** if test structure changes

## Coverage Goals

- **Unit Tests:** 80%+ code coverage
- **Integration Tests:** All major workflows covered
- **E2E Tests:** All user-facing features tested

## Further Reading

- [Pytest Documentation](https://docs.pytest.org/)
- [Playwright for Python](https://playwright.dev/python/)
- [Streamlit Testing](https://docs.streamlit.io/library/advanced-features/testing)

