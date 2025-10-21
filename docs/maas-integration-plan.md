# MaaS Integration Plan for E2E Testing

## Overview

Integrate Red Hat's Model-as-a-Service (MaaS) into GitHub Actions e2e tests to enable full inference testing without requiring local model serving infrastructure.

## Current State

- E2E tests run on Kind cluster with basic connectivity validation
- Model inference tests are skipped due to lack of available models
- Uses llama-stack dependency chart that supports multiple inference providers
- Frontend already supports OpenAI-compatible API keys via `provider_data`

## MaaS Information

- **URL**: https://maas.apps.prod.rhoai.rh-aiservices-bu.com/
- **Type**: OpenAI-compatible API
- **Authentication**: API key (to be stored as GitHub secret)
- **Available Models**: TBD (need to query `/v1/models` endpoint)

## Architecture Changes

### No Application Code Changes Needed ✓

The application already supports external inference providers:
- `frontend/llama_stack_ui/distribution/ui/modules/api.py` has `openai_api_key` support
- llama-stack chart supports multiple providers via environment variables

### Helm Chart Configuration

We'll create a new values file for MaaS-enabled e2e tests:

**File**: `tests/e2e/values-e2e-maas.yaml`

Key configurations:
1. Configure llama-stack to use MaaS as inference provider
2. Pass MaaS API key via environment variable
3. Register MaaS models in llama-stack

### GitHub Actions Changes

1. **GitHub Secret**: `MAAS_API_KEY`
   - Store the MaaS API key securely
   - Reference in workflow: `${{ secrets.MAAS_API_KEY }}`

2. **New Workflow** or **Update Existing**:
   - Option A: Create separate workflow `e2e-tests-maas.yaml`
   - Option B: Add MaaS variant to existing workflow

3. **Environment Variables**:
   - `MAAS_API_KEY`: From GitHub secrets
   - `MAAS_ENDPOINT`: MaaS base URL
   - `SKIP_MODEL_TESTS`: Set to `false` to enable inference tests

## Implementation Steps

### Phase 1: Branch Setup
```bash
# Ensure we're on latest add-e2e-tests
git checkout add-e2e-tests
git fetch upstream
git rebase upstream/main

# Create new branch
git checkout -b e2e-with-maas
```

### Phase 2: Research MaaS API

Need to determine:
1. Exact API endpoint format (likely `https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1`)
2. Available model IDs
3. Authentication header format
4. Rate limits and usage constraints

**Action**: Query MaaS API to understand:
```bash
# List available models
curl https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1/models \
  -H "Authorization: Bearer $MAAS_API_KEY"

# Test chat completion
curl https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1/chat/completions \
  -H "Authorization: Bearer $MAAS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_ID",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Phase 3: Create MaaS Values File

**File**: `tests/e2e/values-e2e-maas.yaml`

Based on current `values-e2e.yaml` with MaaS-specific additions:

```yaml
# ... inherit most from values-e2e.yaml ...

# Configure llama-stack to use MaaS
llama-stack:
  secrets:
    # MaaS API key will be injected from GitHub secret
    OPENAI_API_KEY: ""  # Set via --set flag in helm install
  env:
    # Configure llama-stack to use MaaS endpoint
    - name: OPENAI_BASE_URL
      value: "https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1"
    - name: INFERENCE_PROVIDER
      value: "openai"

# Define models available via MaaS
global:
  models:
    maas-llama-3-2-3b:
      url: "https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1"
      id: "meta-llama/Llama-3.2-3B-Instruct"  # Or actual MaaS model ID
      enabled: true
      provider: "openai"
```

### Phase 4: Update E2E Tests

**File**: `tests/e2e/test_user_workflow.py`

Add comprehensive inference tests:

```python
def test_chat_completion(client, skip_inference):
    """Test chat completion with MaaS"""
    if skip_inference:
        print("⏭️  Skipping chat completion test\n")
        return
    
    print("🤖 Step 6: Testing chat completion...")
    response = client.chat.completions.create(
        model=INFERENCE_MODEL,
        messages=[
            {"role": "user", "content": "Say 'Hello, RAG!' if you can hear me."}
        ],
        max_tokens=50
    )
    assert response.choices[0].message.content, "No response from model"
    print(f"   Model response: {response.choices[0].message.content[:100]}")
    print("✅ Chat completion works\n")

def test_rag_query(client, skip_inference):
    """Test RAG with vector database"""
    if skip_inference:
        print("⏭️  Skipping RAG query test\n")
        return
    
    print("🔍 Step 7: Testing RAG query...")
    # Create vector DB, upload docs, query
    # (Implementation depends on your RAG API)
    print("✅ RAG query works\n")
```

### Phase 5: GitHub Actions Workflow

**Option A: New Workflow** (Recommended)

**File**: `.github/workflows/e2e-tests-maas.yaml`

```yaml
name: E2E Tests with MaaS

on:
  pull_request:
    branches: [main]
    paths:
      - 'frontend/**'
      - 'deploy/helm/**'
      - 'tests/e2e/**'
      - '.github/workflows/e2e-tests-maas.yaml'
  push:
    branches: [main]
  workflow_dispatch:

env:
  MAAS_ENDPOINT: "https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1"
  INFERENCE_MODEL: "meta-llama/Llama-3.2-3B-Instruct"

jobs:
  e2e-test-maas:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      # ... (copy setup steps from e2e-tests.yaml) ...
      
      - name: Install RAG application with MaaS
        run: |
          helm install rag deploy/helm/rag \
            --namespace rag-e2e \
            --values tests/e2e/values-e2e-maas.yaml \
            --set llama-stack.secrets.OPENAI_API_KEY="${{ secrets.MAAS_API_KEY }}" \
            --set llama-stack.env[0].name=OPENAI_BASE_URL \
            --set llama-stack.env[0].value="${{ env.MAAS_ENDPOINT }}" \
            --skip-crds \
            --timeout 20m \
            --wait
      
      - name: Run E2E tests with inference
        env:
          LLAMA_STACK_ENDPOINT: "http://localhost:8321"
          RAG_UI_ENDPOINT: "http://localhost:8501"
          SKIP_MODEL_TESTS: "false"  # Enable inference tests
          INFERENCE_MODEL: ${{ env.INFERENCE_MODEL }}
        run: |
          python tests/e2e/test_user_workflow.py
```

**Option B: Extend Existing Workflow**

Add a matrix strategy to run both basic and MaaS variants:

```yaml
strategy:
  matrix:
    test-type: [basic, maas]
    include:
      - test-type: basic
        values-file: tests/e2e/values-e2e.yaml
        skip-model-tests: "auto"
      - test-type: maas
        values-file: tests/e2e/values-e2e-maas.yaml
        skip-model-tests: "false"
```

### Phase 6: Testing in Kind

The playground chat and upload features **can work in Kind with MaaS** because:

1. **Playground Chat**: 
   - Just needs llama-stack with MaaS connection
   - No special Kubernetes resources required
   - Works via HTTP/REST API calls

2. **Upload Feature**:
   - Requires MinIO (already in e2e setup ✓)
   - Requires pgvector (already in e2e setup ✓)
   - Requires ingestion pipeline (currently disabled)
   - **Decision**: Start with chat, add upload in follow-up

### Phase 7: Documentation

Update documentation files:

1. **tests/e2e/README.md**: Add MaaS testing section
2. **docs/maas-integration-plan.md**: This document
3. **README.md**: Add note about MaaS-enabled e2e tests

## Prerequisites for User

Before implementation, you need to:

1. **Get MaaS API Key**: Register at https://maas.apps.prod.rhoai.rh-aiservices-bu.com/
2. **Test API Key**: Verify it works with curl/postman
3. **Identify Model IDs**: Determine exact model identifiers in MaaS
4. **Check Rate Limits**: Understand usage constraints for CI runs
5. **Add GitHub Secret**: Add `MAAS_API_KEY` to repository secrets

## Questions to Answer

1. What is the exact MaaS API endpoint format?
2. What models are available and their IDs?
3. Are there rate limits we need to consider for CI?
4. Do we need special headers beyond Authorization?
5. Does MaaS support embeddings API for RAG features?

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| MaaS rate limiting | Add retry logic, limit test frequency |
| API key exposure | Use GitHub secrets, never commit keys |
| Service availability | Fall back to basic tests if MaaS unavailable |
| Cost/quota limits | Monitor usage, add alerts |

## Success Criteria

- [ ] E2E tests run successfully with MaaS in GitHub Actions
- [ ] Chat completion tests pass with real inference
- [ ] No secrets leaked in logs or code
- [ ] Documentation updated
- [ ] Tests complete in < 30 minutes
- [ ] Branch ready for PR to main

## Next Steps

1. **Get API Key**: User provides MaaS API key
2. **Research API**: Query MaaS to understand endpoints and models
3. **Create Branch**: Branch off from add-e2e-tests
4. **Implement**: Create values file and update tests
5. **Test Locally**: Verify with kind and port-forward to MaaS
6. **Add to CI**: Create/update GitHub Actions workflow
7. **Document**: Update all relevant documentation
8. **PR**: Submit for review when ready

## Timeline Estimate

- Phase 1 (Branch): 5 minutes
- Phase 2 (Research): 30 minutes
- Phase 3 (Values File): 1 hour
- Phase 4 (Test Updates): 2 hours
- Phase 5 (GitHub Actions): 1 hour
- Phase 6 (Testing): 2 hours
- Phase 7 (Documentation): 1 hour

**Total**: ~8 hours of development time

## References

- MaaS Portal: https://maas.apps.prod.rhoai.rh-aiservices-bu.com/
- OpenAI API Docs: https://platform.openai.com/docs/api-reference
- Llama Stack Docs: https://github.com/meta-llama/llama-stack
- Current E2E Tests: tests/e2e/README.md

