# MaaS Integration - Quick Answers

## Your Questions Answered

### 1. Do we need to make changes in the application to make use of MaaS?

**No application code changes needed! ✅**

Your application already supports external inference providers:
- `frontend/llama_stack_ui/distribution/ui/modules/api.py` has `openai_api_key` support
- llama-stack (the dependency helm chart) supports OpenAI-compatible APIs
- The frontend already passes API keys via `provider_data`

**What we DO need to change:**
- Helm values configuration to point llama-stack to MaaS endpoint
- Pass MaaS API key via environment variables
- Configure model IDs that MaaS provides

### 2. Do we need to modify the helm chart to deploy in "MaaS mode"?

**No helm chart template changes needed, just values! ✅**

We'll create a new values file: `tests/e2e/values-e2e-maas.yaml`

**Key configurations:**
```yaml
llama-stack:
  secrets:
    OPENAI_API_KEY: ""  # Injected from GitHub secret
  env:
    - name: OPENAI_BASE_URL
      value: "https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1"
    - name: INFERENCE_PROVIDER
      value: "openai"

global:
  models:
    maas-model:
      url: "https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1"
      id: "MODEL_ID_FROM_MAAS"  # e.g., meta-llama/Llama-3.2-3B-Instruct
      enabled: true
      provider: "openai"
```

**In GitHub Actions:**
```bash
helm install rag deploy/helm/rag \
  --values tests/e2e/values-e2e-maas.yaml \
  --set llama-stack.secrets.OPENAI_API_KEY="${{ secrets.MAAS_API_KEY }}" \
  ...
```

### 3. Can playground chat and upload features be tested in Kind?

**Yes! Both can work in Kind with MaaS! ✅**

#### Playground Chat:
- ✅ Works in Kind
- ✅ Only needs HTTP connection to MaaS
- ✅ No special Kubernetes resources required
- ✅ All infrastructure (llama-stack, RAG UI) already in e2e setup

#### Upload Feature:
- ✅ Can work in Kind
- ✅ MinIO already in e2e setup
- ✅ pgvector already in e2e setup
- ⚠️ Requires enabling ingestion-pipeline (currently disabled)
- ⚠️ May need embedding model configuration

**Recommendation:**
1. **Phase 1**: Start with playground chat (simpler)
2. **Phase 2**: Add upload feature after chat works

### 4. Should we branch off add-e2e-tests?

**Yes, branch off add-e2e-tests! ✅**

**Recommended workflow:**
```bash
# Option 1: Simple branch (no rebase needed if add-e2e-tests is stable)
git checkout add-e2e-tests
git checkout -b e2e-with-maas

# Option 2: Rebase first if you want latest main changes
git checkout add-e2e-tests
git fetch upstream
git rebase upstream/main
# Resolve any conflicts
git checkout -b e2e-with-maas
```

**Branch strategy:**
```
main
 └── add-e2e-tests (basic connectivity tests, no models)
      └── e2e-with-maas (full inference tests with MaaS)
```

Later merge order: `e2e-with-maas` → `add-e2e-tests` → `main`

Or if add-e2e-tests is already approved: `add-e2e-tests` → `main`, then `e2e-with-maas` → `main`

## What You Need to Provide

Before we start implementation:

1. **MaaS API Key** - Register at https://maas.apps.prod.rhoai.rh-aiservices-bu.com/
2. **Model information** - What models are available? We need exact model IDs
3. **API endpoint format** - Confirm the base URL and any special configuration

## Quick Test to Verify MaaS

Run this to verify your MaaS setup:

```bash
# Set your API key
export MAAS_API_KEY="your-api-key-here"

# List available models
curl https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1/models \
  -H "Authorization: Bearer $MAAS_API_KEY" \
  -H "Content-Type: application/json"

# Test chat completion
curl https://maas.apps.prod.rhoai.rh-aiservices-bu.com/v1/chat/completions \
  -H "Authorization: Bearer $MAAS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_ID",
    "messages": [{"role": "user", "content": "Hello, this is a test!"}],
    "max_tokens": 50
  }'
```

## Implementation Plan Summary

### Phase 1: Setup (10 min)
- Create branch `e2e-with-maas` from `add-e2e-tests`
- Add `MAAS_API_KEY` to GitHub repository secrets

### Phase 2: Configuration (1 hour)
- Create `tests/e2e/values-e2e-maas.yaml`
- Configure llama-stack to use MaaS endpoint
- Set model IDs from MaaS

### Phase 3: Test Updates (2 hours)
- Update `test_user_workflow.py` to test chat completions
- Add RAG query tests
- Ensure tests fail gracefully if MaaS unavailable

### Phase 4: GitHub Actions (1 hour)
- Create `.github/workflows/e2e-tests-maas.yaml`
- Or add to existing workflow with matrix strategy
- Configure secrets and environment variables

### Phase 5: Testing (2 hours)
- Test locally with Kind + MaaS
- Verify in GitHub Actions
- Fix any issues

### Phase 6: Upload Feature (Optional - 3 hours)
- Enable ingestion-pipeline in values
- Configure embedding model
- Add upload tests
- Test document ingestion workflow

## Files to Create/Modify

**New Files:**
- `tests/e2e/values-e2e-maas.yaml` - Helm values for MaaS configuration
- `.github/workflows/e2e-tests-maas.yaml` - GitHub Actions workflow
- `docs/maas-integration-plan.md` - Detailed plan (already created)
- `docs/maas-integration-quickstart.md` - This file

**Modified Files:**
- `tests/e2e/test_user_workflow.py` - Add inference tests
- `tests/e2e/README.md` - Document MaaS testing
- `README.md` - Mention MaaS capability

**No Changes Needed:**
- Application code (`frontend/**`)
- Helm templates (`deploy/helm/rag/templates/**`)
- Dependency charts

## Next Steps - What I Can Do Now

I can help you:

1. ✅ **Create the branch** - Branch off add-e2e-tests
2. ✅ **Create values-e2e-maas.yaml** - Configure for MaaS (need model IDs from you)
3. ✅ **Update test_user_workflow.py** - Add inference tests
4. ✅ **Create GitHub Actions workflow** - MaaS-enabled e2e tests
5. ✅ **Update documentation** - README and guides

**What I need from you:**
- MaaS API key (to test, won't commit it)
- List of available models and their IDs
- Confirmation to proceed

## Risk Assessment

| Item | Risk Level | Notes |
|------|-----------|-------|
| Application changes | 🟢 Low | No changes needed |
| Helm chart changes | 🟢 Low | Values only, no templates |
| MaaS availability | 🟡 Medium | Add retry logic and fallback |
| API key security | 🟡 Medium | Use GitHub secrets, audit workflows |
| Rate limits | 🟡 Medium | Monitor usage, may need throttling |
| Cost/quota | 🟡 Medium | Clarify with MaaS team |

## Success Metrics

After implementation, you'll have:

- ✅ E2E tests that actually test chat completions
- ✅ Automated testing in GitHub Actions with real models
- ✅ No dependency on local inference infrastructure
- ✅ Repeatable, reliable CI pipeline
- ✅ Foundation for testing upload/RAG features
- ✅ Clear documentation for maintenance

## Ready to Start?

Just let me know:
1. Your MaaS API key (for local testing, won't be committed)
2. Available model IDs from MaaS
3. Whether you want me to proceed with implementation

I can create all the files and configurations needed!

