# CI Debugging Notes - MaaS E2E Tests

## Current Issue: Models Not Being Registered

### Problem
Llama-stack is returning 404 on `/models` endpoint, which means:
- ✅ Llama-stack pod is running
- ✅ Llama-stack HTTP service is responding
- ❌ Models are not being loaded/registered

### Root Cause Analysis

**Expected Behavior:**
When configured with `global.models.llama-3-2-3b`, llama-stack should:
1. Load the model configuration from Helm values
2. Register the remote MaaS model internally
3. Return it when `/models` endpoint is queried

**Current Behavior:**
- `/models` returns 404 (or empty list)
- This suggests the llama-stack subchart isn't processing `global.models` correctly

### Configuration Check

From `tests/e2e/values-e2e-maas.yaml`:
```yaml
global:
  models:
    llama-3-2-3b:
      url: "https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1"
      id: "llama-3-2-3b"
      enabled: true
      apiToken: "${MAAS_API_KEY}"  # Set via helm --set
```

Helm install command:
```bash
helm install rag deploy/helm/rag \
  --values tests/e2e/values-e2e-maas.yaml \
  --set global.models.llama-3-2-3b.apiToken="${MAAS_API_KEY}" \
  --set-json llama-stack.initContainers='[]'
```

### Things to Check in Next CI Run

#### 1. Llama Stack Pod Logs (PRIORITY)
```bash
kubectl logs -l app.kubernetes.io/name=llamastack -n rag-e2e --tail=200
```

**Look for:**
- Model registration messages
- Configuration loading
- Any errors about remote models
- MaaS endpoint configuration
- API token validation

**Expected to see:**
```
Loading model configuration...
Registering remote model: llama-3-2-3b
  URL: https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1
  Provider: openai
Model registered successfully
```

#### 2. Llama Stack Environment Variables
```bash
kubectl describe pod -l app.kubernetes.io/name=llamastack -n rag-e2e
```

**Check if these are set:**
- Model configuration as env vars or config files
- API tokens properly passed through
- Any llama-stack-specific configuration

#### 3. Helm Values Rendered
```bash
helm get values rag -n rag-e2e
```

**Verify:**
- `global.models.llama-3-2-3b` is present
- `apiToken` is set (should show as `REDACTED`)
- Configuration is properly structured

### Potential Issues

#### Issue 1: Llama-stack subchart doesn't support external models
**Likelihood:** Medium  
**Fix:** May need to patch the subchart or use a different mechanism

#### Issue 2: Model config not passed to llama-stack correctly
**Likelihood:** High  
**Fix:** May need to add additional configuration or use a different values structure

#### Issue 3: Llama-stack API endpoint mismatch
**Likelihood:** Low (since basic health check works)  
**Fix:** Verify OpenAI client is using correct endpoint format

#### Issue 4: Init process or startup order issue
**Likelihood:** Medium  
**Fix:** May need startup script or wait logic

### Alternative Approaches if Model Registration Fails

#### Option A: Programmatic Model Registration
Use llama-stack-client to register the model programmatically in the test:

```python
from llama_stack_client import LlamaStackClient

# In test setup
llama_client = LlamaStackClient(base_url=LLAMA_STACK_ENDPOINT)

# Register remote model
llama_client.models.register(
    model_id="llama-3-2-3b",
    provider_id="remote::openai",
    provider_model_id="llama-3-2-3b",
    metadata={
        "url": MAAS_ENDPOINT,
        "api_key": os.getenv("MAAS_API_KEY")
    }
)
```

#### Option B: Direct MaaS Connection
Skip llama-stack for inference, connect directly to MaaS:

```python
# Point OpenAI client directly at MaaS
client = OpenAI(
    api_key=os.getenv("MAAS_API_KEY"),
    base_url=MAAS_ENDPOINT  # Direct to MaaS, not llama-stack
)
```

**Pros:** Simpler, tests MaaS integration directly  
**Cons:** Doesn't test llama-stack orchestration layer

#### Option C: Custom Llama Stack Configuration
Create a custom llama-stack configuration file and mount it:

```yaml
# ConfigMap with llama-stack config
apiVersion: v1
kind: ConfigMap
metadata:
  name: llamastack-config
data:
  config.yaml: |
    models:
      - model_id: llama-3-2-3b
        provider: remote::openai
        config:
          url: ${MAAS_ENDPOINT}
          api_key: ${MAAS_API_KEY}
```

### Next Steps

1. **Immediate:** Add debug output to capture llama-stack logs
2. **Analysis:** Examine logs to understand why models aren't loading
3. **Decision:** Based on logs, choose:
   - Fix helm configuration (if config issue)
   - Patch llama-stack subchart (if missing feature)
   - Use programmatic registration (Option A)
   - Change architecture (Option B)

### References

- Llama Stack Chart: https://github.com/rh-ai-quickstart/ai-architecture-charts/tree/main/llama-stack
- Llama Stack Client: https://github.com/meta-llama/llama-stack-client-python
- Model Configuration Format: See `deploy/helm/rag/values.yaml` lines 47-120

---

**Status:** Debugging in progress  
**Last Updated:** 2025-10-21  
**Priority:** HIGH - Blocks MaaS e2e testing
