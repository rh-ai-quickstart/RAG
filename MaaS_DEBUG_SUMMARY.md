# MaaS Integration Debugging Summary

## Issue Identified

**Problem:** E2E test was falsely reporting success even when models weren't available.

**Root Cause:** Llama-stack returns 404 on `/models` endpoint, suggesting models aren't being registered even though configuration is provided via Helm values.

## Changes Made (3 commits)

### 1. Fix False Success Reporting (`d8cf002`)
**File:** `tests/e2e/test_user_workflow.py`

Added proper pass/fail logic:
```python
# If SKIP_MODEL_TESTS is explicitly false, inference MUST work
if SKIP_MODEL_TESTS == "false":
    if not model_available:
        test_passed = False
        failure_reasons.append("SKIP_MODEL_TESTS=false but no models available")
    elif not chat_passed:
        test_passed = False
        failure_reasons.append("Chat completion test failed")

# Exit with appropriate code
if not test_passed:
    raise AssertionError(f"E2E test failed: {', '.join(failure_reasons)}")
```

**Result:** Test will now properly **FAIL** when:
- `SKIP_MODEL_TESTS=false` but no models available
- Models available but chat completion fails

### 2. Enhanced Debugging (`9cd8ab9`)
**Files:** 
- `tests/e2e/test_user_workflow.py`
- `CI_DEBUG_NOTES.md`

Added detailed debugging output:
```python
print(f"   DEBUG: Calling {LLAMA_STACK_ENDPOINT}/models endpoint...")
models = client.models.list()
print(f"   DEBUG: Raw response type: {type(models)}")
print(f"   DEBUG: Number of models in response: {len(models.data)}")
print(f"   DEBUG: Extracted model IDs: {model_ids}")

# Better error handling
except Exception as e:
    print(f"   ❌ Model API check failed: {e}")
    print(f"   DEBUG: Exception type: {type(e).__name__}")
    if hasattr(e, 'response'):
        print(f"   DEBUG: Response status: {e.response.status_code}")
        print(f"   DEBUG: Response body: {e.response.text}")
    print("   Suggestion: Check llama-stack pod logs for model registration errors\n")
```

### 3. Enhanced CI Logging (`110bcda`)
**File:** `.github/workflows/e2e-tests-maas.yaml`

Improved llama-stack debugging:
```yaml
echo "=== Llama Stack logs (CRITICAL - Check model registration) ==="
kubectl logs -l app.kubernetes.io/name=llamastack -n rag-e2e --tail=300

echo "=== Llama Stack pod details ==="
kubectl describe pod -l app.kubernetes.io/name=llamastack -n rag-e2e
```

## What to Look For in Next CI Run

### 1. Test Should Now FAIL (This is Good!)
Since models aren't being registered, the test should:
- ❌ Fail with: "SKIP_MODEL_TESTS=false but no models available"
- Show clear error message explaining why

### 2. Enhanced Debug Output
```
🤖 Step 4: Checking for available models...
   DEBUG: Calling http://localhost:8321/models endpoint...
   DEBUG: Raw response type: <class 'SyncPage[Model]'>
   DEBUG: Number of models in response: 0
   DEBUG: Extracted model IDs: []
   ⚠️  No models returned from llama-stack
   This suggests llama-stack didn't load the model configuration
```

### 3. Llama Stack Logs (MOST IMPORTANT)
Will show 300 lines of llama-stack startup logs. Look for:

**Expected to see:**
```
Loading configuration from /config/llama-stack.yaml
Registering remote model: llama-3-2-3b
  Provider: remote::openai
  URL: https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1
  API Token: ******
Model registered successfully
```

**If missing registration messages:**
→ Llama-stack subchart isn't processing `global.models` config
→ Need to use alternative approach (see CI_DEBUG_NOTES.md)

### 4. Pod Environment
```
kubectl describe pod -l app.kubernetes.io/name=llamastack -n rag-e2e
```

Will show:
- Environment variables set
- Volume mounts (config files)
- Events and errors

## Expected Next Steps Based on Results

### Scenario A: Llama-stack doesn't support remote model config
**If logs show no model registration:**

**Solution:** Use programmatic model registration in test setup:
```python
from llama_stack_client import LlamaStackClient

llama_client = LlamaStackClient(base_url=LLAMA_STACK_ENDPOINT)
llama_client.models.register(
    model_id="llama-3-2-3b",
    provider_id="remote::openai",
    provider_model_id="llama-3-2-3b",
    metadata={
        "url": MAAS_ENDPOINT,
        "api_key": MAAS_API_KEY
    }
)
```

### Scenario B: Config format is wrong
**If logs show config errors:**

**Solution:** Adjust Helm values format to match what llama-stack expects.

### Scenario C: API endpoint mismatch
**If logs show successful registration but OpenAI client fails:**

**Solution:** Use llama-stack native client instead of OpenAI client.

### Scenario D: Init timing issue
**If model registration happens after readiness probe:**

**Solution:** Add model registration wait logic or adjust probes.

## Key Question Being Answered

**"If llama-stack is connected to MaaS, it should list at least one model, right?"**

**Answer:** YES! The `/models` endpoint should return:
```json
{
  "data": [
    {
      "id": "llama-3-2-3b",
      "identifier": "llama-3-2-3b",
      "provider": "remote::openai",
      "metadata": {
        "url": "https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1"
      }
    }
  ]
}
```

Currently getting empty list or 404, which means configuration isn't being loaded.

## Files Changed Summary

```
tests/e2e/test_user_workflow.py       | Enhanced debugging, proper fail logic
.github/workflows/e2e-tests-maas.yaml | Better log collection
CI_DEBUG_NOTES.md                     | Comprehensive debug guide
MaaS_DEBUG_SUMMARY.md                 | This file
```

## Ready to Push

```bash
cd /Users/skattoju/code/RAG
git push origin e2e-with-maas
```

This will:
1. Test will properly fail (revealing the issue)
2. Logs will show what llama-stack is actually doing
3. We can then fix based on evidence, not guessing

---

**Status:** Ready for diagnostic CI run  
**Expected:** Test failure with clear evidence of root cause  
**Goal:** Understand why llama-stack isn't loading model configuration

