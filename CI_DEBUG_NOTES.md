# CI Debug Notes - MaaS Integration

## Current Issue

Models showing as `[None, None]` instead of actual model IDs.

### Symptoms
```
Found 2 model(s): [None, None]
⚠️  Target model 'llama-3-2-3b' not found, but 2 other(s) available
```

### Root Cause Analysis

**Fixed**: Model ID detection
- Changed from `model.id` to `model.identifier` 
- Llama-stack uses different attribute name than OpenAI

**Remaining Issue**: Model registration
- Llama-stack is seeing 2 models but their IDs are None
- This suggests models are registered but without proper identifiers

### Potential Causes

1. **Model configuration not being passed correctly**
   - Check if `global.models.llama-3-2-3b` is reaching llama-stack
   - Verify helm chart processes this correctly

2. **API token not set properly**
   - API token might not be passed to the model config
   - Without token, model might register but fail to initialize

3. **Llama-stack chart issue**
   - The llama-stack dependency chart might not support external models properly
   - Or might require different configuration format

### Debug Commands for Next CI Run

Add these to workflow to see what's happening:

```yaml
- name: Debug llama-stack configuration
  run: |
    echo "=== Llama Stack Pod Describe ==="
    kubectl describe pod -l app.kubernetes.io/name=llamastack -n rag-e2e
    
    echo "=== Llama Stack Environment ==="
    kubectl exec -n rag-e2e deployment/llamastack -- env | grep -i model || true
    
    echo "=== Llama Stack Logs (full) ==="
    kubectl logs -l app.kubernetes.io/name=llamastack -n rag-e2e --tail=200
    
    echo "=== Check llama-stack config file ==="
    kubectl exec -n rag-e2e deployment/llamastack -- cat /root/.llama/config || true
```

### Alternative Approach

If llama-stack chart doesn't support external models well, we could:

1. **Use llama-stack client directly** in tests
   - Register the MaaS model via API after llamastack starts
   - Don't rely on helm chart to register it

2. **Create a custom init container**
   - Have it call llama-stack API to register the model
   - This ensures model is registered before tests run

### Quick Fix to Try

Add model registration to test setup:

```python
# In test_user_workflow.py, before running tests
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=LLAMA_STACK_ENDPOINT)

# Register MaaS model directly
try:
    client.models.register(
        model_id="llama-3-2-3b",
        provider_id="remote::maas",
        provider_model_id="llama-3-2-3b",
        metadata={
            "url": "https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1",
            "api_key": os.getenv("MAAS_API_KEY", "")
        }
    )
    print("✅ Registered MaaS model with llama-stack")
except Exception as e:
    print(f"⚠️  Model registration: {e}")
```

### Next Steps

1. Push current fix (model.identifier)
2. Trigger CI run
3. Check llama-stack logs in failure output
4. Determine if we need to register model programmatically
5. Update tests accordingly

### Expected Behavior

With working configuration, should see:
```
Found 1 model(s): ['llama-3-2-3b']
✅ Target model 'llama-3-2-3b' is available
```

Then tests should run:
```
💬 Step 6: Testing chat completion...
   ✓ Model responded: ...
   ✓ Tokens used: ...
✅ Chat completion test passed
```

