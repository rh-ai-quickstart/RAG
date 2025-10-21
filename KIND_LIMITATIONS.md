# Kind Limitations for E2E Testing

## Issue: Ingestion Pipeline Requires OpenShift

The ingestion pipeline components use OpenShift internal registry images that are not available in Kind clusters.

### Failing Components

1. **rag-ingestion-pipeline** - Main pipeline pod
   - Init container tries to pull: `image-registry.openshift-image-registry.svc:5000/openshift/tools:latest`
   - Result: `ImagePullBackOff`

2. **add-default-ingestion-pipeline** - Pipeline initialization job
   - Same image issue
   - Result: `ImagePullBackOff`

3. **upload-sample-docs-job** - Document upload job
   - Same image issue
   - Result: `ImagePullBackOff`

4. **llamastack init container** - `wait-for-models`
   - Was waiting for local models to be ready
   - Not needed when using external MaaS
   - Fixed with: `waitForModels: false`

### Error Messages

```
Failed to pull image "image-registry.openshift-image-registry.svc:5000/openshift/tools:latest": 
failed to pull and unpack image: failed to resolve reference: failed to do request: 
Head "https://image-registry.openshift-image-registry.svc:5000/v2/openshift/tools/manifests/latest": 
dial tcp: lookup image-registry.openshift-image-registry.svc on 172.18.0.1:53: no such host
```

## Solution for MaaS E2E Tests

### What We Disabled

```yaml
# Disabled for Kind compatibility
ingestion-pipeline:
  enabled: false
  replicaCount: 0
  defaultPipeline:
    enabled: false

minio:
  sampleFileUpload:
    enabled: false

llama-stack:
  waitForModels: false  # Don't wait for local models
```

### What Still Works

✅ **Core Infrastructure**
- RAG UI deployment
- Llama Stack with MaaS connectivity
- pgvector database
- MinIO object storage

✅ **MaaS Integration**
- Model availability check
- Chat completions with real LLM
- Token usage tracking
- Error handling

✅ **Basic E2E Validation**
- Deployment health
- Service connectivity
- API endpoint testing
- Inference with MaaS

### What Doesn't Work in Kind

❌ **Document Ingestion**
- Can't test document upload
- Can't test pipeline processing
- Can't test embedding generation
- Can't create vector DBs automatically

❌ **Full RAG Workflow**
- Can't test end-to-end RAG with uploaded docs
- Can manually create vector DBs via API (if needed)

## Workarounds

### Option 1: Keep Ingestion Disabled (Current)
**Pros:**
- Clean, fast CI
- Tests core MaaS integration
- No OpenShift dependency

**Cons:**
- Can't test document upload
- Can't test full RAG workflow

### Option 2: Use Public Images (Future)
Replace OpenShift internal images with public alternatives:

```yaml
ingestion-pipeline:
  initContainer:
    image: "docker.io/alpine:latest"  # or busybox
  # ... configure to use public images
```

**Status:** Not implemented (would require chart changes)

### Option 3: Test in OpenShift (Ideal)
Run MaaS e2e tests in actual OpenShift environment:

**Pros:**
- Full functionality testing
- Real environment
- Complete RAG validation

**Cons:**
- Requires OpenShift cluster
- More complex setup
- Slower CI

## Recommendations

### For CI/CD (GitHub Actions)
✅ **Use current approach:**
- Basic e2e: No models, no ingestion (fast validation)
- MaaS e2e: External models, no ingestion (inference validation)
- Both run in Kind for speed and simplicity

### For Full Testing (Manual/Staging)
🔄 **Use OpenShift environment:**
- Deploy with all features enabled
- Test document upload manually
- Validate complete RAG workflow
- Use MaaS for consistent inference

### For Development
💻 **Local testing:**
- Use podman-compose for full stack (local)
- Or connect to deployed OpenShift instance
- Test ingestion features there

## What We Learned

1. **OpenShift-specific images** don't work in Kind
2. **Init containers** that wait for local models block MaaS deployments
3. **Kind is great for**:
   - Core infrastructure validation
   - External API integration (like MaaS)
   - Fast CI/CD pipelines

4. **Kind is NOT suitable for**:
   - Components requiring OpenShift internal registry
   - Features needing OpenShift-specific tools
   - Full platform integration testing

## Impact on Testing Strategy

### What MaaS E2E Tests Validate

✅ **High Value Tests:**
- MaaS connectivity and authentication
- Chat completion end-to-end
- Model inference correctness
- Token usage and billing data
- Error handling and retries
- Deployment health and stability

✅ **Still Achievable:**
- Manual RAG testing (use pre-created vector DBs)
- Manual document upload (via UI, outside CI)
- Integration testing (all services working together)

### What Requires OpenShift

⚠️ **Deferred to Manual Testing:**
- Automated document ingestion
- Pipeline processing validation
- Embedding generation testing
- Vector DB auto-creation

## Conclusion

The current MaaS e2e test setup is **optimal for CI/CD:**
- ✅ Fast execution (~20-25 min)
- ✅ Tests core MaaS integration
- ✅ Validates inference capability
- ✅ No OpenShift dependency
- ✅ Easy to maintain

For **full RAG validation**, we should:
- Run manual tests in OpenShift environment
- Test document upload through UI
- Validate complete pipeline in staging
- Keep CI focused on core functionality

This is a **pragmatic approach** that balances:
- Speed and reliability of CI
- Coverage of critical paths
- Cost and complexity
- Development velocity

