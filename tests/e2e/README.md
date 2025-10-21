# E2E Tests for RAG Application

End-to-end deployment and functionality tests for Kind-based CI with OpenShift/MicroShift compatibility.

## Test Variants

### 1. Basic E2E Tests (`e2e-tests.yaml`)
Lightweight deployment validation without model inference:

1. **RAG UI accessibility** - Verifies Streamlit interface is reachable
2. **Backend connection** - Confirms Llama Stack service is operational  
3. **API endpoints** - Validates OpenAI-compatible API responds
4. **Model inference** - Auto-skipped (no models configured)

**Use case**: Fast CI checks for deployment health without inference dependencies.

### 2. MaaS-Enabled E2E Tests (`e2e-tests-maas.yaml`)
Full functionality testing with Red Hat Model-as-a-Service:

1. **All basic tests** - Plus full inference capability
2. **Chat completion** - Tests real model inference via MaaS
3. **RAG queries** - Validates end-to-end RAG workflow
4. **Document ingestion** - Tests upload and vector DB creation

**Use case**: Complete validation of RAG functionality with real model inference.

## Running Locally

### Prerequisites
- [kind](https://kind.sigs.k8s.io/) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI  
- [helm](https://helm.sh/docs/intro/install/) - Package manager
- Python 3.11+

### Quick Start

```bash
# 1. Create Kind cluster
cat > kind-config.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 30080
    hostPort: 8501
  - containerPort: 30081
    hostPort: 8321
EOF

kind create cluster --name rag-e2e --config kind-config.yaml

# 2. Install required CRDs (OpenShift/KServe/Kubeflow compatibility)
# See .github/workflows/e2e-tests.yaml for full CRD definitions
# Minimal stubs needed: Route, InferenceService, ServingRuntime, 
# DataSciencePipelinesApplication, Notebook

# 3. Install RAG application
kubectl create namespace rag-e2e

cd deploy/helm/rag
helm dependency build
cd -

helm install rag deploy/helm/rag \
  --namespace rag-e2e \
  --values tests/e2e/values-e2e.yaml \
  --skip-crds \
  --timeout 20m

# Wait for core services
kubectl wait --for=condition=available --timeout=600s \
  deployment/llamastack deployment/rag -n rag-e2e

# 4. Setup port forwarding
kubectl port-forward -n rag-e2e svc/rag 8501:8501 &
kubectl port-forward -n rag-e2e svc/llamastack 8321:8321 &

# 5. Install test dependencies and run
pip install -r tests/e2e/requirements.txt
python tests/e2e/test_user_workflow.py

# 6. Cleanup
pkill -f "kubectl port-forward"
helm uninstall rag -n rag-e2e
kubectl delete namespace rag-e2e
kind delete cluster --name rag-e2e
rm kind-config.yaml
```

## GitHub Actions

### Basic E2E Tests
Workflow: `.github/workflows/e2e-tests.yaml`

Runs automatically on:
- Pull requests to `main`
- Pushes to `main`  
- Manual trigger via workflow dispatch

Tests deployment health without model inference.

### MaaS-Enabled E2E Tests
Workflow: `.github/workflows/e2e-tests-maas.yaml`

Runs automatically on:
- Pull requests to `main` (if files changed: frontend, helm, tests, workflow)
- Pushes to `main`  
- Manual trigger via workflow dispatch

Tests full RAG functionality with Red Hat MaaS for inference.

**Requirements**: 
- GitHub secret `MAAS_API_KEY` must be configured
- Uses model: `llama-3-2-3b` from MaaS

## Configuration

### Basic Test Configuration (`values-e2e.yaml`)
Lightweight setup for deployment validation:
- Disabled: llm-service, configure-pipeline, ingestion-pipeline, mcp-servers
- CPU-only (no GPU needed)
- Minimal resources (512Mi RAM, 0.5 CPU)
- Core services: RAG UI, Llama Stack, pgvector, MinIO
- No model inference

### MaaS Test Configuration (`values-e2e-maas.yaml`)
Full functionality setup with external inference:
- Enabled: ingestion-pipeline, document upload
- Configured: MaaS as inference provider
- Model: `llama-3-2-3b` via OpenAI-compatible API
- Includes sample document upload for RAG testing
- Tests chat completion and RAG queries

### Environment Variables
- `LLAMA_STACK_ENDPOINT` - Backend API endpoint (default: `http://localhost:8321`)
- `RAG_UI_ENDPOINT` - Frontend UI endpoint (default: `http://localhost:8501`)
- `SKIP_MODEL_TESTS` - Skip model inference tests (`auto`|`true`|`false`, default: `auto`)
- `INFERENCE_MODEL` - Model for inference tests (default: `meta-llama/Llama-3.2-3B-Instruct`)

## Troubleshooting

### Check pod status
```bash
kubectl get pods -n rag-e2e
kubectl logs -l app.kubernetes.io/name=llamastack -n rag-e2e
```

### Check services
```bash
kubectl get services -n rag-e2e
```

### View events
```bash
kubectl get events -n rag-e2e --sort-by='.lastTimestamp'
```

## Adding More Tests

Add test steps in `test_complete_rag_workflow()` in `test_user_workflow.py`:

```python
print("🧪 Step X: Testing feature...")
# Test logic here
assert condition, "Error message"
print("✅ Passed\n")
```

For model inference tests, check `skip_inference` flag to see if models are available.

## CI Expectations

### Basic E2E Tests
- **Duration**: ~15-20 minutes
- **Resources**: 4 CPU cores, 16GB RAM
- **Environment**: Kind with 5 OpenShift/KServe/Kubeflow CRDs
- **Components**: RAG UI + Llama Stack + pgvector + MinIO

### MaaS E2E Tests
- **Duration**: ~20-25 minutes
- **Resources**: 4 CPU cores, 16GB RAM
- **Environment**: Kind + external MaaS connectivity
- **Components**: Full stack + ingestion pipeline
- **Network**: Requires outbound HTTPS to MaaS endpoint

## MaaS Integration

### What is MaaS?

Red Hat's Model-as-a-Service (MaaS) provides OpenAI-compatible API access to hosted LLMs. This eliminates the need for local model serving infrastructure in CI.

**Benefits**:
- No GPU required in CI
- Fast startup (no model downloads)
- Reliable inference
- Production-like testing

### Setting Up MaaS for CI

1. **Get API Key**: Register at https://maas.apps.prod.rhoai.rh-aiservices-bu.com/

2. **Add GitHub Secret**:
   - Go to repo Settings → Secrets → Actions
   - Add `MAAS_API_KEY` with your key

3. **Verify Configuration**:
   ```bash
   # Test MaaS endpoint
   curl https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1/models \
     -H "Authorization: Bearer YOUR_API_KEY"
   ```

4. **Run Workflow**: Manual trigger or push to main

### Testing MaaS Locally with Kind

You can test the MaaS integration locally:

```bash
# 1. Export your MaaS API key
export MAAS_API_KEY="your-api-key"

# 2. Create Kind cluster (same as basic e2e)
# ... (see Quick Start section)

# 3. Install with MaaS values
helm install rag deploy/helm/rag \
  --namespace rag-e2e \
  --values tests/e2e/values-e2e-maas.yaml \
  --set llama-stack.secrets.OPENAI_API_KEY="${MAAS_API_KEY}" \
  --skip-crds \
  --timeout 20m

# 4. Run tests with inference enabled
export SKIP_MODEL_TESTS=false
export INFERENCE_MODEL=llama-3-2-3b
python tests/e2e/test_user_workflow.py
```
