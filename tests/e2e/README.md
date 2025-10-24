# E2E Tests for RAG Application

End-to-end deployment and functionality tests for Kind-based CI with MaaS integration.

## Overview

The E2E tests validate the complete RAG application stack:

1. **RAG UI accessibility** - Verifies Streamlit interface is reachable
2. **Backend connection** - Confirms Llama Stack service is operational  
3. **API endpoints** - Validates OpenAI-compatible API responds
4. **Chat completion** - Tests real model inference via Red Hat MaaS
5. **RAG queries** - Validates end-to-end RAG workflow with vector DB
6. **Token usage tracking** - Monitors inference costs

**Key Features**:
- ✅ Uses Red Hat Model-as-a-Service for inference (no local GPU needed)
- ✅ Runs in Kind cluster (lightweight, fast CI)
- ✅ Full RAG pipeline testing without OpenShift dependencies
- ✅ Detailed logging of requests, responses, and token usage

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

Workflow: `.github/workflows/e2e-tests.yaml`

### Triggers

Runs automatically on:
- Pull requests to `main` (if files changed: frontend/, deploy/helm/, tests/e2e/, .github/workflows/e2e-tests.yaml)
- Pushes to `main`  
- Manual trigger via workflow dispatch

### Requirements

**Required GitHub Secret**: `MAAS_API_KEY`

The workflow will **fail with a clear error message** if this secret is not configured.

#### Setting up the secret:

1. Go to your repository on GitHub
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add the following:
   - **Name**: `MAAS_API_KEY`
   - **Value**: Your Red Hat MaaS API key (e.g., `bba9481a58685eb906c203d9358c3885`)
5. Click **"Add secret"**

#### Error if secret is missing:

```
❌ ERROR: MAAS_API_KEY secret is not configured!

To fix this, add the MAAS_API_KEY secret to your repository:
1. Go to: Settings > Secrets and variables > Actions
2. Click 'New repository secret'
3. Name: MAAS_API_KEY
4. Value: Your Red Hat MaaS API key

For more information, see:
https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions
```

### MaaS Configuration

The workflow uses these environment variables (can be overridden with repository secrets):

- **MAAS_ENDPOINT**: `https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1`
- **MAAS_MODEL_ID**: `llama-3-2-3b`
- **MAAS_API_KEY**: (from GitHub secret)

All configuration is passed to Helm via `--set` flags, so the values file remains environment-agnostic.

## Configuration

### Test Configuration (`values-e2e.yaml`)

The values file provides a base configuration optimized for Kind/CI:

**Disabled components** (require OpenShift):
- `llm-service` - Local model serving
- `configure-pipeline` - Pipeline configuration job
- `ingestion-pipeline` - Document ingestion pipeline
- `mcp-servers` - Model Context Protocol servers

**Resource limits** (optimized for CI):
- Llama Stack: 512Mi RAM, 0.5 CPU
- PGVector: 512Mi RAM, 0.5 CPU  
- MinIO: 256Mi RAM, 0.25 CPU

**MaaS configuration**:
- Model configuration is **injected at deployment time** via Helm `--set` flags
- No hardcoded credentials in version control
- Flexible: can use different models/endpoints by changing workflow env vars

**Core services enabled**:
- RAG UI (Streamlit)
- Llama Stack (orchestration)
- PGVector (vector database)
- MinIO (document storage)

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
export MAAS_API_KEY="your-api-key-here"
export MAAS_MODEL_ID="llama-3-2-3b"
export MAAS_ENDPOINT="https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1"

# 2. Create Kind cluster (same as Quick Start)
# ... (see Quick Start section)

# 3. Install with MaaS configuration injected via --set
helm install rag deploy/helm/rag \
  --namespace rag-e2e \
  --values tests/e2e/values-e2e.yaml \
  --set global.models.${MAAS_MODEL_ID}.url="${MAAS_ENDPOINT}" \
  --set global.models.${MAAS_MODEL_ID}.id="${MAAS_MODEL_ID}" \
  --set global.models.${MAAS_MODEL_ID}.enabled=true \
  --set global.models.${MAAS_MODEL_ID}.apiToken="${MAAS_API_KEY}" \
  --set-json llama-stack.initContainers='[]' \
  --skip-crds \
  --timeout 20m

# 4. Setup port forwarding (same as Quick Start)
# ... (see Quick Start section)

# 5. Run tests with inference enabled
export SKIP_MODEL_TESTS=false
export INFERENCE_MODEL=llama-3-2-3b
export LLAMA_STACK_ENDPOINT=http://localhost:8321
export RAG_UI_ENDPOINT=http://localhost:8501
python tests/e2e/test_user_workflow.py
```
