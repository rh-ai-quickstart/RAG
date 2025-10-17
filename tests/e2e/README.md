# E2E Tests for RAG Application

Lightweight deployment validation tests for Kind-based CI with OpenShift/MicroShift compatibility.

## What It Tests

Core infrastructure and connectivity (no models required):

1. **RAG UI accessibility** - Verifies Streamlit interface is reachable
2. **Backend connection** - Confirms Llama Stack service is operational  
3. **API endpoints** - Validates OpenAI-compatible API responds
4. **Model inference** - Auto-skipped if no models configured (set `SKIP_MODEL_TESTS=false` to force)

This is a **lightweight validation** focused on deployment health, not full functionality testing.

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

The E2E test runs automatically on:
- Pull requests to `main`
- Pushes to `main`  
- Manual trigger via workflow dispatch

View workflow: `.github/workflows/e2e-tests.yaml`

## Configuration

### Test Configuration (`values-e2e.yaml`)
Lightweight setup for CI:
- Disabled: llm-service, configure-pipeline, ingestion-pipeline, mcp-servers
- CPU-only (no GPU needed)
- Minimal resources (512Mi RAM, 0.5 CPU)
- Only core services: RAG UI, Llama Stack, pgvector, MinIO

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

- **Duration**: ~15-20 minutes
- **Resources**: 4 CPU cores, 16GB RAM
- **Environment**: Kind with 5 OpenShift/KServe/Kubeflow CRDs
- **Components**: RAG UI + Llama Stack + pgvector + MinIO only
