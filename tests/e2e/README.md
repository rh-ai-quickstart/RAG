# E2E Tests for RAG Application

End-to-end test that validates the complete RAG user workflow in a kind cluster.

## What It Tests

The test simulates a real user journey through the application:

1. **User opens the RAG UI** - Verifies the Streamlit interface is accessible
2. **Backend connection** - Confirms Llama Stack service is operational
3. **Model availability** - Checks that the LLM is loaded and ready
4. **Basic chat** - Tests simple question/answer functionality
5. **Multi-turn conversation** - Validates conversation history works
6. **Custom system prompts** - Tests user can customize model behavior
7. **Health checks** - Verifies application health endpoints

## Running Locally

### Prerequisites
- [kind](https://kind.sigs.k8s.io/) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [helm](https://helm.sh/docs/intro/install/) - Package manager
- Python 3.11+

### Quick Start

```bash
# 1. Install Python dependencies
pip install -r tests/e2e/requirements.txt

# 2. Create kind cluster with port mappings
kind create cluster --name rag-e2e --config - <<EOF
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

# 3. Install RAG application
helm repo add rag-charts https://rh-ai-quickstart.github.io/ai-architecture-charts
helm repo update

kubectl create namespace rag-e2e

helm install rag deploy/helm/rag \
  --namespace rag-e2e \
  --values tests/e2e/values-e2e.yaml \
  --timeout 20m \
  --wait

# 4. Setup port forwarding
kubectl port-forward -n rag-e2e svc/rag 8501:8501 &
kubectl port-forward -n rag-e2e svc/llamastack 8321:8321 &

# 5. Run the test
export LLAMA_STACK_ENDPOINT=http://localhost:8321
export RAG_UI_ENDPOINT=http://localhost:8501
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct

python tests/e2e/test_user_workflow.py

# 6. Cleanup
pkill -f "kubectl port-forward"
helm uninstall rag -n rag-e2e
kubectl delete namespace rag-e2e
kind delete cluster --name rag-e2e
```

## GitHub Actions

The E2E test runs automatically on:
- Pull requests to `main`
- Pushes to `main`
- Manual trigger via workflow dispatch

View workflow: `.github/workflows/e2e-tests.yaml`

## Configuration

### Test Configuration (`values-e2e.yaml`)
Optimized for CI with:
- CPU-only deployment (no GPU needed)
- Reduced resource limits
- Faster startup times
- Simplified stack (no ingestion pipeline)

### Environment Variables
- `LLAMA_STACK_ENDPOINT` - Backend API endpoint (default: http://localhost:8321)
- `RAG_UI_ENDPOINT` - Frontend UI endpoint (default: http://localhost:8501)
- `INFERENCE_MODEL` - Model to use (default: meta-llama/Llama-3.2-3B-Instruct)

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

To add additional workflow tests, edit `test_user_workflow.py`:

```python
def test_your_workflow():
    """Test description"""
    print("🧪 Testing your feature...")
    # Your test code
    assert condition, "Error message"
    print("✅ Test passed\n")
```

## CI Expectations

- **Startup time**: ~5-10 minutes
- **Test execution**: ~1-2 minutes
- **Total runtime**: ~15-20 minutes
- **Resources needed**: 4 CPU cores, 16GB RAM
