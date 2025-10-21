# MaaS E2E Testing Setup Guide

Quick guide to set up and run E2E tests with Red Hat's Model-as-a-Service (MaaS).

## For Repository Maintainers

### One-Time Setup

1. **Get MaaS API Key**
   - Visit: https://maas.apps.prod.rhoai.rh-aiservices-bu.com/
   - Register and obtain API key

2. **Add GitHub Secret**
   - Go to: Repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `MAAS_API_KEY`
   - Value: Your MaaS API key
   - Click "Add secret"

3. **Verify Setup**
   ```bash
   # Test the API key locally first
   curl https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1/models \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json"
   ```

### Running the Workflow

The MaaS E2E workflow runs automatically on:
- PRs to main (if relevant files changed)
- Pushes to main
- Manual trigger

**Manual trigger**:
1. Go to Actions → "E2E Tests with MaaS"
2. Click "Run workflow"
3. Select branch
4. Click "Run workflow"

## For Local Development

### Prerequisites
- Kind cluster
- Helm 3.x
- Python 3.11+
- MaaS API key

### Quick Local Test

```bash
# 1. Set your API key
export MAAS_API_KEY="your-api-key-here"

# 2. Create Kind cluster with port mappings
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

kind create cluster --name rag-maas-test --config kind-config.yaml

# 3. Install required CRDs
# (See .github/workflows/e2e-tests-maas.yaml for CRD definitions)
kubectl apply -f tests/e2e/crds/  # If you've extracted them

# 4. Create namespace
kubectl create namespace rag-e2e

# 5. Add helm repos and build dependencies
helm repo add rag-charts https://rh-ai-quickstart.github.io/ai-architecture-charts
helm repo update
cd deploy/helm/rag
helm dependency build
cd -

# 6. Install with MaaS configuration
helm install rag deploy/helm/rag \
  --namespace rag-e2e \
  --values tests/e2e/values-e2e-maas.yaml \
  --set llama-stack.secrets.OPENAI_API_KEY="${MAAS_API_KEY}" \
  --skip-crds \
  --timeout 20m \
  --wait

# 7. Expose services
kubectl patch service rag -n rag-e2e \
  -p '{"spec":{"type":"NodePort","ports":[{"port":8501,"nodePort":30080}]}}'
kubectl patch service llamastack -n rag-e2e \
  -p '{"spec":{"type":"NodePort","ports":[{"port":8321,"nodePort":30081}]}}'

# OR use port-forward
kubectl port-forward -n rag-e2e svc/rag 8501:8501 &
kubectl port-forward -n rag-e2e svc/llamastack 8321:8321 &

# 8. Run tests
pip install -r tests/e2e/requirements.txt
export LLAMA_STACK_ENDPOINT=http://localhost:8321
export RAG_UI_ENDPOINT=http://localhost:8501
export INFERENCE_MODEL=llama-3-2-3b
export SKIP_MODEL_TESTS=false

python tests/e2e/test_user_workflow.py

# 9. Cleanup
pkill -f "kubectl port-forward" || true
helm uninstall rag -n rag-e2e
kubectl delete namespace rag-e2e
kind delete cluster --name rag-maas-test
rm kind-config.yaml
```

## Verifying MaaS Connection

### From Your Machine

```bash
# List available models
curl -s https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1/models \
  -H "Authorization: Bearer ${MAAS_API_KEY}" \
  | python3 -m json.tool

# Test chat completion
curl -s https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1/chat/completions \
  -H "Authorization: Bearer ${MAAS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-2-3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

### From Within Kubernetes

```bash
# Shell into llama-stack pod
kubectl exec -it -n rag-e2e deployment/llamastack -- sh

# Test MaaS connection from pod
curl -s https://llama-3-2-3b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1/models \
  -H "Authorization: Bearer ${OPENAI_API_KEY}"
```

## Troubleshooting

### Issue: API Key Not Working

**Symptoms**: 401 Unauthorized errors

**Solutions**:
1. Verify API key is correct
2. Check if key has expired
3. Ensure proper format: `Bearer ${MAAS_API_KEY}`
4. Verify GitHub secret is set correctly

### Issue: MaaS Endpoint Unreachable

**Symptoms**: Connection timeout or refused

**Solutions**:
1. Check network connectivity: `curl https://maas.apps.prod.rhoai.rh-aiservices-bu.com/`
2. Verify firewall/proxy settings
3. In Kind: Ensure cluster has external network access
4. Check if MaaS service is operational

### Issue: Tests Skip Inference

**Symptoms**: Tests report "No models available"

**Solutions**:
1. Check llama-stack logs: `kubectl logs -l app.kubernetes.io/name=llamastack -n rag-e2e`
2. Verify OPENAI_API_KEY is set in llama-stack pod
3. Confirm OPENAI_BASE_URL environment variable
4. Check if model ID matches: `llama-3-2-3b`

### Issue: Ingestion Pipeline Fails

**Symptoms**: Upload/RAG tests fail

**Solutions**:
1. Check if ingestion-pipeline is enabled in values
2. Verify MinIO and pgvector are running
3. Check ingestion pipeline logs
4. Ensure sample documents are uploaded

## What Gets Tested

With MaaS integration enabled, the e2e tests validate:

### ✅ Infrastructure
- RAG UI deployment and accessibility
- Llama Stack backend connectivity
- Database (pgvector) functionality
- Object storage (MinIO) functionality

### ✅ Inference (NEW with MaaS)
- Model availability via MaaS
- Chat completions with real LLM
- Token usage and response handling
- Error handling and retries

### ✅ RAG Workflow (NEW with MaaS)
- Document ingestion pipeline
- Vector database creation
- Embedding generation
- RAG-based query and response

### ⚠️ Not Tested (Yet)
- Multi-turn conversations
- Streaming responses
- Shield/guardrail functionality
- Web search integration
- Advanced RAG features (reranking, etc.)

## Next Steps

To extend testing:

1. **Add vector DB tests**: Create DB, upload docs, query
2. **Test upload UI**: Automate document upload through Streamlit
3. **Test chat UI**: Use Streamlit's testing framework
4. **Add performance tests**: Measure latency, throughput
5. **Test error scenarios**: Rate limits, timeouts, bad inputs

## References

- MaaS Portal: https://maas.apps.prod.rhoai.rh-aiservices-bu.com/
- E2E Tests Documentation: [tests/e2e/README.md](../tests/e2e/README.md)
- Main Documentation: [README.md](../README.md)
- MaaS Integration Plan: [maas-integration-plan.md](maas-integration-plan.md)

