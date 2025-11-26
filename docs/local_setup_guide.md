<!-- omit from toc -->
# Local Deployment
This guide walks you through running a Llama Stack server locally using **Ollama** and **Podman**.

> **⚡ Performance Update**: Ollama now runs directly on your host machine for significantly better performance. The quickstart uses `make setup-ollama` to automate the installation and configuration.

<!-- omit from toc -->
## Table of Contents
- [Prerequisites](#prerequisites)
- [Supported Models](#supported-models)
- [Installing the RAG QuickStart](#installing-the-rag-quickstart)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Start Ollama Server](#2-start-ollama-server)
  - [3. Configure Environment Variables](#3-configure-environment-variables)
  - [4. Run Llama Stack with Podman](#4-run-llama-stack-with-podman)
- [Quick Start with Podman Compose](#quick-start-with-podman-compose)
  - [Setup](#setup)
  - [Useful Commands](#useful-commands)
- [Automatic Document Ingestion](#automatic-document-ingestion)
  - [Configuration](#configuration)
  - [Source Types](#source-types)
    - [GitHub Repository](#github-repository)
    - [S3/MinIO Bucket](#s3minio-bucket)
    - [Direct URLs](#direct-urls)
  - [Managing Ingestion](#managing-ingestion)
- [Using the RAG UI](#using-the-rag-ui)
- [Environment Variables](#environment-variables)
- [Redeploying Changes](#redeploying-changes)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- [Podman](https://podman.io/docs/installation) or Docker
- [podman-compose](https://github.com/containers/podman-compose) - Install with: `pip install podman-compose`
- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [Ollama](https://ollama.com/download) - **runs on host machine for better performance**

You can install Ollama manually or use the automated setup:

** Note: If running on Apple Silicon please read [Apple Silicon Guide](./AppleSiliconReadMe.md)

```bash
# Automated setup (recommended - from deploy/local directory)
cd deploy/local
make setup-ollama

# Or check if all dependencies are installed
cd deploy/local
make check-deps

# Or verify manual installation
podman --version
podman-compose --version
python3 --version
uv --version
ollama --version
```

**Note**: 
- `podman-compose` is required to orchestrate multiple containers - install it with `pip install podman-compose`
- `uv` is used for fast Python package management and virtual environment handling
- `docker` works as well as podman, but these instructions use podman
- **Ollama now runs on your host** instead of in a container for ~3x better inference performance
- Use `make check-deps` to verify all required dependencies (podman, podman-compose, uv, ollama) are installed
- See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed Ollama configuration and troubleshooting

## Supported Models

| Function    | Model Name                             | Hardware    | Local Support |
|-------------|----------------------------------------|-------------|---------------|
| Embedding   | `all-MiniLM-L6-v2`                     | CPU/GPU     | ✓             |
| Generation  | `meta-llama/Llama-3.2-3B-Instruct`     | CPU/GPU     | ✓             |

Note: Local deployment primarily supports CPU and GPU acceleration through Ollama. The models are automatically downloaded when first requested.

## Installing the RAG QuickStart

### 1. Clone Repository

Clone the repo so you have a working copy

```bash
git clone https://github.com/rh-ai-quickstart/RAG
```

### 2. Start Ollama Server

**Option A: Quick Setup (Recommended)**

Use the automated setup (requires navigating to `deploy/local` first):

```bash
cd deploy/local
make setup-ollama
```

This will install Ollama, start the service, and pull the required model.

**Option B: Manual Setup**

If you've already installed Ollama but need to start it manually:

```bash
# On macOS, Ollama usually starts automatically
# On Linux, you may need to start it manually:
ollama serve &

# Pull the model
ollama pull llama3.2:3b-instruct-fp16

# Optional: Pre-load the model with keepalive
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```

Note: The `--keepalive 60m` option keeps the model in memory for 60 minutes for faster inference. After this, proceed to use `make start` or `make up` to start all RAG services.

### 3. Configure Environment Variables

Launch a new terminal window to set the necessary environment variables:

```bash
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export LLAMA_STACK_PORT=8321
```

### 4. Run Llama Stack with Podman

Pull the Docker image:

```bash
podman pull docker.io/llamastack/distribution-ollama
```

Create a local directory for persistent data:

```bash
mkdir -p ~/.llama
```

Run the container:

```bash
podman run -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.containers.internal:11434 \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT
```

Optional: Use a custom network:

```bash
podman network create llama-net
podman run --privileged --network llama-net -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT
```

Verify the container is running:

```bash
podman ps
```

Note: Another option is to connect to a remote Llama Stack Service. Based on the instructions of this quickstart, you should have one running in cluster and you can use `oc port-forward` to make it available on localhost.

```bash
oc get services -l app.kubernetes.io/name=llamastack

oc port-forward svc/llamastack 8321:8321
```

This makes the remote Service available at localhost:8321

## Quick Start with Podman Compose

For a simpler deployment experience, use the provided Makefile commands which use `podman-compose` under the hood to automatically set up Llama Stack and the RAG UI. **Ollama runs directly on your host machine** for better performance instead of in a container.

> **💡 Tip**: Run `make help` in the `deploy/local` directory to see all available commands and their descriptions.

### Setup

1. Navigate to the local deployment directory:

```bash
cd deploy/local
```

2. **First-time setup only** - Install and configure Ollama on your host machine:

```bash
make setup-ollama
```

This will:
- Install Ollama on your host machine (if not already installed)
- Start the Ollama service
- Pull the `llama3.2:3b-instruct-fp16` model

> **Note**: This step is only needed once. After Ollama is installed, it will run as a background service (especially on macOS).

3. Start all containerized services:

```bash
make start
# or use the shorthand alias:
make up
```

This will:
- Check that Ollama is running on the host (will fail if not running)
- Start Llama Stack server using `podman-compose` (connects to host Ollama)
- Run the ingestion service to populate vector databases
- Start the RAG UI

**Important**: `make start` (or `make up`) only starts the **containerized services** using `podman-compose` - it does NOT install or start Ollama. If you haven't run `make setup-ollama` yet, `make start` will fail with an error asking you to start Ollama first.

4. Access the services:
- **RAG UI**: http://localhost:8501
- **Llama Stack API**: http://localhost:8321
- **Ollama API (host)**: http://localhost:11434

**Note**: For detailed information about running Ollama on the host, see [OLLAMA_SETUP.md](OLLAMA_SETUP.md).

### Useful Commands

```bash
# Check if Ollama is running on host
make check-ollama

# View all services status
make status

# View logs for all services
make logs

# View logs for specific service
make logs-ui
make logs-llamastack
make logs-ingestion

# Test all service endpoints
make test-services

# Restart all services
make restart

# Pull latest container images
make pull

# Stop all services (Note: This does NOT stop Ollama on host)
make stop

# Clean up everything (including containers, networks, and volumes)
make cleanup

# Complete reset (cleanup and rebuild everything)
make reset

# View current configuration
make config

# Monitor real-time resource usage
make monitor

# Open shell in llamastack container
make shell-llamastack
```

**Ingestion Commands:**

```bash
# Re-run ingestion service
make ingest

# Edit ingestion configuration
make ingest-config

# List all vector databases
make list-vector-dbs
```

**Development Mode**: Start backend services in containers and run UI locally for faster development:

```bash
make dev
```

This starts Llama Stack in a container (using host Ollama) and runs the UI locally with hot-reload enabled.

**UI Build Commands** (for advanced users):

```bash
# Build the RAG UI container locally
make build-ui

# Build and push UI container to registry
make build-and-push-ui
```

**API Key Management:**

```bash
# Clear saved TAVILY_SEARCH_API_KEY
make clear-tavily-key
```

**Quick Aliases:**

```bash
make up    # Alias for 'make start'
make down  # Alias for 'make stop'
make ps    # Alias for 'make status'
```

## Automatic Document Ingestion

The local deployment includes an automatic ingestion service that creates and populates vector databases from various sources.

### Configuration

Edit `deploy/local/ingestion-config.yaml` to configure your document sources:

```yaml
pipelines:
  my-docs-pipeline:
    enabled: true
    name: "my-docs-db"
    version: "1.0"
    vector_store_name: "my-docs-db-v1-0"
    source: GITHUB  # or S3, URL
    config:
      url: "https://github.com/yourorg/your-repo.git"
      path: "docs"
      branch: "main"
```

### Source Types

#### GitHub Repository

```yaml
source: GITHUB
config:
  url: "https://github.com/rh-ai-quickstart/RAG.git"
  path: "notebooks/hr"
  branch: "main"
  # token: "ghp_xxx"  # Optional for private repos
```

#### S3/MinIO Bucket

```yaml
source: S3
config:
  endpoint: "http://minio:9000"
  bucket: "documents"
  access_key: "minio_user"
  secret_key: "minio_password"
  # prefix: "folder/"  # Optional
```

#### Direct URLs

```yaml
source: URL
config:
  urls:
    - "https://example.com/document1.pdf"
    - "https://example.com/document2.pdf"
```

### Managing Ingestion

**View ingestion progress:**
```bash
make logs-ingestion
```

**Re-run ingestion:**
```bash
make ingest
```

**Edit configuration:**
```bash
make ingest-config
```

**Check created databases:**
```bash
make list-vector-dbs
```

The ingestion service will:
1. Wait for Llama Stack to be ready
2. Fetch documents from configured sources
3. Process PDFs with Docling (advanced chunking)
4. Create vector databases with embeddings
5. Insert all chunks into PGVector

For more details, see the [Ingestion Service README](../ingestion-service/README.md).

## Using the RAG UI

1. Navigate to the frontend directory

```bash
cd frontend
```

2. Run the RAG UI application using the provided start script

```bash
./start.sh
```

This script will automatically:
- Create a virtual environment (if it doesn't exist)
- Install/sync all dependencies with `uv sync`
- Start the Streamlit server with auto-reload enabled
- Watch for file changes and automatically restart

3. Open your browser to the displayed URL (typically `http://localhost:8501`)

4. Upload your PDF documents through the UI and start asking questions!

## Environment Variables

The RAG UI supports the following optional environment variables for configuration:

| Environment Variable       | Description                                    | Default Value             |
|----------------------------|------------------------------------------------|---------------------------|
| `LLAMA_STACK_ENDPOINT`     | The endpoint for the Llama Stack API server    | `http://localhost:8321`   |
| `FIREWORKS_API_KEY`        | API key for Fireworks AI provider (optional)   | (empty string)            |
| `TOGETHER_API_KEY`         | API key for Together AI provider (optional)    | (empty string)            |
| `SAMBANOVA_API_KEY`        | API key for SambaNova provider (optional)      | (empty string)            |
| `OPENAI_API_KEY`           | API key for OpenAI provider (optional)         | (empty string)            |
| `TAVILY_SEARCH_API_KEY`    | API key for Tavily search provider (optional)  | (empty string)            |

**Note**: For the default local deployment using Ollama, you typically only need to ensure `LLAMA_STACK_ENDPOINT` points to your running Llama Stack server. The API keys are only needed if you want to use external AI providers instead of the local Ollama setup.

To set environment variables before starting the UI:

```bash
export LLAMA_STACK_ENDPOINT=http://localhost:8321
./start.sh
```

## Redeploying Changes

Deployment after making changes requires a rebuild of the container image using either `docker` or `podman`. Replace `docker.io` with your target container registry such as `quay.io`.

Note: Use your favorite repository, organization and image name

```bash
export CONTAINER_REGISTRY=docker.io # quay.io
export CONTAINER_ORGANIZATION=yourorg
export IMAGE_NAME=rag
export IMAGE_TAG=1.0.0
```

Build the image:

```bash
podman buildx build --platform linux/amd64,linux/arm64 -t $CONTAINER_REGISTRY/$CONTAINER_ORGANIZATION/$IMAGE_NAME:$IMAGE_TAG -f Containerfile .
```

Login to registry:

```bash
podman login $CONTAINER_REGISTRY
```

Push the image:

```bash
podman push $CONTAINER_REGISTRY/$CONTAINER_ORGANIZATION/$IMAGE_NAME:$IMAGE_TAG
```

Add modification to `deploy/helm/rag/values.yaml`, replacing the placeholders with your values:

```yaml
image:
  repository: {CONTAINER_REGISTRY}/{CONTAINER_ORGANIZATION}/{IMAGE_NAME}
  pullPolicy: IfNotPresent
  tag: {IMAGE_TAG}
```

To redeploy to the cluster run the same `make` command as you did before.

## Troubleshooting

**Check if Ollama is running on host:**

```bash
make check-ollama
# Or manually:
curl http://localhost:11434/api/version
ollama list  # List available models
```

**If Ollama is not running:**

```bash
# Recommended: Use make command to setup/restart Ollama
make setup-ollama

# Or manually start Ollama (on Linux):
ollama serve &

# On macOS, Ollama typically runs as a background service automatically
```

**Check if containers are running:**

```bash
# Using make (recommended - shows podman-compose status):
make status

# Or check directly with podman:
podman ps
```

**Test all services:**

```bash
make test-services
```

**View service logs:**

```bash
make logs              # All services
make logs-llamastack   # Llama Stack only
make logs-ui           # UI only
make logs-ingestion    # Ingestion service
```

**Ollama can't be reached from containers:**

If Llama Stack can't connect to Ollama on the host:

```bash
# First, ensure Ollama is running
make check-ollama

# If that fails, verify Ollama is listening on all interfaces
export OLLAMA_HOST=0.0.0.0
ollama serve &

# Test from container
podman exec -it rag-llamastack curl http://host.containers.internal:11434/api/version

# On Linux, you may need to use the host IP directly:
podman exec -it rag-llamastack curl http://172.17.0.1:11434/api/version
```

**Model not found:**

```bash
# Pull the required model
ollama pull llama3.2:3b-instruct-fp16

# List available models
ollama list
```

**Reinstall dependencies if needed:**

```bash
uv sync --reinstall
```

**Test the client in Python:**

```bash
uv run python -c "from llama_stack_client import LlamaStackClient; print(LlamaStackClient)"
```

**Check uv environment:**

```bash
uv pip list
```

**Run commands in the uv environment:**

```bash
uv run <command>
```

**Reset the entire environment:**

```bash
rm -rf .venv
uv sync
```

**Complete reset of local deployment:**

```bash
# Option 1: Use the reset command (cleanup and rebuild)
make reset

# Option 2: Manual step-by-step reset
make cleanup          # Remove all containers and volumes
make setup-ollama     # Reinstall Ollama and models (if needed)
make start            # Start services fresh
```

For more detailed troubleshooting related to Ollama running on the host, see [OLLAMA_SETUP.md](OLLAMA_SETUP.md).
