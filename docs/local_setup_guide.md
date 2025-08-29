# 🦙 Llama Stack Local Setup Guide

This guide walks you through running a Llama Stack server locally using **Ollama** and **Podman**.

---

## ✅ 1. Prerequisites

Install the following tools:

- [Podman](https://podman.io/docs/installation)
- Python 3.10 or newer
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Ollama](https://ollama.com/download)

Verify tools:

```bash
podman --version
python3 --version
pip --version
ollama --version
```

Note: `pip` might become available only after you setup your venv (see below).  Also, `docker` works as well as podman. 

---

## 🚀 2. Start Ollama Server

Use the following command to launch the Ollama model:

```bash
ollama run llama3.2:3b-instruct-fp16 --keepalive 60m
```

Note: This will keep the model in memory for 60 minutes.

---

## ⚙️ 3. Configure Environment Variables

Launch a new terminal window to set the necessary environment variables:

```bash
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export LLAMA_STACK_PORT=8321
```

---

## 🐳 4. Run Llama Stack with Podman

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

Note:  Another option is to connect to a remote Llama Stack Service.  Based on the instructions of this quickstart, you should have one running in cluster and you can use `oc port-forward` to make it available on localhost.

```bash
oc get services -l app.kubernetes.io/name=llamastack

oc port-forward svc/llamastack 8321:8321
```

This makes the remote Service available at localhost:8321

---

## 🐍 5. Set Up Python Environment

Switch over to the `frontend` directory

```bash
cd frontend
```

Create a Python venv using your Python3.x executable

```bash
python3.11 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate  # macOS/Linux
# Windows:
# llama-stack-demo\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Check installation:

```bash
pip show llama-stack-client
```

```
Output
Name: llama_stack_client
Version: 0.2.9
Summary: The official Python library for the llama-stack-client API
Home-page: https://github.com/meta-llama/llama-stack-client-python
Author:
Author-email: Llama Stack Client <dev-feedback@llama-stack-client.com>
License:
Location: /Users/bsutter/my-projects/redhat/RAG-Blueprint/frontend/.venv/lib/python3.11/site-packages
Requires: anyio, click, distro, httpx, pandas, prompt-toolkit, pyaml, pydantic, rich, sniffio, termcolor, tqdm, typing-extensions
Required-by: llama_stack
```

---

## 📡 6. Configure the Client

Point the client to the local Llama Stack server:

```bash
llama-stack-client configure --endpoint http://localhost:$LLAMA_STACK_PORT
```

Hit enter as there is no API key for ollama and this container based Llama Stack server

```
> Enter the API key (leave empty if no key is needed):
```

List models:

```bash
llama-stack-client models list
```

```
Output
Available Models

┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ model_type        ┃ identifier                                         ┃ provider_resource_id                    ┃ metadata                                        ┃ provider_id       ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ embedding         │ all-MiniLM-L6-v2                                   │ all-minilm:latest                       │ {'embedding_dimension': 384.0}                  │ ollama            │
├───────────────────┼────────────────────────────────────────────────────┼─────────────────────────────────────────┼─────────────────────────────────────────────────┼───────────────────┤
│ llm               │ meta-llama/Llama-3.2-3B-Instruct                   │ llama3.2:3b-instruct-fp16               │                                                 │ ollama            │
└───────────────────┴────────────────────────────────────────────────────┴─────────────────────────────────────────┴─────────────────────────────────────────────────┴───────────────────┘

Total models: 2
```

## 7. Run the RAG UI

```bash
streamlit run llama_stack/distribution/ui/app.py
```

## 8. Redeploy changes

Deployment after making changes requires a rebuild of the container image using either `docker` or `podman`.  Replace `docker.io` with your target container registry such as `quay.io`.

Note: Use your favorite repository, organization and image name

```bash
export CONTAINER_REGISTRY=docker.io # quay.io
export CONTAINER_ORGANIZATION=yourorg
export IMAGE_NAME=rag
export IMAGE_TAG=1.0.0
```

```bash
podman buildx build --platform linux/amd64,linux/arm64 -t $CONTAINER_REGISTRY/$CONTAINER_ORGANIZATION/$IMAGE_NAME:$IMAGE_TAG -f Containerfile .
```

```bash
podman login $CONTAINER_REGISTRY
```

```bash
podman push $CONTAINER_REGISTRY/$CONTAINER_ORGANIZATION/$IMAGE_NAME:$IMAGE_TAG
```

Add modification to `deploy/helm/rag/values.yaml`, replacing the placeholders with your values

```
image:
  repository: {CONTAINER_REGISTRY}/{CONTAINER_ORGANIZATION}/{IMAGE_NAME}
  pullPolicy: IfNotPresent
  tag: {IMAGE_TAG}
```

 To redeploy to the cluster run the same `make` command as you did before.


---

## Troubleshooting Tips

**Check if Podman is running:**

```bash
podman ps
```

**Activate your virtual environment:**

```bash
source .venv/bin/activate
```

**Reinstall the client if needed:**

```bash
pip uninstall llama-stack-client
pip install llama-stack-client
```

**Test the client in Python:**

```bash
python -c "from llama_stack_client import LlamaStackClient; print(LlamaStackClient)"
```

## 10. Running the GUI locally

```bash
cd llama_stack/distribution/ui/
```

```bash
streamlit run app.py
```

