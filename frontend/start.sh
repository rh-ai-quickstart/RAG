#!/bin/bash

cd "$(dirname "$0")"

uv sync && \
uv run streamlit run llama_stack_ui/distribution/ui/app.py --server.runOnSave true --server.fileWatcherType auto