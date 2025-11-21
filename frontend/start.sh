#!/bin/bash

cd "$(dirname "$0")"

# Export TAVILY_SEARCH_API_KEY if it's set
if [ -n "$TAVILY_SEARCH_API_KEY" ]; then
    export TAVILY_SEARCH_API_KEY
fi

uv sync && \
uv run streamlit run llama_stack_ui/distribution/ui/app.py --server.runOnSave true --server.fileWatcherType auto