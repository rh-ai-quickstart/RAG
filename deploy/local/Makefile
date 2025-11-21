# RAG Stack Local Deployment Makefile
# Manages the RAG stack using podman-compose

.PHONY: help start stop restart status logs cleanup upload check-deps build build-ui build-and-push-ui dev

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Configuration
COMPOSE_FILE := podman-compose.yml
PROJECT_NAME := rag
OLLAMA_MODEL := llama3.2:3b-instruct-fp16

# UI Build Configuration
VERSION ?= 0.2.23
TAVILY_SEARCH_API_KEY ?= ""
CONTAINER_REGISTRY ?= quay.io/rh-ai-quickstart
DIST_UI_DIR := $(abspath ../../frontend)
PLATFORM := linux/amd64

# TAVILY API Key persistence
TAVILY_KEY_FILE := $(HOME)/.rag_tavily_key

# Service names for easy reference
SERVICES := llamastack rag-ui

# Ollama configuration (expected to run on host)
OLLAMA_HOST_URL := http://localhost:11434

# Helper function to get TAVILY key (from file, environment, or prompt user)
define get_tavily_key
	@if [ -n "$(TAVILY_SEARCH_API_KEY)" ]; then \
		echo "$(GREEN)[SUCCESS]$(NC) Using TAVILY_SEARCH_API_KEY from environment: $(TAVILY_SEARCH_API_KEY)"; \
		echo "$(TAVILY_SEARCH_API_KEY)" > /tmp/tavily_key_$(1); \
	elif [ -f "$(TAVILY_KEY_FILE)" ]; then \
		TAVILY_KEY=$$(cat "$(TAVILY_KEY_FILE)" 2>/dev/null | tr -d '\n\r'); \
		if [ -n "$$TAVILY_KEY" ]; then \
			echo "$(GREEN)[SUCCESS]$(NC) Using TAVILY_SEARCH_API_KEY from saved file"; \
			echo "$$TAVILY_KEY" > /tmp/tavily_key_$(1); \
		else \
			echo "$(YELLOW)[INFO]$(NC) TAVILY_SEARCH_API_KEY is not set. Web search functionality will be disabled."; \
			echo "$(BLUE)[INFO]$(NC) To enable web search, you can:"; \
			echo "  1. Set environment variable: export TAVILY_SEARCH_API_KEY=your_key_here"; \
			echo "  2. Run with key: TAVILY_SEARCH_API_KEY=your_key_here make $(1)"; \
			echo "  3. Enter key now (press Enter to skip):"; \
			read -p "TAVILY_SEARCH_API_KEY: " tavily_key; \
			if [ -n "$$tavily_key" ]; then \
				echo "$(BLUE)[INFO]$(NC) Would you like to save this API key for future use? (y/N):"; \
				read -p "Save API key to $(TAVILY_KEY_FILE)? " save_key; \
				if [ "$$save_key" = "y" ] || [ "$$save_key" = "Y" ]; then \
					echo "$$tavily_key" > "$(TAVILY_KEY_FILE)"; \
					chmod 600 "$(TAVILY_KEY_FILE)"; \
					echo "$(GREEN)[SUCCESS]$(NC) TAVILY_SEARCH_API_KEY saved and set."; \
				else \
					echo "$(BLUE)[INFO]$(NC) TAVILY_SEARCH_API_KEY set for this session only."; \
				fi; \
				echo "$$tavily_key" > /tmp/tavily_key_$(1); \
			else \
				echo "$(BLUE)[INFO]$(NC) Skipping TAVILY_SEARCH_API_KEY setup."; \
				echo "" > /tmp/tavily_key_$(1); \
			fi; \
		fi; \
	else \
		echo "$(YELLOW)[INFO]$(NC) TAVILY_SEARCH_API_KEY is not set. Web search functionality will be disabled."; \
		echo "$(BLUE)[INFO]$(NC) To enable web search, you can:"; \
		echo "  1. Set environment variable: export TAVILY_SEARCH_API_KEY=your_key_here"; \
		echo "  2. Run with key: TAVILY_SEARCH_API_KEY=your_key_here make $(1)"; \
		echo "  3. Enter key now (press Enter to skip):"; \
		read -p "TAVILY_SEARCH_API_KEY: " tavily_key; \
		if [ -n "$$tavily_key" ]; then \
			echo "$(BLUE)[INFO]$(NC) Would you like to save this API key for future use? (y/N):"; \
			read -p "Save API key to $(TAVILY_KEY_FILE)? " save_key; \
			if [ "$$save_key" = "y" ] || [ "$$save_key" = "Y" ]; then \
				echo "$$tavily_key" > "$(TAVILY_KEY_FILE)"; \
				chmod 600 "$(TAVILY_KEY_FILE)"; \
				echo "$(GREEN)[SUCCESS]$(NC) TAVILY_SEARCH_API_KEY saved and set."; \
			else \
				echo "$(BLUE)[INFO]$(NC) TAVILY_SEARCH_API_KEY set for this session only."; \
			fi; \
			echo "$$tavily_key" > /tmp/tavily_key_$(1); \
		else \
			echo "$(BLUE)[INFO]$(NC) Skipping TAVILY_SEARCH_API_KEY setup."; \
			echo "" > /tmp/tavily_key_$(1); \
		fi; \
	fi
endef

help: ## Show this help message
	@echo "RAG Stack Local Deployment"
	@echo ""
	@echo "Usage: make [TARGET]"
	@echo ""
	@echo "Quick Start:"
	@echo "  $(BLUE)make setup-ollama$(NC) - Install and configure Ollama on host machine (first time)"
	@echo "  $(BLUE)make dev$(NC)          - Start development environment (backend in containers + UI locally)"
	@echo "  $(BLUE)make start$(NC)        - Start full stack in containers"
	@echo "  $(BLUE)make stop$(NC)         - Stop all services"
	@echo ""
	@echo "Important Notes:"
	@echo "  - Ollama now runs on the host machine (not in container) for better performance"
	@echo "  - Run '$(BLUE)make setup-ollama$(NC)' if you haven't installed Ollama yet"
	@echo ""
	@echo "Environment Variables:"
	@echo "  TAVILY_SEARCH_API_KEY - Set your Tavily API key for web search functionality"
	@echo "    Example: TAVILY_SEARCH_API_KEY=your_key_here make dev"
	@echo "    Or export: export TAVILY_SEARCH_API_KEY=your_key_here && make dev"
	@echo ""
	@echo "All Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-deps: ## Check if required dependencies are installed
	@echo "$(BLUE)[INFO]$(NC) Checking dependencies..."
	@command -v podman >/dev/null 2>&1 || { echo "$(RED)[ERROR]$(NC) podman is not installed. Please install podman first."; exit 1; }
	@command -v podman-compose >/dev/null 2>&1 || { echo "$(RED)[ERROR]$(NC) podman-compose is not installed. Install with: pip install podman-compose"; exit 1; }
	@command -v uv >/dev/null 2>&1 || { echo "$(RED)[ERROR]$(NC) uv is not installed. Install with: pip install uv"; exit 1; }
	@command -v ollama >/dev/null 2>&1 || { echo "$(RED)[ERROR]$(NC) ollama is not installed. Run '$(BLUE)make setup-ollama$(NC)' to install it."; exit 1; }
	@echo "$(GREEN)[SUCCESS]$(NC) All dependencies are installed."

setup-ollama: ## Install and configure Ollama on the host machine
	@echo "$(BLUE)[INFO]$(NC) Setting up Ollama on host machine..."
	@if command -v ollama >/dev/null 2>&1; then \
		echo "$(GREEN)[SUCCESS]$(NC) Ollama is already installed."; \
	else \
		echo "$(BLUE)[INFO]$(NC) Installing Ollama..."; \
		if [ "$$(uname)" = "Darwin" ]; then \
			echo "$(BLUE)[INFO]$(NC) Please visit https://ollama.ai/download and install Ollama for macOS"; \
			echo "$(BLUE)[INFO]$(NC) Or use Homebrew: brew install ollama"; \
		elif [ "$$(uname)" = "Linux" ]; then \
			echo "$(BLUE)[INFO]$(NC) Installing Ollama for Linux..."; \
			curl -fsSL https://ollama.ai/install.sh | sh; \
		else \
			echo "$(YELLOW)[WARNING]$(NC) Unsupported OS. Please visit https://ollama.ai/download"; \
		fi; \
	fi
	@echo "$(BLUE)[INFO]$(NC) Checking if Ollama service is running..."
	@sleep 2
	@curl -sf $(OLLAMA_HOST_URL)/api/version > /dev/null 2>&1 || { \
		echo "$(YELLOW)[WARNING]$(NC) Ollama service is not running yet."; \
		echo "$(BLUE)[INFO]$(NC) Starting Ollama service..."; \
		if [ "$$(uname)" = "Darwin" ]; then \
			echo "$(BLUE)[INFO]$(NC) On macOS, Ollama should start automatically. If not, run: ollama serve"; \
		else \
			echo "$(BLUE)[INFO]$(NC) Starting Ollama in background..."; \
			nohup ollama serve > /tmp/ollama.log 2>&1 & \
			sleep 5; \
		fi; \
	}
	@echo "$(BLUE)[INFO]$(NC) Pulling required model: $(OLLAMA_MODEL)..."
	@ollama pull $(OLLAMA_MODEL) || { echo "$(RED)[ERROR]$(NC) Failed to pull model."; exit 1; }
	@echo "$(GREEN)[SUCCESS]$(NC) Ollama setup complete!"
	@echo ""
	@echo "Ollama is now running at: $(OLLAMA_HOST_URL)"
	@echo "Model '$(OLLAMA_MODEL)' is ready to use."

check-ollama: ## Check if Ollama is running on host
	@echo "$(BLUE)[INFO]$(NC) Checking if Ollama is running on host..."
	@curl -sf $(OLLAMA_HOST_URL)/api/version > /dev/null 2>&1 || { \
		echo "$(RED)[ERROR]$(NC) Ollama is not running or not accessible at $(OLLAMA_HOST_URL)"; \
		echo "$(BLUE)[INFO]$(NC) Please start Ollama:"; \
		echo "  - On macOS/Linux: ollama serve (or it may already be running as a service)"; \
		echo "  - On Windows: Ollama typically runs as a background service"; \
		echo ""; \
		echo "$(BLUE)[INFO]$(NC) Then pull the required model:"; \
		echo "  ollama pull $(OLLAMA_MODEL)"; \
		exit 1; \
	}
	@echo "$(GREEN)[SUCCESS]$(NC) Ollama is running on host at $(OLLAMA_HOST_URL)"
	@echo "$(BLUE)[INFO]$(NC) Checking if required model is available..."
	@ollama list | grep -q "$(OLLAMA_MODEL)" || { \
		echo "$(YELLOW)[WARNING]$(NC) Model $(OLLAMA_MODEL) not found locally."; \
		echo "$(BLUE)[INFO]$(NC) Pulling model $(OLLAMA_MODEL)..."; \
		ollama pull $(OLLAMA_MODEL) || { \
			echo "$(RED)[ERROR]$(NC) Failed to pull model. Please run: ollama pull $(OLLAMA_MODEL)"; \
			exit 1; \
		}; \
	}
	@echo "$(GREEN)[SUCCESS]$(NC) Model $(OLLAMA_MODEL) is available."

start: check-deps check-ollama ## Start all RAG stack services
	@echo "$(BLUE)[INFO]$(NC) Starting RAG stack services..."
	@$(call get_tavily_key,start)
	@TAVILY_KEY=$$(cat /tmp/tavily_key_start 2>/dev/null || echo ""); \
	rm -f /tmp/tavily_key_start; \
	if [ "$$(uname)" = "Linux" ]; then \
		echo "$(BLUE)[INFO]$(NC) Detected Linux - using 172.17.0.1 for container networking..."; \
		export OLLAMA_URL="http://172.17.0.1:11434"; \
	else \
		echo "$(BLUE)[INFO]$(NC) Detected macOS/Windows - using host.docker.internal..."; \
		export OLLAMA_URL="http://host.docker.internal:11434"; \
	fi; \
	echo "$(BLUE)[DEBUG]$(NC) Using OLLAMA_URL: $$OLLAMA_URL"; \
	echo "$(BLUE)[DEBUG]$(NC) Resolved TAVILY key: '$$TAVILY_KEY'"; \
	export TAVILY_SEARCH_API_KEY="$$TAVILY_KEY"; \
	podman-compose up -d
	@echo "$(BLUE)[INFO]$(NC) Waiting for services to be healthy..."
	@sleep 30
	@echo "$(GREEN)[SUCCESS]$(NC) RAG stack is starting up. Please wait a few minutes for all services to be ready."
	@echo ""
	@echo "Service URLs:"
	@echo "  - RAG UI: http://localhost:8501"
	@echo "  - Ollama API (host): http://localhost:11434"
	@echo "  - Llama Stack: http://localhost:8321"

stop: ## Stop all RAG stack services
	@echo "$(BLUE)[INFO]$(NC) Stopping RAG stack services..."
	@podman-compose down 2>/dev/null || true
	@echo "$(BLUE)[INFO]$(NC) Cleaning up any manual containers..."
	@podman stop rag-ui-manual 2>/dev/null || true
	@podman rm rag-ui-manual 2>/dev/null || true
	@echo "$(GREEN)[SUCCESS]$(NC) RAG stack services stopped."

restart: ## Restart all RAG stack services
	@echo "$(BLUE)[INFO]$(NC) Restarting RAG stack services..."
	@podman-compose restart
	@echo "$(GREEN)[SUCCESS]$(NC) RAG stack services restarted."

status: ## Show status of all services
	@echo "$(BLUE)[INFO]$(NC) RAG stack service status:"
	@podman-compose ps

logs: ## Show logs for all services (use logs-SERVICE for specific service)
	@echo "$(BLUE)[INFO]$(NC) Showing logs for all services..."
	@podman-compose logs -f

logs-llamastack: ## Show logs for llamastack service
	@echo "$(BLUE)[INFO]$(NC) Showing logs for llamastack..."
	@podman-compose logs -f llamastack

logs-ui: ## Show logs for rag-ui service
	@echo "$(BLUE)[INFO]$(NC) Showing logs for rag-ui..."
	@podman-compose logs -f rag-ui

logs-ingestion: ## Show logs for ingestion service
	@echo "$(BLUE)[INFO]$(NC) Showing logs for ingestion service..."
	@podman logs rag-ingestion

pull: ## Pull all container images
	@echo "$(BLUE)[INFO]$(NC) Pulling container images..."
	@podman-compose pull
	@echo "$(GREEN)[SUCCESS]$(NC) Container images pulled successfully."


test-services: ## Test if all services are responding
	@echo "$(BLUE)[INFO]$(NC) Testing service endpoints..."
	@echo -n "Testing Ollama (host)... "
	@curl -s http://localhost:11434/api/version > /dev/null && echo "$(GREEN)✓$(NC)" || echo "$(RED)✗$(NC)"
	@echo -n "Testing Llama Stack... "
	@curl -s http://localhost:8321/alpha/health > /dev/null && echo "$(GREEN)✓$(NC)" || echo "$(RED)✗$(NC)"
	@echo -n "Testing RAG UI... "
	@curl -s http://localhost:8501 > /dev/null && echo "$(GREEN)✓$(NC)" || echo "$(RED)✗$(NC)"

cleanup: ## Remove all containers, networks, and volumes (including data)
	@echo "$(YELLOW)[WARNING]$(NC) This will remove all containers, networks, and volumes (including data)."
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ] || { echo "$(BLUE)[INFO]$(NC) Cleanup cancelled."; exit 0; }
	@echo "$(BLUE)[INFO]$(NC) Cleaning up RAG stack..."
	@podman-compose down -v 2>/dev/null || true
	@podman stop rag-ui-manual 2>/dev/null || true
	@podman rm rag-ui-manual 2>/dev/null || true
	@podman network rm rag_rag-network -f 2>/dev/null || true
	@echo "$(GREEN)[SUCCESS]$(NC) RAG stack cleaned up."

reset: cleanup ## Complete reset - cleanup and rebuild everything
	@echo "$(BLUE)[INFO]$(NC) Performing complete reset..."
	@$(MAKE) build
	@$(MAKE) start
	@echo "$(GREEN)[SUCCESS]$(NC) RAG stack reset complete."

shell-llamastack: ## Open shell in llamastack container
	@podman exec -it $(PROJECT_NAME)-llamastack-1 /bin/bash


monitor: ## Show real-time resource usage
	@echo "$(BLUE)[INFO]$(NC) Monitoring RAG stack resources (Ctrl+C to exit)..."
	@watch -n 2 'podman-compose ps && echo "" && podman stats --no-stream'

config: ## Show current configuration
	@echo "$(BLUE)[INFO]$(NC) Current RAG stack configuration:"
	@echo "  - Compose file: $(COMPOSE_FILE)"
	@echo "  - Project name: $(PROJECT_NAME)"
	@echo "  - Ollama model: $(OLLAMA_MODEL)"
	@echo "  - Ollama running on: Host machine"
	@echo "  - Services: $(SERVICES)"
	@echo ""
	@echo "$(BLUE)[INFO]$(NC) Service ports:"
	@echo "  - RAG UI: 8501"
	@echo "  - Ollama (host): 11434"
	@echo "  - Llama Stack: 8321"

# UI Build and Deployment Targets
build-ui: ## Build the RAG UI container locally
	@echo "$(BLUE)[INFO]$(NC) Building RAG UI container for $(PLATFORM)..."
	@podman build --platform $(PLATFORM) -t llamastack-dist-ui:$(VERSION) -f $(DIST_UI_DIR)/Containerfile $(DIST_UI_DIR)
	@echo "$(GREEN)[SUCCESS]$(NC) RAG UI container built successfully."

build-and-push-ui: build-ui ## Build and push UI container to registry
	@echo "$(BLUE)[INFO]$(NC) Pushing UI container to registry..."
	@podman login $(CONTAINER_REGISTRY)
	@podman tag llamastack-dist-ui:$(VERSION) $(CONTAINER_REGISTRY)/llamastack-dist-ui:$(VERSION)
	@podman push $(CONTAINER_REGISTRY)/llamastack-dist-ui:$(VERSION)
	@echo "$(GREEN)[SUCCESS]$(NC) UI container pushed to registry."

dev: check-ollama ## Start backend services in containers and run UI locally for development
	@echo "$(BLUE)[INFO]$(NC) Starting development environment..."
	@echo "$(BLUE)[INFO]$(NC) Starting backend services (Llama Stack) in containers..."
	@echo "$(BLUE)[INFO]$(NC) Using Ollama on host machine at $(OLLAMA_HOST_URL)"
	@$(call get_tavily_key,dev)
	@TAVILY_KEY=$$(cat /tmp/tavily_key_dev 2>/dev/null || echo ""); \
	rm -f /tmp/tavily_key_dev; \
	mkdir -p ~/.local; \
	if [ "$$(uname)" = "Linux" ]; then \
		echo "$(BLUE)[INFO]$(NC) Detected Linux - using 172.17.0.1 for container networking..."; \
		export OLLAMA_URL="http://172.17.0.1:11434"; \
	else \
		echo "$(BLUE)[INFO]$(NC) Detected macOS/Windows - using host.docker.internal..."; \
		export OLLAMA_URL="http://host.docker.internal:11434"; \
	fi; \
	echo "$(BLUE)[DEBUG]$(NC) Using OLLAMA_URL: $$OLLAMA_URL"; \
	echo "$(BLUE)[DEBUG]$(NC) Resolved TAVILY key: '$$TAVILY_KEY'"; \
	export TAVILY_SEARCH_API_KEY="$$TAVILY_KEY"; \
	podman-compose up -d llamastack; \
	echo "$(BLUE)[INFO]$(NC) Waiting for backend services to be ready..."; \
	sleep 30; \
	echo "$(GREEN)[SUCCESS]$(NC) Backend services started successfully."; \
	echo ""; \
	echo "Backend services available at:"; \
	echo "  - Ollama API (host): http://localhost:11434"; \
	echo "  - Llama Stack: http://localhost:8321"; \
	echo ""; \
	echo "$(BLUE)[INFO]$(NC) Starting RAG UI locally for development..."; \
	cd ../../frontend && \
	VERSION=$(VERSION) TAVILY_SEARCH_API_KEY="$$TAVILY_KEY" ./start.sh



# API Key Management
clear-tavily-key: ## Clear the saved TAVILY_SEARCH_API_KEY
	@if [ -f "$(TAVILY_KEY_FILE)" ]; then \
		rm "$(TAVILY_KEY_FILE)"; \
		echo "$(GREEN)[SUCCESS]$(NC) Saved TAVILY_SEARCH_API_KEY cleared."; \
	else \
		echo "$(BLUE)[INFO]$(NC) No saved TAVILY_SEARCH_API_KEY found."; \
	fi

ingest: ## Re-run the ingestion service
	@echo "$(BLUE)[INFO]$(NC) Re-running ingestion service..."
	@podman-compose up --build rag-ingestion
	@echo "$(GREEN)[SUCCESS]$(NC) Ingestion completed."

ingest-config: ## Edit the ingestion configuration
	@echo "$(BLUE)[INFO]$(NC) Opening ingestion configuration..."
	@$${EDITOR:-vim} ingestion-config.yaml

list-vector-dbs: ## List all vector databases
	@echo "$(BLUE)[INFO]$(NC) Listing vector databases..."
	@cd ../../client-examples-python && \
	LLAMA_STACK_SERVER=http://localhost:8321 python rag-list-vector-db.py || \
	echo "$(YELLOW)[WARNING]$(NC) Failed to list vector DBs. Is Llama Stack running?"

# Quick aliases for common tasks
up: start ## Alias for start
down: stop ## Alias for stop
ps: status ## Alias for status
