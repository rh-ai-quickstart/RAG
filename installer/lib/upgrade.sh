#!/bin/bash
# ============================================================================
# RAG Quickstart — Upgrade Logic
# ============================================================================
# Upgrade uses the same helm upgrade --install path as install.
# The deploy_quickstart function in install.sh already handles both cases.

upgrade_quickstart() {
  log_status "running" "upgrading" "Running upgrade (helm upgrade --install)..."
  deploy_quickstart
}
