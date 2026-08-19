#!/bin/bash
# ============================================================================
# RAG Quickstart — Uninstallation Logic
# ============================================================================

RAG_CHART_NAME="rag"

cleanup_quickstart() {
  local mode=$1
  local ns="$TARGET_NAMESPACE"

  # Check if namespace exists
  if ! oc get namespace "$ns" &>/dev/null; then
    log_status "running" "uninstalling" "Namespace ${ns} does not exist. Nothing to uninstall."
    return 0
  fi

  # Check namespace phase
  local ns_phase
  ns_phase=$(oc get namespace "$ns" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
  if [[ "$ns_phase" == "Terminating" ]]; then
    log_status "running" "uninstalling" "Namespace ${ns} is already terminating. Waiting..."
    local wait_count=0
    while [[ $wait_count -lt 60 ]]; do
      if ! oc get namespace "$ns" &>/dev/null; then
        log_status "running" "uninstalling" "Namespace ${ns} terminated."
        return 0
      fi
      sleep 5
      wait_count=$((wait_count + 1))
    done
    log_status "running" "uninstalling" "WARNING: Namespace ${ns} still terminating after 5 minutes"
    return 0
  fi

  # Uninstall Helm release
  log_status "running" "uninstalling" "Uninstalling Helm release ${RAG_CHART_NAME}..."
  helm -n "$ns" uninstall "$RAG_CHART_NAME" 2>/dev/null || {
    log_status "running" "uninstalling" "No Helm release '${RAG_CHART_NAME}' found in namespace ${ns}"
  }

  case "$mode" in
    delete-all)
      # Remove PVCs
      log_status "running" "uninstalling" "Removing PVCs from ${ns}..."
      local pvcs
      pvcs=$(oc get pvc -n "$ns" -o custom-columns=NAME:.metadata.name --no-headers 2>/dev/null || echo "")
      if [[ -n "$pvcs" ]]; then
        echo "$pvcs" | while IFS= read -r pvc; do
          [[ -z "$pvc" ]] && continue
          oc delete pvc "$pvc" -n "$ns" --ignore-not-found=true 2>/dev/null || true
        done
      fi

      # Delete remaining pods
      log_status "running" "uninstalling" "Deleting remaining pods in ${ns}..."
      oc delete pods -n "$ns" --all 2>/dev/null || true

      # Delete namespace
      log_status "running" "uninstalling" "Deleting namespace ${ns}..."
      oc delete project "$ns" --ignore-not-found=true 2>/dev/null || true

      # Wait briefly for namespace to start terminating
      local wait_count=0
      while [[ $wait_count -lt 24 ]]; do
        if ! oc get namespace "$ns" &>/dev/null; then
          log_status "running" "uninstalling" "Namespace ${ns} deleted."
          break
        fi
        sleep 5
        wait_count=$((wait_count + 1))
      done
      ;;

    keep-data)
      # Delete remaining pods but keep PVCs
      log_status "running" "uninstalling" "Deleting pods in ${ns} (keeping PVCs)..."
      oc delete pods -n "$ns" --all 2>/dev/null || true

      local pvcs_kept
      pvcs_kept=$(oc get pvc -n "$ns" -o custom-columns=NAME:.metadata.name --no-headers 2>/dev/null || echo "none")
      log_status "running" "uninstalling" "Preserved PVCs: ${pvcs_kept}"
      ;;

    *)
      log_error "Unknown uninstall mode: ${mode}"
      ;;
  esac
}
