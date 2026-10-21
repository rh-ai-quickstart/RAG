#!/bin/bash
# ============================================================================
# RAG Quickstart — Deployment Status Verification
# ============================================================================

RAG_CHART_NAME="rag"

verify_deployment() {
  local ns="$TARGET_NAMESPACE"

  # Check namespace existence
  if ! oc get namespace "$ns" &>/dev/null; then
    log_status "running" "verifying" "Namespace ${ns} does not exist — no deployment found."
    return 0
  fi

  local ns_phase
  ns_phase=$(oc get namespace "$ns" -o jsonpath='{.status.phase}' 2>/dev/null || echo "unknown")
  log_status "running" "verifying" "Namespace ${ns} phase: ${ns_phase}"

  if [[ "$ns_phase" == "Terminating" ]]; then
    log_status "running" "verifying" "Namespace ${ns} is terminating."
    return 0
  fi

  # Check Helm release
  local helm_status
  helm_status=$(helm status "$RAG_CHART_NAME" -n "$ns" -o json 2>/dev/null | jq -r '.info.status' 2>/dev/null || echo "not-installed")
  log_status "running" "verifying" "Helm release '${RAG_CHART_NAME}': ${helm_status}"

  if [[ "$helm_status" == "not-installed" ]]; then
    log_status "running" "verifying" "No Helm release found in namespace ${ns}."
    return 0
  fi

  # Pod status summary
  local pods
  pods=$(oc get pods -n "$ns" -o json 2>/dev/null || echo '{"items":[]}')

  local total ready running
  total=$(echo "$pods" | jq '.items | length' 2>/dev/null || echo "0")
  running=$(echo "$pods" | jq '[.items[] | select(.status.phase == "Running")] | length' 2>/dev/null || echo "0")
  ready=$(echo "$pods" | jq '[.items[] | select(.status.conditions[]? | select(.type == "Ready" and .status == "True"))] | length' 2>/dev/null || echo "0")

  log_status "running" "verifying" "Pods: ${ready} ready, ${running} running, ${total} total"

  # Report any unhealthy pods
  local failed
  failed=$(echo "$pods" | jq -r '[.items[] | select(.status.phase != "Running" and .status.phase != "Succeeded") | .metadata.name + " (" + .status.phase + ")"] | join(", ")' 2>/dev/null || echo "")
  if [[ -n "$failed" ]]; then
    log_status "running" "verifying" "Non-running pods: ${failed}"
  fi

  # Check CrashLoopBackOff / Error containers
  local crash_pods
  crash_pods=$(echo "$pods" | jq -r '[.items[] | .status.containerStatuses // [] | .[] | select(.state.waiting.reason == "CrashLoopBackOff" or .state.waiting.reason == "Error") | .name] | join(", ")' 2>/dev/null || echo "")
  if [[ -n "$crash_pods" ]]; then
    log_status "running" "verifying" "WARNING: Containers in CrashLoopBackOff/Error: ${crash_pods}"
  fi

  # Services
  local svc_count
  svc_count=$(oc get svc -n "$ns" -o json 2>/dev/null | jq '.items | length' 2>/dev/null || echo "0")
  log_status "running" "verifying" "Services: ${svc_count}"

  # Routes
  local routes
  routes=$(oc get routes -n "$ns" -o json 2>/dev/null || echo '{"items":[]}')
  local route_count
  route_count=$(echo "$routes" | jq '.items | length' 2>/dev/null || echo "0")
  log_status "running" "verifying" "Routes: ${route_count}"

  if [[ "$route_count" -gt 0 ]]; then
    local route_hosts
    route_hosts=$(echo "$routes" | jq -r '.items[] | .metadata.name + " -> https://" + .spec.host' 2>/dev/null || echo "")
    if [[ -n "$route_hosts" ]]; then
      log_status "running" "verifying" "Route endpoints: ${route_hosts}"
    fi
  fi

  # PVCs
  local pvcs
  pvcs=$(oc get pvc -n "$ns" -o json 2>/dev/null || echo '{"items":[]}')
  local pvc_count
  pvc_count=$(echo "$pvcs" | jq '.items | length' 2>/dev/null || echo "0")
  if [[ "$pvc_count" -gt 0 ]]; then
    local pvc_summary
    pvc_summary=$(echo "$pvcs" | jq -r '.items[] | .metadata.name + " (" + .status.phase + ")"' 2>/dev/null || echo "")
    log_status "running" "verifying" "PVCs (${pvc_count}): ${pvc_summary}"
  else
    log_status "running" "verifying" "PVCs: none"
  fi
}
