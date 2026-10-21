#!/bin/bash
# ============================================================================
# RAG Quickstart — Installation Logic
# ============================================================================

HELM_CHART_DIR="/installer/charts/rag"
RAG_CHART_NAME="rag"
VALUES_FILE="/installer/charts/rag-values.yaml"

detect_gpu_tolerations() {
  # If GPU_TOLERATIONS is set externally, use it
  if [[ -n "${GPU_TOLERATIONS:-}" ]]; then
    echo "$GPU_TOLERATIONS"
    return
  fi

  local taint_keys
  taint_keys=$(oc get nodes -l nvidia.com/gpu.present=true -o json 2>/dev/null | \
    jq -r '[.items[] | .spec.taints // [] | .[] | select(.effect == "NoSchedule") | .key] | unique | .[]' 2>/dev/null || true)

  if [[ -z "$taint_keys" ]]; then
    # Check for HPU nodes
    taint_keys=$(oc get nodes -l habana.ai/gaudi.present=true -o json 2>/dev/null | \
      jq -r '[.items[] | .spec.taints // [] | .[] | select(.effect == "NoSchedule") | .key] | unique | .[]' 2>/dev/null || true)
  fi

  if [[ -z "$taint_keys" ]]; then
    taint_keys="nvidia.com/gpu"
  fi

  local tolerations="["
  local first=true
  while IFS= read -r key; do
    [[ -z "$key" ]] && continue
    if [[ "$first" == "true" ]]; then
      first=false
    else
      tolerations+=","
    fi
    tolerations+="{\"key\":\"${key}\",\"effect\":\"NoSchedule\",\"operator\":\"Exists\"}"
  done <<< "$taint_keys"
  tolerations+="]"

  echo "$tolerations"
}

build_toleration_helm_args() {
  local tolerations_json=$1
  local model_key=$2
  local args=""

  local count
  count=$(echo "$tolerations_json" | jq 'length' 2>/dev/null || echo "0")

  local i=0
  while [[ $i -lt $count ]]; do
    local key effect operator
    key=$(echo "$tolerations_json" | jq -r ".[$i].key" 2>/dev/null)
    effect=$(echo "$tolerations_json" | jq -r ".[$i].effect" 2>/dev/null)
    operator=$(echo "$tolerations_json" | jq -r ".[$i].operator" 2>/dev/null)
    args+=" --set global.models.${model_key}.tolerations[$i].key=${key}"
    args+=" --set global.models.${model_key}.tolerations[$i].effect=${effect}"
    args+=" --set global.models.${model_key}.tolerations[$i].operator=${operator}"
    i=$((i + 1))
  done

  echo "$args"
}

deploy_quickstart() {
  local ns="$TARGET_NAMESPACE"

  # Create namespace if it doesn't exist
  if ! oc get namespace "$ns" &>/dev/null; then
    log_status "running" "deploying" "Creating namespace ${ns}..."
    oc new-project "$ns" &>/dev/null || oc create namespace "$ns" &>/dev/null
  fi
  oc label namespace "$ns" modelmesh-enabled=false --overwrite &>/dev/null || true

  # Update Helm dependencies
  log_status "running" "deploying" "Updating Helm chart dependencies..."
  helm dependency update "$HELM_CHART_DIR" &>/dev/null || true
  helm dependency build "$HELM_CHART_DIR" 2>/dev/null || true

  # Delete any existing jobs in the target namespace to avoid conflicts
  oc delete jobs -n "$ns" --all 2>/dev/null || true

  # Build Helm arguments
  local HELM_ARGS="-f ${VALUES_FILE}"

  # HF_TOKEN
  if [[ -n "${HF_TOKEN:-}" ]]; then
    log_status "running" "deploying" "Setting Hugging Face token..."
    HELM_ARGS+=" --set llm-service.secret.hf_token=${HF_TOKEN}"
  fi

  # TAVILY_API_KEY
  if [[ -n "${TAVILY_API_KEY:-}" ]]; then
    log_status "running" "deploying" "Setting TAVILY API key..."
    HELM_ARGS+=" --set llama-stack.secrets.TAVILY_SEARCH_API_KEY=${TAVILY_API_KEY}"
  fi

  # DEVICE
  if [[ -n "${DEVICE:-}" ]]; then
    log_status "running" "deploying" "Setting device type: ${DEVICE}..."
    HELM_ARGS+=" --set llm-service.device=${DEVICE}"
  fi

  # LLM model
  if [[ -n "${LLM:-}" ]]; then
    log_status "running" "deploying" "Enabling LLM model: ${LLM}..."
    HELM_ARGS+=" --set global.models.${LLM}.enabled=true"

    if [[ -n "${LLM_ID:-}" ]]; then
      HELM_ARGS+=" --set global.models.${LLM}.id=${LLM_ID}"
    fi
    if [[ -n "${LLM_URL:-}" ]]; then
      HELM_ARGS+=" --set global.models.${LLM}.url=${LLM_URL}"
    fi
    if [[ -n "${LLM_API_TOKEN:-}" ]]; then
      HELM_ARGS+=" --set global.models.${LLM}.apiToken=${LLM_API_TOKEN}"
    fi

    # GPU tolerations for LLM
    if [[ -n "${LLM_TOLERATION:-}" ]]; then
      HELM_ARGS+=" --set global.models.${LLM}.tolerations[0].key=${LLM_TOLERATION}"
      HELM_ARGS+=" --set global.models.${LLM}.tolerations[0].effect=NoSchedule"
      HELM_ARGS+=" --set global.models.${LLM}.tolerations[0].operator=Exists"
    elif [[ -n "${GPU_TOLERATIONS:-}" ]] && [[ "${DEVICE:-gpu}" != "cpu" ]] && [[ "${DEVICE:-gpu}" != "xeon" ]]; then
      local tol_args
      tol_args=$(build_toleration_helm_args "$GPU_TOLERATIONS" "$LLM")
      HELM_ARGS+="$tol_args"
    fi
  fi

  # Safety model
  if [[ -n "${SAFETY:-}" ]]; then
    log_status "running" "deploying" "Enabling safety model: ${SAFETY}..."
    HELM_ARGS+=" --set global.models.${SAFETY}.enabled=true"

    if [[ -n "${SAFETY_ID:-}" ]]; then
      HELM_ARGS+=" --set global.models.${SAFETY}.id=${SAFETY_ID}"
    fi
    if [[ -n "${SAFETY_URL:-}" ]]; then
      HELM_ARGS+=" --set global.models.${SAFETY}.url=${SAFETY_URL}"
    fi
    if [[ -n "${SAFETY_API_TOKEN:-}" ]]; then
      HELM_ARGS+=" --set global.models.${SAFETY}.apiToken=${SAFETY_API_TOKEN}"
    fi

    # GPU tolerations for safety model
    if [[ -n "${SAFETY_TOLERATION:-}" ]]; then
      HELM_ARGS+=" --set global.models.${SAFETY}.tolerations[0].key=${SAFETY_TOLERATION}"
      HELM_ARGS+=" --set global.models.${SAFETY}.tolerations[0].effect=NoSchedule"
      HELM_ARGS+=" --set global.models.${SAFETY}.tolerations[0].operator=Exists"
    elif [[ -n "${GPU_TOLERATIONS:-}" ]] && [[ "${DEVICE:-gpu}" != "cpu" ]] && [[ "${DEVICE:-gpu}" != "xeon" ]]; then
      local tol_args
      tol_args=$(build_toleration_helm_args "$GPU_TOLERATIONS" "$SAFETY")
      HELM_ARGS+="$tol_args"
    fi
  fi

  # RAW_DEPLOYMENT mode
  if [[ -n "${RAW_DEPLOYMENT:-}" ]]; then
    HELM_ARGS+=" --set llm-service.rawDeploymentMode=${RAW_DEPLOYMENT}"
    HELM_ARGS+=" --set llama-stack.rawDeploymentMode=${RAW_DEPLOYMENT}"
  fi

  # Install via Helm
  log_status "running" "deploying" "Running helm upgrade --install ${RAG_CHART_NAME}..."
  eval helm -n "$ns" upgrade --install "$RAG_CHART_NAME" "$HELM_CHART_DIR" $HELM_ARGS
}

check_deployment_status() {
  local ns="$TARGET_NAMESPACE"
  local max_wait=600
  local interval=10
  local waited=0

  log_status "running" "checking-status" "Waiting for llamastack deployment to be ready (timeout: ${max_wait}s)..."

  while [[ $waited -lt $max_wait ]]; do
    local ready
    ready=$(oc get deploy llamastack -n "$ns" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    local desired
    desired=$(oc get deploy llamastack -n "$ns" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")

    if [[ "${ready:-0}" -ge "${desired:-1}" ]] && [[ "${ready:-0}" -gt 0 ]]; then
      log_status "running" "checking-status" "llamastack deployment is ready (${ready}/${desired})"
      return 0
    fi

    log_status "running" "checking-status" "Waiting for llamastack... (${ready:-0}/${desired:-1} ready, ${waited}s elapsed)"
    sleep "$interval"
    waited=$((waited + interval))
  done

  log_error "Timed out waiting for llamastack deployment to be ready after ${max_wait}s"
}

get_endpoints() {
  local ns="$TARGET_NAMESPACE"
  local endpoints="[]"

  local route_host
  route_host=$(oc get route "$RAG_CHART_NAME" -n "$ns" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")

  if [[ -n "$route_host" ]]; then
    endpoints="[{\"name\":\"RAG UI\",\"url\":\"https://${route_host}\"}]"
  fi

  echo "$endpoints"
}
