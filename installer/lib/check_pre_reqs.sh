#!/bin/bash
# ============================================================================
# RAG Quickstart — Prerequisites Validation
# ============================================================================

check_prerequisites() {
  local missing=()

  # --------------------------------------------------------------------------
  # 1. OpenShift cluster connectivity
  # --------------------------------------------------------------------------
  log_status "running" "validating" "Checking OpenShift cluster connectivity..."
  if ! oc whoami &>/dev/null; then
    missing+=("{\"name\":\"openshift-connectivity\",\"message\":\"Cannot connect to OpenShift cluster. Verify oc login.\"}")
  fi

  # --------------------------------------------------------------------------
  # 2. OpenShift version (minimum 4.16)
  # --------------------------------------------------------------------------
  log_status "running" "validating" "Checking OpenShift version..."
  local ocp_version
  ocp_version=$(oc get clusterversion version -o jsonpath='{.status.desired.version}' 2>/dev/null || echo "")
  if [[ -n "$ocp_version" ]]; then
    local major minor
    major=$(echo "$ocp_version" | cut -d. -f1)
    minor=$(echo "$ocp_version" | cut -d. -f2)
    if [[ "$major" -lt 4 ]] || { [[ "$major" -eq 4 ]] && [[ "$minor" -lt 16 ]]; }; then
      missing+=("{\"name\":\"openshift-version\",\"message\":\"OpenShift 4.16+ required, found ${ocp_version}\"}")
    else
      log_status "running" "validating" "OpenShift version: ${ocp_version} (OK)"
    fi
  else
    missing+=("{\"name\":\"openshift-version\",\"message\":\"Could not determine OpenShift version\"}")
  fi

  # --------------------------------------------------------------------------
  # 3. Route API availability (route.openshift.io)
  # --------------------------------------------------------------------------
  log_status "running" "validating" "Checking Route API availability..."
  if ! oc get crd routes.route.openshift.io &>/dev/null; then
    if ! oc api-resources --api-group=route.openshift.io 2>/dev/null | grep -q routes; then
      missing+=("{\"name\":\"route-api\",\"message\":\"route.openshift.io API not available. OpenShift Routes are required.\"}")
    fi
  fi

  # --------------------------------------------------------------------------
  # 4. Storage classes with RWO access mode
  # --------------------------------------------------------------------------
  log_status "running" "validating" "Checking storage classes..."
  local sc_count
  sc_count=$(oc get storageclass -o json 2>/dev/null | jq '[.items[] | select(.metadata.name != null)] | length' 2>/dev/null || echo "0")
  if [[ "$sc_count" -eq 0 ]]; then
    missing+=("{\"name\":\"storage-classes\",\"message\":\"No StorageClasses found. PgVector and MinIO require RWO storage.\"}")
  else
    log_status "running" "validating" "Found ${sc_count} StorageClass(es) (OK)"
  fi

  # --------------------------------------------------------------------------
  # 5. OpenShift AI / Open Data Hub (check via CRD existence)
  # --------------------------------------------------------------------------
  log_status "running" "validating" "Checking for OpenShift AI (Open Data Hub)..."
  if ! oc get crd datasciencepipelinesapplications.datasciencepipelinesapplications.opendatahub.io &>/dev/null; then
    log_status "running" "validating" "WARNING: OpenShift AI (ODH) CRDs not detected. Kubeflow pipelines for batch ingestion may not be available."
  else
    log_status "running" "validating" "OpenShift AI (ODH) CRDs detected (OK)"
  fi

  # --------------------------------------------------------------------------
  # 6. Node resources — report allocatable CPU and memory
  # --------------------------------------------------------------------------
  log_status "running" "validating" "Checking node resources..."
  local node_info
  node_info=$(oc get nodes -o json 2>/dev/null || echo '{"items":[]}')

  local total_cpu
  total_cpu=$(echo "$node_info" | jq '[.items[].status.allocatable.cpu // "0" | if test("m$") then (gsub("m$";"") | tonumber / 1000) else tonumber end] | add | round' 2>/dev/null || echo "0")

  local total_memory_gi
  total_memory_gi=$(echo "$node_info" | jq '[.items[].status.allocatable.memory // "0Ki" | gsub("Ki$";"") | tonumber / 1048576] | add | round' 2>/dev/null || echo "0")

  local worker_count
  worker_count=$(echo "$node_info" | jq '[.items[] | select(.metadata.labels["node-role.kubernetes.io/worker"] == "")] | length' 2>/dev/null || echo "0")

  log_status "running" "validating" "Cluster resources: ${worker_count} worker node(s), ${total_cpu} CPU cores, ${total_memory_gi} GiB memory"

  # --------------------------------------------------------------------------
  # 7. GPU / HPU / Accelerator detection (informational)
  # --------------------------------------------------------------------------
  log_status "running" "validating" "Checking for accelerator nodes..."

  # NVIDIA GPU nodes
  local gpu_nodes
  gpu_nodes=$(echo "$node_info" | jq '[.items[] | select(.metadata.labels["nvidia.com/gpu.present"] == "true")] | length' 2>/dev/null || echo "0")
  if [[ "$gpu_nodes" -gt 0 ]]; then
    log_status "running" "validating" "Found ${gpu_nodes} NVIDIA GPU node(s)"
    # Report NoSchedule taint keys on GPU nodes
    local gpu_taints
    gpu_taints=$(echo "$node_info" | jq -r '[.items[] | select(.metadata.labels["nvidia.com/gpu.present"] == "true") | .spec.taints // [] | .[] | select(.effect == "NoSchedule") | .key] | unique | join(", ")' 2>/dev/null || echo "")
    if [[ -n "$gpu_taints" ]]; then
      log_status "running" "validating" "GPU node NoSchedule taint key(s): ${gpu_taints}"
    fi
  else
    log_status "running" "validating" "No NVIDIA GPU nodes detected"
  fi

  # Intel Gaudi HPU nodes
  local hpu_nodes
  hpu_nodes=$(echo "$node_info" | jq '[.items[] | select(.metadata.labels["habana.ai/gaudi.present"] == "true")] | length' 2>/dev/null || echo "0")
  if [[ "$hpu_nodes" -gt 0 ]]; then
    log_status "running" "validating" "Found ${hpu_nodes} Intel Gaudi HPU node(s)"
    local hpu_taints
    hpu_taints=$(echo "$node_info" | jq -r '[.items[] | select(.metadata.labels["habana.ai/gaudi.present"] == "true") | .spec.taints // [] | .[] | select(.effect == "NoSchedule") | .key] | unique | join(", ")' 2>/dev/null || echo "")
    if [[ -n "$hpu_taints" ]]; then
      log_status "running" "validating" "HPU node NoSchedule taint key(s): ${hpu_taints}"
    fi
  else
    log_status "running" "validating" "No Intel Gaudi HPU nodes detected"
  fi

  # --------------------------------------------------------------------------
  # 8. Helm repository availability
  # --------------------------------------------------------------------------
  log_status "running" "validating" "Checking Helm chart repository..."
  if ! helm repo list 2>/dev/null | grep -q "rh-ai-quickstart"; then
    helm repo add rh-ai-quickstart https://rh-ai-quickstart.github.io/ai-architecture-charts 2>/dev/null || true
  fi
  helm repo update rh-ai-quickstart 2>/dev/null || {
    missing+=("{\"name\":\"helm-repo\",\"message\":\"Cannot reach Helm chart repository at https://rh-ai-quickstart.github.io/ai-architecture-charts\"}")
  }

  # --------------------------------------------------------------------------
  # Result
  # --------------------------------------------------------------------------
  if [[ ${#missing[@]} -gt 0 ]]; then
    local missing_json
    missing_json=$(printf '%s,' "${missing[@]}")
    missing_json="[${missing_json%,}]"
    log_prerequisites_failed "$missing_json"
    return 2
  fi

  return 0
}
