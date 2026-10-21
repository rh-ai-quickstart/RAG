#!/bin/bash
# ============================================================================
# RAG Installer — Deploy Script (Navigator Proxy)
# ============================================================================
# Creates RBAC, runs the installer Job, monitors it, retrieves the
# termination message, and cleans up.
# ============================================================================

set -euo pipefail

# Configuration
REGISTRY="quay.io/rh-ai-quickstart"
IMAGE_NAME="rag-installer"
VERSION="0.2.38"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }

# ============================================================================
# DO NOT MODIFY: Job deployment function
# ============================================================================

deploy_job() {
  local ACTION=$1
  local TARGET_NAMESPACE=$2
  local EXTRA_ENV=$3

  local INSTALLER_NAMESPACE="default"

  # --------------------------------------------------------------------------
  # Create RBAC for installer
  # --------------------------------------------------------------------------
  info "Creating installer RBAC..."

  # ServiceAccount + Role + RoleBinding in default namespace
  cat <<RBAC | oc apply -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rag-installer
  namespace: ${INSTALLER_NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: rag-installer
  namespace: ${INSTALLER_NAMESPACE}
rules:
  - apiGroups: [""]
    resources: ["pods", "configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: rag-installer
  namespace: ${INSTALLER_NAMESPACE}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: rag-installer
subjects:
  - kind: ServiceAccount
    name: rag-installer
    namespace: ${INSTALLER_NAMESPACE}
RBAC

  # ClusterRole + ClusterRoleBinding
  cat <<RBAC | oc apply -f -
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: rag-installer-${TARGET_NAMESPACE}
rules:
  # Cluster-scoped read permissions for prerequisites checking
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list"]
  - apiGroups: ["storage.k8s.io"]
    resources: ["storageclasses"]
    verbs: ["get", "list"]
  - apiGroups: ["config.openshift.io"]
    resources: ["clusterversions"]
    verbs: ["get", "list"]
  - apiGroups: ["apiextensions.k8s.io"]
    resources: ["customresourcedefinitions"]
    verbs: ["get", "list"]
  - apiGroups: ["packages.operators.coreos.com"]
    resources: ["packagemanifests"]
    verbs: ["get", "list"]
  # Namespace management
  - apiGroups: [""]
    resources: ["namespaces"]
    verbs: ["get", "list", "create", "delete"]
  # Namespace-scoped resources for RAG deployment
  - apiGroups: [""]
    resources: ["pods", "pods/log", "services", "configmaps", "secrets", "serviceaccounts", "persistentvolumeclaims"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments", "statefulsets", "replicasets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["route.openshift.io"]
    resources: ["routes"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["rbac.authorization.k8s.io"]
    resources: ["roles", "rolebindings"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  # Project API for namespace creation via oc new-project
  - apiGroups: ["project.openshift.io"]
    resources: ["projectrequests", "projects"]
    verbs: ["get", "list", "create", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: rag-installer-${TARGET_NAMESPACE}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: rag-installer-${TARGET_NAMESPACE}
subjects:
  - kind: ServiceAccount
    name: rag-installer
    namespace: ${INSTALLER_NAMESPACE}
RBAC

  # --------------------------------------------------------------------------
  # Create and monitor the Job
  # --------------------------------------------------------------------------

  local ACTION_SHORT=$(echo $ACTION | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  local TIMESTAMP=$(date +%s | tail -c 7)
  local JOB_NAME="rag-inst-${ACTION_SHORT}-${TIMESTAMP}"

  info "Creating installer Job: $JOB_NAME"
  info "Action: $ACTION"
  info "Target namespace: $TARGET_NAMESPACE"
  info "Installer namespace: $INSTALLER_NAMESPACE"
  info "Image: ${FULL_IMAGE}"

  cat <<EOF | oc apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${INSTALLER_NAMESPACE}
  labels:
    app: rag-installer
    action: $(echo $ACTION | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    target-namespace: ${TARGET_NAMESPACE}
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: rag-installer
        action: $(echo $ACTION | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    spec:
      restartPolicy: Never
      serviceAccountName: rag-installer
      containers:
      - name: installer
        image: ${FULL_IMAGE}
        imagePullPolicy: Always
        terminationMessagePolicy: FallbackToLogsOnError
        env:
        - name: ACTION
          value: "${ACTION}"
        - name: TARGET_NAMESPACE
          value: "${TARGET_NAMESPACE}"
        - name: JOB_NAME
          value: "${JOB_NAME}"
${EXTRA_ENV}
EOF

  echo ""
  info "Job created! Monitoring logs..."
  echo ""

  sleep 3
  oc logs -n "$INSTALLER_NAMESPACE" -f "job/${JOB_NAME}" 2>/dev/null || {
    warn "Job may still be starting. Check logs with:"
    echo "  oc logs -n $INSTALLER_NAMESPACE -f job/${JOB_NAME}"
  }

  # --------------------------------------------------------------------------
  # DO NOT MODIFY: Wait for Job completion (poll both Complete and Failed)
  # --------------------------------------------------------------------------
  echo ""
  info "Waiting for Job to complete..."

  WAIT_COUNT=0
  MAX_WAIT=240  # 20 minutes = 240 * 5 seconds
  while [[ $WAIT_COUNT -lt $MAX_WAIT ]]; do
    JOB_COMPLETE=$(oc get job -n "$INSTALLER_NAMESPACE" "${JOB_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
    JOB_FAILED=$(oc get job -n "$INSTALLER_NAMESPACE" "${JOB_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)

    if [[ "$JOB_COMPLETE" == "True" ]]; then
      info "Job completed successfully"
      break
    elif [[ "$JOB_FAILED" == "True" ]]; then
      warn "Job failed. Check logs above for details."
      break
    fi

    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 1))
  done

  if [[ $WAIT_COUNT -eq $MAX_WAIT ]]; then
    warn "Job did not complete within 20 minutes"
    echo "  Check status: oc get job -n $INSTALLER_NAMESPACE ${JOB_NAME}"
  fi

  # --------------------------------------------------------------------------
  # DO NOT MODIFY: Retrieve termination message (pod first, Job annotation fallback)
  # --------------------------------------------------------------------------
  TERM_MSG=""
  POD_NAME=$(oc get pods -n "$INSTALLER_NAMESPACE" -l "job-name=${JOB_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [[ -n "$POD_NAME" ]]; then
    TERM_MSG=$(oc get pod -n "$INSTALLER_NAMESPACE" "$POD_NAME" -o jsonpath='{.status.containerStatuses[0].state.terminated.message}' 2>/dev/null)
  fi
  if [[ -z "$TERM_MSG" ]]; then
    TERM_MSG=$(oc get job -n "$INSTALLER_NAMESPACE" "${JOB_NAME}" -o jsonpath='{.metadata.annotations.rag-installer/termination-message}' 2>/dev/null)
  fi
  if [[ -n "$TERM_MSG" ]]; then
    echo ""
    info "Termination message:"
    echo "  $TERM_MSG"
  fi

  echo ""
  info "Job complete! Check status with:"
  echo "  oc get job -n $INSTALLER_NAMESPACE ${JOB_NAME}"
  echo "  oc describe job -n $INSTALLER_NAMESPACE ${JOB_NAME}"

  # --------------------------------------------------------------------------
  # DO NOT MODIFY: Clean up all installer RBAC
  # --------------------------------------------------------------------------
  info "Cleaning up installer RBAC..."

  oc delete serviceaccount rag-installer -n default --ignore-not-found=true 2>/dev/null || true
  oc delete role rag-installer -n default --ignore-not-found=true 2>/dev/null || true
  oc delete rolebinding rag-installer -n default --ignore-not-found=true 2>/dev/null || true
  oc delete secret -l "kubernetes.io/service-account.name=rag-installer" -n default --ignore-not-found=true 2>/dev/null || true

  oc delete clusterrolebinding "rag-installer-${TARGET_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
  oc delete clusterrole "rag-installer-${TARGET_NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
}

# ============================================================================
# Main case statement
# ============================================================================

case "${1:-}" in
  check_pre_reqs)
    NAMESPACE="${2:-${NAMESPACE:-}}"
    [[ -z "$NAMESPACE" ]] && error "Namespace required. Usage: ./deploy.sh check_pre_reqs <namespace>"
    deploy_job "CHECK_PRE_REQS" "$NAMESPACE" ""
    ;;

  status)
    NAMESPACE="${2:-${NAMESPACE:-}}"
    [[ -z "$NAMESPACE" ]] && error "Namespace required. Usage: ./deploy.sh status <namespace>"
    deploy_job "STATUS" "$NAMESPACE" ""
    ;;

  install)
    NAMESPACE="${2:-${NAMESPACE:-}}"
    [[ -z "$NAMESPACE" ]] && error "Namespace required. Usage: ./deploy.sh install <namespace>"

    # Prompt for installation parameters
    echo ""
    echo "RAG Installer — Installation Configuration"
    echo "============================================"
    echo ""

    # HF_TOKEN
    HF_TOKEN_VAL="${HF_TOKEN:-}"
    if [[ -z "$HF_TOKEN_VAL" ]]; then
      echo "Hugging Face token (required for model downloads):"
      echo "  Get yours at: https://huggingface.co/settings/tokens"
      read -sp "  HF_TOKEN: " HF_TOKEN_VAL
      echo ""
    fi

    # LLM model
    LLM_VAL="${LLM:-}"
    if [[ -z "$LLM_VAL" ]]; then
      echo ""
      echo "LLM model to enable (e.g., llama-3-2-3b-instruct):"
      read -p "  LLM: " LLM_VAL
    fi

    # Device type
    DEVICE_VAL="${DEVICE:-gpu}"
    echo ""
    echo "Device type [cpu/gpu/hpu/xeon] (default: gpu):"
    read -p "  DEVICE [$DEVICE_VAL]: " device_input
    if [[ -n "$device_input" ]]; then
      DEVICE_VAL="$device_input"
    fi

    # GPU taint override
    GPU_TOL_ENV=""
    if [[ "$DEVICE_VAL" == "gpu" ]] || [[ "$DEVICE_VAL" == "hpu" ]]; then
      echo ""
      echo "GPU/HPU taint key override (comma-separated, or 'auto' to detect):"
      echo "  Examples: nvidia.com/gpu, habana.ai/gaudi"
      read -p "  Taint keys [auto]: " taint_input
      if [[ -n "$taint_input" ]] && [[ "$taint_input" != "auto" ]]; then
        tol_json="["
        first=true
        IFS=',' read -ra KEYS <<< "$taint_input"
        for key in "${KEYS[@]}"; do
          key=$(echo "$key" | xargs)
          if [[ "$first" == "true" ]]; then
            first=false
          else
            tol_json+=","
          fi
          tol_json+="{\"key\":\"${key}\",\"effect\":\"NoSchedule\",\"operator\":\"Exists\"}"
        done
        tol_json+="]"
        GPU_TOL_ENV="        - name: GPU_TOLERATIONS
          value: '${tol_json}'"
      fi
    fi

    # LLM toleration (direct override)
    LLM_TOL_VAL="${LLM_TOLERATION:-}"

    # Safety model
    SAFETY_VAL="${SAFETY:-}"

    # TAVILY
    TAVILY_VAL="${TAVILY_API_KEY:-}"

    # Build env vars for the Job
    INSTALL_ENV="        - name: INSTALL_MODE
          value: \"demo\""

    if [[ -n "$HF_TOKEN_VAL" ]]; then
      INSTALL_ENV+="
        - name: HF_TOKEN
          value: \"${HF_TOKEN_VAL}\""
    fi
    if [[ -n "$LLM_VAL" ]]; then
      INSTALL_ENV+="
        - name: LLM
          value: \"${LLM_VAL}\""
    fi
    if [[ -n "$DEVICE_VAL" ]]; then
      INSTALL_ENV+="
        - name: DEVICE
          value: \"${DEVICE_VAL}\""
    fi
    if [[ -n "$LLM_TOL_VAL" ]]; then
      INSTALL_ENV+="
        - name: LLM_TOLERATION
          value: \"${LLM_TOL_VAL}\""
    fi
    if [[ -n "$SAFETY_VAL" ]]; then
      INSTALL_ENV+="
        - name: SAFETY
          value: \"${SAFETY_VAL}\""
    fi
    if [[ -n "${SAFETY_TOLERATION:-}" ]]; then
      INSTALL_ENV+="
        - name: SAFETY_TOLERATION
          value: \"${SAFETY_TOLERATION}\""
    fi
    if [[ -n "$TAVILY_VAL" ]]; then
      INSTALL_ENV+="
        - name: TAVILY_API_KEY
          value: \"${TAVILY_VAL}\""
    fi
    if [[ -n "${LLM_URL:-}" ]]; then
      INSTALL_ENV+="
        - name: LLM_URL
          value: \"${LLM_URL}\""
    fi
    if [[ -n "${LLM_API_TOKEN:-}" ]]; then
      INSTALL_ENV+="
        - name: LLM_API_TOKEN
          value: \"${LLM_API_TOKEN}\""
    fi
    if [[ -n "${LLM_ID:-}" ]]; then
      INSTALL_ENV+="
        - name: LLM_ID
          value: \"${LLM_ID}\""
    fi
    if [[ -n "$GPU_TOL_ENV" ]]; then
      INSTALL_ENV+="
${GPU_TOL_ENV}"
    fi

    deploy_job "INSTALL" "$NAMESPACE" "$INSTALL_ENV"
    ;;

  upgrade)
    NAMESPACE="${2:-${NAMESPACE:-}}"
    [[ -z "$NAMESPACE" ]] && error "Namespace required. Usage: ./deploy.sh upgrade <namespace>"

    # Reuse install env vars for upgrade
    UPGRADE_ENV="        - name: INSTALL_MODE
          value: \"demo\""

    if [[ -n "${HF_TOKEN:-}" ]]; then
      UPGRADE_ENV+="
        - name: HF_TOKEN
          value: \"${HF_TOKEN}\""
    fi
    if [[ -n "${LLM:-}" ]]; then
      UPGRADE_ENV+="
        - name: LLM
          value: \"${LLM}\""
    fi
    if [[ -n "${DEVICE:-}" ]]; then
      UPGRADE_ENV+="
        - name: DEVICE
          value: \"${DEVICE}\""
    fi
    if [[ -n "${LLM_TOLERATION:-}" ]]; then
      UPGRADE_ENV+="
        - name: LLM_TOLERATION
          value: \"${LLM_TOLERATION}\""
    fi
    if [[ -n "${SAFETY:-}" ]]; then
      UPGRADE_ENV+="
        - name: SAFETY
          value: \"${SAFETY}\""
    fi
    if [[ -n "${TAVILY_API_KEY:-}" ]]; then
      UPGRADE_ENV+="
        - name: TAVILY_API_KEY
          value: \"${TAVILY_API_KEY}\""
    fi

    deploy_job "UPGRADE" "$NAMESPACE" "$UPGRADE_ENV"
    ;;

  uninstall_keep_data)
    NAMESPACE="${2:-${NAMESPACE:-}}"
    [[ -z "$NAMESPACE" ]] && error "Namespace required. Usage: ./deploy.sh uninstall_keep_data <namespace>"
    deploy_job "UNINSTALL_KEEP_DATA" "$NAMESPACE" ""
    ;;

  uninstall_delete_all)
    NAMESPACE="${2:-${NAMESPACE:-}}"
    [[ -z "$NAMESPACE" ]] && error "Namespace required. Usage: ./deploy.sh uninstall_delete_all <namespace>"
    deploy_job "UNINSTALL_DELETE_ALL" "$NAMESPACE" ""
    ;;

  "")
    echo "RAG Installer - Deploy Jobs to Cluster"
    echo ""
    echo "Usage: ./deploy.sh <action> <namespace>"
    echo ""
    echo "Actions:"
    echo "  check_pre_reqs <namespace>          - Validate prerequisites"
    echo "  status <namespace>                   - Check deployment status"
    echo "  install <namespace>                  - Deploy installation"
    echo "  upgrade <namespace>                  - Upgrade existing deployment"
    echo "  uninstall_keep_data <namespace>      - Uninstall (keep data)"
    echo "  uninstall_delete_all <namespace>     - Uninstall (delete all)"
    echo ""
    echo "Environment variables (for non-interactive install):"
    echo "  HF_TOKEN            - Hugging Face token"
    echo "  LLM                 - LLM model name (e.g., llama-3-2-3b-instruct)"
    echo "  SAFETY              - Safety model name (e.g., llama-guard-3-8b)"
    echo "  DEVICE              - Device type: cpu, gpu, hpu, xeon (default: gpu)"
    echo "  LLM_TOLERATION      - GPU/HPU taint key for LLM"
    echo "  SAFETY_TOLERATION   - GPU/HPU taint key for safety model"
    echo "  TAVILY_API_KEY      - TAVILY search API key"
    echo "  LLM_URL             - Remote LLM endpoint URL"
    echo "  LLM_API_TOKEN       - Remote LLM API token"
    echo "  LLM_ID              - Remote LLM model ID"
    echo ""
    echo "Image: ${FULL_IMAGE}"
    ;;

  *)
    error "Unknown action: $1"
    ;;
esac
