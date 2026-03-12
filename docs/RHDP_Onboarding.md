# RHDP Onboarding

This document describes the intended repository layout for RHDP onboarding and how each bootstrap area fits into the GitOps flow for the RAG quickstart.

## Target Directory Structure

```text
tenant/
  bootstrap/     # Helm chart to create Argo CD Application for the RAG quickstart
infra/
  bootstrap/     # Helm chart to install required cluster operators/infrastructure
platform/
  bootstrap/     # Helm chart to create platform CRs and OpenShift UI plugins
```

## 1) `tenant/bootstrap`

Purpose:
- Owns tenant-scoped onboarding for the quickstart.
- Creates an Argo CD `Application` object.
- That `Application` points to the RAG chart and drives deployment/sync of the quickstart application.

Typical responsibilities:
- Argo CD `Application` definition (repo URL, path, revision, destination namespace).
- Passing Helm values to the RAG chart.
- Optional tenant-related RBAC needed for reconciliation.

## 2) `infra/bootstrap`

Purpose:
- Owns cluster/operator prerequisites required by the quickstart.
- Installs platform services and operators needed before tenant app deployment.

Typical responsibilities:
- Operator subscriptions or chart-based operator installs.
- Shared infra dependencies (for example: data science operators, serving stack prerequisites, pipeline dependencies, etc.).
- Cluster-wide setup needed by multiple tenants.

## 3) `platform/bootstrap`

Purpose:
- Owns platform-level custom resources and UI integrations.
- Creates cluster/platform objects that are not tenant app manifests.

Typical responsibilities:
- Custom resources used by the platform layer.
- OpenShift console/UI plugins and related configuration.
- Platform controls and integrations shared across workloads.

## Recommended Flow

1. `infra/bootstrap` provisions required operators and cluster prerequisites.
2. `platform/bootstrap` applies platform-level CRs/plugins.
3. `tenant/bootstrap` creates the Argo CD `Application` that deploys the RAG quickstart for a tenant namespace.

This separation keeps ownership clear:
- Infra team manages prerequisites.
- Platform team manages shared platform objects and plugins.
- Tenant/app team manages quickstart app onboarding through GitOps `Application` objects.
