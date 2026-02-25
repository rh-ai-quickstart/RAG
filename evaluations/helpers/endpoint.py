"""
Shared utility for auto-detecting the RAG UI endpoint.

Used by both conftest.py (pytest runs) and test_conversations_ui.py (__main__ mode).
"""

import os
import subprocess


def get_rag_ui_endpoint() -> str:
    """
    Automatically determine RAG UI endpoint.

    Priority:
    1. RAG_UI_ENDPOINT env var (explicit override)
    2. Construct from NAMESPACE env var (e.g., https://rag-{namespace}.apps...)
    3. Query OpenShift for route in current namespace
    4. Fall back to localhost:8501
    """
    # Check for explicit override
    if os.getenv("RAG_UI_ENDPOINT"):
        return os.getenv("RAG_UI_ENDPOINT")

    # Try to construct from NAMESPACE
    namespace = os.getenv("NAMESPACE")
    if namespace:
        try:
            result = subprocess.run(
                [
                    "oc",
                    "get",
                    "route",
                    "rag",
                    "-n",
                    namespace,
                    "-o",
                    "jsonpath={.spec.host}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                route_host = result.stdout.strip()
                endpoint = f"https://{route_host}"
                print(
                    f"Auto-detected RAG UI endpoint from namespace '{namespace}': {endpoint}"
                )
                return endpoint
        except Exception as e:
            print(f"Warning: Could not query route for namespace '{namespace}': {e}")

    # Try to detect current namespace from oc context
    try:
        result = subprocess.run(
            ["oc", "project", "-q"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            current_namespace = result.stdout.strip()

            result = subprocess.run(
                ["oc", "get", "route", "rag", "-o", "jsonpath={.spec.host}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                route_host = result.stdout.strip()
                endpoint = f"https://{route_host}"
                print(
                    f"Auto-detected RAG UI endpoint from current namespace '{current_namespace}': {endpoint}"
                )
                return endpoint
    except Exception as e:
        print(f"Warning: Could not detect namespace from oc context: {e}")

    # Fall back to localhost (assumes port-forward is running)
    print("No RAG_UI_ENDPOINT or NAMESPACE set. Defaulting to http://localhost:8501")
    print("Make sure to run: oc port-forward svc/rag 8501:8501 -n <namespace>")
    return "http://localhost:8501"
