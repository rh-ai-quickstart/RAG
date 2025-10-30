"""
Pytest configuration for Playwright e2e tests
"""
import pytest
import os
import requests
import time


RAG_UI_ENDPOINT = os.getenv("RAG_UI_ENDPOINT", "http://localhost:8501")
LLAMA_STACK_ENDPOINT = os.getenv("LLAMA_STACK_ENDPOINT", "http://localhost:8321")


def wait_for_service(url, name, max_retries=30, retry_delay=2):
    """Wait for a service to be ready"""
    print(f"⏳ Waiting for {name} at {url}...")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 404]:
                print(f"✅ {name} is ready!")
                return True
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                print(f"   Attempt {attempt + 1}/{max_retries} failed, retrying...")
                time.sleep(retry_delay)
    return False


@pytest.fixture(scope="session", autouse=True)
def check_services():
    """Check that required services are running before tests"""
    rag_ui_ready = wait_for_service(RAG_UI_ENDPOINT, "RAG UI", max_retries=10)
    llama_stack_ready = wait_for_service(LLAMA_STACK_ENDPOINT, "Llama Stack", max_retries=10)
    
    if not rag_ui_ready:
        pytest.skip(f"RAG UI not available at {RAG_UI_ENDPOINT}")
    
    if not llama_stack_ready:
        print(f"⚠️  Warning: Llama Stack not available at {LLAMA_STACK_ENDPOINT}")
        print("   Some tests may be skipped.")


@pytest.fixture(scope="function")
def page_with_retry(page):
    """Page fixture with retry logic for flaky tests"""
    page.set_default_timeout(30000)  # 30 seconds
    yield page

