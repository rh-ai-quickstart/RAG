"""
Pytest configuration for conversation evaluation tests.
Reuses fixtures and patterns from tests/e2e_ui/
"""

import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page

# Add project root and evaluations/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Ensure Playwright Chromium is installed in evaluations/bin/
from helpers.playwright_setup import ensure_playwright_browsers
from helpers.endpoint import get_rag_ui_endpoint

ensure_playwright_browsers()


def pytest_addoption(parser):
    parser.addoption(
        "--subdir",
        default=None,
        help="Subdirectory under conversations/ to run (e.g. 'legal'). Runs all subdirectories by default.",
    )


# Configuration
RAG_UI_ENDPOINT = get_rag_ui_endpoint()
TEST_TIMEOUT = 60000  # 60 seconds for responses


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context - matches e2e_ui setup"""
    return {
        **browser_context_args,
        "viewport": {
            "width": 1920,
            "height": 1080,
        },
    }


@pytest.fixture(autouse=True)
def reset_chat_before_test(page: Page):
    """
    Navigate to RAG UI and reset chat state before each test.
    This ensures clean state between tests (no conversation history).
    """
    # Navigate to app
    max_retries = 3
    for attempt in range(max_retries):
        try:
            page.goto(RAG_UI_ENDPOINT, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=60000)

            # Wait for Streamlit to initialize
            time.sleep(2)

            # Verify page loaded
            if page.url.startswith(RAG_UI_ENDPOINT):
                break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Navigation attempt {attempt + 1} failed: {e}, retrying...")
            time.sleep(2)

    # Click "Clear Chat & Reset Config" button to ensure clean state
    try:
        clear_button = page.get_by_text("Clear Chat", exact=False).first
        if clear_button.is_visible():
            clear_button.click()
            page.wait_for_load_state("networkidle")
            time.sleep(2)
    except Exception:
        # If button not found, continue - app might already be clean
        pass

    yield page

    # Cleanup after test (optional)
    # Can add cleanup logic here if needed
