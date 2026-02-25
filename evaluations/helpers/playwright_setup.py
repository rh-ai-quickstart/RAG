"""
Playwright browser setup for project-local installation.

Ensures Chromium is installed in evaluations/bin/ and configures
PLAYWRIGHT_BROWSERS_PATH so Playwright finds the browsers there.
"""

import os
import subprocess
import sys
from pathlib import Path

# evaluations/bin/
BROWSERS_PATH = Path(__file__).parent.parent / "bin"


def _chromium_installed() -> bool:
    """Check if Chromium is installed in the project-local path."""
    if not BROWSERS_PATH.exists():
        return False
    return any(BROWSERS_PATH.glob("chromium-*")) or any(
        BROWSERS_PATH.glob("chromium_headless_shell-*")
    )


def ensure_playwright_browsers() -> None:
    """
    Set PLAYWRIGHT_BROWSERS_PATH and install Chromium if not present.

    Must be called before Playwright launches a browser. Sets the
    PLAYWRIGHT_BROWSERS_PATH environment variable to evaluations/bin/
    and installs Chromium there if it is missing.
    """
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(BROWSERS_PATH)

    if not _chromium_installed():
        print(f"Installing Playwright Chromium to {BROWSERS_PATH}/ (one-time setup)...")
        BROWSERS_PATH.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
        )
        print("Playwright Chromium installed.")
        print()
