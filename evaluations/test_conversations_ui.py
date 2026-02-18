"""
Playwright-based conversation evaluation tests.
Reads test definitions from JSON files and executes them through the RAG UI.
"""

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest
from playwright.sync_api import Page, expect
from markdownify import markdownify as md

# Configuration
CONVERSATIONS_DIR = Path(__file__).parent / "conversations"
RESULTS_DIR = Path(__file__).parent / "results" / "conversation_results"
TEST_TIMEOUT = 60000  # 60 seconds
RESPONSE_WAIT_MAX = 120  # 2 minutes max for AI response
ELEMENT_TIMEOUT = 60000  # 60 seconds for element operations (inner_text, etc.)


@pytest.fixture(scope="session", autouse=True)
def cleanup_old_results(request):
    """
    Clean up old test results before running new tests.
    Runs once at the start of the test session.
    If pytest is run with -k filter, only cleans up matching files.
    """
    # Check if pytest is running with a -k filter
    k_filter = request.config.getoption("-k", default=None)

    if RESULTS_DIR.exists():
        # Determine which files to clean up
        if k_filter:
            # Extract the test name from the filter (e.g., "hr_benefits_test_agent.json")
            # The filter might be complex, but we'll try to extract a simple filename pattern
            # Common patterns: "test.json", "[test.json]", "test and not other"
            import re

            # Try to extract a simple filename pattern
            match = re.search(r"([a-zA-Z0-9_-]+\.json)", k_filter)
            if match:
                pattern = match.group(1).replace(".json", "")
                print(f"\n🧹 Cleaning up results for filtered test: {pattern}")
                json_files = list(RESULTS_DIR.glob(f"{pattern}_*.json"))
                screenshot_dirs = []
                if (RESULTS_DIR / "screenshots").exists():
                    screenshot_dirs = [
                        d
                        for d in (RESULTS_DIR / "screenshots").glob("*")
                        if pattern in d.name
                    ]
            else:
                # If we can't parse the filter, just do full cleanup
                print(
                    f"\n🧹 Cleaning up old test results from {RESULTS_DIR} (couldn't parse filter)"
                )
                json_files = list(RESULTS_DIR.glob("*.json"))
                screenshot_dirs = (
                    list((RESULTS_DIR / "screenshots").glob("*"))
                    if (RESULTS_DIR / "screenshots").exists()
                    else []
                )
        else:
            # No filter - clean up everything
            print(f"\n🧹 Cleaning up old test results from {RESULTS_DIR}")
            json_files = list(RESULTS_DIR.glob("*.json"))
            screenshot_dirs = (
                list((RESULTS_DIR / "screenshots").glob("*"))
                if (RESULTS_DIR / "screenshots").exists()
                else []
            )

        total_files = len(json_files)
        total_screenshot_dirs = len(screenshot_dirs)

        if total_files > 0 or total_screenshot_dirs > 0:
            print(f"   Removing {total_files} result file(s)")
            print(f"   Removing {total_screenshot_dirs} screenshot directory(ies)")

            # Remove JSON result files
            for json_file in json_files:
                json_file.unlink()

            # Remove screenshot directories
            for screenshot_dir in screenshot_dirs:
                if screenshot_dir.is_dir():
                    shutil.rmtree(screenshot_dir)

            print("✅ Cleanup complete\n")
        else:
            print("   No old results to clean up\n")

    yield  # Tests run here

    # After all tests complete
    print(f"\n📊 New test results saved to: {RESULTS_DIR}")

    # Count new results
    new_json_files = list(RESULTS_DIR.glob("*.json"))
    new_screenshot_dirs = (
        list((RESULTS_DIR / "screenshots").glob("*"))
        if (RESULTS_DIR / "screenshots").exists()
        else []
    )

    print(f"   Generated {len(new_json_files)} result file(s)")
    print(f"   Generated {len(new_screenshot_dirs)} screenshot directory(ies)")


class ConversationTestRunner:
    """Handles running conversation tests through the Playwright UI"""

    def __init__(self, page: Page, screenshot_dir: Optional[Path] = None):
        self.page = page
        self.conversation_history = []
        self.screenshot_dir = screenshot_dir

        # Create screenshot directory if provided
        if self.screenshot_dir:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def select_mode(self, mode: str):
        """Select Direct or Agent-based mode"""
        if mode.lower() == "agent":
            agent_radio = self.page.get_by_text("Agent-based", exact=False).first
            if agent_radio.is_visible():
                agent_radio.click()
                time.sleep(1)
        else:
            # Direct mode is usually default, but click it to be sure
            direct_radio = self.page.get_by_text("Direct", exact=True).first
            if direct_radio.is_visible():
                direct_radio.click()
                time.sleep(1)

    def select_vector_dbs(self, vector_db_names: List[str]):
        """Select document collections (vector databases) from Streamlit multiselect"""
        if not vector_db_names:
            return

        print(f"Attempting to select vector DBs: {vector_db_names}")

        try:
            # Find the multiselect using Streamlit's test ID
            multiselect = self.page.get_by_test_id("stMultiSelect")

            if not multiselect.is_visible():
                print("Warning: Vector DB multiselect not visible")
                return

            # Click the dropdown arrow to open the options list
            # Streamlit uses an img with name 'open' for the dropdown arrow
            dropdown_arrow = multiselect.get_by_role("img", name="open")
            dropdown_arrow.click()
            print("✓ Opened vector DB dropdown")
            time.sleep(1)

            # Now select each vector DB from the dropdown options
            for vdb_name in vector_db_names:
                try:
                    # Find the option by role and name
                    # Use partial match since the option might have additional attributes
                    option = self.page.get_by_role("option", name=vdb_name)

                    if option.is_visible():
                        option.click()
                        print(f"✓ Selected vector DB: {vdb_name}")
                        time.sleep(0.5)
                    else:
                        print(f"⚠ Vector DB option not visible: {vdb_name}")
                except Exception as e:
                    print(f"Warning: Could not select '{vdb_name}': {e}")

            # Click outside to close the dropdown (optional - it may auto-close)
            # Click on a safe element like the Configuration heading
            try:
                config_heading = self.page.get_by_text(
                    "Configuration", exact=True
                ).first
                if config_heading.is_visible():
                    config_heading.click()
                    time.sleep(0.5)
            except Exception:
                pass

            print("✓ Completed vector DB selection")

        except Exception as e:
            print(f"Error selecting vector DBs: {e}")
            import traceback

            traceback.print_exc()
            print("Continuing with test - vector DBs may not be selected properly")

    def select_tools(self, tools: List[str]):
        """Select tools for agent mode (e.g., 'rag', 'web_search')"""
        if not tools:
            return

        for tool in tools:
            # Convert 'rag' to 'builtin::rag' format if needed
            tool_name = tool if "::" in tool else f"builtin::{tool}"

            try:
                # Look for the tool in the Available ToolGroups section
                # The UI shows tools as pills/buttons
                tool_element = self.page.get_by_text(
                    tool_name.split("::")[-1], exact=False
                ).first
                if tool_element.is_visible():
                    tool_element.click()
                    time.sleep(0.5)
            except Exception as e:
                print(f"Warning: Could not select tool '{tool}': {e}")

    def set_sampling_params(self, sampling_params: Dict[str, Any]):
        """Set sampling parameters like temperature, max_tokens"""
        if not sampling_params:
            return

        # Temperature
        if "temperature" in sampling_params:
            # Streamlit sliders - this is tricky, might need to use keyboard
            # For now, we'll skip detailed slider manipulation
            # The UI default should be reasonable
            pass

        # Max tokens
        if "max_tokens" in sampling_params:
            # Similar to temperature - skip for now
            pass

    def send_message(self, content: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Send a message and wait for response.
        Returns tuple of (assistant's response text, actual RAG content dict or None).
        """
        # Find chat input
        chat_input = self.page.get_by_placeholder("Ask a question...", exact=False)
        expect(chat_input).to_be_visible(timeout=TEST_TIMEOUT)

        # Send message
        chat_input.fill(content)
        chat_input.press("Enter")

        # Wait for Streamlit to process
        self.page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Wait for assistant response
        response_text = self._wait_for_assistant_response(content)

        # Extract actual RAG content if available
        actual_rag_content = self._extract_actual_rag_content()

        return response_text, actual_rag_content

    def _wait_for_assistant_response(self, user_message: str) -> str:
        """Wait for and extract assistant response"""
        max_wait = RESPONSE_WAIT_MAX
        wait_time = 0

        print("⏳ Waiting for assistant response to complete...")

        # First, wait for the "Running..." indicator to disappear
        # This indicates the response generation is complete
        while wait_time < max_wait:
            # Check if the running indicator is present
            running_indicator = self.page.locator('img[alt="Running..."]')

            if running_indicator.count() == 0:
                # Running indicator is gone, response should be complete
                print("✓ Response generation complete")
                break

            time.sleep(2)
            wait_time += 2

            if wait_time % 10 == 0:
                print(f"  Still waiting... ({wait_time}s elapsed)")

        # Wait for the response text to stabilize (no more streaming)
        # Check if text content stops changing
        print("⏳ Waiting for response text to stabilize...")
        last_text = ""
        stable_count = 0

        for _ in range(20):  # Check for up to 20 seconds
            chat_messages = self.page.locator('[data-testid="stChatMessage"]').all()
            if chat_messages:
                current_text = (
                    chat_messages[-1].inner_text(timeout=ELEMENT_TIMEOUT)
                    if chat_messages
                    else ""
                )

                if current_text == last_text:
                    stable_count += 1
                    if (
                        stable_count >= 5
                    ):  # Text unchanged for 5 consecutive checks (5 seconds)
                        print("✓ Response text stabilized")
                        break
                else:
                    stable_count = 0
                    last_text = current_text

            time.sleep(1)

        # Now extract the complete response
        # Find all chat message containers
        chat_messages = self.page.locator('[data-testid="stChatMessage"]').all()

        # Look for the most recent assistant message
        # It should come after the user message we just sent
        for container in reversed(chat_messages):  # Check newest first
            if container.is_visible():
                # Get the message content div (skip avatar)
                content_div = container.locator(
                    '[data-testid="stChatMessageContent"]'
                ).first

                # Check if this is a user message or greeting
                text_check = content_div.inner_text(timeout=ELEMENT_TIMEOUT).strip()
                if user_message in text_check or text_check == "How can I help you?":
                    continue

                # Get all stMarkdown containers (these contain the actual AI response)
                # Skip the search results group which is in a different structure
                markdown_containers = content_div.locator(
                    '[data-testid="stMarkdown"]'
                ).all()

                # Collect markdown from all containers except search results
                all_markdown = []
                for md_container in markdown_containers:
                    html_content = md_container.inner_html()

                    # Skip if this contains search results JSON
                    if '"source":' in html_content or "0:{" in html_content:
                        continue

                    # Convert HTML to markdown (preserves lists, bold, etc.)
                    markdown_text = md(html_content, heading_style="ATX")
                    all_markdown.append(markdown_text)

                # Join all markdown sections
                combined_markdown = "\n\n".join(all_markdown)

                # Filter out UI elements line by line
                lines = combined_markdown.split("\n")
                cleaned_lines = []

                for line in lines:
                    line = line.strip()

                    # Skip empty lines
                    if not line:
                        continue
                    # Skip UI element markers
                    if line in ["smart_toy", "keyboard_arrow_right", "face"]:
                        continue
                    # Skip search status indicators
                    if "🛠" in line and ("Searching" in line or "Searched" in line):
                        continue
                    # Skip search results header
                    if "📄 Search Results from" in line:
                        continue

                    # This is actual content
                    cleaned_lines.append(line)

                # Join all content lines
                if cleaned_lines:
                    response_text = "\n".join(cleaned_lines)

                    # Remove streaming cursor character
                    response_text = response_text.replace("▌", "").strip()

                    if len(response_text) > 20:  # Substantial response
                        print(
                            f"✅ Extracted complete response ({len(response_text)} chars)"
                        )
                        print(f"   Preview: {response_text[:150]}...")
                        return response_text

        # No response found
        print(f"❌ Warning: No response received within {max_wait} seconds")
        return "[ERROR: No response received]"

    def _extract_actual_rag_content(self) -> Optional[Dict[str, Any]]:
        """
        Extract actual RAG chunks from the search results expander.
        Returns dict with structure: {"chunks": [str, str, ...]} or None if no results.
        """
        print("  🔍 Attempting to extract actual RAG content...")
        try:
            # Find all expanders first to debug
            all_expanders = self.page.locator('[data-testid="stExpander"]').all()
            print(f"  Found {len(all_expanders)} expander(s) on page")

            # Look for ALL search results expanders (one per vector database)
            search_expanders = []
            for exp in all_expanders:
                exp_text = exp.inner_text(timeout=ELEMENT_TIMEOUT)
                print(f"    Expander text: {exp_text[:100]}...")
                if "Search Results from" in exp_text or "📄" in exp_text:
                    search_expanders.append(exp)
                    print("    ✓ Found search results expander")

            if not search_expanders:
                print("  ❌ No search results expander found")
                return None

            print(f"  Found {len(search_expanders)} search result expander(s)")

            # Process ALL expanders and collect chunks from each
            all_chunks = []

            for exp_idx, expander in enumerate(search_expanders):
                print(
                    f"\n  Processing expander {exp_idx + 1}/{len(search_expanders)}..."
                )

                # Try to expand it - click on the summary/header
                try:
                    # Find the details/summary element and click it
                    summary = expander.locator("summary").first
                    if summary.is_visible():
                        print("    Expanding search results...")
                        summary.click()
                        time.sleep(1)  # Wait for expansion
                except Exception as e:
                    print(
                        f"    Note: Couldn't click expander (may already be open): {e}"
                    )

                # Find JSON content - look for the data-value attribute which contains actual JSON
                json_elements = expander.locator('[data-testid="stJson"]').all()
                print(f"    Found {len(json_elements)} stJson element(s) in expander")

                if not json_elements:
                    print(
                        "    ⚠️  No stJson element found in this expander, skipping..."
                    )
                    continue

                # Get the element
                json_element = json_elements[0]

                # Try to get the actual JSON data
                search_results = []
                try:
                    # Approach 1: Try data-value attribute
                    json_data_attr = json_element.get_attribute("data-value")
                    if json_data_attr:
                        print(
                            f"    Found data-value attribute ({len(json_data_attr)} chars)"
                        )
                        search_results = json.loads(json_data_attr)
                        print(
                            f"    ✅ Parsed JSON from data-value: {len(search_results)} items"
                        )
                    # Approach 2: Look for a data- attribute that contains JSON
                    elif json_element.get_attribute("data-json"):
                        json_data = json_element.get_attribute("data-json")
                        search_results = json.loads(json_data)
                        print(
                            f"    ✅ Parsed JSON from data-json: {len(search_results)} items"
                        )
                    # Approach 3: Evaluate JavaScript to extract the React props/state
                    else:
                        print("    Trying to extract via JavaScript evaluation...")
                        # Try to get React props which might have the JSON data
                        raw_json = json_element.evaluate("""
                            (element) => {
                                // Try to find React fiber and get props
                                const key = Object.keys(element).find(k => k.startsWith('__react'));
                                if (key && element[key] && element[key].memoizedProps) {
                                    const props = element[key].memoizedProps;
                                    // For st.json, the data is in props.src
                                    if (props.src) return JSON.stringify(props.src);
                                    if (props.value) return JSON.stringify(props.value);
                                    if (props.data) return JSON.stringify(props.data);
                                    if (props.children) return JSON.stringify(props.children);
                                }
                                return null;
                            }
                        """)

                        if raw_json:
                            search_results = json.loads(raw_json)
                            print(
                                f"    ✅ Parsed JSON from React props: {len(search_results)} items"
                            )
                        else:
                            raise ValueError("Could not extract JSON via JavaScript")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"    Direct extraction failed: {e}, trying text parsing...")
                    # Fallback: parse the formatted text
                    import re

                    json_text = json_element.evaluate(
                        "el => el.textContent || el.innerText"
                    )

                    # Fix Streamlit's st.json format by adding missing commas
                    # But we need to be careful not to mess up quotes inside strings
                    # Split into array elements first
                    elements = []
                    # Find each numbered object like "0:{...}" "1:{...}"
                    pattern = r"(\d+):\s*(\{[^}]*\})"
                    for match in re.finditer(pattern, json_text, re.DOTALL):
                        num, obj_text = match.groups()
                        # Fix missing comma between "field""field"
                        # Only fix at the property level
                        fixed = re.sub(
                            r'("[^"]*")\s*("[a-zA-Z_]+")', r"\1,\2", obj_text
                        )
                        try:
                            obj = json.loads(fixed)
                            elements.append(obj)
                            print(
                                f"      Extracted element {num}: source={obj.get('source', '?')}, text_len={len(obj.get('text', ''))}"
                            )
                        except json.JSONDecodeError:
                            # If that didn't work, try without inner quotes fix
                            try:
                                obj = json.loads(obj_text)
                                elements.append(obj)
                            except Exception:
                                print(f"      Warning: Could not parse element {num}")

                    if elements:
                        search_results = elements
                        print(
                            f"    ✅ Extracted {len(search_results)} results via text parsing"
                        )
                    else:
                        print("    ❌ Text parsing also failed")
                        search_results = []

                except json.JSONDecodeError as e:
                    print(f"    ❌ Final JSON parsing failed: {e}")
                    # Try one more approach - get all the individual JSON objects
                    print("    Attempting to extract individual objects...")
                    import re

                    json_text = json_element.inner_text(timeout=ELEMENT_TIMEOUT)

                    # Find each object between curly braces
                    objects = []
                    depth = 0
                    current_obj = ""
                    in_string = False
                    escape_next = False

                    for char in json_text:
                        if escape_next:
                            current_obj += char
                            escape_next = False
                            continue

                        if char == "\\":
                            escape_next = True
                            current_obj += char
                            continue

                        if char == '"' and not escape_next:
                            in_string = not in_string

                        if not in_string:
                            if char == "{":
                                depth += 1
                            elif char == "}":
                                depth -= 1

                        if depth > 0:
                            current_obj += char
                            if depth == 1 and char == "{":
                                current_obj = char  # Start fresh
                        elif depth == 0 and current_obj and char == "}":
                            current_obj += char
                            # Try to parse this object
                            try:
                                # Clean up formatting
                                clean_obj = re.sub(r"^\d+:", "", current_obj.strip())
                                obj = json.loads(clean_obj)
                                objects.append(obj)
                                print(
                                    f"      Extracted object {len(objects)}: {len(str(obj))} chars"
                                )
                            except Exception:
                                pass
                            current_obj = ""

                    search_results = objects
                    print(
                        f"    Total extracted via manual parsing: {len(search_results)} objects"
                    )

                if not isinstance(search_results, list):
                    print(
                        f"    ⚠️  Search results not a list: {type(search_results)}, skipping..."
                    )
                    continue

                print(
                    f"    Parsed {len(search_results)} search result(s) from this expander"
                )

                # Extract chunks from results
                # Results might be React elements with props.src containing the actual data
                expander_chunks = []

                for i, result in enumerate(search_results):
                    if not result:
                        continue

                    # Check if this is a React element wrapper with props.src
                    if (
                        isinstance(result, dict)
                        and "props" in result
                        and "src" in result["props"]
                    ):
                        # Extract the actual data from props.src
                        src_data = result["props"]["src"]
                        if isinstance(src_data, list):
                            for item in src_data:
                                if isinstance(item, dict) and "text" in item:
                                    expander_chunks.append(item["text"])
                            print(
                                f"      Extracted {len(src_data)} chunks from React props.src"
                            )
                    # Otherwise check if this is direct data
                    elif isinstance(result, dict) and "text" in result:
                        expander_chunks.append(result["text"])
                        print(f"      Chunk {i + 1}: {len(result['text'])} chars")

                if expander_chunks:
                    print(
                        f"    ✅ Extracted {len(expander_chunks)} chunks from this expander"
                    )
                    all_chunks.extend(expander_chunks)
                else:
                    print("    ⚠️  No chunks found in this expander")

            # After processing all expanders
            if not all_chunks:
                print("  ❌ No chunks found across all search result expanders")
                return None

            print(
                f"\n  ✅ Successfully extracted {len(all_chunks)} total RAG chunks from {len(search_expanders)} expander(s)"
            )

            return {"chunks": all_chunks}

        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to parse RAG JSON: {e}")
            if json_text:
                print(f"  JSON text was: {json_text[:200]}...")
            return None
        except Exception as e:
            print(f"  ❌ Failed to extract RAG content: {e}")
            import traceback

            traceback.print_exc()
            return None

    def run_conversation(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete conversation test.
        Returns result in the format: {metadata, conversation}
        """
        metadata = test_config.get("metadata", {})
        config = test_config.get("config", {})
        messages = test_config.get("messages", [])

        # Configure test
        mode = config.get("mode", "direct")
        vector_dbs = config.get("vector_dbs", [])
        tools = config.get("tools", [])
        sampling_params = config.get("sampling_params", {})

        print(f"\n{'=' * 60}")
        print(f"Running test: {metadata.get('description', 'No description')}")
        print(f"Mode: {mode}")
        print(f"Vector DBs: {vector_dbs}")
        print(f"Tools: {tools}")
        print(f"{'=' * 60}\n")

        # Take screenshot before any configuration
        if self.screenshot_dir:
            try:
                screenshot_path = self.screenshot_dir / "01_before_configuration.png"
                self.page.screenshot(path=str(screenshot_path))
                print(f"📸 Screenshot: {screenshot_path}")
            except Exception as e:
                print(f"Warning: Screenshot failed: {e}")

        # Set up UI configuration
        self.select_mode(mode)
        time.sleep(1)

        self.select_vector_dbs(vector_dbs)
        time.sleep(1)

        if mode.lower() == "agent":
            self.select_tools(tools)
            time.sleep(1)

        self.set_sampling_params(sampling_params)
        time.sleep(1)

        # Take screenshot after all configuration is complete
        if self.screenshot_dir:
            try:
                screenshot_path = self.screenshot_dir / "02_after_configuration.png"
                self.page.screenshot(path=str(screenshot_path))
                print(f"📸 Screenshot: {screenshot_path}")
            except Exception as e:
                print(f"Warning: Screenshot failed: {e}")

        # Run conversation
        conversation = []
        message_num = 1

        for msg in messages:
            if msg["role"] != "user":
                print(f"Warning: Skipping non-user message: {msg}")
                continue

            user_content = msg["content"]
            expected_rag_content = msg.get("expected_rag_content")
            expected_output = msg.get("expected_output")

            print(f"USER: {user_content}")
            if expected_rag_content:
                num_chunks = len(expected_rag_content.get("chunks", []))
                print(f"  Expected RAG chunks: {num_chunks}")

            # Add user message to conversation
            user_message = {"role": "user", "content": user_content}

            # Include expected output if present
            if expected_output:
                user_message["expected_output"] = expected_output

            # Include expected RAG content if present
            if expected_rag_content:
                user_message["expected_rag_content"] = expected_rag_content

            # Send message and get response (including actual RAG content)
            assistant_response, actual_rag_content = self.send_message(user_content)
            print(f"ASSISTANT: {assistant_response[:200]}...")

            # Include actual RAG content if retrieved
            if actual_rag_content:
                num_actual_chunks = len(actual_rag_content.get("chunks", []))
                print(f"  Actual RAG chunks: {num_actual_chunks}")
                user_message["actual_rag_content"] = actual_rag_content

            conversation.append(user_message)

            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": assistant_response})

            # Take screenshot after each message exchange
            if self.screenshot_dir:
                try:
                    screenshot_path = (
                        self.screenshot_dir
                        / f"03_message_{message_num:02d}_complete.png"
                    )
                    self.page.screenshot(path=str(screenshot_path))
                    print(f"📸 Screenshot: {screenshot_path}")
                except Exception as e:
                    print(f"Warning: Screenshot failed: {e}")

            message_num += 1

            # Small delay between multi-turn messages
            time.sleep(2)

        # Build result - include config for traceability
        result = {
            "metadata": metadata,
            "config": config,  # Include test configuration
            "conversation": conversation,
        }

        return result


def find_test_files(subdir: Optional[str] = None) -> List[Path]:
    """Find all test JSON files in conversations directory (or a subdirectory)."""
    base = CONVERSATIONS_DIR / subdir if subdir else CONVERSATIONS_DIR
    if not base.exists():
        return []
    return sorted(base.rglob("*.json"))


def load_test_config(filepath: Path) -> Dict[str, Any]:
    """Load test configuration from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(
    result: Dict[str, Any], test_filename: str, timestamp: str = None
) -> Path:
    """Save test result with timestamp"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = Path(test_filename).stem
    result_filename = f"{base_name}_{timestamp}.json"
    result_path = RESULTS_DIR / result_filename

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result_path


# Pytest test generation
def pytest_generate_tests(metafunc):
    """
    Dynamically generate test cases from JSON files.
    This creates one test per JSON file in conversations/
    """
    if "test_config_file" in metafunc.fixturenames:
        subdir = metafunc.config.getoption("--subdir", default=None)
        test_files = find_test_files(subdir)
        if not test_files:
            # Create a dummy test to show warning
            metafunc.parametrize("test_config_file", [None])
        else:
            # Parametrize with test file paths
            # Use file name (with .json) as test ID for precise matching with -k
            metafunc.parametrize(
                "test_config_file", test_files, ids=[f.name for f in test_files]
            )


class TestConversations:
    """Conversation evaluation tests"""

    def test_conversation_from_json(self, page: Page, test_config_file: Path):
        """
        Run conversation test from JSON file.
        Test name will be the JSON filename.
        """
        if test_config_file is None:
            pytest.skip("No test files found in evaluations/conversations/")

        # Load test configuration
        test_config = load_test_config(test_config_file)

        # Generate result filename with timestamp (before running test)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = test_config_file.stem
        result_filename = f"{base_name}_{timestamp}"

        # Create screenshot directory: evaluations/results/conversation_results/screenshots/{result_filename}
        screenshot_dir = RESULTS_DIR / "screenshots" / result_filename
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📂 Screenshots will be saved to: {screenshot_dir}")

        # Run test with screenshot directory
        runner = ConversationTestRunner(page, screenshot_dir=screenshot_dir)
        result = runner.run_conversation(test_config)

        # Save result with the same base filename
        result_path = save_result(result, test_config_file.name, timestamp)

        print(f"\n📊 Result saved to: {result_path}")
        print(f"📸 Screenshots in: {screenshot_dir}")

        # Assertions
        assert len(result["conversation"]) > 0, "Conversation should not be empty"

        # Check that we got responses for all user messages
        user_messages = [m for m in result["conversation"] if m["role"] == "user"]
        assistant_messages = [
            m for m in result["conversation"] if m["role"] == "assistant"
        ]

        assert len(assistant_messages) == len(user_messages), (
            f"Should have {len(user_messages)} assistant responses, got {len(assistant_messages)}"
        )

        # Check that none of the responses are error messages
        for msg in assistant_messages:
            assert "[ERROR:" not in msg["content"], (
                f"Assistant response contains error: {msg['content']}"
            )


# Manual test running (for debugging)
if __name__ == "__main__":
    import urllib.request
    from helpers.playwright_setup import ensure_playwright_browsers
    from helpers.endpoint import get_rag_ui_endpoint

    ensure_playwright_browsers()

    endpoint = get_rag_ui_endpoint()
    print(f"Using RAG UI endpoint: {endpoint}")
    print()

    # Check if RAG UI is accessible
    try:
        urllib.request.urlopen(endpoint, timeout=5)
    except Exception:
        print(f"Warning: Cannot reach RAG UI at {endpoint}")
        print()
        if endpoint == "http://localhost:8501":
            print("Tip: Start port-forward in another terminal:")
            print("   oc port-forward svc/rag 8501:8501 -n $NAMESPACE")
            print()
            print("   Or set NAMESPACE environment variable:")
            print("   export NAMESPACE=<your-namespace>")
            print()
        print("Options:")
        print("  1. Set NAMESPACE: export NAMESPACE=<your-namespace>")
        print("  2. Port-forward: oc port-forward svc/rag 8501:8501 -n <namespace>")
        print(
            "  3. Override endpoint: export RAG_UI_ENDPOINT=https://rag-<namespace>.apps..."
        )
        print()
        response = input("Continue anyway? (y/N) ").strip().lower()
        if response != "y":
            sys.exit(1)

    print("Tip: Pass --headed to see the browser in action")
    print("Tip: Pass --screenshot=on for debugging screenshots")
    print("Tip: Pass --slowmo=1000 to slow down browser actions")
    print()

    pytest.main([__file__, "-v", "-s", f"--base-url={endpoint}"] + sys.argv[1:])
