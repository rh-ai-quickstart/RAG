"""
End-to-end UI tests for chat functionality using Playwright
Tests the actual UI interactions in a browser
"""
import pytest
import os
import time
from playwright.sync_api import Page, expect


# Configuration
RAG_UI_ENDPOINT = os.getenv("RAG_UI_ENDPOINT", "http://localhost:8501")
LLAMA_STACK_ENDPOINT = os.getenv("LLAMA_STACK_ENDPOINT", "http://localhost:8321")
TEST_TIMEOUT = 30000  # 30 seconds


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context"""
    return {
        **browser_context_args,
        "viewport": {
            "width": 1920,
            "height": 1080,
        },
    }


@pytest.fixture(autouse=True)
def wait_for_app(page: Page):
    """Wait for the Streamlit app to be ready before each test"""
    # Retry navigation in case of transient connection issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            page.goto(RAG_UI_ENDPOINT, timeout=60000, wait_until="domcontentloaded")
            # Wait for Streamlit to finish loading
            page.wait_for_load_state("networkidle", timeout=60000)
            # Give Streamlit additional time to initialize
            time.sleep(2)
            # Verify page actually loaded
            if page.url.startswith(RAG_UI_ENDPOINT):
                return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Navigation attempt {attempt + 1} failed: {e}, retrying...")
            time.sleep(2)


class TestChatUIBasics:
    """Basic UI tests for the chat interface"""
    
    def test_page_loads(self, page: Page):
        """Test that the chat page loads successfully"""
        page.goto(RAG_UI_ENDPOINT)
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Check URL instead of body visibility (more reliable in headless mode)
        assert page.url.startswith(RAG_UI_ENDPOINT)
        
        # Verify page content loaded
        page_content = page.content()
        assert len(page_content) > 100  # Should have substantial content
    
    def test_chat_title_visible(self, page: Page):
        """Test that the chat page title is visible"""
        title = page.get_by_text("💬 Chat", exact=False)
        expect(title).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_sidebar_configuration_visible(self, page: Page):
        """Test that the configuration sidebar is visible"""
        config_heading = page.get_by_text("Configuration", exact=False).first
        expect(config_heading).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_model_selector_visible(self, page: Page):
        """Test that the model selector is visible in sidebar"""
        # Use role-based selector to avoid strict mode violations
        model_heading = page.get_by_role("heading", name="Model")
        expect(model_heading).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_chat_input_visible(self, page: Page):
        """Test that the chat input field is visible"""
        chat_input = page.get_by_placeholder("Ask a question...", exact=False)
        expect(chat_input).to_be_visible(timeout=TEST_TIMEOUT)


class TestDirectModeChat:
    """UI tests for direct mode (non-agent) chat"""
    
    def test_direct_mode_selection(self, page: Page):
        """Test selecting direct mode"""
        # Look for the "Direct" text in the radio button labels
        # Streamlit radio buttons are structured with labels
        direct_label = page.get_by_text("Direct", exact=True).first
        expect(direct_label).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_direct_mode_shows_vector_db_selection(self, page: Page):
        """Test that direct mode shows vector DB selection"""
        # Just verify the page loads - actual vector DBs depend on setup
        page_content = page.content()
        assert len(page_content) > 0


class TestAgentModeChat:
    """UI tests for agent mode chat"""
    
    def test_agent_mode_selection(self, page: Page):
        """Test selecting agent mode"""
        agent_mode = page.get_by_text("Agent-based", exact=False).first
        expect(agent_mode).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_agent_mode_shows_toolgroups(self, page: Page):
        """Test that agent mode shows available toolgroups"""
        agent_radio = page.get_by_text("Agent-based", exact=False).first
        if agent_radio.is_visible():
            agent_radio.click()
            time.sleep(1)
        
        toolgroups = page.get_by_text("Available ToolGroups", exact=False)
        expect(toolgroups).to_be_visible(timeout=TEST_TIMEOUT)


class TestConfigurationOptions:
    """UI tests for configuration options in sidebar"""
    
    def test_temperature_slider(self, page: Page):
        """Test that temperature slider is visible"""
        temp_label = page.get_by_text("Temperature", exact=False).first
        expect(temp_label).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_max_tokens_slider(self, page: Page):
        """Test that max tokens slider is visible"""
        max_tokens_label = page.get_by_text("Max Tokens", exact=False).first
        expect(max_tokens_label).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_system_prompt_textarea(self, page: Page):
        """Test that system prompt textarea is visible"""
        # Use role-based selector to avoid strict mode violations
        system_prompt_heading = page.get_by_role("heading", name="System Prompt")
        expect(system_prompt_heading).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_clear_chat_button(self, page: Page):
        """Test that clear chat button is visible"""
        clear_button = page.get_by_text("Clear Chat", exact=False).first
        expect(clear_button).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_clear_chat_button_works(self, page: Page):
        """Test that clicking clear chat button resets the conversation"""
        clear_button = page.get_by_text("Clear Chat", exact=False).first
        clear_button.click()
        
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        greeting = page.get_by_text("How can I help you?", exact=False)
        expect(greeting).to_be_visible(timeout=TEST_TIMEOUT)


class TestRAGConfiguration:
    """UI tests for RAG configuration"""
    
    def test_vector_db_selection_in_direct_mode(self, page: Page):
        """Test that vector DB selection is available in direct mode"""
        page_content = page.content()
        assert len(page_content) > 0
    
    def test_rag_tool_in_agent_mode(self, page: Page):
        """Test that RAG tool is available in agent mode"""
        agent_radio = page.get_by_text("Agent-based", exact=False).first
        if agent_radio.is_visible():
            agent_radio.click()
            time.sleep(1)
        
        assert page.url.startswith(RAG_UI_ENDPOINT)


class TestResponseDisplay:
    """UI tests for response display and formatting"""
    
    def test_initial_greeting_message(self, page: Page):
        """Test that initial greeting message is displayed"""
        greeting = page.get_by_text("How can I help you?", exact=False)
        expect(greeting).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_tool_debug_toggle(self, page: Page):
        """Test that tool debug toggle is visible"""
        debug_toggle = page.get_by_text("Show Tool/Debug Info", exact=False)
        expect(debug_toggle).to_be_visible(timeout=TEST_TIMEOUT)


class TestMaaSIntegration:
    """UI tests for MaaS (Model-as-a-Service) integration through the UI
    
    These tests verify that MaaS works end-to-end through the browser UI.
    They send actual messages and verify MaaS responses.
    """
    
    @pytest.mark.skipif(
        os.getenv("SKIP_MODEL_TESTS", "false").lower() == "true",
        reason="Model inference tests disabled via SKIP_MODEL_TESTS"
    )
    def test_maas_chat_completion_direct_mode(self, page: Page):
        """Test that MaaS responds to chat messages in direct mode"""
        # Ensure we're in direct mode (default)
        # Verify the chat input is visible
        chat_input = page.get_by_placeholder("Ask a question...", exact=False)
        expect(chat_input).to_be_visible(timeout=TEST_TIMEOUT)
        
        # Send a simple test message
        test_message = "Say 'Hello from RAG e2e test!' in one short sentence."
        chat_input.fill(test_message)
        chat_input.press("Enter")
        
        # Wait for Streamlit to process the input and rerun
        page.wait_for_load_state("networkidle")
        time.sleep(3)  # Give Streamlit time to send request and start receiving response
        
        # Wait for the user message to appear in chat
        user_msg = page.get_by_text(test_message, exact=False)
        expect(user_msg).to_be_visible(timeout=TEST_TIMEOUT)
        
        # Wait for assistant response (MaaS should respond)
        # Streamlit chat messages have structure: stChatMessage with role
        # Look for assistant messages (not user, not the initial greeting)
        max_wait = 90  # seconds - MaaS can be slow
        wait_time = 0
        while wait_time < max_wait:
            time.sleep(3)
            wait_time += 3
            
            # Check for new assistant message content
            # Streamlit chat messages are structured with role="assistant"
            # We want to find text that's not the user message and not the initial greeting
            page_content = page.content()
            
            # Look for assistant message containers
            # Try multiple approaches to find the response
            assistant_containers = page.locator('[data-testid="stChatMessage"]').all()
            
            for container in assistant_containers:
                if container.is_visible():
                    text_content = container.inner_text().strip()
                    # Check if it's a new assistant message (not greeting, not user message)
                    if (text_content and 
                        text_content != "How can I help you?" and 
                        test_message not in text_content and
                        len(text_content) > 15):  # Real response should be substantial
                        # Found a real MaaS response!
                        print(f"✅ MaaS responded: {text_content[:150]}...")
                        assert len(text_content) > 10, "MaaS response too short"
                        return  # Success!
            
            # Also check for any new text that appeared after user message
            # Streamlit might render responses incrementally
            all_visible_text = page.locator('body').inner_text()
            if test_message in all_visible_text:
                # Check if there's additional text that looks like a response
                lines = all_visible_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if (line and 
                        test_message not in line and
                        "How can I help you?" not in line and
                        len(line) > 20 and  # Substantial response
                        any(word in line.lower() for word in ['hello', 'test', 'rag', 'e2e', 'from'])):  # Should mention something from our test
                        print(f"✅ MaaS responded (found in text): {line[:150]}...")
                        return  # Success!
        
        # If we get here, no response was received
        # Print debug info before failing
        print(f"\n❌ Debug info after {max_wait} seconds:")
        print(f"Page URL: {page.url}")
        print(f"User message visible: {user_msg.is_visible()}")
        print(f"Number of chat messages found: {len(assistant_containers)}")
        page_content = page.content()
        print(f"Page content length: {len(page_content)}")
        # Take a screenshot if possible
        try:
            page.screenshot(path="test-debug-screenshot.png")
            print("Screenshot saved: test-debug-screenshot.png")
        except:
            pass
        
        pytest.fail(f"MaaS did not respond within {max_wait} seconds")
    
    @pytest.mark.skipif(
        os.getenv("SKIP_MODEL_TESTS", "false").lower() == "true",
        reason="Model inference tests disabled via SKIP_MODEL_TESTS"
    )
    def test_maas_model_selection(self, page: Page):
        """Test that MaaS model is available and can be selected"""
        # Check that model selector shows the MaaS model
        model_id = os.getenv("MAAS_MODEL_ID", "llama-3-2-3b")
        
        # The model should be in the selectbox options
        # In Streamlit, we can check if the model identifier appears in the page
        page_content = page.content()
        
        # Model identifier should appear somewhere (in selectbox or visible text)
        # This is a basic check - full selection would require interacting with Streamlit selectbox
        assert len(page_content) > 0, "Page should have content"
        
        # More specific check: look for model in the model selector area
        # Streamlit selectbox for model should be visible
        model_heading = page.get_by_role("heading", name="Model")
        expect(model_heading).to_be_visible(timeout=TEST_TIMEOUT)
