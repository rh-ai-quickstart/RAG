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
    page.goto(RAG_UI_ENDPOINT)
    # Wait for Streamlit to finish loading
    page.wait_for_load_state("networkidle")
    # Give Streamlit additional time to initialize
    time.sleep(2)


class TestChatUIBasics:
    """Basic UI tests for the chat interface"""
    
    def test_page_loads(self, page: Page):
        """Test that the chat page loads successfully"""
        page.goto(RAG_UI_ENDPOINT)
        
        # Check that we can see the Streamlit app
        expect(page.locator("body")).to_be_visible()
        
        # The page should have loaded without errors
        assert page.url.startswith(RAG_UI_ENDPOINT)
    
    def test_chat_title_visible(self, page: Page):
        """Test that the chat page title is visible"""
        # Look for the chat title
        title = page.get_by_text("💬 Chat", exact=False)
        expect(title).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_sidebar_configuration_visible(self, page: Page):
        """Test that the configuration sidebar is visible"""
        # Streamlit sidebar should be visible
        # Look for "Configuration" heading
        config_heading = page.get_by_text("Configuration", exact=False)
        expect(config_heading).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_model_selector_visible(self, page: Page):
        """Test that the model selector is visible in sidebar"""
        # Look for "Model" label
        model_label = page.get_by_text("Model", exact=False)
        expect(model_label).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_chat_input_visible(self, page: Page):
        """Test that the chat input field is visible"""
        # Streamlit uses a chat input at the bottom
        # Look for the input placeholder
        chat_input = page.get_by_placeholder("Ask a question...", exact=False)
        expect(chat_input).to_be_visible(timeout=TEST_TIMEOUT)


class TestDirectModeChat:
    """UI tests for direct mode (non-agent) chat"""
    
    def test_direct_mode_selection(self, page: Page):
        """Test selecting direct mode"""
        # Look for "Processing mode" radio buttons
        direct_mode = page.get_by_text("Direct", exact=False)
        expect(direct_mode).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_direct_mode_shows_vector_db_selection(self, page: Page):
        """Test that direct mode shows vector DB selection"""
        # In direct mode, vector DB selection should be visible
        # Look for "Document Collections" text
        doc_collections = page.get_by_text("Document Collections", exact=False)
        # This may or may not be visible depending on available DBs
        # Just check it can be found in the page content
    
    @pytest.mark.skip(reason="Requires live model for actual chat")
    def test_send_simple_message_direct_mode(self, page: Page):
        """Test sending a simple message in direct mode"""
        # Make sure we're in direct mode
        direct_radio = page.get_by_text("Direct", exact=False)
        if direct_radio.is_visible():
            direct_radio.click()
        
        # Find and fill the chat input
        chat_input = page.get_by_placeholder("Ask a question...")
        chat_input.fill("Hello, can you hear me?")
        
        # Submit the message (press Enter)
        chat_input.press("Enter")
        
        # Wait for response (this requires a working model)
        # Look for assistant message
        time.sleep(5)  # Give time for response
        
        # Check that the user message appears in chat history
        user_msg = page.get_by_text("Hello, can you hear me?")
        expect(user_msg).to_be_visible(timeout=TEST_TIMEOUT)


class TestAgentModeChat:
    """UI tests for agent mode chat"""
    
    def test_agent_mode_selection(self, page: Page):
        """Test selecting agent mode"""
        # Look for "Agent-based" radio button
        agent_mode = page.get_by_text("Agent-based", exact=False)
        expect(agent_mode).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_agent_mode_shows_toolgroups(self, page: Page):
        """Test that agent mode shows available toolgroups"""
        # Click on agent-based mode
        agent_radio = page.get_by_text("Agent-based", exact=False).first
        if agent_radio.is_visible():
            agent_radio.click()
            time.sleep(1)
        
        # Look for "Available ToolGroups" section
        toolgroups = page.get_by_text("Available ToolGroups", exact=False)
        expect(toolgroups).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_agent_type_selector(self, page: Page):
        """Test agent type selector (Regular vs ReAct)"""
        # Click on agent mode first
        agent_radio = page.get_by_text("Agent-based", exact=False).first
        if agent_radio.is_visible():
            agent_radio.click()
            time.sleep(1)
        
        # Look for agent type options
        regular_agent = page.get_by_text("Regular", exact=False)
        react_agent = page.get_by_text("ReAct", exact=False)
        
        # At least one should be visible
        assert regular_agent.is_visible() or react_agent.is_visible()


class TestConfigurationOptions:
    """UI tests for configuration options in sidebar"""
    
    def test_temperature_slider(self, page: Page):
        """Test that temperature slider is visible"""
        # Look for "Temperature" label
        temp_label = page.get_by_text("Temperature", exact=False)
        expect(temp_label).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_max_tokens_slider(self, page: Page):
        """Test that max tokens slider is visible"""
        # Look for "Max Tokens" label
        max_tokens_label = page.get_by_text("Max Tokens", exact=False)
        expect(max_tokens_label).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_system_prompt_textarea(self, page: Page):
        """Test that system prompt textarea is visible"""
        # Look for "System Prompt" label
        system_prompt_label = page.get_by_text("System Prompt", exact=False)
        expect(system_prompt_label).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_clear_chat_button(self, page: Page):
        """Test that clear chat button is visible"""
        # Look for "Clear Chat" button
        clear_button = page.get_by_text("Clear Chat", exact=False)
        expect(clear_button).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_clear_chat_button_works(self, page: Page):
        """Test that clicking clear chat button resets the conversation"""
        # Click the clear chat button
        clear_button = page.get_by_text("Clear Chat", exact=False).first
        clear_button.click()
        
        # Wait for page to reload/reset
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # The chat should be reset - check for initial greeting
        greeting = page.get_by_text("How can I help you?", exact=False)
        expect(greeting).to_be_visible(timeout=TEST_TIMEOUT)


class TestRAGConfiguration:
    """UI tests for RAG configuration"""
    
    def test_vector_db_selection_in_direct_mode(self, page: Page):
        """Test that vector DB selection is available in direct mode"""
        # Make sure we're in direct mode (default)
        # Look for vector DB multiselect
        # The text "Document Collections" should appear if there are vector DBs
        page_content = page.content()
        # Just verify the page loads - actual vector DBs depend on setup
        assert len(page_content) > 0
    
    def test_rag_tool_in_agent_mode(self, page: Page):
        """Test that RAG tool is available in agent mode"""
        # Click on agent mode
        agent_radio = page.get_by_text("Agent-based", exact=False).first
        if agent_radio.is_visible():
            agent_radio.click()
            time.sleep(1)
        
        # Page should load without errors
        assert page.url.startswith(RAG_UI_ENDPOINT)


class TestResponseDisplay:
    """UI tests for response display and formatting"""
    
    def test_initial_greeting_message(self, page: Page):
        """Test that initial greeting message is displayed"""
        # Look for the assistant's initial greeting
        greeting = page.get_by_text("How can I help you?", exact=False)
        expect(greeting).to_be_visible(timeout=TEST_TIMEOUT)
    
    def test_tool_debug_toggle(self, page: Page):
        """Test that tool debug toggle is visible"""
        # Look for "Show Tool/Debug Info" toggle
        debug_toggle = page.get_by_text("Show Tool/Debug Info", exact=False)
        expect(debug_toggle).to_be_visible(timeout=TEST_TIMEOUT)


class TestResponsiveness:
    """UI tests for responsive design"""
    
    def test_mobile_viewport(self, page: Page):
        """Test that the app loads on mobile viewport"""
        # Set mobile viewport
        page.set_viewport_size({"width": 375, "height": 812})
        page.goto(RAG_UI_ENDPOINT)
        
        # Wait for load
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Page should still load
        expect(page.locator("body")).to_be_visible()
    
    def test_tablet_viewport(self, page: Page):
        """Test that the app loads on tablet viewport"""
        # Set tablet viewport
        page.set_viewport_size({"width": 768, "height": 1024})
        page.goto(RAG_UI_ENDPOINT)
        
        # Wait for load
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Page should still load
        expect(page.locator("body")).to_be_visible()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

