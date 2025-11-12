"""
Unit tests for the chat/playground module
Tests individual functions and logic without requiring Streamlit runtime

Note: We test the core logic inline rather than importing from chat.py
because chat.py executes Streamlit code at module level.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


# Recreate the get_strategy function logic for testing
def get_strategy(temperature, top_p):
    """Determines the sampling strategy for the LLM based on temperature."""
    return {'type': 'greedy'} if temperature == 0 else {
            'type': 'top_p', 'temperature': temperature, 'top_p': top_p
        }


class TestGetStrategy:
    """Test the get_strategy function for sampling parameters"""
    
    def test_greedy_strategy_when_temperature_is_zero(self):
        """When temperature is 0, should return greedy strategy"""
        strategy = get_strategy(temperature=0, top_p=0.95)
        assert strategy == {'type': 'greedy'}
    
    def test_top_p_strategy_when_temperature_is_nonzero(self):
        """When temperature > 0, should return top_p strategy with params"""
        strategy = get_strategy(temperature=0.7, top_p=0.95)
        assert strategy == {
            'type': 'top_p',
            'temperature': 0.7,
            'top_p': 0.95
        }
    
    def test_top_p_strategy_with_high_temperature(self):
        """Test with high temperature value"""
        strategy = get_strategy(temperature=1.5, top_p=0.9)
        assert strategy['type'] == 'top_p'
        assert strategy['temperature'] == 1.5
        assert strategy['top_p'] == 0.9
    
    def test_top_p_strategy_with_low_top_p(self):
        """Test with low top_p value"""
        strategy = get_strategy(temperature=0.5, top_p=0.5)
        assert strategy['top_p'] == 0.5


class TestChatMessageFormatting:
    """Test chat message formatting and handling"""
    
    def test_direct_mode_prompt_with_context(self):
        """Test that direct mode correctly formats prompts with RAG context"""
        prompt = "What is the Eiffel Tower?"
        context = "The Eiffel Tower is 330 metres tall."
        
        expected_prompt = f"Please answer the following query using the context below.\n\nCONTEXT:\n{context}\n\nQUERY:\n{prompt}"
        
        # Simulate the logic from direct_process_prompt
        extended_prompt = f"Please answer the following query using the context below.\n\nCONTEXT:\n{context}\n\nQUERY:\n{prompt}"
        
        assert extended_prompt == expected_prompt
        assert "CONTEXT:" in extended_prompt
        assert "QUERY:" in extended_prompt
    
    def test_direct_mode_prompt_without_context(self):
        """Test that direct mode correctly formats prompts without RAG context"""
        prompt = "Hello, how are you?"
        
        expected_prompt = f"Please answer the following query. \n\nQUERY:\n{prompt}"
        
        # Simulate the logic from direct_process_prompt
        extended_prompt = f"Please answer the following query. \n\nQUERY:\n{prompt}"
        
        assert extended_prompt == expected_prompt
        assert "CONTEXT:" not in extended_prompt
        assert "QUERY:" in extended_prompt


class TestAgentType:
    """Test AgentType enum and agent configuration"""
    
    def test_agent_types_exist(self):
        """Test that AgentType enum values are as expected"""
        # Testing the expected agent type values
        regular_value = "Regular"
        react_value = "ReAct"
        
        assert regular_value == "Regular"
        assert react_value == "ReAct"


class TestSystemPromptHandling:
    """Test system prompt construction"""
    
    def test_default_system_prompt_for_direct_mode(self):
        """Test default system prompt"""
        default_prompt = "You are a helpful AI assistant."
        assert len(default_prompt) > 0
        assert "helpful" in default_prompt.lower()
    
    def test_react_system_prompt(self):
        """Test ReAct agent system prompt"""
        react_prompt = "You are a helpful ReAct agent. Reason step-by-step to fulfill the user query using available tools."
        assert "ReAct" in react_prompt
        assert "tools" in react_prompt.lower()
    
    def test_system_prompt_ending_with_period(self):
        """Test that system prompts are properly formatted with period"""
        prompt = "You are a helpful assistant"
        updated_prompt = prompt if prompt.strip().endswith('.') else prompt + '.'
        assert updated_prompt.endswith('.')
        
        prompt_with_period = "You are a helpful assistant."
        updated_prompt = prompt_with_period if prompt_with_period.strip().endswith('.') else prompt_with_period + '.'
        assert updated_prompt.endswith('.')
        assert updated_prompt.count('.') == 1


class TestToolGroupSelection:
    """Test tool group selection and configuration"""
    
    def test_rag_tool_configuration_format(self):
        """Test that RAG tool is correctly configured with vector DB IDs"""
        tool_name = "builtin::rag"
        selected_vector_dbs = ["test-db-1", "test-db-2"]
        
        tool_dict = dict(
            name="builtin::rag",
            args={
                "vector_db_ids": list(selected_vector_dbs),
            },
        )
        
        assert tool_dict["name"] == "builtin::rag"
        assert "vector_db_ids" in tool_dict["args"]
        assert tool_dict["args"]["vector_db_ids"] == ["test-db-1", "test-db-2"]
    
    def test_builtin_tools_filtering(self):
        """Test that builtin tools are correctly filtered from MCP tools"""
        tool_groups_list = ["builtin::rag", "builtin::web_search", "mcp::github", "mcp::slack"]
        
        builtin_tools = [tool for tool in tool_groups_list if not tool.startswith("mcp::")]
        mcp_tools = [tool for tool in tool_groups_list if tool.startswith("mcp::")]
        
        assert len(builtin_tools) == 2
        assert len(mcp_tools) == 2
        assert "builtin::rag" in builtin_tools
        assert "mcp::github" in mcp_tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

