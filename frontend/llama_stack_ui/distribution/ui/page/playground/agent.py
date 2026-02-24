# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Agent mode implementation for chat with automatic tool calling.
"""

import logging

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api
from llama_stack_ui.distribution.ui.modules.utils import clean_text, get_vector_db_name


logger = logging.getLogger(__name__)


def build_response_tools(toolgroup_selection, selected_vector_dbs, top_k, client):
    """
    Convert toolgroup selections to LlamaStack Responses API compatible tool format.

    Args:
        toolgroup_selection: List of selected toolgroup IDs
        selected_vector_dbs: List of selected vector database names
        top_k: Maximum file search results per query
        client: LlamaStack client instance

    Returns:
        List of tools in Responses API format (works for both Agent and Direct modes)
    """
    agent_tools = []

    for toolgroup_name in toolgroup_selection:
        if toolgroup_name == "builtin::rag":
            if len(selected_vector_dbs) > 0:
                vector_dbs = client.vector_stores.list() or []
                vector_db_ids = [
                    vector_db.id for vector_db in vector_dbs
                    if get_vector_db_name(vector_db) in selected_vector_dbs
                ]
                # Use file_search tool format
                agent_tools.append({
                    "type": "file_search",
                    "max_num_results": top_k,
                    "vector_store_ids": list(vector_db_ids),
                })
        elif "web_search" in toolgroup_name or "search" in toolgroup_name.lower():
            # Convert search tools to web_search format
            agent_tools.append({"type": "web_search"})
        elif toolgroup_name.startswith("mcp::"):
            # For MCP tools, get server info
            try:
                toolgroups = client.toolgroups.list()
                for toolgroup in toolgroups:
                    if str(toolgroup.identifier) == toolgroup_name:
                        agent_tools.append({
                            "type": "mcp",
                            "server_label": toolgroup.args.get(
                                "name", str(toolgroup.identifier)
                            ),
                            "server_url": toolgroup.mcp_endpoint.uri,
                        })
                        break
            except Exception as e:
                logger.logger.debug("Failed to get MCP server info for %s: %s", toolgroup_name, e)
        else:
            # For other toolgroups, get individual tools and convert to function format
            try:
                tools_in_group = client.tools.list(toolgroup_id=toolgroup_name)
                for tool in tools_in_group:
                    # Convert to function tool dict
                    agent_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.parameters or {}
                        }
                    })
            except Exception as e:
                logger.logger.debug("Failed to get tools for %s: %s", toolgroup_name, e)

    return agent_tools


# ============================================================================
# Agent Mode - Chunk Handlers
# ============================================================================

def handle_agent_file_search_chunk(state, selected_vector_dbs):
    """Handle file_search tool chunk in agent mode."""
    if state.tool_used:
        return

    # Show tool status message in persistent container
    if selected_vector_dbs:
        db_label = "vector store" if len(selected_vector_dbs) == 1 else "vector stores"
        status_msg = (
            f"🛠 :grey[_Using file_search tool with {db_label}: "
            f"{', '.join(selected_vector_dbs)}_]"
        )
    else:
        status_msg = "🛠 :grey[_Using file_search tool..._]"

    state.tool_status = status_msg
    with state.containers.tool_status:
        st.markdown(status_msg)


def handle_agent_web_search_chunk(state):
    """Handle web_search tool chunk in agent mode."""
    if state.tool_used:
        return

    status_msg = "🛠 :grey[_Using web_search tool..._]"
    state.tool_status = status_msg
    with state.containers.tool_status:
        st.markdown(status_msg)


def handle_agent_output_item_done(chunk, state):
    """Handle response.output_item.done - tool execution completion with results."""
    if not hasattr(chunk, 'item'):
        return

    item = chunk.item
    item_type = getattr(item, 'type', None)

    if item_type == "file_search_call":
        # File search results
        if hasattr(item, 'results') and item.results:
            display_results = []
            for r in item.results:
                text = getattr(r, 'text', '')
                attrs = getattr(r, 'attributes', {})
                source = attrs.get('source') or getattr(r, 'filename', 'unknown')
                display_results.append({"source": source, "text": clean_text(text)})
            state.tool_results.append({
                'title': '📄 File Search Results',
                'type': 'json',
                'content': display_results
            })
            with state.containers.tool_results:
                with st.expander("📄 File Search Results", expanded=False):
                    st.json(display_results)

    elif item_type == "web_search_call":
        # Web search - API doesn't expose raw results, just status
        pass

    elif item_type == "function_call":
        # Function call output
        if hasattr(item, 'output') and item.output:
            tool_name = getattr(item, 'name', 'function')
            state.tool_results.append({
                'title': f'🔧 Tool Output: {tool_name}',
                'type': 'code',
                'content': str(item.output)
            })
            with state.containers.tool_results:
                with st.expander(f"🔧 Tool Output: {tool_name}", expanded=False):
                    st.code(str(item.output))

    elif item_type == "mcp_call":
        # MCP call output
        if hasattr(item, 'output') and item.output:
            tool_name = getattr(item, 'name', 'mcp')
            state.tool_results.append({
                'title': f'🔧 MCP Tool Output: {tool_name}',
                'type': 'code',
                'content': str(item.output)
            })
            with state.containers.tool_results:
                with st.expander(f"🔧 MCP Tool Output: {tool_name}", expanded=False):
                    st.code(str(item.output))

    elif item_type and item_type.endswith("_call"):
        # Generic handler for any other tool call types
        if hasattr(item, 'results') and item.results:
            formatted_name = item_type.replace("_", " ").title()
            state.tool_results.append({
                'title': f'🔧 {formatted_name} Results',
                'type': 'json',
                'content': item.results
            })
            with state.containers.tool_results:
                with st.expander(f"🔧 {formatted_name} Results", expanded=False):
                    st.json(item.results)
        elif hasattr(item, 'output') and item.output:
            formatted_name = item_type.replace("_", " ").title()
            state.tool_results.append({
                'title': f'🔧 {formatted_name} Output',
                'type': 'json',
                'content': item.output
            })
            with state.containers.tool_results:
                with st.expander(f"🔧 {formatted_name} Output", expanded=False):
                    st.json(item.output)


def handle_chunk_error(chunk):
    """Handle error chunk and return whether to stop streaming."""
    error_msg = "Unknown error"
    error_code = None

    # Try to get error from chunk.error first
    if hasattr(chunk, 'error') and chunk.error:
        if hasattr(chunk.error, 'message'):
            error_msg = chunk.error.message
        if hasattr(chunk.error, 'code'):
            error_code = chunk.error.code
    # Fallback to chunk attributes
    elif hasattr(chunk, 'error_message'):
        error_msg = chunk.error_message

    error_display = f"❌ Error: {error_msg}"
    if error_code:
        error_display += f" (Code: {error_code})"

    st.error(error_display)
    logger.debug("Response failed: %s", error_msg)
    return True  # Stop streaming


def handle_chunk_completed(chunk):
    """Handle completed chunk."""
    logger.debug("Response completed successfully")
    if hasattr(chunk, 'stop_reason'):
        logger.debug("Stop reason: %s", chunk.stop_reason)


def handle_chunk_done(chunk, state):
    """Handle done chunk and finalize response."""
    has_output = (
        hasattr(chunk, 'response') and
        hasattr(chunk.response, 'output_text') and
        chunk.response.output_text
    )
    if has_output:
        state.full_response = chunk.response.output_text


def process_chunk_by_type(chunk, state, selected_vector_dbs):
    """Process a single chunk based on its type. Returns True to stop streaming."""
    chunk_type = chunk.type

    # Handle file_search tool
    if chunk_type == "response.file_search_call.in_progress":
        handle_agent_file_search_chunk(state, selected_vector_dbs)

    # Handle web_search tool
    elif chunk_type == "response.web_search_call.in_progress":
        handle_agent_web_search_chunk(state)

    elif chunk_type in ("response.web_search_call.searching",
                         "response.web_search_call.completed"):
        pass  # Just for event tracking

    # Handle tool results
    elif chunk_type == "response.output_item.done":
        handle_agent_output_item_done(chunk, state)

    # Handle reasoning
    elif chunk_type == "response.reasoning_text.delta":
        if hasattr(chunk, 'delta') and chunk.delta:
            state.update_reasoning(chunk.delta)

    # Handle message content
    elif chunk_type == "response.output_text.delta":
        if hasattr(chunk, 'delta') and chunk.delta:
            state.update_message(chunk.delta)

    # Handle errors
    elif chunk_type == "response.failed":
        return handle_chunk_error(chunk)

    # Handle completion
    elif chunk_type == "response.completed":
        handle_chunk_completed(chunk)

    # Handle done
    elif chunk_type == "response.done":
        handle_chunk_done(chunk, state)

    return False  # Continue streaming


# ============================================================================
# Agent Mode - Main Functions
# ============================================================================

def stream_agent_response(response, state, selected_vector_dbs):
    """
    Stream and process chunks from Responses API.
    Updates state containers as chunks arrive.
    """
    chunk_count = 0

    for chunk in response:
        chunk_count += 1
        logger.debug("Chunk #%s: type=%s", chunk_count, getattr(chunk, 'type', 'NO_TYPE'))
        logger.debug("  -> Full chunk: %s", chunk)

        # Some server failures arrive as an error payload with type=None.
        if hasattr(chunk, 'error') and chunk.error:
            if isinstance(chunk.error, dict):
                error_msg = chunk.error.get("message", "Unknown error")
            else:
                error_msg = str(chunk.error)
            st.error(
                f"❌ Error: {error_msg}"
            )
            logger.debug("Response stream error: %s", error_msg)
            break

        if hasattr(chunk, 'type'):
            should_stop = process_chunk_by_type(chunk, state, selected_vector_dbs)
            if should_stop:
                break


def save_agent_response_to_session(state):
    """Save agent response to session state."""
    state.finalize_reasoning()
    state.finalize_message()

    response_dict = {
        "role": "assistant",
        "content": state.full_response,
        "stop_reason": "end_of_message"
    }

    if state.reasoning_text:
        response_dict["reasoning"] = state.reasoning_text
    if state.tool_status:
        response_dict["tool_status"] = state.tool_status
    if state.tool_results:
        response_dict["tool_results"] = state.tool_results

    st.session_state.messages.append(response_dict)


def agent_process_prompt(prompt, state, config):
    """Agent-based mode: Use Responses API with automatic tool calling."""
    # Build tools list from selected toolgroups
    tools = build_response_tools(
        config.toolgroup_selection,
        config.selected_vector_dbs,
        config.sampling.top_k,
        llama_stack_api.client,
    ) if config.toolgroup_selection else None

    # Build request for Responses API
    request_kwargs = {
        "model": config.model,
        "instructions": config.system_prompt,
        "input": prompt,
        "conversation": config.conversation_id,
        "temperature": config.sampling.temperature,
        "max_infer_iters": config.sampling.max_infer_iters,
        "stream": True,
        "max_output_tokens": config.sampling.max_tokens,

    }
    logger.debug("************************************************")
    logger.debug("Request Args: %s", request_kwargs)
    logger.debug("************************************************")

    # Add tools if available
    if tools:
        request_kwargs["tools"] = tools

    logger.debug("Request: %s", request_kwargs)
    try:
        response = llama_stack_api.client.responses.create(**request_kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
        st.error(f"❌ Error: {str(e)}")
        logger.debug("Agent mode create() error: %s", e)
        return

    # Stream response and update UI
    stream_agent_response(response, state, config.selected_vector_dbs)

    # Save response to session
    save_agent_response_to_session(state)
