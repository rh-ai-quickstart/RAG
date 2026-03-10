# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Chat page for LlamaStack UI with RAG support.
Provides both Direct mode (manual RAG) and Agent-based mode (automatic tool calling).
"""

import logging
from dataclasses import dataclass

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api
from llama_stack_ui.distribution.ui.modules.utils import (
    get_suggestions_for_databases,
    get_vector_db_name,
)
from llama_stack_ui.distribution.ui.page.playground.agent import (
    agent_process_prompt,
)
from llama_stack_ui.distribution.ui.page.playground.direct import (
    direct_process_prompt,
)


logger = logging.getLogger(__name__)

def render_tool_results(tool_results):
    """Render tool results from a message."""
    for tool_result in tool_results:
        with st.expander(tool_result['title'], expanded=False):
            if tool_result['type'] == 'json':
                st.json(tool_result['content'])
            else:
                st.code(tool_result['content'])


def render_message(msg):
    """Render a single message in chat history."""
    with st.chat_message(msg['role']):
        # Display tool status if present
        if msg.get('tool_status'):
            st.markdown(msg['tool_status'])

        # Display tool results if present
        if msg.get('tool_results'):
            render_tool_results(msg['tool_results'])

        # Display reasoning if present (right before the answer)
        if msg.get('reasoning'):
            with st.expander("🧠 Reasoning", expanded=False):
                st.markdown(msg['reasoning'])

        # Display the final answer
        st.markdown(msg['content'])


def render_history():
    """Renders the chat history from the session state."""
    # Initialize messages in the session state if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        render_message(msg)


def fetch_models_and_tools():
    """Fetch and categorize models and toolgroups from LlamaStack."""
    client = llama_stack_api.client

    # Fetch models
    models = client.models.list()
    model_list = [model.id for model in models if model.custom_metadata.get("model_type") == "llm"]
    

    # Fetch and categorize toolgroups
    tool_groups = client.toolgroups.list()
    logger.debug("Raw tool groups from LlamaStack: %s", tool_groups)
    tool_groups_list = [tool_group.identifier for tool_group in tool_groups]
    logger.debug("Tool group identifiers: %s", tool_groups_list)

    mcp_tools_list = [tool for tool in tool_groups_list if tool.startswith("mcp::")]
    logger.debug("MCP tools: %s", mcp_tools_list)

    builtin_tools_list = [tool for tool in tool_groups_list if not tool.startswith("mcp::")]
    logger.debug("Built-in tools: %s", builtin_tools_list)

    return model_list, builtin_tools_list, mcp_tools_list


def render_toolgroup_selection(builtin_tools_list, mcp_tools_list, selected_vector_dbs,
                                on_toolgroup_change, on_reset):
    """Render toolgroup selection UI and return selected toolgroups."""
    st.subheader("Available ToolGroups")

    # Initialize toolgroup selector if not present
    if "toolgroup_selector" not in st.session_state:
        # Default: include RAG if a vector DB is selected
        if selected_vector_dbs:
            st.session_state["toolgroup_selector"] = ["builtin::rag"]
        else:
            st.session_state["toolgroup_selector"] = []

    # Built-in tools selection (web_search, etc.)
    toolgroup_selection = st.pills(
        label="Built-in tools",
        options=builtin_tools_list,
        selection_mode="multi",
        key="toolgroup_selector",
        on_change=on_toolgroup_change,
        format_func=lambda tool: "".join(tool.split("::")[1:]),
        help="List of built-in tools from your llama stack server.",
    )

    # MCP tools selection (if available)
    if mcp_tools_list:
        mcp_selection = st.pills(
            label="MCP Servers",
            options=mcp_tools_list,
            selection_mode="multi",
            on_change=on_reset,
            format_func=lambda tool: "".join(tool.split("::")[1:]),
            help="List of MCP servers registered to your llama stack server.",
        )
        toolgroup_selection = list(toolgroup_selection) + list(mcp_selection)

    # Display active tools summary
    client = llama_stack_api.client
    grouped_tools = {}
    total_tools = 0

    for toolgroup_id in toolgroup_selection:
        tools = client.tools.list(toolgroup_id=toolgroup_id)
        logger.debug("Raw tools from toolgroup '%s': %s", toolgroup_id, tools)
        grouped_tools[toolgroup_id] = [tool.name for tool in tools]
        total_tools += len(tools)

    logger.debug("Grouped tools summary: %s", grouped_tools)

    if total_tools > 0:
        st.markdown(f"Active Tools: 🛠 {total_tools}")

        for group_id, tools in grouped_tools.items():
            with st.expander(f"🔧 Tools from `{group_id}`"):
                for idx, tool in enumerate(tools, start=1):
                    st.markdown(f"{idx}. `{tool.split(':')[-1]}`")

    return toolgroup_selection


def reset_agent():
    """Reset the agent by clearing session state and cache."""
    st.session_state.clear()
    st.cache_resource.clear()


def create_vector_db_callbacks(processing_mode, vector_dbs):
    """Create callbacks for vector DB and toolgroup synchronization."""
    def on_vector_db_change():
        """When vector DB changes, update toolgroup selection"""
        if processing_mode != "Agent-based":
            return

        selected_vdbs = st.session_state.get("chat_vector_db_selector", [])
        current_toolgroups = st.session_state.get("toolgroup_selector", [])

        if not selected_vdbs:
            # Remove RAG from toolgroups
            if "builtin::rag" in current_toolgroups:
                filtered = [t for t in current_toolgroups if t != "builtin::rag"]
                st.session_state["toolgroup_selector"] = filtered
        else:
            # Add RAG to toolgroups if not present
            if "builtin::rag" not in current_toolgroups:
                updated = list(current_toolgroups) + ["builtin::rag"]
                st.session_state["toolgroup_selector"] = updated

    def on_toolgroup_change():
        """When toolgroup changes, update vector DB selection"""
        current_toolgroups = st.session_state.get("toolgroup_selector", [])
        selected_vdbs = st.session_state.get("chat_vector_db_selector", [])

        if "builtin::rag" not in current_toolgroups:
            # RAG deselected, clear vector DB selection
            st.session_state["chat_vector_db_selector"] = []
        elif "builtin::rag" in current_toolgroups and not selected_vdbs:
            # RAG selected but no vector DB, select first available
            if vector_dbs:
                first_vdb = get_vector_db_name(vector_dbs[0])
                st.session_state["chat_vector_db_selector"] = [first_vdb]

    return on_vector_db_change, on_toolgroup_change


def render_sidebar_configuration(model_list, builtin_tools_list, mcp_tools_list):
    """Render sidebar configuration and return selected parameters."""
    st.title("Configuration")
    st.subheader("Model")
    model = st.selectbox(
        label="Model",
        options=model_list,
        on_change=reset_agent,
        label_visibility="collapsed",
    )

    # Processing Mode
    processing_mode = st.radio(
        "Processing mode",
        ["Direct", "Agent-based"],
        index=0,
        captions=[
            "Passes vector store search results as context to LLM",
            "Uses Responses API with tool calling",
        ],
        on_change=reset_agent,
        help="Choose how requests are processed.",
    )

    # Vector Database Selection
    vector_dbs = list(llama_stack_api.client.vector_stores.list() or [])
    on_vector_db_change, on_toolgroup_change = create_vector_db_callbacks(
        processing_mode, vector_dbs
    )

    selected_vector_dbs = render_vector_db_selector(
        vector_dbs, processing_mode, on_vector_db_change
    )

    # Toolgroup Selection (Agent-based mode only)
    toolgroup_selection = []
    if processing_mode == "Agent-based":
        toolgroup_selection = render_toolgroup_selection(
            builtin_tools_list, mcp_tools_list, selected_vector_dbs,
            on_toolgroup_change, reset_agent
        )

    # Sampling Parameters
    st.subheader("Sampling Parameters")
    temperature = st.slider(
        "Temperature",
        0.0, 2.0, 0.1, 0.05,
        on_change=reset_agent,
        help="Controls randomness. Higher values = more random.",
    )
    top_k = st.slider(
        "Top K",
        1, 50, 5, 1,
        on_change=reset_agent,
        help="Number of documents retrieved per vector store in Direct mode.",
    )
    max_tokens = st.slider(
        "Max Tokens",
        1, 4096, 512, 64,
        on_change=reset_agent,
        help="Maximum number of tokens to generate.",
    )
    max_infer_iters = st.slider(
        "Max Inference Iterations",
        1, 50, 10, 1,
        on_change=reset_agent,
        help="Maximum number of inference iterations before stopping",
    )

    # System Prompt
    st.subheader("System Prompt")
    default_prompt = "You are a helpful AI assistant."
    system_prompt = st.text_area(
        "System Prompt", value=default_prompt, on_change=reset_agent, height=100
    )

    if st.button("Clear Chat & Reset Config", use_container_width=True):
        reset_agent()
        st.rerun()

    return {
        'model': model,
        'processing_mode': processing_mode,
        'selected_vector_dbs': selected_vector_dbs,
        'toolgroup_selection': toolgroup_selection,
        'temperature': temperature,
        'top_k': top_k,
        'max_infer_iters': max_infer_iters,
        'max_tokens': max_tokens,
        'system_prompt': system_prompt,
    }


def render_vector_db_selector(vector_dbs, processing_mode, on_vector_db_change):
    """Render vector database selector and return selected databases."""
    selected_vector_dbs = []

    # Initialize vector DB selector if not present
    if "chat_vector_db_selector" not in st.session_state:
        st.session_state["chat_vector_db_selector"] = []

    if not vector_dbs:
        return selected_vector_dbs

    vector_db_names = [get_vector_db_name(vector_db) for vector_db in vector_dbs]
    selected_vector_dbs = st.multiselect(
        label="Select Document Collections for RAG queries",
        options=vector_db_names,
        key="chat_vector_db_selector",
        on_change=on_vector_db_change,
        help=(
            "Select one or more vector databases to use for retrieval, "
            "or leave empty for normal chat"
        )
    )

    # Store the selected DBs for Direct mode
    if selected_vector_dbs:
        if processing_mode == "Direct":
            st.session_state["direct_vector_dbs"] = [
                vdb for vdb in vector_dbs
                if get_vector_db_name(vdb) in selected_vector_dbs
            ]
    else:
        # Clear direct_vector_dbs if nothing is selected
        if "direct_vector_dbs" in st.session_state:
            del st.session_state["direct_vector_dbs"]

    return selected_vector_dbs


def initialize_session_state():
    """Initialize session state variables."""
    if "conversation_id" not in st.session_state:
        conversation = llama_stack_api.client.conversations.create()
        st.session_state["conversation_id"] = conversation.id
        logger.debug("Created new conversation: %s", conversation.id)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "How can I help you?",
                "stop_reason": "end_of_turn",
            }
        ]

    if "show_more_questions" not in st.session_state:
        st.session_state["show_more_questions"] = False

    if "selected_question" not in st.session_state:
        st.session_state["selected_question"] = None


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class SamplingParams:
    """Sampling parameters for model inference."""
    temperature: float
    top_k: int
    max_infer_iters: int
    max_tokens: int


@dataclass
class ChatConfig:
    """Configuration for chat processing."""
    model: str
    processing_mode: str
    system_prompt: str
    conversation_id: str
    toolgroup_selection: list
    selected_vector_dbs: list
    sampling: SamplingParams


# ============================================================================
# UI Container Classes
# ============================================================================

class Containers:
    """Simple container for UI elements.

    Note: Containers are created in visual display order (top to bottom).
    """
    def __init__(self):
        # Create containers in visual order: tools -> reasoning -> message
        self.tool_status = st.container()
        self.tool_results = st.container()
        self.reasoning = st.empty()
        self.message = st.empty()

class ResponseState:
    """
    State container for assistant response UI components and data.
    Shared by both Agent-based and Direct modes.
    """
    def __init__(self):
        # UI containers (grouped)
        self.containers = Containers()

        # Reasoning state
        self.reasoning_text = ""
        self.reasoning_placeholder = None

        # Tool state
        self.tool_status = None
        self.tool_results = []

        # Response content
        self.full_response = ""

    @property
    def has_reasoning(self):
        """Check if reasoning has been started."""
        return self.reasoning_placeholder is not None

    @property
    def tool_used(self):
        """Check if any tool has been used."""
        return self.tool_status is not None

    def update_reasoning(self, delta_text):
        """Add reasoning text and update display."""
        self.reasoning_text += delta_text

        # Create reasoning expander on first delta
        if not self.has_reasoning:
            with self.containers.reasoning.container():
                reasoning_expander = st.expander("🧠 Reasoning", expanded=True)
                self.reasoning_placeholder = reasoning_expander.empty()

        # Update reasoning text with cursor
        if self.reasoning_placeholder:
            self.reasoning_placeholder.markdown(self.reasoning_text + "▌")

    def finalize_reasoning(self):
        """Remove cursor from reasoning display."""
        if self.reasoning_placeholder and self.reasoning_text:
            self.reasoning_placeholder.markdown(self.reasoning_text)

    def update_message(self, delta_text):
        """Add message text and update display."""
        self.full_response += delta_text
        self.containers.message.markdown(self.full_response + "▌")

    def finalize_message(self):
        """Remove cursor from message display."""
        self.containers.message.markdown(self.full_response)


# ============================================================================
# Suggested Questions UI
# ============================================================================

def render_question_button(question, db_name, idx):
    """Render a single question button."""
    button_key = f"question_btn_{idx}_{hash(question)}"
    if st.button(
        question,
        key=button_key,
        use_container_width=True,
        help=f"From: {db_name}"
    ):
        st.session_state.selected_question = question
        st.rerun()


def render_question_grid(suggestions, num_to_show):
    """Render questions in a grid layout."""
    cols_per_row = 2
    for i in range(0, num_to_show, cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < num_to_show:
                question, db_name = suggestions[idx]
                with cols[j]:
                    render_question_button(question, db_name, idx)


def render_show_more_button(suggestions):
    """Render show more/less button if needed."""
    if len(suggestions) <= 4:
        return

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if st.session_state.show_more_questions:
            if st.button("Show Less", use_container_width=True):
                st.session_state.show_more_questions = False
                st.rerun()
        else:
            button_text = f"Show More ({len(suggestions) - 4} more)"
            if st.button(button_text, use_container_width=True):
                st.session_state.show_more_questions = True
                st.rerun()


def display_suggested_questions(selected_vector_dbs):
    """Display suggested questions based on selected databases."""
    if not selected_vector_dbs:
        return

    vector_dbs = list(llama_stack_api.client.vector_stores.list() or [])
    suggestions = get_suggestions_for_databases(selected_vector_dbs, vector_dbs)

    if not suggestions:
        return

    st.markdown("### 💡 Suggested Questions")

    # Determine how many questions to show
    num_to_show = (
        len(suggestions) if st.session_state.show_more_questions
        else min(4, len(suggestions))
    )

    # Display questions and controls
    render_question_grid(suggestions, num_to_show)
    render_show_more_button(suggestions)
    st.markdown("---")


# ============================================================================
# Main Prompt Processing
# ============================================================================

def process_prompt(prompt, config):
    """Process user prompt and generate response."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create assistant message context and setup shared containers once
    with st.chat_message("assistant"):
        state = ResponseState()

        # Call the appropriate mode-specific function
        if config.processing_mode == "Direct":
            direct_process_prompt(prompt, state, config)
        elif config.processing_mode == "Agent-based":
            agent_process_prompt(prompt, state, config)

    st.rerun()


def tool_chat_page():
    """Main chat page with RAG support in Direct and Agent-based modes."""
    st.title("💬 Chat")

    # Fetch models and tools
    model_list, builtin_tools_list, mcp_tools_list = fetch_models_and_tools()

    # Render sidebar and get configuration
    with st.sidebar:
        sidebar_config = render_sidebar_configuration(
            model_list, builtin_tools_list, mcp_tools_list
        )

    # Initialize session state
    initialize_session_state()

    # Create chat configuration object
    chat_config = ChatConfig(
        model=sidebar_config['model'],
        processing_mode=sidebar_config['processing_mode'],
        system_prompt=sidebar_config['system_prompt'],
        conversation_id=st.session_state["conversation_id"],
        toolgroup_selection=sidebar_config['toolgroup_selection'],
        selected_vector_dbs=sidebar_config['selected_vector_dbs'],
        sampling=SamplingParams(
            temperature=sidebar_config['temperature'],
            top_k=sidebar_config['top_k'],
            max_infer_iters=sidebar_config['max_infer_iters'],
            max_tokens=sidebar_config['max_tokens'],
        )
    )

    # Display chat history
    render_history()

    # Display suggested questions
    display_suggested_questions(chat_config.selected_vector_dbs)

    # Handle selected question from suggestions
    if st.session_state.selected_question:
        prompt = st.session_state.selected_question
        st.session_state.selected_question = None
        process_prompt(prompt, chat_config)

    # Handle manual chat input
    if prompt := st.chat_input(placeholder="Ask a question..."):
        process_prompt(prompt, chat_config)


tool_chat_page()
