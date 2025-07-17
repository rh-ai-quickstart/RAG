# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import enum
import json
import uuid

import streamlit as st
from llama_stack_client import Agent, AgentEventLogger
from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.agents.react.tool_parser import ReActOutput
from llama_stack.apis.common.content_types import ToolCallDelta
from llama_stack.distribution.ui.modules.api import llama_stack_api


class AgentType(enum.Enum):
    REGULAR = "Regular"
    REACT = "ReAct"

def get_strategy(temperature, top_p):
    """Determines the sampling strategy for the LLM based on temperature."""
    return {'type': 'greedy'} if temperature == 0 else {
            'type': 'top_p', 'temperature': temperature, 'top_p': top_p
        }


def render_history(tool_debug):
    """Renders the chat history from session state.
    Also displays debug events for assistant messages if tool_debug is enabled.
    """
    # Initialize messages in session state if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    # Initialize debug_events in session state if not present
    if 'debug_events' not in st.session_state:
         st.session_state.debug_events = []

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

            # Display debug events expander for assistant messages (excluding the initial greeting)
            if msg['role'] == 'assistant' and tool_debug and i > 0:
                # Debug events are stored per assistant turn.
                # The index for debug_events corresponds to the assistant message turn.
                # messages: [A_initial, U_1, A_1, U_2, A_2, ...]
                # debug_events: [events_for_A_1, events_for_A_2, ...]
                # For A_1 (msg index 2), debug_events index is (2//2)-1 = 0.
                debug_event_list_index = (i // 2) - 1
                if 0 <= debug_event_list_index < len(st.session_state.debug_events):
                    current_turn_events_list = st.session_state.debug_events[debug_event_list_index]

                    if current_turn_events_list: # Only show expander if there are events
                        with st.expander("Tool/Debug Events", expanded=False):
                            if isinstance(current_turn_events_list, list) and len(current_turn_events_list) > 0:
                                for event_idx, event_item in enumerate(current_turn_events_list):
                                    with st.container():
                                        if isinstance(event_item, dict):
                                            st.json(event_item, expanded=False)
                                        elif isinstance(event_item, str):
                                            st.text_area(
                                                label=f"Debug Event {event_idx + 1}",
                                                value=event_item,
                                                height=100,
                                                disabled=True,
                                                key=f"debug_event_msg{i}_item{event_idx}" # Unique key for each text area
                                            )
                                        else:
                                            st.write(event_item) # Fallback for other data types
                                        if event_idx < len(current_turn_events_list) - 1:
                                            st.divider()
                            elif isinstance(current_turn_events_list, list) and not current_turn_events_list:
                                st.caption("No debug events recorded for this turn.")
                            else: # Should not happen with current logic
                                st.write("Debug data for this turn (unexpected format):")
                                st.write(current_turn_events_list)

def tool_chat_page():
    st.title("💬 Chat")

    client = llama_stack_api.client
    models = client.models.list()
    model_list = [model.identifier for model in models if model.api_model_type == "llm"]

    tool_groups = client.toolgroups.list()
    tool_groups_list = [tool_group.identifier for tool_group in tool_groups]
    mcp_tools_list = [tool for tool in tool_groups_list if tool.startswith("mcp::")]
    builtin_tools_list = [tool for tool in tool_groups_list if not tool.startswith("mcp::")]

    selected_vector_dbs = []

    def reset_agent():
        st.session_state.clear()
        st.cache_resource.clear()

    with st.sidebar:
        st.title("Configuration")
        st.subheader("Model")
        model = st.selectbox(label="Model", options=model_list, on_change=reset_agent, label_visibility="collapsed")

        ## Added mode 
        processing_mode = st.radio(
            "Processing mode",
            ["Direct", "Agent-based"],
            index=0, # Default to Direct
            captions=[
                "Directly calls the model with optional RAG.",
                "Uses an Agent (Regular or ReAct) with tools.",
            ],
            on_change=reset_agent,
            help="Choose how requests are processed. 'Direct' bypasses agents, 'Agent-based' uses them.",
        )

        
        toolgroup_selection = []
        if processing_mode == "Direct":
            vector_dbs = llama_stack_api.client.vector_dbs.list() or []
            if not vector_dbs:
                st.info("No vector databases available for selection.")
            vector_dbs = [vector_db.identifier for vector_db in vector_dbs]
            selected_vector_dbs = st.multiselect(
                label="Select Document Collections to use in RAG queries",
                options=vector_dbs,
                on_change=reset_agent,
            )
        if processing_mode == "Agent-based":
            st.subheader("Available ToolGroups")

            toolgroup_selection = st.pills(
                label="Built-in tools",
                options=builtin_tools_list,
                selection_mode="multi",
                on_change=reset_agent,
                format_func=lambda tool: "".join(tool.split("::")[1:]),
                help="List of built-in tools from your llama stack server.",
            )

            if "builtin::rag" in toolgroup_selection:
                vector_dbs = llama_stack_api.client.vector_dbs.list() or []
                if not vector_dbs:
                    st.info("No vector databases available for selection.")
                vector_dbs = [vector_db.identifier for vector_db in vector_dbs]
                selected_vector_dbs = st.multiselect(
                    label="Select Document Collections to use in RAG queries",
                    options=vector_dbs,
                    on_change=reset_agent,
                )

            # Display mcp list only if there are mcp tools
            if len(mcp_tools_list) > 0:
                mcp_selection = st.pills(
                    label="MCP Servers",
                    options=mcp_tools_list,
                    selection_mode="multi",
                    on_change=reset_agent,
                    format_func=lambda tool: "".join(tool.split("::")[1:]),
                    help="List of MCP servers registered to your llama stack server.",
                )

                toolgroup_selection.extend(mcp_selection)

            grouped_tools = {}
            total_tools = 0

            for toolgroup_id in toolgroup_selection:
                tools = client.tools.list(toolgroup_id=toolgroup_id)
                grouped_tools[toolgroup_id] = [tool.identifier for tool in tools]
                total_tools += len(tools)

            st.markdown(f"Active Tools: 🛠 {total_tools}")

            for group_id, tools in grouped_tools.items():
                with st.expander(f"🔧 Tools from `{group_id}`"):
                    for idx, tool in enumerate(tools, start=1):
                        st.markdown(f"{idx}. `{tool.split(':')[-1]}`")

            # st.subheader("Agent Configurations")
            # st.subheader("Agent Type")
            # agent_type = st.radio(
            #     label="Select Agent Type",
            #     options=["Regular", "ReAct"],
            #     on_change=reset_agent,
            # )

            # if agent_type == "ReAct":
            #     agent_type = AgentType.REACT
            # else:
            #     agent_type = AgentType.REGULAR

            agent_type = AgentType.REGULAR
        
        if processing_mode == "Agent-based":
            input_shields = []
            output_shields = []

            st.subheader("Security Shields")
            shields_available = client.shields.list()
            shield_options = [s.identifier for s in shields_available if hasattr(s, 'identifier')]
            input_shields = st.multiselect("Input Shields", options=shield_options, on_change=reset_agent)
            output_shields = st.multiselect("Output Shields", options=shield_options, on_change=reset_agent)

        st.subheader("Sampling Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.05, on_change=reset_agent)
        top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.05, on_change=reset_agent)
        max_tokens = st.slider("Max Tokens", 1, 4096, 512, 64, on_change=reset_agent)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.0, 0.05, on_change=reset_agent)

        st.subheader("System Prompt")
        default_prompt = "You are a helpful AI assistant."
        if processing_mode == "Agent-based" and agent_type == AgentType.REACT:
            default_prompt = "You are a helpful ReAct agent. Reason step-by-step to fulfill the user query using available tools."
        system_prompt = st.text_area(
            "System Prompt", value=default_prompt, on_change=reset_agent, height=100
        )

        st.subheader("Response Handling")
        #stream_opt = st.toggle("Stream Response", value=True, on_change=reset_agent)
        tool_debug = st.toggle("Show Tool/Debug Info", value=False)

        if st.button("Clear Chat & Reset Config", use_container_width=True):
            reset_agent()
            st.rerun()
    

    updated_toolgroup_selection = []
    if processing_mode == "Agent-based":
        for i, tool_name in enumerate(toolgroup_selection):
            if tool_name == "builtin::rag":
                if len(selected_vector_dbs) > 0:
                    tool_dict = dict(
                        name="builtin::rag",
                        args={
                            "vector_db_ids": list(selected_vector_dbs),
                        },
                    )
                    updated_toolgroup_selection.append(tool_dict)
            else:
                updated_toolgroup_selection.append(tool_name)

    @st.cache_resource
    def create_agent():
        if "agent_type" in st.session_state and st.session_state.agent_type == AgentType.REACT:
            return ReActAgent(
                client=client,
                model=model,
                tools=updated_toolgroup_selection,
                response_format={
                    "type": "json_schema",
                    "json_schema": ReActOutput.model_json_schema(),
                },
                sampling_params={
                    "strategy": get_strategy(temperature, top_p),
                    "max_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                },
                input_shields= input_shields,
                output_shields= output_shields,
            )
        else:
            updated_system_prompt = system_prompt.strip()
            updated_system_prompt = updated_system_prompt if updated_system_prompt.strip().endswith('.') else updated_system_prompt + '.'
            return Agent(
                client,
                model=model,
                instructions=f"{updated_system_prompt} When you use a tool always respond with a summary of the result.",
                tools=updated_toolgroup_selection,
                sampling_params={
                    "strategy": get_strategy(temperature, top_p),
                    "max_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                },
                input_shields= input_shields,
                output_shields= output_shields,
            )

    if processing_mode == "Agent-based":
        st.session_state.agent_type = agent_type
        agent = create_agent()

        if "agent_session_id" not in st.session_state:
            st.session_state["agent_session_id"] = agent.create_session(session_name=f"tool_demo_{uuid.uuid4()}")

        session_id = st.session_state["agent_session_id"]

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?", "stop_reason": "end_of_turn"}]
    
    if "debug_events" not in st.session_state: # Per-turn debug logs
        st.session_state["debug_events"] = []

    render_history(tool_debug) # Display current chat history and any past debug events

    def response_generator(turn_response, debug_events_list):
        if st.session_state.get("agent_type") == AgentType.REACT:
            return _handle_react_response(turn_response)
        else:
            return _handle_regular_response(turn_response, debug_events_list)

    def _handle_react_response(turn_response):
        current_step_content = ""
        final_answer = None
        tool_results = []

        for response in turn_response:
            if not hasattr(response.event, "payload"):
                yield (
                    "\n\n🚨 :red[_Llama Stack server Error:_]\n"
                    "The response received is missing an expected `payload` attribute.\n"
                    "This could indicate a malformed response or an internal issue within the server.\n\n"
                    f"Error details: {response}"
                )
                return

            payload = response.event.payload

            if payload.event_type == "step_progress" and hasattr(payload.delta, "text"):
                current_step_content += payload.delta.text
                continue

            if payload.event_type == "step_complete":
                step_details = payload.step_details

                if step_details.step_type == "inference":
                    yield from _process_inference_step(current_step_content, tool_results, final_answer)
                    current_step_content = ""
                elif step_details.step_type == "tool_execution":
                    tool_results = _process_tool_execution(step_details, tool_results)
                    current_step_content = ""
                else:
                    current_step_content = ""

        if not final_answer and tool_results:
            yield from _format_tool_results_summary(tool_results)

    def _process_inference_step(current_step_content, tool_results, final_answer):
        try:
            react_output_data = json.loads(current_step_content)
            thought = react_output_data.get("thought")
            action = react_output_data.get("action")
            answer = react_output_data.get("answer")

            if answer and answer != "null" and answer is not None:
                final_answer = answer

            if thought:
                with st.expander("🤔 Thinking...", expanded=False):
                    st.markdown(f":grey[__{thought}__]")

            if action and isinstance(action, dict):
                tool_name = action.get("tool_name")
                tool_params = action.get("tool_params")
                with st.expander(f'🛠 Action: Using tool "{tool_name}"', expanded=False):
                    st.json(tool_params)

            if answer and answer != "null" and answer is not None:
                yield f"\n\n✅ **Final Answer:**\n{answer}"

        except json.JSONDecodeError:
            yield f"\n\nFailed to parse ReAct step content:\n```json\n{current_step_content}\n```"
        except Exception as e:
            yield f"\n\nFailed to process ReAct step: {e}\n```json\n{current_step_content}\n```"

        return final_answer

    def _process_tool_execution(step_details, tool_results):
        try:
            if hasattr(step_details, "tool_responses") and step_details.tool_responses:
                for tool_response in step_details.tool_responses:
                    tool_name = tool_response.tool_name
                    content = tool_response.content
                    tool_results.append((tool_name, content))
                    with st.expander(f'⚙️ Observation (Result from "{tool_name}")', expanded=False):
                        try:
                            parsed_content = json.loads(content)
                            st.json(parsed_content)
                        except json.JSONDecodeError:
                            st.code(content, language=None)
            else:
                with st.expander("⚙️ Observation", expanded=False):
                    st.markdown(":grey[_Tool execution step completed, but no response data found._]")
        except Exception as e:
            with st.expander("⚙️ Error in Tool Execution", expanded=False):
                st.markdown(f":red[_Error processing tool execution: {str(e)}_]")

        return tool_results

    def _format_tool_results_summary(tool_results):
        yield "\n\n**Here's what I found:**\n"
        for tool_name, content in tool_results:
            try:
                parsed_content = json.loads(content)

                if tool_name == "web_search" and "top_k" in parsed_content:
                    yield from _format_web_search_results(parsed_content)
                elif "results" in parsed_content and isinstance(parsed_content["results"], list):
                    yield from _format_results_list(parsed_content["results"])
                elif isinstance(parsed_content, dict) and len(parsed_content) > 0:
                    yield from _format_dict_results(parsed_content)
                elif isinstance(parsed_content, list) and len(parsed_content) > 0:
                    yield from _format_list_results(parsed_content)
            except json.JSONDecodeError:
                yield f"\n**{tool_name}** was used but returned complex data. Check the observation for details.\n"
            except (TypeError, AttributeError, KeyError, IndexError) as e:
                print(f"Error processing {tool_name} result: {type(e).__name__}: {e}")

    def _format_web_search_results(parsed_content):
        for i, result in enumerate(parsed_content["top_k"], 1):
            if i <= 3:
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                content_text = result.get("content", "").strip()
                yield f"\n- **{title}**\n  {content_text}\n  [Source]({url})\n"

    def _format_results_list(results):
        for i, result in enumerate(results, 1):
            if i <= 3:
                if isinstance(result, dict):
                    name = result.get("name", result.get("title", "Result " + str(i)))
                    description = result.get("description", result.get("content", result.get("summary", "")))
                    yield f"\n- **{name}**\n  {description}\n"
                else:
                    yield f"\n- {result}\n"

    def _format_dict_results(parsed_content):
        yield "\n```\n"
        for key, value in list(parsed_content.items())[:5]:
            if isinstance(value, str) and len(value) < 100:
                yield f"{key}: {value}\n"
            else:
                yield f"{key}: [Complex data]\n"
        yield "```\n"

    def _format_list_results(parsed_content):
        yield "\n"
        for _, item in enumerate(parsed_content[:3], 1):
            if isinstance(item, str):
                yield f"- {item}\n"
            elif isinstance(item, dict) and "text" in item:
                yield f"- {item['text']}\n"
            elif isinstance(item, dict) and len(item) > 0:
                first_value = next(iter(item.values()))
                if isinstance(first_value, str) and len(first_value) < 100:
                    yield f"- {first_value}\n"

    def _handle_regular_response(turn_response, debug_events_list):

        # Use itertools.tee to duplicate the stream for UI and debug logging
        # This is crucial because a generator can only be consumed once.
        from itertools import tee
        ui_stream, debug_log_stream = tee(turn_response, 2)

        for response in ui_stream:
            if hasattr(response.event, "payload"):
                if response.event.payload.event_type == "step_progress":
                    if hasattr(response.event.payload.delta, "text"):
                        yield response.event.payload.delta.text
                if response.event.payload.event_type == "step_complete":
                    if response.event.payload.step_details.step_type == "tool_execution":
                        if response.event.payload.step_details.tool_calls:
                            tool_name = str(response.event.payload.step_details.tool_calls[0].tool_name)
                            yield f'\n\n🛠 :grey[_Using "{tool_name}" tool:_]\n\n'
                        else:
                            yield "No tool_calls present in step_details"
                    if response.event.payload.step_details.step_type == "shield_call":
                        if response.event.payload.step_details.violation:
                            yield response.event.payload.step_details.violation.user_message
            else:
                yield f"Error occurred in the Llama Stack Cluster: {response}"
                debug_events_list.append({"type": "warning", "source": "_handle_regular_response", "details": "Unexpected event structure", "event": str(response)[:200]})

        # Process the debug log stream separately
        # AgentEventLogger helps parse and structure these events
        for log_entry in AgentEventLogger().log(debug_log_stream):
            if log_entry.role == "tool_execution": # Or other relevant roles
                debug_events_list.append({"type": "tool_log", "content": log_entry.content})
            # Add other log types as needed for debugging

    def agent_process_prompt(prompt, debug_events_list):
        # Send the prompt to the agent
        turn_response = agent.create_turn(
            session_id=session_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        response_content = st.write_stream(response_generator(turn_response, debug_events_list))
        st.session_state.messages.append({"role": "assistant", "content": response_content})


    def direct_process_prompt(prompt, debug_events_list):
        # Query the vector DB
        if selected_vector_dbs:
            with st.spinner("Retrieving context (RAG)..."):
                try:
                    rag_response = llama_stack_api.client.tool_runtime.rag_tool.query(
                        content=prompt, vector_db_ids=list(selected_vector_dbs) 
                    )
                    prompt_context = rag_response.content
                    debug_events_list.append({
                        "type": "rag_query_direct_mode", "query": prompt,
                        "vector_dbs": selected_vector_dbs,
                        "context_length": len(prompt_context) if prompt_context else 0,
                        "context_preview": (str(prompt_context[:200]) + "..." if prompt_context else "None")
                    })
                except Exception as e:
                    st.warning(f"RAG Error (Direct Mode): {e}")
                    debug_events_list.append({"type": "error", "source": "rag_direct_mode", "content": str(e)})
        else:
            prompt_context = None
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            retrieval_response = ""

            # Construct the extended prompt
            if prompt_context:
                extended_prompt = f"Please answer the following query using the context below.\n\nCONTEXT:\n{prompt_context}\n\nQUERY:\n{prompt}"
            else:
                extended_prompt = f"Please answer the following query. \n\nQUERY:\n{prompt}"

            # Run inference directly
            #st.session_state.messages.append({"role": "user", "content": extended_prompt})
            messages_for_direct_api = (
                [{'role': 'system', 'content': system_prompt}] +
                [{'role': 'user', 'content': extended_prompt}]
            )
            response = llama_stack_api.client.inference.chat_completion(
                messages=messages_for_direct_api,
                model_id=model,
                sampling_params={
                    "strategy": get_strategy(temperature, top_p),
                    "max_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                },
                stream=True,
            )

            # Display assistant response
            for chunk in response:
                if chunk.event:
                    response_delta = chunk.event.delta
                    if isinstance(response_delta, ToolCallDelta):
                        retrieval_response += response_delta.tool_call.replace("====", "").strip()
                        #retrieval_message_placeholder.info(retrieval_response)
                    else:
                        full_response += chunk.event.delta.text
                        message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        response_dict = {"role": "assistant", "content": full_response, "stop_reason": "end_of_message"}
        st.session_state.messages.append(response_dict)
        #st.session_state.displayed_messages.append(response_dict)

    if prompt := st.chat_input(placeholder="Ask a question..."):
        # Append user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Prepare for assistant's response
        # Each assistant turn gets its own list for debug events
        st.session_state.debug_events.append([])
        current_turn_debug_events_list = st.session_state.debug_events[-1] # Get the list for this turn

        st.session_state.prompt = prompt
        if processing_mode == "Agent-based":
            agent_process_prompt(st.session_state.prompt, current_turn_debug_events_list)
        else:  # rag_mode == "Direct"
            direct_process_prompt(st.session_state.prompt, current_turn_debug_events_list)
        #st.session_state.prompt = None
        st.rerun()

tool_chat_page()
