import json
import pandas as pd
import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api

"""
Native Evaluation page: select a benchmark, configure eval candidate, and run full generation + scoring.
"""

def select_benchmark_1():
    """
    Step 1: Let user choose a registered benchmark (eval task).
    Show error if none available.
    """
    # 1. Choose an Eval Task
    st.subheader("1. Choose An Eval Task")

    raw_benchmarks = llama_stack_api.client.benchmarks.list()
    if not raw_benchmarks:
        st.error("No benchmarks available. Please register a benchmark first.")
        return

    benchmarks = {et.identifier: et for et in raw_benchmarks}
    benchmark_names = list(benchmarks.keys())
    selected_benchmark = st.selectbox(
        "Choose an eval task.",
        options=benchmark_names,
        help="Each eval task is parameterized by a dataset and scoring functions.",
    )
    with st.expander("View Eval Task"):
        st.json(benchmarks[selected_benchmark], expanded=True)

    st.session_state["selected_benchmark"] = selected_benchmark
    st.session_state["benchmarks"] = benchmarks
    if st.button("Confirm", key="confirm_1"):
        st.session_state["selected_benchmark_1_next"] = True


def define_eval_candidate_2():
    """
    Step 2: After benchmark selection, define generation candidate (model or agent).
    Store configuration in session state.
    """
    # 2. Define Eval Candidate
    if not st.session_state.get("selected_benchmark_1_next"):
        return

    st.subheader("2. Define Eval Candidate")
    st.info(
        "Define generation configuration: choose 'model' for inference API or 'agent' for agent API."
    )

    with st.expander("Define Eval Candidate", expanded=True):
        candidate_type = st.radio("Candidate Type", ["model", "agent"])
        # fetch models
        raw_models = llama_stack_api.client.models.list()
        model_ids = [m.identifier for m in raw_models] or []
        if not model_ids:
            st.error("No models available in Llama Stack API client.")
            return

        selected_model = st.selectbox("Choose a model", model_ids)

        # sampling params
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.05)
        max_tokens = st.slider("Max Tokens", 0, 4096, 512, 1)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.0, 0.1)

        if candidate_type == "model":
            strategy = (
                {"type": "greedy"}
                if temperature == 0.0
                else {"type": "top_p", "temperature": temperature, "top_p": top_p}
            )
            eval_candidate = {
                "type": "model",
                "model": selected_model,
                "sampling_params": {
                    "strategy": strategy,
                    "max_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                },
            }
        else:  # agent
            system_prompt = st.text_area(
                "System Prompt",
                value="You are a helpful AI assistant.",
            )
            tools_json = st.text_area(
                "Tools Configuration (JSON)",
                value=json.dumps(
                    [
                        {
                            "type": "brave_search",
                            "engine": "brave",
                            "api_key": "ENTER_BRAVE_API_KEY_HERE",
                        }
                    ],
                    indent=2,
                ),
                height=200,
            )
            try:
                tools = json.loads(tools_json)
            except json.JSONDecodeError:
                st.error("Invalid JSON for tools configuration")
                tools = []

            eval_candidate = {
                "type": "agent",
                "config": {
                    "model": selected_model,
                    "instructions": system_prompt,
                    "tools": tools,
                    "tool_choice": "auto",
                    "tool_prompt_format": "json",
                    "input_shields": [],
                    "output_shields": [],
                    "enable_session_persistence": False,
                },
            }

        st.session_state["eval_candidate"] = eval_candidate

    if st.button("Confirm", key="confirm_2"):
        st.session_state["selected_eval_candidate_2_next"] = True


def run_evaluation_3():
    """
    Step 3: Run evaluation across dataset rows using the selected candidate.
    Display progress and results in a DataFrame.
    """
    # 3. Run Evaluation
    if not st.session_state.get("selected_eval_candidate_2_next"):
        return

    st.subheader("3. Run Evaluation")
    st.info("Review configurations and click 'Run Evaluation' to start.")

    benchmarks = st.session_state.get("benchmarks", {})
    selected_benchmark = st.session_state.get("selected_benchmark")
    if not benchmarks or not selected_benchmark:
        st.error("Missing benchmark selection. Please complete step 1.")
        return

    eval_candidate = st.session_state.get("eval_candidate")
    if not eval_candidate:
        st.error("Missing eval candidate. Please complete step 2.")
        return

    dataset_id = benchmarks[selected_benchmark].dataset_id
    rows_iter = llama_stack_api.client.datasets.iterrows(dataset_id=dataset_id)
    data_rows = getattr(rows_iter, "data", [])
    if not data_rows:
        st.error(f"No data found for dataset '{dataset_id}'.")
        return

    total = len(data_rows)
    num_rows = st.number_input(
        "Number of Examples to Evaluate",
        min_value=1,
        max_value=total,
        value=min(5, total),
    )

    benchmark_config = {"type": "benchmark", "eval_candidate": eval_candidate, "scoring_params": {}}

    with st.expander("View Benchmark", expanded=True):
        st.json(benchmarks[selected_benchmark], expanded=True)
    with st.expander("View Config", expanded=True):
        st.json(benchmark_config, expanded=True)

    if st.button("Run Evaluation"):
        progress_bar = st.progress(0)
        text_cont = st.empty()
        res_cont = st.empty()
        output = {}

        for i, row in enumerate(data_rows[:num_rows]):
            progress = (i + 1) / num_rows
            progress_bar.progress(progress)
            text_cont.write(f"Processing row {i + 1}/{num_rows}")

            eval_res = llama_stack_api.client.eval.evaluate_rows(
                benchmark_id=selected_benchmark,
                input_rows=[row],
                scoring_functions=benchmarks[selected_benchmark].scoring_functions,
                benchmark_config=benchmark_config,
            )

            # accumulate inputs
            for k, v in row.items():
                output.setdefault(k, []).append(v)
            # accumulate generations & scores
            for gen_key, gen_val in eval_res.generations[0].items():
                output.setdefault(gen_key, []).append(gen_val)
            for fn in benchmarks[selected_benchmark].scoring_functions:
                score = eval_res.scores[fn].score_rows[0]
                output.setdefault(fn, []).append(score)

            res_cont.json(eval_res, expanded=2)

        st.success("Evaluation complete!")
        df_out = pd.DataFrame(output)
        st.subheader("Evaluation Results")
        st.dataframe(df_out)


def native_evaluation_page():
    # Compose the three steps in order
    select_benchmark_1()
    define_eval_candidate_2()
    run_evaluation_3()


native_evaluation_page()