# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import pandas as pd
import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api
from llama_stack_ui.distribution.ui.modules.utils import process_dataset


"""
Application Evaluation page: upload a dataset, select scoring functions, and run scoring.
"""


def application_evaluation_page():
    """
    1) Upload and preview a CSV/XLS dataset
    2) Choose scoring functions and set parameters
    3) Run evaluation on selected rows and display results
    """

    # Step 1: File uploader
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.error("No file uploaded")  # show error if no file
        return

    # Process uploaded file
    df = process_dataset(uploaded_file)
    if df is None:
        st.error("Error processing file")  # abort on processing error
        return

    # Display dataset information
    st.success("Dataset loaded successfully!")

    # Preview the loaded DataFrame
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Step 2: select scoring functions
    st.subheader("Select Scoring Functions")
    scoring_functions = llama_stack_api.client.scoring_functions.list()
    if not scoring_functions:
        st.info("No scoring functions available.")
        return
    scoring_functions = {sf.identifier: sf for sf in scoring_functions}
    scoring_functions_names = list(scoring_functions.keys())
    selected_scoring_functions = st.multiselect(
        "Choose one or more scoring functions",
        options=scoring_functions_names,
        help="Choose one or more scoring functions.",
    )

    available_models = llama_stack_api.client.models.list()
    available_models = [m.identifier for m in available_models]

    scoring_params = {}
    if selected_scoring_functions:
        st.write("Selected:")
        for scoring_fn_id in selected_scoring_functions:
            scoring_fn = scoring_functions[scoring_fn_id]
            st.write(f"- **{scoring_fn_id}**: {scoring_fn.description}")
            new_params = None
            if scoring_fn.params:
                new_params = {}
                for param_name, param_value in scoring_fn.params.to_dict().items():
                    if param_name == "type":
                        new_params[param_name] = param_value
                        continue

                    if param_name == "judge_model":
                        value = st.selectbox(
                            f"Select **{param_name}** for {scoring_fn_id}",
                            options=available_models,
                            index=0,
                            key=f"{scoring_fn_id}_{param_name}",
                        )
                        new_params[param_name] = value
                    else:
                        value = st.text_area(
                            f"Enter value for **{param_name}** in {scoring_fn_id} in valid JSON format",
                            value=json.dumps(param_value, indent=2),
                            height=80,
                        )
                        try:
                            new_params[param_name] = json.loads(value)
                        except json.JSONDecodeError:
                            st.error(f"Invalid JSON for **{param_name}** in {scoring_fn_id}")

                st.json(new_params)
            scoring_params[scoring_fn_id] = new_params

        # Step 3: configure and run evaluation
        total_rows = len(df)
        num_rows = st.slider("Number of rows to evaluate", 1, total_rows, total_rows)

        if st.button("Run Evaluation"):
            progress_text = "Running evaluation..."
            progress_bar = st.progress(0, text=progress_text)
            rows = df.to_dict(orient="records")
            if num_rows < total_rows:
                rows = rows[:num_rows]

            # Containers for progress and live results
            progress_text_container = st.empty()
            results_container = st.empty()
            output_res = {}
            for i, r in enumerate(rows):
                # Update progress
                progress = i / len(rows)
                progress_bar.progress(progress, text=progress_text)

                # Run scoring for current row
                score_res = llama_stack_api.run_scoring(
                    r,
                    scoring_function_ids=selected_scoring_functions,
                    scoring_params=scoring_params,
                )

                # Aggregate inputs and scores
                for k in r.keys():
                    output_res.setdefault(k, []).append(r[k])

                for fn_id in selected_scoring_functions:
                    output_res.setdefault(fn_id, []).append(
                        score_res.results[fn_id].score_rows[0]
                    )

                # Display each processed row
                progress_text_container.write(
                    f"Processed row {i + 1} / {len(rows)}"
                )
                results_container.json(
                    score_res.to_json(),
                    expanded=2,
                )

            progress_bar.progress(1.0, text="Evaluation complete!")

            # Show final DataFrame of results
            if output_res:
                output_df = pd.DataFrame(output_res)
                st.subheader("Evaluation Results")
                st.dataframe(output_df)


application_evaluation_page()
