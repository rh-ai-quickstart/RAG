# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api


def scoring_functions():
    """
    Inspect available scoring functions and display details for a selected one.
    """
    st.header("Scoring Functions")
    # Retrieve scoring functions
    sf_list = llama_stack_api.client.scoring_functions.list()
    if not sf_list:
        st.info("No scoring functions found.")
        return
    scoring_functions_info = {s.identifier: s.to_dict() for s in sf_list}

    # Let user select and view a scoring function
    selected_scoring_function = st.selectbox("Select a scoring function", list(scoring_functions_info.keys()))
    st.json(scoring_functions_info[selected_scoring_function], expanded=True)
