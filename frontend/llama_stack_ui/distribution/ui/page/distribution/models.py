# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api


def models():
    """
    Inspect available models and display details for a selected one.
    """
    st.header("Models")
    try:
        with st.spinner("Loading models..."):
            model_list = llama_stack_api.client.models.list()
        if not model_list:
            st.info("No models available.")
            return
        models_info = {m.id: m.to_dict() for m in model_list}
        selected_model = st.selectbox("Select a model", list(models_info.keys()))
        st.json(models_info[selected_model], expanded=True)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
