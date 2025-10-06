# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api


def datasets():
    """
    Inspect available datasets and display details for the selected one.
    """
    st.header("Datasets")

    # Fetch all datasets
    dataset_list = llama_stack_api.client.datasets.list()
    if not dataset_list:
        st.info("No datasets found.")
        return
    datasets_info = {d.identifier: d.to_dict() for d in dataset_list}

    # Let user select and view a dataset
    selected_dataset = st.selectbox("Select a dataset", list(datasets_info.keys()))
    st.json(datasets_info[selected_dataset], expanded=True)
