# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api


def vector_dbs():
    """
    Inspect available vector databases and display details for the selected one.
    """
    st.header("Vector Databases")
    # Fetch all vector databases
    vdb_list = llama_stack_api.client.vector_dbs.list()
    if not vdb_list:
        st.info("No vector databases found.")
        return
    # Build info dict and allow selection
    vdb_info = {v.identifier: v.to_dict() for v in vdb_list}
    selected_vector_db = st.selectbox("Select a vector database", list(vdb_info.keys()))
    st.json(vdb_info[selected_vector_db], expanded=True)
