# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api


def shields():
    """
    Inspect available shields and display details for a selected one.
    """
    st.header("Shields")
    # Retrieve all shields
    shields_list = llama_stack_api.client.shields.list()
    if not shields_list:
        st.info("No shields available.")
        return
    shields_info = {s.identifier: s.to_dict() for s in shields_list}

    # Let user select and view shield details
    selected_shield = st.selectbox("Select a shield", list(shields_info.keys()))
    st.json(shields_info[selected_shield])
