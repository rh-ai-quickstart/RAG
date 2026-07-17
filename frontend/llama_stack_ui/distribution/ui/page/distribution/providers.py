# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api


def providers():
    """
    Inspect available API providers by API type and display details.
    """
    st.header("API Providers")
    try:
        with st.spinner("Loading providers..."):
            providers_list = llama_stack_api.client.providers.list()
        if not providers_list:
            st.info("No API providers registered.")
            return

        api_to_providers: dict[str, list] = {}
        for p in providers_list:
            api_to_providers.setdefault(p.api, []).append(p)

        for api_name, api_providers in api_to_providers.items():
            st.markdown(f"###### {api_name}")
            st.dataframe([p.to_dict() for p in api_providers], use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load providers: {e}")
