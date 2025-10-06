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
    # Retrieve all providers
    providers_list = llama_stack_api.client.providers.list()
    if not providers_list:
        st.info("No API providers registered.")
        return

    # Group providers by API name
    api_to_providers: dict[str, list] = {}
    for p in providers_list:
        api_to_providers.setdefault(p.api, []).append(p)

    # Display each group with its providers
    for api_name, providers in api_to_providers.items():
        st.markdown(f"###### {api_name}")
        st.dataframe([p.to_dict() for p in providers], width=500)



