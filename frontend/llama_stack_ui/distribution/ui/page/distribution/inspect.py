# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Inspect page: allows browsing and viewing details of Llama Stack resources.
"""
from streamlit_option_menu import option_menu
import streamlit as st

from llama_stack_ui.distribution.ui.page.distribution.datasets import datasets
from llama_stack_ui.distribution.ui.page.distribution.eval_tasks import benchmarks
from llama_stack_ui.distribution.ui.page.distribution.models import models
from llama_stack_ui.distribution.ui.page.distribution.scoring_functions import scoring_functions
from llama_stack_ui.distribution.ui.page.distribution.shields import shields
from llama_stack_ui.distribution.ui.page.distribution.providers import providers
from llama_stack_ui.distribution.ui.page.distribution.vector_dbs import vector_dbs

def inspect_page():
    """
    Display a horizontal menu to select a resource and show its details.
    """
    st.header("🔍 Inspect")
    options = [
        "API Providers",
        "Models",
        "Vector Databases",
        "Shields"
    ]
    icons = ["plug", "magic", "memory", "shield"]
    selected_resource = option_menu(
        None,
        options,
        icons=icons,
        orientation="horizontal",
        styles={
            "nav-link": {
                "font-size": "12px",
            },
        },
    )

    if selected_resource == "Vector Databases":
        vector_dbs()
    elif selected_resource == "Models":
        models()
    elif selected_resource == "Shields":
        shields()
    elif selected_resource == "API Providers":
        providers()


inspect_page()
