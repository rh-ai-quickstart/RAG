# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import streamlit as st

def main():
    # Define available pages: path and icon
    pages = {
        "Chat": ("page/playground/chat.py", "💬"),
        "Upload Documents": ("page/upload/upload.py", "📄"),
        "Inspect": ("page/distribution/inspect.py", "🔍"),
    }

    # Build navigation items dynamically
    nav_items = [
        st.Page(path, title=name, icon=icon, default=(name == "Chat"))
        for name, (path, icon) in pages.items()
    ]
    # Render navigation
    pg = st.navigation({"Playground": nav_items}, expanded=False)
    pg.run()


if __name__ == "__main__":
    main()

