# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Upload documents page for managing vector databases and document ingestion."""

import traceback

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api
from llama_stack_ui.distribution.ui.modules.utils import get_vector_db_name


def _init_upload_page_session_state():
    """Initialize all session state variables needed by the upload page."""
    defaults = {
        "creation_status": None,
        "creation_message": "",
        "selected_vector_db": "",
        "newly_created_vdb": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "vector_db_selector" not in st.session_state:
        st.session_state["vector_db_selector"] = st.session_state["selected_vector_db"]


def _show_status(status_key, message_key):
    """Show and clear a status message from session state."""
    status = st.session_state[status_key]
    if status == "success":
        st.success(st.session_state[message_key])
    elif status == "error":
        st.error(st.session_state[message_key])
    else:
        return
    st.session_state[status_key] = None
    st.session_state[message_key] = ""


def _build_dropdown_options(vdb_list):
    """Build dropdown options from the vector database list.

    Returns:
        tuple: (create_new_option, dropdown_options)
    """
    create_new_option = "➕ Create New"

    if vdb_list:
        existing_vdbs = [get_vector_db_name(v) for v in vdb_list]
        return [create_new_option] + existing_vdbs, create_new_option

    return [create_new_option], create_new_option


def _sync_vector_db_selection(dropdown_options, vdb_list):
    """Sync the vector database selection state with available options."""
    # Priority 1: Auto-select a newly created database
    newly_created = st.session_state["newly_created_vdb"]
    if newly_created and newly_created in dropdown_options:
        st.session_state["selected_vector_db"] = newly_created
        st.session_state["vector_db_selector"] = newly_created
        st.session_state["newly_created_vdb"] = None
        return

    # Priority 2: Keep the previously selected database if it still exists
    selected = st.session_state["selected_vector_db"]
    if selected and selected in dropdown_options:
        st.session_state["vector_db_selector"] = selected
        return

    # Priority 3: Smart default
    if vdb_list:
        first_db = dropdown_options[0]
        st.session_state["selected_vector_db"] = first_db
        st.session_state["vector_db_selector"] = first_db
    else:
        st.session_state["selected_vector_db"] = dropdown_options[0]
        st.session_state["vector_db_selector"] = dropdown_options[0]


def upload_page():
    """Page to upload documents and manage vector databases for RAG."""
    st.title("📄 Upload Documents")

    _init_upload_page_session_state()
    _show_status("creation_status", "creation_message")

    vdb_list = llama_stack_api.client.vector_stores.list()
    dropdown_options, create_new_option = _build_dropdown_options(vdb_list)
    _sync_vector_db_selection(dropdown_options, vdb_list)

    def on_vector_db_change():
        st.session_state["selected_vector_db"] = st.session_state["vector_db_selector"]

    selected_vector_db = st.selectbox(
        "Select a vector database",
        dropdown_options,
        key="vector_db_selector",
        on_change=on_vector_db_change,
        help="Your selection will be remembered when you navigate to other pages"
    )

    if selected_vector_db != st.session_state["selected_vector_db"]:
        st.session_state["selected_vector_db"] = selected_vector_db

    if selected_vector_db == create_new_option:
        _show_create_vector_db_ui()
    elif selected_vector_db:
        selected_vdb_obj = None
        for vdb in vdb_list:
            if get_vector_db_name(vdb) == selected_vector_db:
                selected_vdb_obj = vdb
                break

        _show_existing_documents_table(selected_vector_db, selected_vdb_obj)
        st.subheader(f"📁 Upload Documents to '{selected_vector_db}'")
        _show_document_upload_ui(selected_vector_db, selected_vdb_obj)


def _show_create_vector_db_ui():
    """Display UI for creating a new vector database."""
    st.subheader("Create New Vector Database")

    if "new_vdb_name" not in st.session_state:
        st.session_state["new_vdb_name"] = ""

    new_vdb_name = st.text_input(
        "Add New Vector Database",
        value=st.session_state["new_vdb_name"],
        help="Enter a unique name for the new vector database",
        key="new_vdb_name_input"
    )

    st.session_state["new_vdb_name"] = new_vdb_name

    if st.button("Add", type="primary", disabled=not new_vdb_name.strip()):
        _create_vector_database(new_vdb_name.strip())


def _create_vector_database(vdb_name):
    """Create a new vector database using the LlamaStack API.

    Args:
        vdb_name (str): Name for the new vector database
    """
    try:
        st.session_state["creation_status"] = None
        st.session_state["creation_message"] = ""

        if not vdb_name or not vdb_name.strip():
            st.session_state["creation_status"] = "error"
            st.session_state["creation_message"] = "Vector database name cannot be empty."
            return

        existing_vdbs = llama_stack_api.client.vector_stores.list()
        existing_names = [get_vector_db_name(vdb) for vdb in existing_vdbs]
        if vdb_name in existing_names:
            st.session_state["creation_status"] = "error"
            st.session_state["creation_message"] = (
                f"Vector database '{vdb_name}' already exists. "
                "Please choose a different name."
            )
            return

        with st.spinner(f"Creating vector database '{vdb_name}'..."):
            _vector_db = llama_stack_api.client.vector_stores.create(
                name=vdb_name,
            )

        st.session_state["creation_status"] = "success"
        st.session_state["creation_message"] = (
            f"Vector database '{vdb_name}' created successfully!"
        )
        st.session_state["newly_created_vdb"] = vdb_name
        st.session_state["new_vdb_name"] = ""
        st.rerun()

    except Exception as e:  # pylint: disable=broad-exception-caught
        st.session_state["creation_status"] = "error"
        st.session_state["creation_message"] = f"Error creating vector database: {str(e)}"


def _show_document_upload_ui(vector_db_name, vector_db_obj=None):
    """Display UI for uploading documents to an existing vector database.

    Args:
        vector_db_name (str): Name of the selected vector database
        vector_db_obj: The actual vector database object with identifier
    """
    if "upload_status" not in st.session_state:
        st.session_state["upload_status"] = None
    if "upload_message" not in st.session_state:
        st.session_state["upload_message"] = ""

    _show_status("upload_status", "upload_message")

    upload_key = f"processed_files_{vector_db_name}"
    if upload_key not in st.session_state:
        st.session_state[upload_key] = set()

    uploaded_files = st.file_uploader(
        "Browse and select files to upload (files will upload automatically)",
        accept_multiple_files=True,
        type=["txt", "pdf", "doc", "docx", "md"],
        key=f"uploader_{vector_db_name}",
        help=(
            "Select one or more documents - they will be uploaded "
            "automatically to this vector database"
        ),
    )

    if uploaded_files:
        file_set_id = frozenset([f.name + str(f.size) for f in uploaded_files])

        if file_set_id not in st.session_state[upload_key]:
            st.session_state[upload_key].add(file_set_id)

            if vector_db_obj and hasattr(vector_db_obj, 'id'):
                vector_db_id = vector_db_obj.id
            else:
                vector_db_id = vector_db_name

            _upload_documents_to_database(
                vector_db_name, uploaded_files, vector_db_id
            )


def _upload_documents_to_database(vector_db_name, uploaded_files, vector_db_id=None):
    """Upload documents to an existing vector database.

    Args:
        vector_db_name (str): Name of the target vector database
        uploaded_files: List of uploaded files from Streamlit file uploader
        vector_db_id (str): The actual database identifier for API calls
    """
    try:
        st.session_state["upload_status"] = None
        st.session_state["upload_message"] = ""

        if not uploaded_files:
            st.session_state["upload_status"] = "error"
            st.session_state["upload_message"] = "No files selected for upload."
            return

        actual_db_id = vector_db_id or vector_db_name
        uploaded_file_ids = []

        with st.spinner(f"Uploading {len(uploaded_files)} file(s)..."):
            for uploaded_file in uploaded_files:
                file_response = llama_stack_api.client.files.create(
                    file=uploaded_file,
                    purpose="assistants"
                )
                llama_stack_api.client.vector_stores.files.create(
                    vector_store_id=actual_db_id,
                    file_id=file_response.id,
                )
                uploaded_file_ids.append(file_response.id)

        st.session_state["upload_status"] = "success"
        st.session_state["upload_message"] = (
            f"Successfully uploaded {len(uploaded_files)} document(s) "
            f"to '{vector_db_name}'!"
        )
        st.rerun()

    except Exception as e:  # pylint: disable=broad-exception-caught
        st.session_state["upload_status"] = "error"
        st.session_state["upload_message"] = f"Error uploading documents: {str(e)}"
        st.rerun()


def _get_documents_from_vector_store(vector_store_id):
    """Get files from a vector store using the Files API.

    Args:
        vector_store_id (str): The vector store identifier

    Returns:
        list: List of file objects, or None if query fails
    """
    try:
        files_response = llama_stack_api.client.vector_stores.files.list(
            vector_store_id=vector_store_id
        )

        if hasattr(files_response, 'data'):
            return files_response.data
        return list(files_response) if files_response else None

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error listing files from vector store: {e}")
        return None


def _delete_file_from_vector_store(vector_store_id, file_id):
    """Delete a file from a vector store using the Files API.

    Args:
        vector_store_id (str): The vector store identifier
        file_id (str): The file ID to delete

    Returns:
        tuple: (success: bool, error_message: str)
    """
    try:
        llama_stack_api.client.vector_stores.files.delete(
            file_id=file_id,
            vector_store_id=vector_store_id
        )
        return True, None
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, str(e)


def _get_file_sources(files):
    """Retrieve the source name for each file.

    Prefers attributes["source"], falls back to filename from Files API.

    Args:
        files: List of vector store file objects

    Returns:
        dict: Mapping of file_id to source name
    """
    source_names = {}
    for file_obj in files:
        file_id = getattr(file_obj, 'id', None)
        if not file_id:
            continue
        attrs = getattr(file_obj, 'attributes', None) or {}
        source = attrs.get("source")
        if not source:
            try:
                file_info = llama_stack_api.client.files.retrieve(file_id)
                source = getattr(file_info, 'filename', None)
            except Exception:  # pylint: disable=broad-exception-caught
                source = None
        source_names[file_id] = source
    return source_names


def _render_documents_table(files, source_names):
    """Render the documents table with source and document ID columns.

    Args:
        files: List of vector store file objects
        source_names (dict): Mapping of file_id to source name
    """
    # Add CSS for bordered table rows
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] {
        border-bottom: 1px solid #444;
        padding: 8px 0;
    }
    div[data-testid="stHorizontalBlock"]:first-of-type {
        border-top: 1px solid #444;
        background-color: rgba(255, 255, 255, 0.05);
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display table header
    col1, col2, col3 = st.columns([0.5, 3, 3])
    with col1:
        st.markdown("**#**")
    with col2:
        st.markdown("**Source**")
    with col3:
        st.markdown("**Document ID**")

    # Display each file in a row
    for idx, file_obj in enumerate(files, start=1):
        col1, col2, col3 = st.columns([0.5, 3, 3])
        file_id = getattr(file_obj, 'id', 'unknown')
        source = source_names.get(file_id) or "unknown"

        with col1:
            st.write(idx)
        with col2:
            st.write(source)
        with col3:
            st.write(file_id)


def _show_existing_documents_table(vector_db_name, vector_db_obj=None):
    """Display information about documents in the selected vector database.

    Args:
        vector_db_name (str): Display name of the selected vector database
        vector_db_obj: The actual vector database object with identifier
    """
    try:
        if vector_db_obj and hasattr(vector_db_obj, 'id'):
            vector_db_id = vector_db_obj.id
        else:
            vector_db_id = vector_db_name

        if "delete_status" not in st.session_state:
            st.session_state["delete_status"] = None
        if "delete_message" not in st.session_state:
            st.session_state["delete_message"] = ""

        _show_status("delete_status", "delete_message")

        with st.spinner("Checking for documents..."):
            files = _get_documents_from_vector_store(vector_db_id)

            if files:
                st.subheader(f"📄 Documents in '{vector_db_name}'")
                source_names = _get_file_sources(files)
                _render_documents_table(files, source_names)

    except Exception as e:  # pylint: disable=broad-exception-caught
        st.error(f"Error loading document information: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


upload_page()
