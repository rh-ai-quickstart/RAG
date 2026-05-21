# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Upload documents page for managing vector databases and document ingestion."""

import traceback

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api
from llama_stack_ui.distribution.ui.modules.local_extractors import (
    LOCAL_SUPPORTED_EXTENSIONS,
    PROVIDER_SUPPORTED_EXTENSIONS,
    create_text_file_from_extracted_content,
    extract_text,
)
from llama_stack_ui.distribution.ui.modules.utils import get_vector_db_name


def _init_upload_page_session_state():
    """Initialize all session state variables needed by the upload page."""
    defaults = {
        "creation_status": None,
        "creation_message": "",
        "selected_vector_db": "",
        "newly_created_vdb": None,
        "extraction_method": "provider",
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

    # Priority 3: Smart default — pick the first real DB, not "Create New"
    if vdb_list:
        first_real_db = dropdown_options[1]
        st.session_state["selected_vector_db"] = first_real_db
        st.session_state["vector_db_selector"] = first_real_db
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

    Shows an extraction method toggle that determines which file types are
    accepted and how they are processed before ingestion.

    Args:
        vector_db_name (str): Name of the selected vector database
        vector_db_obj: The actual vector database object with identifier
    """
    if "upload_status" not in st.session_state:
        st.session_state["upload_status"] = None
    if "upload_message" not in st.session_state:
        st.session_state["upload_message"] = ""

    _show_status("upload_status", "upload_message")

    local_label = (
        "Docling ("
        + ", ".join(LOCAL_SUPPORTED_EXTENSIONS) + ")"
    )
    provider_label = (
        "LlamaStack Provider ("
        + ", ".join(PROVIDER_SUPPORTED_EXTENSIONS) + ")"
    )
    method_options = [provider_label, local_label]

    selected_label = st.radio(
        "Extraction method",
        method_options,
        key="extraction_method_radio",
        horizontal=False,
        help="Local extraction converts .docx/.xlsx to text in the browser. "
             "LlamaStack Provider sends files directly to the server.",
    )

    is_local = selected_label == local_label
    st.session_state["extraction_method"] = "local" if is_local else "provider"

    if is_local:
        accepted_types = [ext.lstrip(".") for ext in LOCAL_SUPPORTED_EXTENSIONS]
    else:
        accepted_types = [ext.lstrip(".") for ext in PROVIDER_SUPPORTED_EXTENSIONS]

    upload_key = f"processed_files_{vector_db_name}"
    if upload_key not in st.session_state:
        st.session_state[upload_key] = set()

    uploaded_files = st.file_uploader(
        "Browse and select files to upload (files will upload automatically)",
        accept_multiple_files=True,
        type=accepted_types,
        key=f"uploader_{vector_db_name}_{st.session_state['extraction_method']}",
        help=(
            "Select one or more documents — they will be uploaded "
            "automatically to this vector database"
        ),
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name + str(f.size) not in st.session_state[upload_key]
        ]

        if new_files:
            for f in new_files:
                st.session_state[upload_key].add(f.name + str(f.size))

            if vector_db_obj and hasattr(vector_db_obj, 'id'):
                vector_db_id = vector_db_obj.id
            else:
                vector_db_id = vector_db_name

            _upload_documents_to_database(
                vector_db_name,
                new_files,
                vector_db_id,
                extraction_method=st.session_state["extraction_method"],
            )

def _upload_documents_to_database(vector_db_name, uploaded_files, vector_db_id=None, extraction_method="provider"):
    """Upload documents to an existing vector database.

    When extraction_method is "local", files are first converted to plain text
    using the local extractors and the resulting .txt content is uploaded.
    When "provider", files are sent directly to the LlamaStack server.

    Args:
        vector_db_name (str): Name of the target vector database
        uploaded_files: List of uploaded files from Streamlit file uploader
        vector_db_id (str): The actual database identifier for API calls
        extraction_method (str): "local" for client-side extraction, "provider" for server-side
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

        spinner_msg = (
            f"Extracting and uploading {len(uploaded_files)} file(s)..."
            if extraction_method == "local"
            else f"Uploading {len(uploaded_files)} file(s)..."
        )

        with st.spinner(spinner_msg):
            for uploaded_file in uploaded_files:
                original_filename = uploaded_file.name

                if extraction_method == "local":
                    text_content = extract_text(uploaded_file, original_filename)
                    file_to_upload = create_text_file_from_extracted_content(
                        text_content, original_filename
                    )
                else:
                    file_to_upload = uploaded_file

                file_response = llama_stack_api.client.files.create(
                    file=file_to_upload,
                    purpose="assistants"
                )

                vs_file_kwargs = {
                    "vector_store_id": actual_db_id,
                    "file_id": file_response.id,
                }
                if extraction_method == "local":
                    vs_file_kwargs["attributes"] = {"source": original_filename}

                llama_stack_api.client.vector_stores.files.create(**vs_file_kwargs)
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
        st.warning(f"Could not list files: {e}")
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


def _render_documents_table(files, source_names, vector_store_id):
    """Render the documents table with source, document ID, and delete button.

    Args:
        files: List of vector store file objects
        source_names (dict): Mapping of file_id to source name
        vector_store_id (str): The vector store identifier (for delete calls)
    """
    st.markdown("""
    <style>
    .doc-table-container div[data-testid="stHorizontalBlock"] {
        border-bottom: 1px solid rgba(128, 128, 128, 0.3);
        padding: 8px 0;
    }
    .doc-table-container div[data-testid="stHorizontalBlock"]:first-of-type {
        border-top: 1px solid rgba(128, 128, 128, 0.3);
        background-color: rgba(128, 128, 128, 0.06);
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="doc-table-container">', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([0.5, 3, 3, 1])
        with col1:
            st.markdown("**#**")
        with col2:
            st.markdown("**Source**")
        with col3:
            st.markdown("**Document ID**")
        with col4:
            st.markdown("**Actions**")

        for idx, file_obj in enumerate(files, start=1):
            col1, col2, col3, col4 = st.columns([0.5, 3, 3, 1])
            file_id = getattr(file_obj, 'id', 'unknown')
            source = source_names.get(file_id) or "unknown"

            with col1:
                st.write(idx)
            with col2:
                st.write(source)
            with col3:
                st.write(file_id)
            with col4:
                if st.button("🗑️", key=f"delete_{file_id}", help=f"Delete {source}"):
                    success, error_msg = _delete_file_from_vector_store(vector_store_id, file_id)
                    if success:
                        st.session_state["delete_status"] = "success"
                        st.session_state["delete_message"] = f"Deleted '{source}' successfully."
                    else:
                        st.session_state["delete_status"] = "error"
                        st.session_state["delete_message"] = f"Failed to delete '{source}': {error_msg}"
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


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

            st.subheader(f"📄 Documents in '{vector_db_name}'")
            if files:
                source_names = _get_file_sources(files)
                _render_documents_table(files, source_names, vector_db_id)
            else:
                st.info("No documents found in this vector database. Upload some below!")

    except Exception as e:  # pylint: disable=broad-exception-caught
        st.error(f"Error loading document information: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


upload_page()
