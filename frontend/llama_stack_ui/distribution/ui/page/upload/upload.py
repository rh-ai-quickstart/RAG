# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import traceback

import streamlit as st

from llama_stack_ui.distribution.ui.modules.api import llama_stack_api
from llama_stack_ui.distribution.ui.modules.utils import get_vector_db_name


def upload_page():
    """
    Page to upload documents and manage vector databases for RAG.
    Supports creating new vector databases and uploading documents to existing ones.
    """
    st.title("📄 Upload Documents")

    # Initialize session state for creation status messages
    if "creation_status" not in st.session_state:
        st.session_state["creation_status"] = None
    if "creation_message" not in st.session_state:
        st.session_state["creation_message"] = ""

    # Initialize session state for selected vector database
    # This persists the selection when navigating away and back to this page
    if "selected_vector_db" not in st.session_state:
        st.session_state["selected_vector_db"] = ""

    # Initialize the widget key to match our tracked selection
    # This ensures the selectbox displays the correct value on page load
    if "vector_db_selector" not in st.session_state:
        st.session_state["vector_db_selector"] = st.session_state["selected_vector_db"]

    # Initialize newly created VDB tracker
    if "newly_created_vdb" not in st.session_state:
        st.session_state["newly_created_vdb"] = None

    # Show status messages at the top level (before dropdown)
    if st.session_state["creation_status"] == "success":
        st.success(st.session_state["creation_message"])
        # Clear the message after showing it
        st.session_state["creation_status"] = None
        st.session_state["creation_message"] = ""
    elif st.session_state["creation_status"] == "error":
        st.error(st.session_state["creation_message"])
        # Clear the message after showing it
        st.session_state["creation_status"] = None
        st.session_state["creation_message"] = ""

    # Fetch all vector databases
    vdb_list = llama_stack_api.client.vector_stores.list()

    # Build dropdown options based on whether databases exist
    dropdown_options = []

    # Define the "Create New" option with emoji for visibility
    create_new_option = "➕ Create New"

    if vdb_list:
        # When databases exist: list actual DBs first, then "Create New" LAST
        existing_vdbs = {get_vector_db_name(v): v.to_dict() for v in vdb_list}
        dropdown_options.extend(list(existing_vdbs.keys()))
        dropdown_options.append(create_new_option)  # Add "Create New" as LAST item
    else:
        # When NO databases exist: only show "Create New"
        dropdown_options = [create_new_option]

    # Sync session state for widget - ensure it shows the right value
    # Priority 1: If a database was just created, auto-select it (highest priority)
    if st.session_state["newly_created_vdb"]:
        newly_created_name = st.session_state["newly_created_vdb"]
        if newly_created_name in dropdown_options:
            # Update both session variables to sync state
            st.session_state["selected_vector_db"] = newly_created_name
            st.session_state["vector_db_selector"] = newly_created_name
            st.session_state["newly_created_vdb"] = None
    # Priority 2: Use the previously selected database from session if it still exists
    elif (
        st.session_state["selected_vector_db"]
        and st.session_state["selected_vector_db"] in dropdown_options
    ):
        # Sync widget state with our tracked state
        st.session_state["vector_db_selector"] = st.session_state["selected_vector_db"]
    # Priority 3: If no saved selection or saved selection doesn't exist, use smart default
    else:
        if vdb_list:
            # When databases exist: default to FIRST actual database (not "Create New")
            first_db = dropdown_options[0]  # First item is first actual database
            st.session_state["selected_vector_db"] = first_db
            st.session_state["vector_db_selector"] = first_db
        else:
            # When NO databases exist: default to "Create New"
            st.session_state["selected_vector_db"] = create_new_option
            st.session_state["vector_db_selector"] = create_new_option

    # Vector database selection dropdown with persistent selection
    # Using key parameter to bind directly to session state - NO index parameter to avoid conflicts
    def on_vector_db_change():
        """Callback to update session state when selection changes"""
        st.session_state["selected_vector_db"] = st.session_state["vector_db_selector"]

    selected_vector_db = st.selectbox(
        "Select a vector database",
        dropdown_options,
        key="vector_db_selector",  # Key binds to session state (session state controls the value)
        on_change=on_vector_db_change,  # Callback updates our tracking variable
        help="Your selection will be remembered when you navigate to other pages"
    )

    # Ensure session state is updated (in case callback didn't fire)
    if selected_vector_db != st.session_state["selected_vector_db"]:
        st.session_state["selected_vector_db"] = selected_vector_db

    # Get the actual vector database object for API calls (do this before using it)
    selected_vdb_obj = None
    if selected_vector_db and selected_vector_db != create_new_option:
        for vdb in vdb_list:
            if get_vector_db_name(vdb) == selected_vector_db:
                selected_vdb_obj = vdb
                break

    if selected_vector_db == create_new_option:
        # Show vector database creation UI
        _show_create_vector_db_ui()
    elif selected_vector_db and selected_vector_db != create_new_option:
        # Show existing documents in the database (heading will show only if documents exist)
        _show_existing_documents_table(selected_vector_db, selected_vdb_obj)

        # Add Browse functionality for uploading documents to this database
        st.subheader(f"📁 Upload Documents to '{selected_vector_db}'")
        _show_document_upload_ui(selected_vector_db, selected_vdb_obj)
    # If empty string is selected, show nothing (clean default state)


def _show_create_vector_db_ui():
    """
    Display UI for creating a new vector database.
    """
    st.subheader("Create New Vector Database")

    # Initialize session state for creation form
    if "new_vdb_name" not in st.session_state:
        st.session_state["new_vdb_name"] = ""

    # Vector database name input
    new_vdb_name = st.text_input(
        "Add New Vector Database",
        value=st.session_state["new_vdb_name"],
        help="Enter a unique name for the new vector database",
        key="new_vdb_name_input"
    )

    # Update session state
    st.session_state["new_vdb_name"] = new_vdb_name

    # Add button
    if st.button("Add", type="primary", disabled=not new_vdb_name.strip()):
        _create_vector_database(new_vdb_name.strip())


def _create_vector_database(vdb_name):
    """
    Create a new vector database using the LlamaStack API.

    Args:
        vdb_name (str): Name for the new vector database
    """
    try:
        # Reset status
        st.session_state["creation_status"] = None
        st.session_state["creation_message"] = ""

        # Validate input
        if not vdb_name or not vdb_name.strip():
            st.session_state["creation_status"] = "error"
            st.session_state["creation_message"] = "Vector database name cannot be empty."
            return

        # Check for duplicate names
        existing_vdbs = llama_stack_api.client.vector_stores.list()
        existing_names = [get_vector_db_name(vdb) for vdb in existing_vdbs]
        if vdb_name in existing_names:
            st.session_state["creation_status"] = "error"
            st.session_state["creation_message"] = (
                f"Vector database '{vdb_name}' already exists. "
                "Please choose a different name."
            )
            return

        # Create the vector database using the new simplified API
        # Note: embedding settings (dimension, model, provider) must be
        # configured at the server level
        with st.spinner(f"Creating vector database '{vdb_name}'..."):
            _vector_db = llama_stack_api.client.vector_stores.create(
                name=vdb_name,
            )

        # Success
        st.session_state["creation_status"] = "success"
        st.session_state["creation_message"] = f"Vector database '{vdb_name}' created successfully!"

        # Mark this database to be auto-selected after refresh
        st.session_state["newly_created_vdb"] = vdb_name

        # Clear the input field
        st.session_state["new_vdb_name"] = ""

        # Trigger page refresh to update the dropdown - this will show the message at the top
        st.rerun()

    except Exception as e:
        st.session_state["creation_status"] = "error"
        st.session_state["creation_message"] = f"Error creating vector database: {str(e)}"


def _show_document_upload_ui(vector_db_name, vector_db_obj=None):
    """
    Display UI for uploading documents to an existing vector database.

    Args:
        vector_db_name (str): Name of the selected vector database
    """
    # Initialize session state for upload status
    if "upload_status" not in st.session_state:
        st.session_state["upload_status"] = None
    if "upload_message" not in st.session_state:
        st.session_state["upload_message"] = ""

    # Show upload status messages
    if st.session_state["upload_status"] == "success":
        st.success(st.session_state["upload_message"])
        # Clear after showing
        st.session_state["upload_status"] = None
        st.session_state["upload_message"] = ""
    elif st.session_state["upload_status"] == "error":
        st.error(st.session_state["upload_message"])
        # Clear after showing
        st.session_state["upload_status"] = None
        st.session_state["upload_message"] = ""

    # Initialize session state to track processed files
    upload_key = f"processed_files_{vector_db_name}"
    if upload_key not in st.session_state:
        st.session_state[upload_key] = set()

    # File uploader
    uploaded_files = st.file_uploader(
        "Browse and select files to upload (files will upload automatically)",
        accept_multiple_files=True,
        type=["txt", "pdf", "doc", "docx"],
        key=f"uploader_{vector_db_name}",  # Unique key per database
        help=(
            "Select one or more documents - they will be uploaded "
            "automatically to this vector database"
        ),
    )

    # Auto-upload when files are selected
    if uploaded_files:
        # Create a unique identifier for this set of files
        file_set_id = frozenset([f.name + str(f.size) for f in uploaded_files])

        # Only process if this is a new set of files
        if file_set_id not in st.session_state[upload_key]:
            # Mark as processed IMMEDIATELY before upload to prevent re-triggering
            st.session_state[upload_key].add(file_set_id)

            # Get the correct database ID for upload
            if vector_db_obj and hasattr(vector_db_obj, 'id'):
                vector_db_id = vector_db_obj.id
            else:
                vector_db_id = vector_db_name

            # Upload automatically
            _upload_documents_to_database(
                vector_db_name, uploaded_files, vector_db_id
            )


def _upload_documents_to_database(vector_db_name, uploaded_files, vector_db_id=None):
    """
    Upload documents to an existing vector database.

    Args:
        vector_db_name (str): Name of the target vector database
        uploaded_files: List of uploaded files from Streamlit file uploader
    """
    try:
        # Reset status
        st.session_state["upload_status"] = None
        st.session_state["upload_message"] = ""

        if not uploaded_files:
            st.session_state["upload_status"] = "error"
            st.session_state["upload_message"] = "No files selected for upload."
            return

        # Upload files using the new Files API + Vector Stores API
        actual_db_id = vector_db_id or vector_db_name
        uploaded_file_ids = []

        with st.spinner(f"Uploading {len(uploaded_files)} file(s)..."):
            for uploaded_file in uploaded_files:
                # Step 1: Upload file to Files API
                file_response = llama_stack_api.client.files.create(
                    file=uploaded_file,
                    purpose="assistants"
                )

                # Step 2: Attach file to vector store
                llama_stack_api.client.vector_stores.files.create(
                    vector_store_id=actual_db_id,
                    file_id=file_response.id,
                )

                uploaded_file_ids.append(file_response.id)

        # Success
        st.session_state["upload_status"] = "success"
        st.session_state["upload_message"] = (
            f"Successfully uploaded {len(uploaded_files)} document(s) "
            f"to '{vector_db_name}'!"
        )

        # Trigger refresh to show the success message
        st.rerun()

    except Exception as e:
        st.session_state["upload_status"] = "error"
        st.session_state["upload_message"] = f"Error uploading documents: {str(e)}"
        st.rerun()


def _get_documents_from_vector_store(vector_store_id):
    """
    Get files from a vector store using the Files API.

    Args:
        vector_store_id (str): The vector store identifier

    Returns:
        list: List of file objects, or None if query fails
    """
    try:
        # List files in the vector store
        files_response = llama_stack_api.client.vector_stores.files.list(
            vector_store_id=vector_store_id
        )

        # Extract file information
        if hasattr(files_response, 'data'):
            return files_response.data
        else:
            return list(files_response) if files_response else None

    except Exception as e:
        print(f"Error listing files from vector store: {e}")
        return None


def _delete_file_from_vector_store(vector_store_id, file_id):
    """
    Delete a file from a vector store using the Files API.

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
    except Exception as e:
        return False, str(e)


def _show_existing_documents_table(vector_db_name, vector_db_obj=None):
    """
    Display information about documents in the selected vector database.

    Args:
        vector_db_name (str): Display name of the selected vector database
        vector_db_obj: The actual vector database object with identifier
    """
    try:
        # Get the correct vector database ID
        if vector_db_obj and hasattr(vector_db_obj, 'id'):
            vector_db_id = vector_db_obj.id
        else:
            vector_db_id = vector_db_name  # Fallback to display name

        # Initialize session state for deletion status
        if "delete_status" not in st.session_state:
            st.session_state["delete_status"] = None
        if "delete_message" not in st.session_state:
            st.session_state["delete_message"] = ""

        # Show deletion status messages (before checking documents, so last delete shows)
        if st.session_state["delete_status"] == "success":
            st.success(st.session_state["delete_message"])
            st.session_state["delete_status"] = None
            st.session_state["delete_message"] = ""
        elif st.session_state["delete_status"] == "error":
            st.error(st.session_state["delete_message"])
            st.session_state["delete_status"] = None
            st.session_state["delete_message"] = ""

        with st.spinner("Checking for documents..."):
            # Get files from vector store using the Files API
            files = _get_documents_from_vector_store(vector_db_id)

            if files:
                # Success! We have the files
                # Show heading for documents section
                st.subheader(f"📄 Documents in '{vector_db_name}'")
                st.info("ℹ️ File deletion is currently not available.")

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
                col1, col2, col3 = st.columns([0.5, 5, 0.5])
                with col1:
                    st.markdown("**#**")
                with col2:
                    st.markdown("**Filename**")
                with col3:
                    st.markdown("**Del**")

                # Display each file in a row with delete button
                for idx, file_obj in enumerate(files, start=1):
                    col1, col2, col3 = st.columns([0.5, 5, 0.5])

                    # Extract file information
                    file_id = getattr(file_obj, 'id', 'unknown')
                    # Try to get filename from metadata or use file_id
                    filename = file_id

                    with col1:
                        st.write(idx)

                    with col2:
                        st.write(filename)

                    with col3:
                        delete_key = f"delete_{vector_db_name}_{file_id}_{idx}"

                        # Disable delete button
                        st.button(
                            "✕",
                            key=delete_key,
                            disabled=True,
                            help="Delete is currently not available",
                        )

            # else: Database appears empty or pgvector query not available
            # For newly created databases, this is expected - just show nothing
            # The upload section below will allow users to add documents

    except Exception as e:
        st.error(f"Error loading document information: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


upload_page()
