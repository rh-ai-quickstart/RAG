import streamlit as st
from llama_stack_client import RAGDocument
from modules.api import llama_stack_api
from modules.utils import data_url_from_file

def upload_page():
    """
    Page to upload documents and create a vector database for RAG.
    """
    st.title("📄 Upload")
    # File/Directory Upload Section
    st.subheader("Create Vector DB")
    # Let user select files to ingest
    uploaded_files = st.file_uploader(
        "Upload file(s) or directory",
        accept_multiple_files=True,
        type=["txt", "pdf", "doc", "docx"],  # supported file types
    )
    # Process uploaded files
    if uploaded_files:
        # Show upload success and prompt for DB name
        st.success(f"Successfully uploaded {len(uploaded_files)} files")
        vector_db_name = st.text_input(
            "Vector Database Name",
            value="rag_vector_db",
            help="Enter a unique identifier for this vector database",
        )
        if st.button("Create Vector Database"):
            # Convert uploaded files into RAGDocument instances
            documents = [
                RAGDocument(
                    document_id=uploaded_file.name,
                    content=data_url_from_file(uploaded_file),
                )
                for i, uploaded_file in enumerate(uploaded_files)
            ]

            # Determine provider for vector IO
            providers = llama_stack_api.client.providers.list()
            vector_io_provider = None
            for x in providers:
                if x.api == "vector_io":
                    vector_io_provider = x.provider_id

            # Register new vector database
            llama_stack_api.client.vector_dbs.register(
                vector_db_id=vector_db_name,
                embedding_dimension=384,
                embedding_model="all-MiniLM-L6-v2",
                provider_id=vector_io_provider,
            )

            # Insert documents into the vector database
            llama_stack_api.client.tool_runtime.rag_tool.insert(
                vector_db_id=vector_db_name,
                documents=documents,
                chunk_size_in_tokens=512,
            )
            st.success("Vector database created successfully!")
            # Reset form fields
            uploaded_files.clear()
            vector_db_name = ""
upload_page()