"""
Unit tests for the upload module
Tests document upload and vector DB creation logic
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the frontend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../frontend'))

# Mock all external dependencies before any imports from the upload module
# This is required because @patch decorators try to import the target module
mock_streamlit = MagicMock()
mock_streamlit.session_state = {}
sys.modules['streamlit'] = mock_streamlit
sys.modules['pandas'] = MagicMock()

# Mock llama_stack_client with a proper RAGDocument mock
mock_llama_stack_client = MagicMock()
def mock_rag_document(**kwargs):
    """Create a dict-like RAGDocument mock"""
    return kwargs
mock_llama_stack_client.RAGDocument = mock_rag_document
sys.modules['llama_stack_client'] = mock_llama_stack_client

# Now we can safely import modules that will be patched
# Pre-import the modules so @patch can find them
from llama_stack_ui.distribution.ui.modules import api, utils
from llama_stack_ui.distribution.ui.page.upload import upload as upload_module


class TestGetDocumentsFromVectorStore:
    """Unit tests for _get_documents_from_vector_store function"""

    def test_get_documents_success_with_data_attribute(self):
        """Test successful retrieval when response has a .data attribute"""
        mock_file1 = MagicMock(id='file-001')
        mock_file2 = MagicMock(id='file-002')
        mock_file3 = MagicMock(id='file-003')

        mock_response = MagicMock()
        mock_response.data = [mock_file1, mock_file2, mock_file3]

        mock_client = MagicMock()
        mock_client.vector_stores.files.list.return_value = mock_response

        mock_api = MagicMock()
        mock_api.client = mock_client

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            result = upload_module._get_documents_from_vector_store("vs-123")

            assert result == [mock_file1, mock_file2, mock_file3]
            assert len(result) == 3
            mock_client.vector_stores.files.list.assert_called_once_with(
                vector_store_id="vs-123"
            )

    def test_get_documents_success_iterable_response(self):
        """Test successful retrieval when response is an iterable (no .data)"""
        mock_file1 = MagicMock(id='file-001')
        mock_file2 = MagicMock(id='file-002')

        # Use a plain list as the response — no .data attribute, but iterable and truthy
        mock_response = [mock_file1, mock_file2]

        mock_client = MagicMock()
        mock_client.vector_stores.files.list.return_value = mock_response

        mock_api = MagicMock()
        mock_api.client = mock_client

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            result = upload_module._get_documents_from_vector_store("vs-456")

            # list() on a list returns a copy, so compare contents
            assert len(result) == 2

    def test_get_documents_empty_result(self):
        """Test that empty/falsy response returns None"""
        # An empty list is falsy and has no .data attribute
        mock_response = []

        mock_client = MagicMock()
        mock_client.vector_stores.files.list.return_value = mock_response

        mock_api = MagicMock()
        mock_api.client = mock_client

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            result = upload_module._get_documents_from_vector_store("empty-store")

            assert result is None

    def test_get_documents_connection_error(self):
        """Test that exceptions return None"""
        mock_client = MagicMock()
        mock_client.vector_stores.files.list.side_effect = Exception("Connection refused")

        mock_api = MagicMock()
        mock_api.client = mock_client

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            result = upload_module._get_documents_from_vector_store("error-store")

            assert result is None


class TestDeleteFileFromVectorStore:
    """Unit tests for _delete_file_from_vector_store function"""

    def test_delete_file_success(self):
        """Test successful deletion of a file from vector store"""
        mock_client = MagicMock()
        mock_client.vector_stores.files.delete.return_value = None

        mock_api = MagicMock()
        mock_api.client = mock_client

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            success, error = upload_module._delete_file_from_vector_store(
                "vs-123", "file-001"
            )

            assert success is True
            assert error is None
            mock_client.vector_stores.files.delete.assert_called_once_with(
                file_id="file-001",
                vector_store_id="vs-123"
            )

    def test_delete_file_error(self):
        """Test deletion with an API error"""
        mock_client = MagicMock()
        mock_client.vector_stores.files.delete.side_effect = Exception(
            "File not found"
        )

        mock_api = MagicMock()
        mock_api.client = mock_client

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            success, error = upload_module._delete_file_from_vector_store(
                "vs-123", "nonexistent-file"
            )

            assert success is False
            assert error is not None
            assert "File not found" in error

    def test_delete_file_connection_error(self):
        """Test deletion with a connection error"""
        mock_client = MagicMock()
        mock_client.vector_stores.files.delete.side_effect = Exception(
            "Connection refused"
        )

        mock_api = MagicMock()
        mock_api.client = mock_client

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            success, error = upload_module._delete_file_from_vector_store(
                "vs-123", "file-001"
            )

            assert success is False
            assert "Connection refused" in error


class TestCreateVectorDatabase:
    """Unit tests for _create_vector_database function"""

    def test_create_vector_database_success(self):
        """Test successful creation of vector database"""
        mock_client = MagicMock()
        mock_client.vector_stores.list.return_value = []
        mock_client.vector_stores.create.return_value = MagicMock()

        mock_api = MagicMock()
        mock_api.client = mock_client

        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            with patch.object(upload_module, 'st', mock_st):
                upload_module._create_vector_database("new-test-db")

                mock_client.vector_stores.create.assert_called_once_with(
                    name="new-test-db",
                )
                assert mock_st.session_state.get("creation_status") == "success"
                assert "new-test-db" in mock_st.session_state.get(
                    "creation_message", ""
                )

    def test_create_vector_database_duplicate_name(self):
        """Test that duplicate names are rejected"""
        existing_db = MagicMock()
        existing_db.name = "existing-db"
        existing_db.id = "vs-existing"

        mock_client = MagicMock()
        mock_client.vector_stores.list.return_value = [existing_db]

        mock_api = MagicMock()
        mock_api.client = mock_client

        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            with patch.object(upload_module, 'st', mock_st):
                upload_module._create_vector_database("existing-db")

                # Creation should NOT be called for duplicates
                mock_client.vector_stores.create.assert_not_called()

                # Error status should be set
                assert mock_st.session_state.get("creation_status") == "error"
                assert "already exists" in mock_st.session_state.get(
                    "creation_message", ""
                )

    def test_create_vector_database_empty_name(self):
        """Test that empty names are rejected"""
        mock_client = MagicMock()

        mock_api = MagicMock()
        mock_api.client = mock_client

        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            with patch.object(upload_module, 'st', mock_st):
                upload_module._create_vector_database("")

                mock_client.vector_stores.create.assert_not_called()
                assert mock_st.session_state.get("creation_status") == "error"
                assert "empty" in mock_st.session_state.get(
                    "creation_message", ""
                ).lower()

    def test_create_vector_database_api_error(self):
        """Test error handling when API call fails"""
        mock_client = MagicMock()
        mock_client.vector_stores.list.return_value = []
        mock_client.vector_stores.create.side_effect = Exception(
            "API unavailable"
        )

        mock_api = MagicMock()
        mock_api.client = mock_client

        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch.object(upload_module, 'llama_stack_api', mock_api):
            with patch.object(upload_module, 'st', mock_st):
                upload_module._create_vector_database("new-db")

                assert mock_st.session_state.get("creation_status") == "error"
                assert "API unavailable" in mock_st.session_state.get(
                    "creation_message", ""
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
