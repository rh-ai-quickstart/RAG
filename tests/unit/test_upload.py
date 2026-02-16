"""
Unit tests for the upload module
Tests document upload and vector DB creation logic
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add the frontend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../frontend'))

# Mock all external dependencies before any imports from the upload module
# This is required because @patch decorators try to import the target module
mock_streamlit = MagicMock()
mock_streamlit.session_state = {}
sys.modules['streamlit'] = mock_streamlit
sys.modules['asyncpg'] = MagicMock()
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


class MockAsyncContextManager:
    """Mock async context manager for pool.acquire()"""
    def __init__(self, conn):
        self.conn = conn
    
    async def __aenter__(self):
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


def create_mock_pool_with_connection(mock_conn):
    """
    Helper to create a mock connection pool that yields the given connection.
    
    Args:
        mock_conn: The mock connection to return from pool.acquire()
        
    Returns:
        MagicMock: A mock pool with proper acquire() context manager
    """
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = MockAsyncContextManager(mock_conn)
    return mock_pool


class TestGetDocumentsFromPgvector:
    """Unit tests for _get_documents_from_pgvector function"""
    
    def test_get_documents_success(self):
        """Test successful retrieval of documents from pgvector"""
        # Setup mock connection
        mock_conn = AsyncMock()
        mock_rows = [
            {'document_id': 'document1.pdf'},
            {'document_id': 'document2.txt'},
            {'document_id': 'document3.docx'},
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        
        # Create mock pool
        mock_pool = create_mock_pool_with_connection(mock_conn)
        
        # Patch _get_pg_pool to return our mock pool
        async def mock_get_pool():
            return mock_pool
        
        with patch.object(upload_module, '_get_pg_pool', mock_get_pool):
            # Call the actual function
            result = upload_module._get_documents_from_pgvector("my-test-db")
            
            # Verify the result
            assert result == ['document1.pdf', 'document2.txt', 'document3.docx']
            
            # Verify acquire was called (connection borrowed from pool)
            mock_pool.acquire.assert_called_once()
    
    def test_get_documents_empty_result(self):
        """Test that empty results return None"""
        # Setup mock connection with empty result
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        
        mock_pool = create_mock_pool_with_connection(mock_conn)
        
        async def mock_get_pool():
            return mock_pool
        
        with patch.object(upload_module, '_get_pg_pool', mock_get_pool):
            result = upload_module._get_documents_from_pgvector("empty-db")
            
            # Empty result should return None
            assert result is None
    
    def test_get_documents_connection_error(self):
        """Test that connection errors return None"""
        # Setup mock pool that raises an exception on acquire
        async def mock_get_pool():
            raise Exception("Connection refused")
        
        with patch.object(upload_module, '_get_pg_pool', mock_get_pool):
            result = upload_module._get_documents_from_pgvector("error-db")
            
            # Error should return None
            assert result is None
    
    def test_get_documents_filters_null_ids(self):
        """Test that null document IDs are filtered out"""
        mock_conn = AsyncMock()
        mock_rows = [
            {'document_id': 'valid1.pdf'},
            {'document_id': None},  # Should be filtered
            {'document_id': 'valid2.txt'},
            {'document_id': None},  # Should be filtered
        ]
        mock_conn.fetch = AsyncMock(return_value=mock_rows)
        
        mock_pool = create_mock_pool_with_connection(mock_conn)
        
        async def mock_get_pool():
            return mock_pool
        
        with patch.object(upload_module, '_get_pg_pool', mock_get_pool):
            result = upload_module._get_documents_from_pgvector("mixed-db")
            
            # Only valid IDs should be returned
            assert result == ['valid1.pdf', 'valid2.txt']
            assert len(result) == 2


class TestDeleteDocumentFromPgvector:
    """Unit tests for _delete_document_from_pgvector function"""
    
    def test_delete_document_success(self):
        """Test successful deletion of document from pgvector"""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 5")
        
        mock_pool = create_mock_pool_with_connection(mock_conn)
        
        async def mock_get_pool():
            return mock_pool
        
        with patch.object(upload_module, '_get_pg_pool', mock_get_pool):
            success, count, error = upload_module._delete_document_from_pgvector(
                "my-test-db", 
                "document.pdf"
            )
            
            assert success is True
            assert count == 5
            assert error is None
            mock_pool.acquire.assert_called_once()
    
    def test_delete_document_not_found(self):
        """Test deletion when document doesn't exist"""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")
        
        mock_pool = create_mock_pool_with_connection(mock_conn)
        
        async def mock_get_pool():
            return mock_pool
        
        with patch.object(upload_module, '_get_pg_pool', mock_get_pool):
            success, count, error = upload_module._delete_document_from_pgvector(
                "my-test-db", 
                "nonexistent.pdf"
            )
            
            assert success is True
            assert count == 0
            assert error is None
    
    def test_delete_document_connection_error(self):
        """Test deletion with connection error"""
        async def mock_get_pool():
            raise Exception("Connection refused")
        
        with patch.object(upload_module, '_get_pg_pool', mock_get_pool):
            success, count, error = upload_module._delete_document_from_pgvector(
                "my-test-db", 
                "document.pdf"
            )
            
            assert success is False
            assert count == 0
            assert error is not None
            assert "Connection refused" in str(error)


class TestCreateVectorDatabase:
    """Unit tests for _create_vector_database function"""
    
    def test_create_vector_database_success(self):
        """Test successful creation of vector database"""
        # Mock the API client
        mock_client = MagicMock()
        mock_client.vector_dbs.list.return_value = []
        mock_client.providers.list.return_value = [
            MagicMock(api="vector_io", provider_id="pgvector")
        ]
        mock_client.vector_dbs.register.return_value = MagicMock()
        
        mock_api = MagicMock()
        mock_api.client = mock_client
        
        # Mock session state
        mock_st = MagicMock()
        mock_st.session_state = {}
        
        with patch.object(upload_module, 'llama_stack_api', mock_api):
            with patch.object(upload_module, 'st', mock_st):
                upload_module._create_vector_database("new-test-db")
                
                # Verify registration was called with correct parameters
                mock_client.vector_dbs.register.assert_called_once()
                call_kwargs = mock_client.vector_dbs.register.call_args[1]
                assert call_kwargs['vector_db_id'] == "new-test-db"
                assert call_kwargs['embedding_model'] == "all-MiniLM-L6-v2"
                assert call_kwargs['embedding_dimension'] == 384
                assert call_kwargs['provider_id'] == "pgvector"
    
    def test_create_vector_database_duplicate_name(self):
        """Test that duplicate names are rejected"""
        # Mock existing database with same name
        existing_db = MagicMock()
        existing_db.identifier = "existing-db"
        
        mock_client = MagicMock()
        mock_client.vector_dbs.list.return_value = [existing_db]
        
        mock_api = MagicMock()
        mock_api.client = mock_client
        
        mock_st = MagicMock()
        mock_st.session_state = {}
        
        with patch.object(upload_module, 'llama_stack_api', mock_api):
            with patch.object(upload_module, 'st', mock_st):
                upload_module._create_vector_database("existing-db")
                
                # Registration should NOT be called for duplicates
                mock_client.vector_dbs.register.assert_not_called()
                
                # Error status should be set
                assert mock_st.session_state.get("creation_status") == "error"
    
    def test_create_vector_database_no_provider(self):
        """Test error when no vector_io provider exists"""
        mock_client = MagicMock()
        mock_client.vector_dbs.list.return_value = []
        mock_client.providers.list.return_value = [
            MagicMock(api="inference", provider_id="ollama")  # No vector_io
        ]
        
        mock_api = MagicMock()
        mock_api.client = mock_client
        
        mock_st = MagicMock()
        mock_st.session_state = {}
        
        with patch.object(upload_module, 'llama_stack_api', mock_api):
            with patch.object(upload_module, 'st', mock_st):
                upload_module._create_vector_database("new-db")
                
                # Registration should NOT be called without provider
                mock_client.vector_dbs.register.assert_not_called()
                
                # Error status should be set
                assert mock_st.session_state.get("creation_status") == "error"


class TestConnectionPool:
    """Unit tests for the connection pool functionality"""
    
    def test_pool_is_reused(self):
        """Test that the same pool is returned on subsequent calls"""
        # Reset the global pool
        upload_module._pg_pool = None
        
        mock_pool = AsyncMock()
        
        # Mock asyncpg.create_pool
        async def mock_create_pool(**kwargs):
            return mock_pool
        
        with patch.object(upload_module.asyncpg, 'create_pool', mock_create_pool):
            # Get pool twice
            async def get_pools():
                pool1 = await upload_module._get_pg_pool()
                pool2 = await upload_module._get_pg_pool()
                return pool1, pool2
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            pool1, pool2 = loop.run_until_complete(get_pools())
            
            # Should be the same pool instance
            assert pool1 is pool2
        
        # Clean up
        upload_module._pg_pool = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
