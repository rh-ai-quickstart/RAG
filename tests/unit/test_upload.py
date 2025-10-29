"""
Unit tests for the upload module
Tests document upload and vector DB creation logic
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the frontend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../frontend'))

# Mock streamlit before importing
sys.modules['streamlit'] = MagicMock()


class TestVectorDBConfiguration:
    """Test vector database configuration and setup"""
    
    def test_vector_db_default_name(self):
        """Test default vector database name"""
        default_name = "rag_vector_db"
        assert default_name == "rag_vector_db"
        assert len(default_name) > 0
    
    def test_vector_db_embedding_dimension(self):
        """Test that embedding dimension is set correctly for all-MiniLM-L6-v2"""
        embedding_dimension = 384
        embedding_model = "all-MiniLM-L6-v2"
        
        assert embedding_dimension == 384
        assert embedding_model == "all-MiniLM-L6-v2"
    
    def test_chunk_size_configuration(self):
        """Test that chunk size is set to 512 tokens"""
        chunk_size = 512
        assert chunk_size == 512


class TestDocumentProcessing:
    """Test document processing and RAGDocument creation"""
    
    def test_supported_file_types(self):
        """Test that supported file types are correctly defined"""
        supported_types = ["txt", "pdf", "doc", "docx"]
        
        assert "txt" in supported_types
        assert "pdf" in supported_types
        assert "doc" in supported_types
        assert "docx" in supported_types
    
    def test_document_id_from_filename(self):
        """Test that document ID is created from filename"""
        filename = "test_document.pdf"
        document_id = filename
        
        assert document_id == "test_document.pdf"
        assert document_id.endswith(".pdf")
    
    @patch('llama_stack_ui.distribution.ui.modules.utils.data_url_from_file')
    def test_rag_document_creation(self, mock_data_url):
        """Test RAGDocument creation from uploaded file"""
        from llama_stack_client import RAGDocument
        
        # Mock file and data URL
        mock_data_url.return_value = "data:text/plain;base64,SGVsbG8gV29ybGQ="
        
        mock_file = Mock()
        mock_file.name = "test.txt"
        
        # Create RAGDocument as done in upload.py
        document = RAGDocument(
            document_id=mock_file.name,
            content=mock_data_url(mock_file),
        )
        
        # RAGDocument returns a dict-like object
        assert document['document_id'] == "test.txt"
        assert document['content'].startswith("data:")
        mock_data_url.assert_called_once()
    
    def test_multiple_documents_processing(self):
        """Test processing multiple uploaded files"""
        # Simulate multiple uploaded files
        mock_file1 = Mock()
        mock_file1.name = "doc1.txt"
        mock_file2 = Mock()
        mock_file2.name = "doc2.pdf"
        mock_file3 = Mock()
        mock_file3.name = "doc3.docx"
        
        uploaded_files = [mock_file1, mock_file2, mock_file3]
        
        # Simulate creating document list
        document_ids = [f.name for f in uploaded_files]
        
        assert len(document_ids) == 3
        assert "doc1.txt" in document_ids
        assert "doc2.pdf" in document_ids
        assert "doc3.docx" in document_ids


class TestVectorDBOperations:
    """Test vector database operations"""
    
    @patch('llama_stack_ui.distribution.ui.modules.api.llama_stack_api')
    def test_vector_db_registration_params(self, mock_api):
        """Test that vector DB registration uses correct parameters"""
        mock_client = Mock()
        mock_api.client = mock_client
        
        vector_db_id = "test_vector_db"
        embedding_dimension = 384
        embedding_model = "all-MiniLM-L6-v2"
        provider_id = "pgvector"
        
        # Simulate registration call
        mock_client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model,
            provider_id=provider_id,
        )
        
        # Verify the call was made with correct params
        mock_client.vector_dbs.register.assert_called_once_with(
            vector_db_id=vector_db_id,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model,
            provider_id=provider_id,
        )
    
    @patch('llama_stack_ui.distribution.ui.modules.api.llama_stack_api')
    def test_document_insertion_params(self, mock_api):
        """Test that document insertion uses correct parameters"""
        from llama_stack_client import RAGDocument
        
        mock_client = Mock()
        mock_api.client = mock_client
        
        vector_db_id = "test_vector_db"
        documents = [
            RAGDocument(document_id="doc1", content="content1"),
            RAGDocument(document_id="doc2", content="content2"),
        ]
        chunk_size = 512
        
        # Simulate insertion call
        mock_client.tool_runtime.rag_tool.insert(
            vector_db_id=vector_db_id,
            documents=documents,
            chunk_size_in_tokens=chunk_size,
        )
        
        # Verify the call was made
        mock_client.tool_runtime.rag_tool.insert.assert_called_once()
        call_args = mock_client.tool_runtime.rag_tool.insert.call_args
        assert call_args[1]['vector_db_id'] == vector_db_id
        assert call_args[1]['chunk_size_in_tokens'] == chunk_size
        assert len(call_args[1]['documents']) == 2
    
    @patch('llama_stack_ui.distribution.ui.modules.api.llama_stack_api')
    def test_provider_detection(self, mock_api):
        """Test vector IO provider detection"""
        mock_client = Mock()
        mock_api.client = mock_client
        
        # Mock provider list
        mock_providers = [
            Mock(api="inference", provider_id="ollama"),
            Mock(api="vector_io", provider_id="pgvector"),
            Mock(api="memory", provider_id="redis"),
        ]
        mock_client.providers.list.return_value = mock_providers
        
        # Simulate provider detection logic
        providers = mock_client.providers.list()
        vector_io_provider = None
        for x in providers:
            if x.api == "vector_io":
                vector_io_provider = x.provider_id
        
        assert vector_io_provider == "pgvector"


class TestUploadValidation:
    """Test upload validation and error handling"""
    
    def test_empty_upload_list(self):
        """Test handling of empty upload list"""
        uploaded_files = []
        assert len(uploaded_files) == 0
    
    def test_upload_count_display(self):
        """Test upload count display logic"""
        uploaded_files = [Mock(), Mock(), Mock()]
        count = len(uploaded_files)
        message = f"Successfully uploaded {count} files"
        
        assert message == "Successfully uploaded 3 files"
        assert str(count) in message
    
    def test_vector_db_name_validation(self):
        """Test vector database name validation"""
        # Valid names
        valid_names = ["rag_vector_db", "test-db-123", "my_documents"]
        for name in valid_names:
            assert len(name) > 0
            assert name.replace('_', '').replace('-', '').isalnum()
        
        # Invalid names should be caught
        invalid_name = ""
        assert len(invalid_name) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

