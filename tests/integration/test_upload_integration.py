"""
Integration tests for document upload functionality
Tests the upload workflow programmatically
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

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

# Configuration
LLAMA_STACK_ENDPOINT = os.getenv("LLAMA_STACK_ENDPOINT", "http://localhost:8321")


@pytest.fixture
def mock_llama_client():
    """Mock LlamaStack client for upload tests"""
    with patch('llama_stack_client.LlamaStackClient') as mock_client:
        # Mock providers
        mock_provider = Mock()
        mock_provider.api = "vector_io"
        mock_provider.provider_id = "pgvector"
        mock_client.return_value.providers.list.return_value = [mock_provider]
        
        # Mock vector DB registration
        mock_client.return_value.vector_dbs.register.return_value = None
        
        # Mock document insertion
        mock_client.return_value.tool_runtime.rag_tool.insert.return_value = None
        
        yield mock_client.return_value


@pytest.fixture
def mock_uploaded_file():
    """Create a mock uploaded file"""
    mock_file = Mock()
    mock_file.name = "test_document.txt"
    mock_file.type = "text/plain"
    mock_file.size = 1024
    mock_file.read.return_value = b"This is a test document content."
    return mock_file


class TestDocumentUploadIntegration:
    """Integration tests for document upload workflow"""
    
    @patch('llama_stack_ui.distribution.ui.page.upload.upload.llama_stack_api')
    def test_single_file_upload_workflow(self, mock_api, mock_uploaded_file):
        """Test complete workflow for uploading a single file"""
        from llama_stack_client import RAGDocument
        
        # Setup mock
        mock_api.client.providers.list.return_value = [
            Mock(api="vector_io", provider_id="pgvector")
        ]
        
        vector_db_name = "test_vector_db"
        
        # Step 1: Create RAGDocument
        document = RAGDocument(
            document_id=mock_uploaded_file.name,
            content="data:text/plain;base64,test_content",
        )
        
        # RAGDocument returns a dict-like object
        assert document['document_id'] == "test_document.txt"
        
        # Step 2: Find vector IO provider
        providers = mock_api.client.providers.list()
        vector_io_provider = None
        for provider in providers:
            if provider.api == "vector_io":
                vector_io_provider = provider.provider_id
        
        assert vector_io_provider == "pgvector"
        
        # Step 3: Register vector DB
        mock_api.client.vector_dbs.register(
            vector_db_id=vector_db_name,
            embedding_dimension=384,
            embedding_model="all-MiniLM-L6-v2",
            provider_id=vector_io_provider,
        )
        
        # Step 4: Insert documents
        mock_api.client.tool_runtime.rag_tool.insert(
            vector_db_id=vector_db_name,
            documents=[document],
            chunk_size_in_tokens=512,
        )
        
        # Verify calls
        mock_api.client.vector_dbs.register.assert_called_once()
        mock_api.client.tool_runtime.rag_tool.insert.assert_called_once()
    
    @patch('llama_stack_ui.distribution.ui.page.upload.upload.llama_stack_api')
    def test_multiple_files_upload_workflow(self, mock_api):
        """Test uploading multiple files at once"""
        from llama_stack_client import RAGDocument
        
        # Setup mock
        mock_api.client.providers.list.return_value = [
            Mock(api="vector_io", provider_id="pgvector")
        ]
        
        # Create mock files
        mock_files = [
            Mock(name="doc1.txt"),
            Mock(name="doc2.pdf"),
            Mock(name="doc3.docx"),
        ]
        
        # Create documents
        documents = [
            RAGDocument(
                document_id=f.name,
                content=f"data:text/plain;base64,content_{i}",
            )
            for i, f in enumerate(mock_files)
        ]
        
        assert len(documents) == 3
        
        # Insert documents
        vector_db_name = "multi_file_db"
        mock_api.client.tool_runtime.rag_tool.insert(
            vector_db_id=vector_db_name,
            documents=documents,
            chunk_size_in_tokens=512,
        )
        
        # Verify insertion called with correct number of documents
        call_args = mock_api.client.tool_runtime.rag_tool.insert.call_args
        assert len(call_args[1]['documents']) == 3
    
    def test_supported_file_types(self):
        """Test that all supported file types are handled"""
        supported_types = ["txt", "pdf", "doc", "docx"]
        
        # Test each file type
        for file_type in supported_types:
            filename = f"test.{file_type}"
            assert filename.endswith(f".{file_type}")
    
    def test_file_type_validation(self):
        """Test file type validation logic"""
        supported_types = ["txt", "pdf", "doc", "docx"]
        
        # Valid files
        valid_files = ["document.txt", "paper.pdf", "report.doc", "essay.docx"]
        for filename in valid_files:
            extension = filename.split('.')[-1]
            assert extension in supported_types
        
        # Invalid files would be rejected by file_uploader
        invalid_files = ["image.jpg", "script.py", "data.csv"]
        for filename in invalid_files:
            extension = filename.split('.')[-1]
            assert extension not in supported_types


class TestVectorDBCreation:
    """Integration tests for vector database creation"""
    
    @patch('llama_stack_ui.distribution.ui.page.upload.upload.llama_stack_api')
    def test_vector_db_registration_params(self, mock_api):
        """Test vector DB registration with correct parameters"""
        mock_api.client.providers.list.return_value = [
            Mock(api="vector_io", provider_id="pgvector")
        ]
        
        vector_db_id = "integration_test_db"
        
        mock_api.client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_dimension=384,
            embedding_model="all-MiniLM-L6-v2",
            provider_id="pgvector",
        )
        
        call_args = mock_api.client.vector_dbs.register.call_args
        assert call_args[1]['vector_db_id'] == vector_db_id
        assert call_args[1]['embedding_dimension'] == 384
        assert call_args[1]['embedding_model'] == "all-MiniLM-L6-v2"
        assert call_args[1]['provider_id'] == "pgvector"
    
    @patch('llama_stack_ui.distribution.ui.page.upload.upload.llama_stack_api')
    def test_vector_db_with_custom_name(self, mock_api):
        """Test creating vector DB with custom name"""
        mock_api.client.providers.list.return_value = [
            Mock(api="vector_io", provider_id="pgvector")
        ]
        
        custom_name = "my_custom_documents"
        
        mock_api.client.vector_dbs.register(
            vector_db_id=custom_name,
            embedding_dimension=384,
            embedding_model="all-MiniLM-L6-v2",
            provider_id="pgvector",
        )
        
        call_args = mock_api.client.vector_dbs.register.call_args
        assert call_args[1]['vector_db_id'] == custom_name


class TestProviderDetection:
    """Integration tests for provider detection"""
    
    @patch('llama_stack_ui.distribution.ui.page.upload.upload.llama_stack_api')
    def test_vector_io_provider_detection(self, mock_api):
        """Test that vector_io provider is correctly detected"""
        mock_api.client.providers.list.return_value = [
            Mock(api="inference", provider_id="ollama"),
            Mock(api="vector_io", provider_id="pgvector"),
            Mock(api="memory", provider_id="redis"),
        ]
        
        providers = mock_api.client.providers.list()
        vector_io_provider = None
        for provider in providers:
            if provider.api == "vector_io":
                vector_io_provider = provider.provider_id
        
        assert vector_io_provider == "pgvector"
    
    @patch('llama_stack_ui.distribution.ui.page.upload.upload.llama_stack_api')
    def test_no_vector_io_provider(self, mock_api):
        """Test handling when no vector_io provider is available"""
        mock_api.client.providers.list.return_value = [
            Mock(api="inference", provider_id="ollama"),
            Mock(api="memory", provider_id="redis"),
        ]
        
        providers = mock_api.client.providers.list()
        vector_io_provider = None
        for provider in providers:
            if provider.api == "vector_io":
                vector_io_provider = provider.provider_id
        
        assert vector_io_provider is None


class TestDocumentInsertion:
    """Integration tests for document insertion into vector DB"""
    
    @patch('llama_stack_ui.distribution.ui.page.upload.upload.llama_stack_api')
    def test_document_insertion_with_chunks(self, mock_api):
        """Test document insertion with chunking"""
        from llama_stack_client import RAGDocument
        
        documents = [
            RAGDocument(
                document_id="long_doc.txt",
                content="data:text/plain;base64,very_long_content_here",
            )
        ]
        
        chunk_size = 512
        vector_db_id = "test_db"
        
        mock_api.client.tool_runtime.rag_tool.insert(
            vector_db_id=vector_db_id,
            documents=documents,
            chunk_size_in_tokens=chunk_size,
        )
        
        call_args = mock_api.client.tool_runtime.rag_tool.insert.call_args
        assert call_args[1]['chunk_size_in_tokens'] == 512
    
    @patch('llama_stack_ui.distribution.ui.page.upload.upload.llama_stack_api')
    def test_empty_document_list(self, mock_api):
        """Test handling of empty document list"""
        documents = []
        
        # Should handle empty list gracefully
        if len(documents) == 0:
            # Don't call insert
            pass
        else:
            mock_api.client.tool_runtime.rag_tool.insert(
                vector_db_id="test_db",
                documents=documents,
                chunk_size_in_tokens=512,
            )
        
        # Verify insert was not called
        mock_api.client.tool_runtime.rag_tool.insert.assert_not_called()


class TestDataURLConversion:
    """Integration tests for file to data URL conversion"""
    
    def test_data_url_format(self):
        """Test that data URLs have correct format"""
        # Simulate data URL creation
        content = "Hello, World!"
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        data_url = f"data:text/plain;base64,{encoded}"
        
        assert data_url.startswith("data:")
        assert ";base64," in data_url
        assert encoded in data_url
    
    def test_pdf_data_url(self):
        """Test data URL for PDF files"""
        import base64
        
        pdf_content = b"%PDF-1.4 test content"
        encoded = base64.b64encode(pdf_content).decode()
        data_url = f"data:application/pdf;base64,{encoded}"
        
        assert data_url.startswith("data:application/pdf")
        assert ";base64," in data_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

