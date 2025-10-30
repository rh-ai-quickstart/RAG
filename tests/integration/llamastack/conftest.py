"""
Pytest fixtures for LlamaStack integration tests
"""
import os
import pytest
import requests
import time
from openai import OpenAI
from llama_stack_client import LlamaStackClient


# Configuration
LLAMA_STACK_ENDPOINT = os.getenv("LLAMA_STACK_ENDPOINT", "http://localhost:8321")
RAG_UI_ENDPOINT = os.getenv("RAG_UI_ENDPOINT", "http://localhost:8501")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
SKIP_MODEL_TESTS = os.getenv("SKIP_MODEL_TESTS", "auto").lower()
MAX_RETRIES = 30
RETRY_DELAY = 10


def wait_for_endpoint(url, name, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Wait for an endpoint to become available"""
    print(f"⏳ Waiting for {name} to be ready at {url}...")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 404]:  # 404 is ok for some endpoints
                print(f"✅ {name} is ready! (attempt {attempt + 1}/{max_retries})")
                return True
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"   Attempt {attempt + 1}/{max_retries} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"{name} not ready after {max_retries} attempts: {str(e)}")
    return False


@pytest.fixture(scope="session")
def llama_stack_endpoint():
    """Llama Stack API endpoint"""
    return LLAMA_STACK_ENDPOINT


@pytest.fixture(scope="session")
def rag_ui_endpoint():
    """RAG UI endpoint"""
    return RAG_UI_ENDPOINT


@pytest.fixture(scope="session")
def client(llama_stack_endpoint):
    """
    OpenAI-compatible client for Llama Stack
    This is used by test_user_workflow.py
    """
    # Wait for Llama Stack to be ready
    wait_for_endpoint(llama_stack_endpoint, "Llama Stack API")
    
    # Initialize OpenAI client pointing to Llama Stack
    return OpenAI(
        base_url=f"{llama_stack_endpoint}/v1",
        api_key="not-needed"  # Llama Stack doesn't require API key by default
    )


@pytest.fixture(scope="session")
def llama_stack_client(llama_stack_endpoint):
    """
    Native LlamaStackClient
    This is used by test_rag_with_vectordb.py
    """
    # Wait for Llama Stack to be ready
    wait_for_endpoint(llama_stack_endpoint, "Llama Stack API")
    
    return LlamaStackClient(base_url=llama_stack_endpoint)


@pytest.fixture(scope="session")
def model_id():
    """Model ID to use for inference tests"""
    return INFERENCE_MODEL


@pytest.fixture(scope="session")
def skip_inference(client, model_id):
    """
    Determine if we should skip inference tests based on model availability
    """
    if SKIP_MODEL_TESTS == "true":
        print("\n⚠️  Skipping model tests (SKIP_MODEL_TESTS=true)")
        return True
    elif SKIP_MODEL_TESTS == "false":
        print("\n✅ Running model tests (SKIP_MODEL_TESTS=false)")
        return False
    
    # Auto-detect mode: check if model is available
    print(f"\n🔍 Auto-detecting model availability...")
    print(f"   Model: {model_id}")
    
    try:
        # Try a simple completion to check if model works
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print(f"   ✅ Model {model_id} is available and working")
        return False
    except Exception as e:
        print(f"   ⚠️  Model {model_id} not available: {e}")
        print(f"   Skipping inference tests")
        return True


@pytest.fixture(scope="session")
def vector_db_id(llama_stack_client):
    """
    Create and return a test vector database ID
    Used by test_rag_with_vectordb.py
    """
    from llama_stack_client.types import Document as RAGDocument
    
    vector_db_id = "e2e-test-db"
    
    print(f"\n📚 Setting up vector database: {vector_db_id}")
    
    try:
        # Register vector database
        print("   Registering vector DB...")
        llama_stack_client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_dimension=384,  # all-MiniLM-L6-v2 dimension
            embedding_model="all-MiniLM-L6-v2",
            provider_id="pgvector"
        )
        print("   ✓ Vector DB registered")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"   ℹ️  Vector DB already exists, reusing...")
        else:
            print(f"   ⚠️  Vector DB registration error: {e}")
    
    # Sample documents for testing RAG
    sample_documents = [
        {
            "id": "doc-1",
            "content": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. The tower is 330 metres tall and was completed in 1889.",
            "metadata": {"source": "test-data", "topic": "landmarks"}
        },
        {
            "id": "doc-2", 
            "content": "Python is a high-level, interpreted programming language with dynamic semantics. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant indentation.",
            "metadata": {"source": "test-data", "topic": "programming"}
        },
        {
            "id": "doc-3",
            "content": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was built to protect Chinese states against invasions. Construction began in the 7th century BC and continued for over 2000 years.",
            "metadata": {"source": "test-data", "topic": "landmarks"}
        },
        {
            "id": "doc-4",
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "metadata": {"source": "test-data", "topic": "technology"}
        },
    ]
    
    # Prepare documents
    documents = [
        RAGDocument(
            document_id=doc["id"],
            content=doc["content"],
            mime_type="text/plain",
            metadata=doc["metadata"]
        )
        for doc in sample_documents
    ]
    
    print(f"   Inserting {len(documents)} test documents...")
    try:
        llama_stack_client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=512,
        )
        print(f"   ✓ Inserted {len(documents)} documents successfully")
    except Exception as e:
        print(f"   ⚠️  Insert warning: {e}")
        print("   Continuing with query tests...")
    
    return vector_db_id

