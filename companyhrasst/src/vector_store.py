import os
from pathlib import Path
from typing import List, Optional

# Embeddings import with fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

# FAISS Vector Store import with fallback
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

from langchain_core.documents import Document
from src.config import EMBEDDING_MODEL_NAME, VECTOR_DB_DIR, DATA_DIR, TOP_K_RESULTS
from src.document_loader import load_documents_from_directory, split_documents

class VectorStoreManager:
    """Manages creation, loading, saving, and querying of FAISS vector store."""
    
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store: Optional[FAISS] = None

    def build_from_directory(self, data_directory: Path = DATA_DIR) -> FAISS:
        """Load documents from data directory and build FAISS index."""
        documents = load_documents_from_directory(data_directory)
        if not documents:
            raise ValueError(f"No documents found in {data_directory}")
            
        chunks = split_documents(documents)
        print(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks.")
        
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.save_local(VECTOR_DB_DIR)
        return self.vector_store

    def save_local(self, target_dir: Path = VECTOR_DB_DIR):
        """Save FAISS index to local directory."""
        if self.vector_store:
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(target_dir))
            print(f"Vector database saved to {target_dir}")

    def load_local(self, target_dir: Path = VECTOR_DB_DIR) -> bool:
        """Load FAISS index from local directory if present."""
        target_dir = Path(target_dir)
        index_file = target_dir / "index.faiss"
        if index_file.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(target_dir), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Successfully loaded existing FAISS index.")
                return True
            except Exception as e:
                print(f"Error loading existing index: {e}")
                return False
        return False

    def get_or_create_vector_store(self, data_directory: Path = DATA_DIR) -> FAISS:
        """Load existing vector store or build a new one if not available."""
        if self.vector_store is not None:
            return self.vector_store
            
        if self.load_local(VECTOR_DB_DIR):
            return self.vector_store
            
        print("Building fresh vector store...")
        return self.build_from_directory(data_directory)

    def similarity_search(self, query: str, k: int = TOP_K_RESULTS) -> List[Document]:
        """Perform similarity search for query."""
        if self.vector_store is None:
            self.get_or_create_vector_store()
        return self.vector_store.similarity_search(query, k=k)
