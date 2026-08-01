import os
from typing import List
from pathlib import Path

# Langchain imports with fallbacks
try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
except ImportError:
    from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_documents_from_directory(directory_path: Path) -> List[Document]:
    """Load all txt, md, and pdf documents from a directory."""
    documents = []
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"Directory {directory_path} does not exist.")
        return []

    # Process .md and .txt files
    for file_path in directory_path.glob("**/*"):
        if file_path.suffix.lower() in [".md", ".txt"]:
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source_filename"] = file_path.name
                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        elif file_path.suffix.lower() == ".pdf":
            try:
                loader = PyPDFLoader(str(file_path))
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source_filename"] = file_path.name
                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading PDF {file_path}: {e}")
                
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
