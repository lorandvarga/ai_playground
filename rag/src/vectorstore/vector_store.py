import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import Config


def create_vectorstore(documents: List[Document]) -> FAISS:
    """
    Create a new FAISS vector store from documents.

    Args:
        documents: List of Document objects to index

    Returns:
        FAISS vector store instance

    Raises:
        ValueError: If documents list is empty
        Exception: If there's an error creating the vector store
    """
    
    if not documents:
        raise ValueError("Cannot create vector store from empty document list")

    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY
        )

        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)

        return vectorstore

    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")


def load_vectorstore() -> Optional[FAISS]:
    """
    Load existing FAISS vector store from disk.

    Returns:
        FAISS vector store instance if it exists, None otherwise

    Raises:
        Exception: If there's an error loading the vector store
    """
    try:
        if not os.path.exists(Config.VECTOR_STORE_PATH):
            return None

        # Check if the index file exists
        index_file = os.path.join(Config.VECTOR_STORE_PATH, "index.faiss")
        if not os.path.exists(index_file):
            return None

        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY
        )

        # Load vector store
        vectorstore = FAISS.load_local(
            Config.VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception as e:
        raise Exception(f"Error loading vector store: {str(e)}")


def add_documents(vectorstore: FAISS, documents: List[Document]) -> FAISS:
    """
    Add new documents to an existing vector store.

    Args:
        vectorstore: Existing FAISS vector store
        documents: List of Document objects to add

    Returns:
        Updated FAISS vector store instance

    Raises:
        ValueError: If documents list is empty
        Exception: If there's an error adding documents
    """
    if not documents:
        raise ValueError("Cannot add empty document list to vector store")

    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        # Add documents to vector store
        vectorstore.add_documents(splits)

        return vectorstore

    except Exception as e:
        raise Exception(f"Error adding documents to vector store: {str(e)}")


def save_vectorstore(vectorstore: FAISS) -> None:
    """
    Persist FAISS vector store to disk.

    Args:
        vectorstore: FAISS vector store to save

    Raises:
        Exception: If there's an error saving the vector store
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(Config.VECTOR_STORE_PATH, exist_ok=True)

        # Save vector store
        vectorstore.save_local(Config.VECTOR_STORE_PATH)

    except Exception as e:
        raise Exception(f"Error saving vector store: {str(e)}")


def get_retriever(vectorstore: FAISS, k: int = 4):
    """
    Create a retriever from the vector store for similarity search.

    Args:
        vectorstore: FAISS vector store
        k: Number of documents to retrieve

    Returns:
        Retriever instance configured for similarity search
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})
