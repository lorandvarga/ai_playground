from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of Document objects.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of Document objects with content and metadata

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If there's an error parsing the PDF
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading PDF file {file_path}: {str(e)}")


def load_txt(file_path: str) -> List[Document]:
    """
    Load a text file and return a list of Document objects.

    Args:
        file_path: Path to the text file

    Returns:
        List of Document objects with content and metadata

    Raises:
        FileNotFoundError: If the text file doesn't exist
        Exception: If there's an error loading the text file
    """
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        raise FileNotFoundError(f"Text file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading text file {file_path}: {str(e)}")
