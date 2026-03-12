#!/usr/bin/env python3

import argparse
import sys
import os
import shutil
from config import Config
from src.loaders.document_loader import load_pdf, load_txt
from src.loaders.api_loader import load_from_api
from src.vectorstore.vector_store import (
    create_vectorstore,
    load_vectorstore,
    add_documents,
    save_vectorstore
)
from src.rag.chain import create_rag_chain, query


def ingest_documents(args):
    """Ingest documents into the vector store"""
    try:
        # Validate configuration
        Config.validate()

        documents = []

        # Load documents based on input type
        if args.pdf:
            print(f"Loading PDF: {args.pdf}")
            documents = load_pdf(args.pdf)
            print(f"Loaded {len(documents)} pages from PDF")

        elif args.txt:
            print(f"Loading text file: {args.txt}")
            documents = load_txt(args.txt)
            print(f"Loaded text file")

        elif args.api:
            print(f"Fetching data from API: {args.api}")
            documents = load_from_api(args.api)
            print(f"Fetched data from API")

        else:
            print("Error: Please specify --pdf, --txt, or --api")
            return

        if not documents:
            print("No documents loaded")
            return

        # Load or create vector store
        print("Processing documents...")
        vectorstore = load_vectorstore()

        if vectorstore is None:
            print("Creating new vector store...")
            vectorstore = create_vectorstore(documents)
        else:
            print("Adding to existing vector store...")
            vectorstore = add_documents(vectorstore, documents)

        # Save vector store
        print("Saving vector store...")
        save_vectorstore(vectorstore)

        print(f"Successfully ingested {len(documents)} document(s)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)


def query_documents(args):
    """Query the vector store"""
    try:
        # Validate configuration
        Config.validate()

        # Load vector store
        print("Loading vector store...")
        vectorstore = load_vectorstore()

        if vectorstore is None:
            print("Error: No vector store found. Please ingest documents first using the 'ingest' command.")
            sys.exit(1)

        # Create RAG chain
        print("Initializing RAG chain...")
        qa_chain = create_rag_chain(vectorstore)

        # Query
        print(f"\nQuestion: {args.question}\n")
        response = query(qa_chain, args.question)

        # Display answer
        print("Answer:")
        print(response["result"])

        # Display sources
        if response.get("source_documents"):
            print("\n" + "="*50)
            print("Sources:")
            for i, doc in enumerate(response["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page")
                print(f"\n{i}. Source: {source}")
                if page is not None:
                    print(f"   Page: {page + 1}")
                print(f"   Content: {doc.page_content[:200]}...")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during query: {e}")
        sys.exit(1)


def reset_vectorstore(args):
    """Reset the vector store"""
    try:
        if os.path.exists(Config.VECTOR_STORE_PATH):
            print(f"Removing vector store at {Config.VECTOR_STORE_PATH}...")
            shutil.rmtree(Config.VECTOR_STORE_PATH)
            print("Vector store reset successfully")
        else:
            print("No vector store found to reset")

    except Exception as e:
        print(f"Error resetting vector store: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Retrieval Augmented Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Ingest a PDF:
    python main.py ingest --pdf ./data/pdfs/document.pdf

  Ingest a text file:
    python main.py ingest --txt ./data/texts/notes.txt

  Ingest from API:
    python main.py ingest --api https://api.example.com/data

  Query documents:
    python main.py query "What are the key findings?"

  Reset vector store:
    python main.py reset
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--pdf", type=str, help="Path to PDF file")
    ingest_group.add_argument("--txt", type=str, help="Path to text file")
    ingest_group.add_argument("--api", type=str, help="API endpoint URL")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", type=str, help="Question to ask")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset the vector store")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Route to appropriate handler
    if args.command == "ingest":
        ingest_documents(args)
    elif args.command == "query":
        query_documents(args)
    elif args.command == "reset":
        reset_vectorstore(args)


if __name__ == "__main__":
    main()
