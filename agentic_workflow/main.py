import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from src.crew import create_document_insights_crew


def extract_pdf_content(pdf_path):
    """Extract text and metadata from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text_content = []

        # Extract text from each page
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                text_content.append(f"--- Page {page_num} ---\n{page_text}")

        # Check for images/charts
        num_images = sum(len(page.images) for page in reader.pages)

        return {
            "filename": os.path.basename(pdf_path),
            "num_pages": len(reader.pages),
            "num_images": num_images,
            "content": "\n\n".join(text_content)
        }
    except Exception as e:
        return {
            "filename": os.path.basename(pdf_path),
            "error": str(e)
        }


def load_documents(documents_dir):
    """Load all PDF documents from the documents directory."""
    documents_dir = Path(documents_dir)
    pdf_files = list(documents_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {documents_dir}. "
            "Please download the sample PDFs as described in the README."
        )

    documents = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file.name}...")
        doc_data = extract_pdf_content(pdf_file)
        documents.append(doc_data)

    return documents


def format_documents_for_analysis(documents):
    """Format extracted documents for agent analysis."""
    formatted = []

    for doc in documents:
        if "error" in doc:
            formatted.append(
                f"Document: {doc['filename']}\n"
                f"Error: {doc['error']}\n"
            )
        else:
            formatted.append(
                f"Document: {doc['filename']}\n"
                f"Pages: {doc['num_pages']}\n"
                f"Contains {doc['num_images']} images/charts\n\n"
                f"Content:\n{doc['content']}\n"
                f"{'=' * 80}\n"
            )

    return "\n\n".join(formatted)


def main():
    """Main entry point for the document insights workflow."""
    # Load environment variables
    load_dotenv()

    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please copy .env.example to .env and add your API key."
        )

    # Setup paths
    project_root = Path(__file__).parent
    documents_dir = project_root / "documents"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("CrewAI Document Insights Workflow")
    print("=" * 80)

    # Load documents
    print("\n1. Loading PDF documents...")
    try:
        documents = load_documents(documents_dir)
        print(f"   Loaded {len(documents)} documents")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        return

    # Format documents for analysis
    print("\n2. Preparing documents for analysis...")
    documents_content = format_documents_for_analysis(documents)

    # Create and run crew
    print("\n3. Starting CrewAI workflow...\n")
    crew = create_document_insights_crew(documents_content)

    try:
        result = crew.kickoff()

        # Save results
        print("\n4. Saving results...")
        output_file = output_dir / "insights.txt"
        with open(output_file, "w") as f:
            f.write(str(result))

        print(f"   Results saved to: {output_file}")
        print("\n" + "=" * 80)
        print("Workflow completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during workflow execution: {e}")
        raise


if __name__ == "__main__":
    main()
