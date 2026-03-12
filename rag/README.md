# RAG Pipeline - Retrieval Augmented Generation

A command-line RAG (Retrieval Augmented Generation) pipeline built with LangChain that enables semantic search and question-answering over multiple data sources including PDF documents, text files, and REST API data.

## Quick Start

Get up and running:

```bash
# 1. Navigate to project directory

# 2. Create and activate virtual environment
pyenv local 3.12
pyenv virtualenv 3.12 rag
pyenv activate rag

# 3. Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure OpenAI API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Try the sample text file
python3 main.py reset
python3 main.py ingest --txt ./examples/sample.txt
python3 main.py query "What is machine learning?"

# 6. Try the sample pdf file
python3 main.py reset
python3 main.py ingest --pdf ./examples/paper.pdf
python3 main.py query "What is the main contribution of this paper?"
python3 main.py query "What is the vanishing gradient problem?"
python3 main.py query "How deep were the networks tested in this paper?"
python3 main.py query "Who is Albert Einstein?"

# 7. Try the api:
python3 main.py reset
python3 main.py ingest --api https://jsonplaceholder.typicode.com/users/1
python3 main.py query "What is the name of the poster?"
```
