# CrewAI Document Insights Workflow

A learning project demonstrating agentic workflow with CrewAI for analyzing PDF documents with charts and generating strategic insights.
This project uses two CrewAI agents working sequentially:
1. **Document Analyzer Agent** - Extracts key information from PDFs including chart data
2. **Insights Generator Agent** - Synthesizes findings into actionable insights


## Quick Start

Get up and running:

```bash
# 1. Navigate to project directory

# 2. Create and activate virtual environment
pyenv local 3.12
pyenv virtualenv 3.12 agentic_workflow
pyenv activate agentic_workflow

# 3. Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure OpenAI API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Run the workflow
python main.py
```
