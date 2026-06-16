# AI Finance Assistant

## Overview

<!-- TODO: Describe the project purpose and key capabilities -->

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd ai-finance-assistant

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and populate environment variables
cp .env.example .env
```

## Usage

<!-- TODO: Describe how to run the application -->

```bash
streamlit run src/web_app/app.py
```

## Architecture

<!-- TODO: Describe the system architecture, agents, RAG pipeline, and data flow -->

| Component | Description |
|-----------|-------------|
| `src/agents/` | LangGraph agent definitions |
| `src/core/` | Core LLM and model utilities |
| `src/data/` | Data loading and processing |
| `src/rag/` | Retrieval-Augmented Generation pipeline |
| `src/web_app/` | Streamlit web interface |
| `src/utils/` | Shared utilities |
| `src/workflow/` | LangGraph workflow orchestration |

## Testing

```bash
pytest tests/ -v --cov=src
```

<!-- TODO: Describe test strategy and how to add new tests -->

## Running the Application

### Quick start (two terminals)

**Terminal 1 — Start the API service:**
```bash
uv run uvicorn api.service:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Start the Streamlit UI:**
```bash
uv run streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

### Prerequisites

The Qdrant collection must be populated before running the service. If starting fresh:
```bash
uv run python -m src.rag.pipeline.rag_pipeline --action embed
```

Refer to the RAG pipeline documentation for full embedding options.

### Optional: override the Qdrant collection at runtime
```bash
COLLECTION_NAME=fin_c250_o25_bge-small_bm42 uv run uvicorn api.service:app --port 8000
```
