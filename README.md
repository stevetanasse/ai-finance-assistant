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

## Command-Line Utilities

### `src/rag/pipeline/rag_pipeline.py`

Orchestrates the offline RAG pipeline: downloading source URLs, scraping them to plain
text, chunking the text, and embedding the chunks into a Qdrant collection. Each
`--action` runs that stage and every stage before it, reusing cached results from prior
runs unless `--force-refresh` is set.

```bash
uv run python -m src.rag.pipeline.rag_pipeline \
  --cache-path rag_caches \
  --action embed \
  --url-path urls.txt \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --dense-embed bge-small-en-v1.5 \
  --sparse-embed bm42
```

| Argument | Required | Description |
|---|---|---|
| `--cache-path` | Yes | Parent folder that will contain the pipeline's cache (downloaded HTML, scraped text, chunks, and Qdrant storage). |
| `--action` | Yes | Pipeline stage to run: `download`, `scrape`, `chunk`, or `embed`. Each stage implies all prior stages (e.g. `embed` also downloads, scrapes, and chunks). |
| `--url-path` | Yes | A single URL, or a path to a text file with one URL per line (blank lines and lines starting with `#` are ignored). |
| `--chunk-size` | Only for `chunk`/`embed` | Number of characters per chunk. |
| `--chunk-overlap` | Only for `chunk`/`embed` | Number of overlapping characters between consecutive chunks. Must be less than `--chunk-size`. |
| `--dense-embed` | Only for `embed` | Dense embedding model name. Valid values: `bge-small`, `bge-small-en-v1.5`. |
| `--sparse-embed` | Only for `embed` | Sparse embedding model name. Valid values: `bm42`. |
| `--force-refresh` | No | Re-download and re-scrape URLs even if already cached (default: off). |
| `--verbose` | No | Print per-URL progress and the resolved Qdrant collection name to stdout (default: off). |

Exits with code `0` on success and `1` on any failure (missing/invalid arguments, a
URL file that doesn't exist or is empty, or an unhandled error during processing).

### `evals/runner.py`

Runs the golden evaluation dataset (`evals/evaluation_dataset.json`) against the
compiled LangGraph application as a LangSmith Experiment, scoring each example with
three evaluators (`answer_correctness`, `refusal_correctness`, `citation_presence`) and
printing a summary of per-metric averages and any zero-score failures.

```bash
uv run python evals/runner.py --experiment-prefix finance-assistant --collection fin_c500_o50_bge-small_bm42
```

| Argument | Required | Description |
|---|---|---|
| `--experiment-prefix` | No | Prefix for the LangSmith Experiment name (default: `finance-assistant`). |
| `--collection` | No | Qdrant collection name to query. Defaults to the name derived from `config.yaml` when omitted. |

Requires `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and
`LANGSMITH_TRACING` to be set (e.g. via `.env`); exits early with an error if any are
missing. See [evals/README.md](evals/README.md) for the dataset schema and metric
definitions.

### `run_workflow.py`

Interactive command-line REPL for the LangGraph application — type a question, get a
response, repeat. Useful for manually exercising the graph without starting the
FastAPI/Streamlit stack. All turns share a single fixed thread ID for the life of the
process, so conversational context is preserved until you exit.

```bash
uv run python run_workflow.py --collection fin_c500_o50_bge-small_bm42
```

| Argument | Required | Description |
|---|---|---|
| `--collection` | No | Qdrant collection name to query. Defaults to the name derived from `config.yaml` when omitted. |

Type `quit` at the prompt to exit. After each response, the assistant also prints the
running `call_counts` dict (how many times each graph node has executed so far).
