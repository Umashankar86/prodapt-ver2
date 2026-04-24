# Agentic System P0

Agentic  P0 is a small Python retrieval system that combines local document search, structured SQL lookup, and web search behind an LLM-driven planning loop. It is built to answer questions with a visible trail: planned subgoals, tool calls, evidence, citations, and the final response.

The system is intentionally simple to run and inspect. The core agent lives in `src/agentic_rag_p0`, local corpora live under `data/`, and generated indexes, logs, and caches live under `artifacts/`.

Supporting submission docs:

- [DESIGN.md](DESIGN.md)
- [EVALUATION.md](EVALUATION.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Submission Artifacts

- Design document: [DESIGN.md](DESIGN.md)
- Evaluation report: [EVALUATION.md](EVALUATION.md)
- Demo video: [Google Drive demo](https://drive.google.com/file/d/1udB1fpjJp7R6sWpCBp4VLkzQzBrxvTc8/view?usp=sharing)

## What It Does

- Searches local PDFs and text files with page and filename citations.
- Loads CSV files into SQLite and exposes read-only SQL querying.
- Uses Tavily for web search when local evidence is missing or stale.
- Uses Gemini through Vertex AI for question understanding, planning, routing, sufficiency checks, and answer composition.
- Records each run as structured JSON, including trace steps, evidence items, subgoal status, and citation mapping.
- Caps each agent run at 8 tool calls to keep execution bounded.

## Corpus Choice

This repo uses the public-company financials corpus from the assignment:

- Unstructured: annual reports for Infosys, TCS, and Wipro
- Structured: 4 years of financial metrics per company in CSV form
- Web: live search for current market and company information

## Project Layout

```text
data/
  docs/                 # Local PDFs and text documents
  structured/           # CSV files loaded into SQLite
artifacts/
  docs_index.json       # Built document index
  structured.db         # Built SQLite database
  llm_responses.json    # Run logs
  llm_cache.json        # Optional LLM cache
src/agentic_rag_p0/
  agent_service.py      # Main agent loop
  prompt_builders.py    # LLM prompts
  document_tool.py      # Local document indexing/search
  data_tool.py          # SQLite build/query tool
  web_tool.py           # Tavily web search wrapper
  cli.py                # Command line entrypoint
tests/
docs/
  ARCHITECTURE.md      # Architecture notes and diagram
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```env
VERTEX_PROJECT_ID=your-project-id
VERTEX_LOCATION=us-central1
GEMINI_FAST_MODEL=gemini-2.5-flash
GEMINI_PRO_MODEL=gemini-2.5-pro
TAVILY_API_KEY=your-tavily-key

AGENTIC_RAG_DOCS_DIR=data/docs
AGENTIC_RAG_STRUCTURED_DIR=data/structured
AGENTIC_RAG_DOC_INDEX=artifacts/docs_index.json
AGENTIC_RAG_SQLITE_DB=artifacts/structured.db
AGENTIC_RAG_ENABLE_LLM_CACHE=false
AGENTIC_RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

You can also start from:

```bash
cp .env.example .env
```

`pypdf` is included in `requirements.txt`. If `pdftotext` is installed on the machine, the document loader will prefer it for PDF extraction.

## Build Indexes

Build the local document index:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli build-doc-index
```

Build the structured SQLite database from CSV files:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli build-db
```

Inspect the database schema:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli db-schema
```

## Run The Agent

Ask a question and return the full JSON trace plus a readable final answer:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli ask "How did Infosys' and TCS' operating margins compare in FY25, and what drove each?"
```

Return only the plan, tools called, and final answer:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli ask "How did Infosys' and TCS' operating margins compare in FY25, and what drove each?" --answer-only
```

Representative questions:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli ask "What was Infosys' operating margin in FY2025?"
PYTHONPATH=src python3 -m agentic_rag_p0.cli ask "What is the current stock price of Infosys?"
PYTHONPATH=src python3 -m agentic_rag_p0.cli ask "How did Infosys' and TCS' operating margins compare in FY25, and what drove each?"
```

## Use Tools Directly

Search local documents:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli search-docs "TCS operating margin FY2025 utilization productivity realization"
```

Query structured data:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli query-data "SELECT company, fiscal_year, operating_margin_pct FROM public_company_financials_india_it_4yr WHERE fiscal_year = 'FY2025'"
```

Search the web:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli web-search "Infosys FY2025 operating margin"
```

## Agent Flow

The agent follows a compact loop:

1. Normalize and classify the question.
2. Build a plan and subgoals.
3. Choose the next tool.
4. Generate tool input.
5. Run the tool and store evidence.
6. Check whether evidence is sufficient.
7. Continue, answer, or refuse.

The final output includes the normalized question, plan summary, final answer, citations, citation map, trace, subgoals, evidence, tool count, and status.

For the full component view and sequence diagram, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Retrieval Notes

- `search_docs` is for the local unstructured corpus and can target specific document files.
- `query_data` is for known structured facts in CSV-backed SQLite tables.
- `web_search` is a fallback for recent, external, or locally missing information.
- The routing loop prefers local evidence when metadata indicates a relevant local document exists.
- Document and page retrieval use learned SentenceTransformers embeddings, then hybrid ranking emphasizes exact metric/topic terms while keeping generic intent words lighter to reduce noisy retrieval.

## Evaluation

The repository includes an `EVALUATION.md` report and a lightweight evaluation runner in `src/agentic_rag_p0/evaluation.py`.

Run the current unit-test-backed evaluation:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run the evaluation helper against a JSON question set:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from agentic_rag_p0.evaluation import run_evaluation
summary = run_evaluation(
    Path("evaluation/questions.example.json"),
    Path("artifacts/evaluation_summary.json"),
)
print(summary)
PY
```

The expected question-set format is a JSON list like:

```json
[
  {
    "question": "What was Infosys' operating margin in FY2025?",
    "expected_status": "answered"
  },
  {
    "question": "Which company should I invest in?",
    "expected_status": "refused"
  }
]
```

A starter file is included at `evaluation/questions.example.json`.

## Tests

Run the full test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run a focused test file:

```bash
PYTHONPATH=src python3 -m unittest tests.test_doc_loop_guard -v
```

## Development Notes

- Generated artifacts can change during local runs; review them before committing.
- SQL execution is restricted to read-only statements.
- LLM response caching is optional and controlled by `AGENTIC_RAG_ENABLE_LLM_CACHE`.
- The CLI uses `PYTHONPATH=src` because this repo is a lightweight source tree rather than a packaged install.

## Known Failure Modes

- Prompt payloads are large because corpus metadata is embedded into several LLM stages, which can increase latency and occasionally contribute to provider rate-limit failures.
- Multi-tool questions that require `search_docs` depend on the local embedding stack being installed and healthy.
- Refusal behavior is still prompt-driven rather than fully deterministic in code, so borderline out-of-scope questions can vary by model output.
- Web-search answers are only as good as the returned snippets and publication metadata; recent market questions may need manual source review.
