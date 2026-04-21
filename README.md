# Agentic RAG P0

This repository now covers a solid P0, the core P1 loop, and key P2 quality features for the CIT Agentic RAG assignment. It gives you three standalone tools and a Gemini-driven agent runner that can choose tools, gather evidence, retry when needed, and return a traced answer.

- `search_docs`: lexical search over local documents with filename and page citations
- `query_data`: read-only SQL over CSV-backed SQLite tables
- `web_search`: live web search through Tavily with normalized output
- `ask`: natural-language question entrypoint backed by Vertex AI Gemini

## What this P0 includes

- Pure Python CLI for indexing and tool testing
- Vertex Gemini orchestration for question understanding, planning, routing, and answer composition
- Document chunking with source metadata
- CSV-to-SQLite loader for structured data
- Read-only SQL guardrails
- LLM prompt caching
- Retry and reformulation for failed tool calls
- Evidence ledger with confidence and source mapping
- Consistent JSON outputs for all three tools
- Trace output with plan, steps, subgoals, evidence, and citations
- Repo hygiene files for a clean handoff into P1

## Expected project layout

```text
data/
  docs/
    report_1.pdf
    notes.txt
  structured/
    financials.csv
    company_metadata.csv
artifacts/
src/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`pypdf` is optional if your machine already has `pdftotext`. The code prefers `pdftotext` for PDF extraction and falls back to `pypdf` when available.

Fill in Vertex and Tavily values in `.env`:

```env
VERTEX_PROJECT_ID=project-1f82cdb3-be94-4df0-acd
VERTEX_LOCATION=us-central1
GEMINI_FAST_MODEL=gemini-2.5-flash
GEMINI_PRO_MODEL=gemini-2.5-pro
TAVILY_API_KEY=your_key_here
```

## Build the document index

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli build-doc-index
```

Override paths if needed:

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli build-doc-index \
  --docs-dir /absolute/path/to/docs \
  --index-path /absolute/path/to/docs_index.json
```

## Build the structured database

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli build-db
```

## Test each tool

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli search-docs "operating margin FY24"
PYTHONPATH=src python3 -m agentic_rag_p0.cli query-data "SELECT company, revenue FROM financials LIMIT 5"
PYTHONPATH=src python3 -m agentic_rag_p0.cli db-schema
PYTHONPATH=src python3 -m agentic_rag_p0.cli web-search "Infosys stock price"
```

## Run the P1 agent

```bash
PYTHONPATH=src python3 -m agentic_rag_p0.cli ask "How did Infosys operating margin compare with TCS in FY24, and what reason did each company give?"
```

The `ask` command returns structured JSON with:

- normalized question
- plan summary
- final answer
- citations
- steps used
- trace entries
- subgoals
- evidence pool
- citation map

## Tool contracts

### `search_docs`

- Input: natural-language query string
- Output: top chunks with `content`, `filename`, `page_number`, `chunk_id`, `score`
- Responsibility: search only the local unstructured corpus

### `query_data`

- Input: read-only SQL query
- Output: `columns`, `rows`, `row_count`
- Responsibility: answer only from structured CSV-backed SQLite tables

### `web_search`

- Input: short search string under 10 words
- Output: top results with `title`, `snippet`, `url`, `published_date`
- Responsibility: fetch only recent external information

## P1 flow

The P1 loop is implemented in plain Python and keeps the control flow readable:

1. Question understanding
2. Planning and subgoal creation
3. Next-action selection
4. Tool execution
5. Evidence recording
6. Sufficiency check
7. Final answer or refusal

The loop enforces a hard cap of 8 tool calls.

## P2 upgrades

- Evidence items carry normalized claims, confidence, raw result snapshots, and subgoal links.
- Final answers use evidence IDs to derive exact citations.
- Failed tool calls trigger one reformulation/retry path before the run gives up.
- LLM generations are cached to `artifacts/llm_cache.json`.
- Structured queries are single-statement, read-only, and truncated to 200 rows.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Notes

- `query_data` intentionally blocks non-read-only SQL.
- `web_search` uses Tavily when `TAVILY_API_KEY` is set.
- The P1 agent uses Vertex AI Gemini through `google-genai`.
- If you want better semantic retrieval later, you can swap the document scorer without changing the CLI contract.
# prodapt-ver2
