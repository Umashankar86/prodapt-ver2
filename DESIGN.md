# Design Document

## Goal

This project implements a small agentic RAG system over a mixed corpus:

- local annual reports for Infosys, TCS, and Wipro
- structured financial data in SQLite
- live web search for current or recent information

The design target is not maximum autonomy. The target is an explainable loop that can be read top-to-bottom, traced after every run, and stopped safely when evidence is weak or the tool budget is exhausted.

## Implemented Project Features

### Mixed-corpus question answering

- The agent can answer from three sources in one workflow: local annual reports, CSV-backed SQLite tables, and live web results.
- It supports single-tool questions such as a direct financial metric lookup, and multi-tool questions such as margin comparisons plus management commentary.
- It also supports a refusal path for out-of-scope requests such as investment advice.

### Local document retrieval features

- The document index supports `.pdf`, `.txt`, and `.md` files.
- PDFs are read page-by-page, chunked into overlapping windows, and stored with page numbers for citations.
- Indexing produces a global manifest plus per-document embedding containers, so each PDF or text document also has its own stored chunk index.
- The build also creates per-document page-topic stores so the system can reason over page-level topical summaries in addition to raw chunks.
- Depending on the environment, embeddings are stored in FAISS indexes or NumPy vector files, with matching JSON manifests for chunk metadata.
- Retrieval uses learned embeddings plus additional lexical/profile signals so exact finance terms and years stay important.
- Each chunk is enriched with metadata such as section type, temporal markers, subject hints, commentary score, and boilerplate score.
- The agent can target a specific local document when the question or active subgoal clearly maps to one company report.
- Search queries are weighted so must-have terms, should-have terms, and contextual terms influence ranking differently.

### Structured data features

- CSV files are automatically loaded into SQLite with inferred column types.
- SQL execution is restricted to a single read-only `SELECT` or `WITH` statement.
- Write operations and unsafe statements such as `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `ATTACH`, and `PRAGMA` are blocked.
- Large query results are truncated to a fixed maximum row count to keep outputs bounded.
- When structured evidence is used in the final answer, the agent can append a compact Markdown table for clarity.

### Web retrieval features

- Web search is reserved for recent or external information not covered by the local corpus.
- Queries are intentionally kept short to improve retrieval quality.
- Returned results are re-ranked to prefer higher-quality sources such as investor-relations pages, filings, releases, presentations, and PDF documents.
- Social and weak-source domains are de-prioritized.
- For non-recent questions, the agent tries an unsearched local document before falling back to the web.

### Agent orchestration features

- The loop performs question understanding, planning, tool selection, tool-input generation, evidence storage, sufficiency checking, and final answer composition.
- The planner tracks subgoals, likely tools, answer requirements, and risks.
- A simple structured-question shortcut lets the agent answer straightforward metrics with less unnecessary looping.
- Tool calls can be retried once with a reformulated input after an execution failure.
- Sufficiency checks can mark evidence as usable or weak and can force the next tool action when the next step is obvious.

### Grounding and observability

- Every run returns the normalized question, plan summary, final answer, citations, citation map, subgoals, evidence, trace steps, and final status.
- Evidence is stored with tool input, source reference, normalized claim, confidence, and usability status.
- Each LLM call is logged to `artifacts/llm_responses.json` with prompt stage, duration, prompt preview, response preview, and error status when relevant.
- Optional JSON caching is available for LLM responses to make repeated runs cheaper and easier to inspect.

### Safety and bounded behavior

- The agent has a hard limit of 8 tool calls enforced in code.
- It can stop early when enough grounded evidence has been collected.
- If the tool budget is exhausted, it produces a bounded answer or refusal instead of guessing.
- Final citations are tied to the actual evidence ids used during answer composition.

## Tool Schemas

### `search_docs`

- Purpose: semantic search over the local unstructured corpus
- Input: natural-language query string
- Output: top relevant chunks with `filename`, `page_number`, `chunk_id`, `content`, and score
- Why it exists: management commentary, MD&A explanations, and document-grounded qualitative answers do not fit well in a structured table

### `query_data`

- Purpose: read-only lookup over CSV-backed SQLite tables
- Input: generated SQL `SELECT` or `WITH` query
- Output: `columns`, `rows`, `row_count`, and truncation flag
- Why it exists: numeric comparisons, trends, and year-by-year tabular questions are most reliable from structured data

### `web_search`

- Purpose: live retrieval for current or external information
- Input: short search query string
- Output: top web result snippets with URL and publication date when available
- Why it exists: recent market moves, current prices, and post-report developments are outside the local corpus

## Agent Loop

The agent loop is implemented directly in `src/agentic_rag_p0/agent_service.py`.

1. `understand_question`
   The LLM normalizes the question and classifies whether it looks like a tool-using query, a direct-answer case, or a refusal.
2. `plan`
   The LLM creates subgoals, likely tools, and answer requirements.
3. `choose_next_action`
   The LLM decides the next action from `search_docs`, `query_data`, `web_search`, `answer`, or `refuse`.
4. `build_tool_input`
   A tool-specific prompt generates SQL, a retrieval query, or a web query.
5. `run_tool`
   The selected tool executes and returns normalized output.
6. `store evidence`
   Results are appended to the evidence list with source references and summary text.
7. `check_sufficiency`
   The LLM decides whether the current evidence is enough, whether another tool is needed, or whether the question should be refused.
8. `compose_answer`
   The final composer uses only gathered evidence and returns answer text plus the evidence ids used.

## Trace and Observability

Every run returns:

- normalized question
- plan summary
- final answer
- citations
- citation map
- trace steps
- subgoals
- evidence
- total steps used
- final status

In addition, `artifacts/llm_responses.json` records raw LLM calls with prompt and response metadata. This is the primary debugging source for prompt-stage failures and routing mistakes.

## Infinite-Loop Prevention

The loop is capped at 8 tool calls.

- The cap is enforced in code, not only in prompts.
- The agent can stop early on `answer` or `refuse`.
- Tool retries are bounded.
- Sufficiency checks are used after each successful tool call to avoid unnecessary extra steps.
- If the cap is reached, the agent returns either a bounded partial/refusal-style outcome instead of guessing.

## Citation Strategy

The answer composer cites only evidence ids that were actually used.

- `search_docs` citations point to document filename and page number
- `query_data` citations point to the SQLite source and referenced table
- `web_search` citations point to the source URL

This keeps the final answer grounded in concrete retrieved evidence rather than generic tool names.

## Known Limitations

- Prompt sizes are large because metadata is passed through several planning stages.
- Refusal behavior is prompt-driven, so some borderline questions may not classify perfectly.
- Multi-tool document questions depend on the local embedding environment being available.
- Web snippets can be incomplete or noisy for fast-moving market topics.
