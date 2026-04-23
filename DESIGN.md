# Design Document

## Goal

This project implements a small agentic RAG system over a mixed corpus:

- local annual reports for Infosys, TCS, and Wipro
- structured financial data in SQLite
- live web search for current or recent information

The design target is not maximum autonomy. The target is an explainable loop that can be read top-to-bottom, traced after every run, and stopped safely when evidence is weak or the tool budget is exhausted.

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
