# Architecture

## System Overview

The system has two halves:

- an offline preparation flow that turns raw PDFs and CSV files into local retrieval artifacts
- a runtime agent flow that decides when to use document search, SQL lookup, or live web search for each question

```mermaid
flowchart LR
    subgraph Prep[Offline Preparation]
        direction TB
        DocsInput[data/docs]
        StructuredInput[data/structured]
        BuildDocs[build-doc-index<br/>upgrade-doc-metadata]
        BuildDb[build-db]
        DocArtifacts[Document artifacts<br/>docs_index.json<br/>docs_index_metadata.json<br/>per-document stores<br/>page-topic stores]
        DbArtifact[SQLite database<br/>artifacts/structured.db]

        DocsInput --> BuildDocs --> DocArtifacts
        StructuredInput --> BuildDb --> DbArtifact
    end

    subgraph Runtime[Runtime Question Answering]
        direction TB
        User[User]
        CLI[CLI entrypoints<br/>ask search-docs query-data web-search]
        Runner[AgentRunner]
        Service[AgentService]
        State[AgentState<br/>subgoals evidence trace]
        LLM[GeminiClient]
        SearchDocs[search_docs]
        QueryData[query_data]
        WebSearch[web_search]
        Tavily[Tavily API]
        Logs[Observability<br/>llm_responses.json<br/>optional llm_cache.json]
        Output[Final answer<br/>citations<br/>trace]

        User --> CLI --> Runner --> Service
        Service --> State
        Service --> LLM --> Logs
        Service --> SearchDocs
        Service --> QueryData
        Service --> WebSearch
        WebSearch --> Tavily
        Service --> Output --> User
    end

    SearchDocs --> DocArtifacts
    QueryData --> DbArtifact
```

### What The Diagram Shows

- `build-doc-index` converts the raw annual-report corpus into one global manifest plus per-document chunk stores and page-topic stores for retrieval.
- `build-db` loads the structured CSV data into SQLite so the agent can issue read-only SQL queries.
- At runtime, the CLI hands the user question to `AgentRunner`, which delegates the full loop to `AgentService`.
- `AgentService` is the orchestrator: it tracks subgoals, evidence, trace state, calls the LLM for planning/routing/composition, and invokes the three tools.
- `search_docs` reads from the built document artifacts, `query_data` reads from SQLite, and `web_search` calls Tavily for recent information.
- Every run produces a final answer plus citations and trace output, while LLM calls are logged to `artifacts/llm_responses.json` and can optionally use `artifacts/llm_cache.json`.

## Runtime Sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Runner as AgentRunner
    participant Agent as AgentService
    participant LLM as GeminiClient
    participant SQL as query_data
    participant Docs as search_docs
    participant Web as web_search
    participant Tavily

    User->>CLI: ask(question)
    CLI->>Runner: run(question)
    Runner->>Agent: run(question)
    Agent->>LLM: understand_question
    LLM-->>Agent: normalized question + mode
    Agent->>LLM: plan
    LLM-->>Agent: subgoals + likely_tools

    loop up to 8 tool calls
        Agent->>LLM: choose_next_action
        LLM-->>Agent: next action
        alt query_data
            Agent->>LLM: build SQL input
            LLM-->>Agent: SQL
            Agent->>SQL: execute query
            SQL-->>Agent: rows/columns
        else search_docs
            Agent->>LLM: build doc query
            LLM-->>Agent: weighted query + filters
            Agent->>Docs: search local corpus
            Docs-->>Agent: top chunks with page refs
        else web_search
            Agent->>LLM: build web query
            LLM-->>Agent: search string
            Agent->>Web: search web
            Web->>Tavily: request results
            Tavily-->>Web: ranked snippets
            Web-->>Agent: snippets + URLs + dates
        else answer or refuse
            LLM-->>Agent: stop tool loop
        end

        Agent->>LLM: check_sufficiency
        LLM-->>Agent: continue or stop
    end

    Agent->>LLM: compose final answer
    LLM-->>Agent: answer + used evidence ids
    Agent-->>CLI: AgentRunResult
    CLI-->>User: answer + trace
```

## Component Roles

### CLI

- Parses commands
- Builds indexes and databases
- Runs tool-only commands
- Runs full agent questions
- Loads runtime settings from `.env`

### AgentService

- Owns the control loop
- Maintains state, subgoals, evidence, and trace
- Decides when to stop
- Applies the 8-step hard cap
- Routes between document search, SQL lookup, and web search

### GeminiClient

- Handles all LLM-facing stages
- Logs every LLM prompt/response pair to `artifacts/llm_responses.json`
- Supports optional prompt-response caching

### `search_docs`

- Uses chunked local annual reports
- Retrieves document passages with filename and page references
- Supports targeted document filters
- Uses the global manifest plus per-document stores and page-topic stores

### `query_data`

- Loads CSV-backed financial data into SQLite
- Executes read-only SQL generated by the agent
- Returns normalized table output

### `web_search`

- Handles live or recent questions
- Returns result snippets, URLs, and dates when available

## Data Artifacts

- `artifacts/docs_index.json`
  Main document index manifest
- `artifacts/docs_index_metadata.json`
  Corpus metadata used in prompt routing
- `artifacts/docs_index_stores/`
  Per-document chunk stores with FAISS or NumPy-backed embeddings
- `artifacts/docs_index_page_stores/`
  Per-document page-topic stores used for page-aware retrieval narrowing
- `artifacts/structured.db`
  SQLite database built from CSV files
- `artifacts/llm_cache.json`
  Optional JSON cache for repeated LLM calls
- `artifacts/llm_responses.json`
  Raw LLM-call log for debugging prompt stages

## Termination and Safety

- Hard cap: 8 tool calls
- Read-only SQL restriction
- Explicit refusal path
- Sufficiency check after every tool call
- Trace returned on every run for post-mortem debugging
