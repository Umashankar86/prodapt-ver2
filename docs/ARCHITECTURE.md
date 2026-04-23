# Architecture

## System Overview

The system is organized around a lightweight orchestrator that decides when to use local documents, structured financial data, or live web search.

```mermaid
flowchart LR
    User[User]
    Env[.env and Settings]

    subgraph CLI[CLI entrypoints]
        Ask[ask]
        SearchDocsCmd[search-docs]
        QueryDataCmd[query-data]
        WebSearchCmd[web-search]
        BuildIndexCmd[build-doc-index and upgrade-doc-metadata]
        BuildDbCmd[build-db]
    end

    subgraph Inputs[Source data]
        DocsDir[data/docs]
        StructuredDir[data/structured]
    end

    subgraph DocArtifacts[Document retrieval artifacts]
        GlobalDocIndex[artifacts/docs_index.json]
        DocMetadata[artifacts/docs_index_metadata.json]
        DocStores[artifacts/docs_index_stores]
        PageStores[artifacts/docs_index_page_stores]
        FaissOrNpy[FAISS indexes or NumPy vectors]
    end

    subgraph StructuredArtifacts[Structured data artifact]
        SQLite[artifacts/structured.db]
    end

    subgraph Runtime[Agent runtime]
        Runner[AgentRunner]
        Service[AgentService]
        State[AgentState with subgoals evidence and trace]
        Prompts[Prompt builders]
        Support[agent_support helpers]
        LLM[GeminiClient]
        Cache[artifacts/llm_cache.json optional]
        LlmLog[artifacts/llm_responses.json]
        SearchDocsTool[search_docs]
        QueryDataTool[query_data]
        WebTool[web_search]
        Tavily[Tavily Search API]
        Result[AgentRunResult]
    end

    subgraph Output[User-visible output]
        FinalAnswer[Final answer]
        Citations[Citations and citation map]
        TraceOut[Trace evidence and subgoals]
    end

    User --> Ask
    User --> SearchDocsCmd
    User --> QueryDataCmd
    User --> WebSearchCmd
    Env --> Ask
    Env --> SearchDocsCmd
    Env --> QueryDataCmd
    Env --> WebSearchCmd
    Env --> BuildIndexCmd
    Env --> BuildDbCmd

    DocsDir --> BuildIndexCmd
    BuildIndexCmd --> GlobalDocIndex
    BuildIndexCmd --> DocMetadata
    BuildIndexCmd --> DocStores
    BuildIndexCmd --> PageStores
    BuildIndexCmd --> FaissOrNpy

    StructuredDir --> BuildDbCmd
    BuildDbCmd --> SQLite

    Ask --> Runner
    Runner --> Service
    Service --> State
    Service --> Prompts
    Service --> Support
    Service --> LLM
    LLM --> Cache
    LLM --> LlmLog

    Service --> SearchDocsTool
    Service --> QueryDataTool
    Service --> WebTool

    SearchDocsTool --> GlobalDocIndex
    SearchDocsTool --> DocMetadata
    SearchDocsTool --> DocStores
    SearchDocsTool --> PageStores
    SearchDocsTool --> FaissOrNpy
    QueryDataTool --> SQLite
    WebTool --> Tavily

    SearchDocsCmd --> SearchDocsTool
    QueryDataCmd --> QueryDataTool
    WebSearchCmd --> WebTool

    Service --> Result
    Result --> FinalAnswer
    Result --> Citations
    Result --> TraceOut
    FinalAnswer --> User
    Citations --> User
    TraceOut --> User
```

## Runtime Sequence

```mermaid
sequenceDiagram
    autonumber
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
        else stop
            break
        end

        Note over Agent: Store evidence, update trace, refresh subgoals
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
