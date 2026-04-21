# Evaluation

## Current coverage

The local test suite validates the following categories:

- Single-tool document retrieval and grounded answer composition
- Multi-tool reasoning across structured data and documents
- Retry and reformulation after a failed SQL generation
- Direct-answer path for trivial no-tool questions
- Refusal path for out-of-scope recommendation requests
- Structured-query safety for multi-statement blocking
- Structured-query truncation for large result sets
- Document indexing and lexical retrieval sanity

Run it with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Latest local result

- 8 tests run
- 8 tests passed

## Failure analysis

Observed risk areas that are not fully exercised by the local fake-LLM tests:

- Live Gemini outputs may differ from the strict JSON shapes used in tests.
- Live Tavily results may be noisier or missing publication dates.
- Citation quality depends on the quality of retrieved chunks and SQL generated from your real schema.
- The current retry logic reformulates once; very noisy corpora may need stronger query rewriting later.

## What still needs live validation

- End-to-end run against your actual document corpus
- End-to-end run against your actual CSV schema
- End-to-end run with Vertex Gemini credentials configured
- End-to-end run with Tavily configured for recent-information questions
