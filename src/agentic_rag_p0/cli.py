from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AgentRunner
from .config import load_settings
from .data_tool import build_sqlite_db, get_db_schema, query_data
from .document_tool import build_doc_index, search_docs, upgrade_doc_metadata
from .web_tool import web_search


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def _print_clean_answer(answer: str, plan_summary: str = "", trace: list[dict] | None = None) -> None:
    if plan_summary:
        print("PLAN")
        print("----")
        print(plan_summary)
        print()
    tool_actions = [
        step.get("action", "")
        for step in (trace or [])
        if step.get("action") not in {"answer", "refuse"} and step.get("status") == "ok"
    ]
    if tool_actions:
        print("TOOLS CALLED")
        print("------------")
        print(" -> ".join(tool_actions))
        print()
    print("FINAL ANSWER")
    print("------------")
    print(answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone P0 tools for Agentic RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_doc_index_parser = subparsers.add_parser("build-doc-index")
    build_doc_index_parser.add_argument("--docs-dir", type=Path)
    build_doc_index_parser.add_argument("--index-path", type=Path)

    upgrade_doc_metadata_parser = subparsers.add_parser("upgrade-doc-metadata")
    upgrade_doc_metadata_parser.add_argument("--index-path", type=Path)

    search_docs_parser = subparsers.add_parser("search-docs")
    search_docs_parser.add_argument("query")
    search_docs_parser.add_argument("--index-path", type=Path)
    search_docs_parser.add_argument("--top-k", type=int, default=3)

    build_db_parser = subparsers.add_parser("build-db")
    build_db_parser.add_argument("--structured-dir", type=Path)
    build_db_parser.add_argument("--db-path", type=Path)

    query_data_parser = subparsers.add_parser("query-data")
    query_data_parser.add_argument("sql")
    query_data_parser.add_argument("--db-path", type=Path)

    schema_parser = subparsers.add_parser("db-schema")
    schema_parser.add_argument("--db-path", type=Path)

    web_search_parser = subparsers.add_parser("web-search")
    web_search_parser.add_argument("query")
    web_search_parser.add_argument("--top-k", type=int, default=3)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--answer-only", action="store_true")
    ask_parser.add_argument("--answer-at-end", action="store_true")

    return parser


def main() -> None:
    settings = load_settings()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-doc-index":
        docs_dir = args.docs_dir or settings.docs_dir
        index_path = args.index_path or settings.doc_index_path
        _print_json(build_doc_index(docs_dir=docs_dir, index_path=index_path))
        return

    if args.command == "upgrade-doc-metadata":
        index_path = args.index_path or settings.doc_index_path
        _print_json(upgrade_doc_metadata(index_path=index_path))
        return

    if args.command == "search-docs":
        index_path = args.index_path or settings.doc_index_path
        _print_json(
            {
                "query": args.query,
                "results": [
                    result.to_dict()
                    for result in search_docs(index_path=index_path, query=args.query, top_k=args.top_k)
                ],
            }
        )
        return

    if args.command == "build-db":
        structured_dir = args.structured_dir or settings.structured_dir
        db_path = args.db_path or settings.sqlite_db_path
        _print_json(build_sqlite_db(structured_dir=structured_dir, db_path=db_path))
        return

    if args.command == "query-data":
        db_path = args.db_path or settings.sqlite_db_path
        _print_json(query_data(db_path=db_path, sql=args.sql).to_dict())
        return

    if args.command == "db-schema":
        db_path = args.db_path or settings.sqlite_db_path
        _print_json(get_db_schema(db_path=db_path))
        return

    if args.command == "web-search":
        _print_json(
            {
                "query": args.query,
                "results": [
                    result.to_dict()
                    for result in web_search(
                        query=args.query,
                        tavily_api_key=settings.tavily_api_key,
                        top_k=args.top_k,
                    )
                ],
            }
        )
        return

    if args.command == "ask":
        runner = AgentRunner(settings)
        result = runner.run(args.question)
        if args.answer_only:
            _print_clean_answer(result.final_answer, result.plan_summary, result.trace)
            return
        _print_json(result.to_dict())
        print()
        _print_clean_answer(result.final_answer, result.plan_summary, result.trace)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
