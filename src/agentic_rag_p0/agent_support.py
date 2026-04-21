from __future__ import annotations

import json
import re

from .agent_state import AgentState
QUERY_STOPWORDS = {
    "what", "which", "who", "whom", "when", "where", "why", "how", "did", "does", "do", "give", "gave",
    "provide", "provided", "for", "from", "with", "that", "this", "its", "their", "there", "about", "into",
    "the", "a", "an", "in", "on", "of", "to", "and", "or", "by",
}


def summarize_tool_result(action: str, results: object) -> str:
    serialized = json.dumps(results, ensure_ascii=True)
    if len(serialized) <= 500:
        return serialized
    return f"{action} returned {len(serialized)} chars of structured output."


def normalize_claim(text: str) -> str:
    return " ".join(text.split())[:180]


def confidence_from_score(score: float) -> float:
    return max(0.1, min(0.95, round(0.35 + score / 10, 2)))


def extract_table_references_from_sql(sql: str) -> str:
    table_names = re.findall(r"\bfrom\s+([a-zA-Z0-9_]+)|\bjoin\s+([a-zA-Z0-9_]+)", sql, re.IGNORECASE)
    flattened = sorted({name for pair in table_names for name in pair if name})
    return ",".join(flattened) if flattened else "query_result"


def state_snapshot(state: AgentState) -> str:
    pending = [subgoal.description for subgoal in state.subgoals if subgoal.status != "done"]
    return f"status={state.status}; pending={pending}; evidence={len(state.evidence)}"


def is_simple_structured_question(question: str) -> bool:
    lowered = question.lower()
    metric_keywords = [
        "operating margin",
        "revenue",
        "net profit",
        "eps",
        "headcount",
        "compare",
        "highest",
        "lowest",
        "what was",
    ]
    doc_keywords = ["reason", "why", "commentary", "explain", "management said", "drove", "drive", "driver", "drivers", "influence", "influenced", "factor", "factors"]
    return any(keyword in lowered for keyword in metric_keywords) and not any(
        keyword in lowered for keyword in doc_keywords
    )


def needs_commentary(question: str) -> bool:
    lowered = question.lower()
    commentary_keywords = ["reason", "why", "explain", "management said", "drove", "drive", "driver", "drivers", "influence", "influenced", "factor", "factors"]
    return any(keyword in lowered for keyword in commentary_keywords)


def local_docs_cover_question(question: str, corpus_metadata: dict[str, object]) -> bool:
    documents = corpus_metadata.get("documents", {}).get("documents", []) if isinstance(corpus_metadata, dict) else []
    if not isinstance(documents, list) or not documents:
        return False
    strong_entities = {
        token.lower() for token in re.findall(r"[A-Za-z0-9_]+", question) if token.isupper() and len(token) > 1
    }
    wanted = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", question)
        if len(token) > 2 and token.lower() not in QUERY_STOPWORDS | {"reason", "reasons", "commentary", "management"}
    }
    if not wanted:
        return True
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        haystack = " ".join(
            str(part)
            for key in ("filename", "subject_hint", "keyword_coverage", "metrics_mentioned", "temporal_markers")
            for part in ([doc.get(key)] if not isinstance(doc.get(key), list) else doc.get(key))
        ).lower()
        doc_tokens = set(re.findall(r"[a-z0-9_]+", haystack))
        if strong_entities and not strong_entities.intersection(doc_tokens):
            continue
        if wanted.intersection(doc_tokens):
            return True
    return False


def has_usable_web_evidence(state: AgentState) -> bool:
    return any(item.source_tool == "web_search" and item.usable for item in state.evidence)
