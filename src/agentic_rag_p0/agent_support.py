from __future__ import annotations

import json
import re

from .agent_state import AgentState

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


def has_usable_web_evidence(state: AgentState) -> bool:
    return any(item.source_tool == "web_search" and item.usable for item in state.evidence)
