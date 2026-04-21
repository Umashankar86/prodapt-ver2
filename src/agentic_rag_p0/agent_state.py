from __future__ import annotations

from dataclasses import dataclass, field

from .models import EvidenceItem, Subgoal, TraceStep


@dataclass
class AgentState:
    question: str
    normalized_question: str = ""
    plan_summary: str = ""
    answer_requirements: list[str] = field(default_factory=list)
    subgoals: list[Subgoal] = field(default_factory=list)
    likely_tools: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    continuity_summary: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    steps_used: int = 0
    status: str = "running"
    final_answer: str = ""
    citations: list[str] = field(default_factory=list)
    recent_search_signatures: list[str] = field(default_factory=list)
    local_doc_attempted: bool = False
    free_web_redirect_available: bool = False
    web_fallback_used: bool = False
    forced_next_action: str = ""
    forced_next_rationale: str = ""
