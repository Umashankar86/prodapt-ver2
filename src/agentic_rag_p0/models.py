from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DocChunk:
    chunk_id: str
    filename: str
    page_number: int | None
    content: str
    token_count: int
    doc_id: str = ""
    chunk_index: int = 0
    section_title: str = ""
    section_type: str = "document_text"
    char_start: int = 0
    char_end: int = 0
    companies_mentioned: list[str] | None = None
    years_mentioned: list[str] | None = None
    metrics_mentioned: list[str] | None = None
    commentary_score: float = 0.0
    boilerplate_score: float = 0.0
    contains_financial_statement: bool = False
    contains_management_commentary: bool = False
    contains_forward_looking: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocSearchResult:
    chunk_id: str
    filename: str
    page_number: int | None
    content: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[object]]
    row_count: int
    truncated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WebResult:
    title: str
    snippet: str
    url: str
    published_date: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Subgoal:
    description: str
    status: str = "pending"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceItem:
    evidence_id: str
    source_tool: str
    source_reference: str
    summary: str
    related_subgoal: str
    tool_input: str = ""
    normalized_claim: str = ""
    confidence: float = 0.5
    raw_result: str = ""
    usability_flag: str = "usable"
    usable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TraceStep:
    step_number: int
    action: str
    rationale: str
    tool_input: str
    result_summary: str
    status: str
    retry_count: int = 0
    state_before: str = ""
    state_after: str = ""
    tool_skipped_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentRunResult:
    question: str
    normalized_question: str
    plan_summary: str
    final_answer: str
    citations: list[str]
    citation_map: list[dict]
    steps_used: int
    status: str
    trace: list[dict]
    subgoals: list[dict]
    evidence: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)
