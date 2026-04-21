import json
import re
from .agent_state import AgentState
from .agent_support import confidence_from_score, extract_table_references_from_sql, has_usable_web_evidence, is_simple_structured_question, local_docs_cover_question, needs_commentary, normalize_claim, state_snapshot, summarize_tool_result
from .config import Settings
from .data_tool import get_db_metadata, get_db_schema, query_data
from .document_tool import get_doc_index_metadata, search_docs
from .llm import GeminiClient
from .models import AgentRunResult, EvidenceItem, Subgoal, TraceStep
from .prompt_builders import build_cap_final_answer_prompt, build_choose_next_action_prompt, build_compose_answer_prompt, build_direct_answer_prompt, build_plan_prompt, build_query_data_input_prompt, build_reformulate_tool_input_prompt, build_sufficiency_with_evidence_review_prompt, build_tool_input_prompt, build_understand_question_prompt
from .web_tool import web_search
MAX_TOOL_CALLS = 8
TOOL_DESCRIPTIONS = {"search_docs": "Use when the answer should come from the local unstructured corpus. Input must be a short natural-language retrieval query. Output is top chunks with filename and page citations.", "query_data": "Use when the answer depends on structured CSV-backed tables. Input must be a read-only SQL SELECT/WITH query over the SQLite database. Output is rows, columns, and row_count.", "web_search": "Use only for recent or external information not covered by local docs/data. Input must be under 10 words. Output is snippets with URL and publication date."}
class AgentService:
    max_tool_calls = MAX_TOOL_CALLS; tool_descriptions = TOOL_DESCRIPTIONS; initialize_state = staticmethod(lambda q: AgentState(question=q)); _is_simple_structured_question = staticmethod(is_simple_structured_question)
    def __init__(self, settings: Settings, llm_client: GeminiClient | None = None) -> None: self.settings, self.llm = settings, llm_client or GeminiClient(settings)
    def _meta(self) -> dict[str, object]:
        try: structured = get_db_metadata(self.settings.sqlite_db_path, sample_rows=5)
        except Exception as exc: structured = {"error": str(exc)}; 
        try: documents = get_doc_index_metadata(self.settings.doc_index_path, sample_chunks=5)
        except Exception as exc: documents = {"error": str(exc)}; 
        return {"structured": structured, "documents": documents}
    def understand_question(self, q: str) -> dict: return self.llm.generate_json(build_understand_question_prompt(q, self._meta()), model_id=self.settings.gemini_fast_model)
    def plan(self, q: str, u: dict) -> dict: return self.llm.generate_json(build_plan_prompt(q, u, self._meta(), self.tool_descriptions), model_id=self.settings.gemini_fast_model)
    def hydrate_understanding(self, s: AgentState, u: dict) -> None: s.normalized_question = u.get("normalized_question", s.question).strip() or s.question
    def hydrate_plan(self, s: AgentState, p: dict) -> None:
        s.plan_summary, s.answer_requirements, s.likely_tools, s.risks = p.get("plan_summary", ""), p.get("answer_requirements", []), p.get("likely_tools", []), p.get("risks", [])
        s.subgoals = [Subgoal(**g) for g in p.get("subgoals", [])] or [Subgoal(description="Answer the user's question")]
        if self._is_simple_structured_question(s.normalized_question) and set(s.likely_tools or ["query_data"]) <= {"query_data"} and len(s.subgoals) == 1 and not needs_commentary(s.normalized_question): s.subgoals = [Subgoal(description="Retrieve the needed value(s) from the structured dataset.", status="pending", notes="Single structured lookup should answer this question.")]
    def _open(self, s: AgentState) -> bool: return any(g.status != "done" for g in s.subgoals)
    def choose_next_action(self, s: AgentState) -> dict:
        if needs_commentary(s.normalized_question) and not s.evidence and not local_docs_cover_question(s.normalized_question, self._meta()): return {"action": "web_search", "rationale": "The local corpus metadata does not show clear coverage for this commentary question, so switching to web search.", "refusal_reason": ""}
        if s.local_doc_attempted and s.web_fallback_used and self._open(s) and has_usable_web_evidence(s): return {"action": "answer", "rationale": "Local docs and one web fallback have already been attempted, so finalize using the best remaining grounded evidence.", "refusal_reason": ""}
        d = self.llm.generate_json(build_choose_next_action_prompt(s.normalized_question, s.plan_summary, s.likely_tools, [g.to_dict() for g in s.subgoals], [e.to_dict() for e in s.evidence[-5:]], self._meta(), s.steps_used, self.max_tool_calls), model_id=self.settings.gemini_fast_model)
        return d
    def _format_tool_input(self, tool_input: object) -> str:
        if isinstance(tool_input, dict):
            return json.dumps(tool_input, ensure_ascii=True)
        return str(tool_input)
    def _format_table_cell(self, value: object) -> str:
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", " ").strip()
    def _query_data_markdown_table(self, evidence: EvidenceItem) -> str:
        try:
            payload = json.loads(evidence.raw_result)
        except json.JSONDecodeError:
            return ""
        columns = payload.get("columns", [])
        rows = payload.get("rows", [])
        if not isinstance(columns, list) or not isinstance(rows, list) or not columns or not rows:
            return ""
        header = "| " + " | ".join(self._format_table_cell(col) for col in columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = []
        for row in rows:
            values = row if isinstance(row, list) else [row]
            padded = list(values[: len(columns)]) + [""] * max(0, len(columns) - len(values))
            body.append("| " + " | ".join(self._format_table_cell(value) for value in padded[: len(columns)]) + " |")
        table = "\n".join([header, divider, *body])
        return f"Structured data:\n{table}\n(Note: results truncated.)" if payload.get("truncated") else f"Structured data:\n{table}"
    def _append_structured_tables(self, answer: str, used_ids: set[str], evidence_items: list[EvidenceItem]) -> str:
        if "\n|" in answer:
            return answer
        tables = []
        for evidence in evidence_items:
            if evidence.evidence_id not in used_ids or evidence.source_tool != "query_data":
                continue
            try:
                payload = json.loads(evidence.raw_result)
            except json.JSONDecodeError:
                continue
            columns = payload.get("columns", [])
            row_count = payload.get("row_count", 0)
            if row_count <= 0 or not isinstance(columns, list):
                continue
            if row_count <= 1 and len(columns) <= 1:
                continue
            table = self._query_data_markdown_table(evidence)
            if table:
                tables.append(table)
        if not tables:
            return answer
        suffix = "\n\n".join(tables)
        return f"{answer}\n\n{suffix}" if answer.strip() else suffix
    def _document_filters_for_text(self, text: str, document_catalog: list[dict]) -> list[str]:
        if not text or not document_catalog:
            return []
        lowered = text.lower()
        matches: list[str] = []
        for document in document_catalog:
            if not isinstance(document, dict):
                continue
            filename = str(document.get("filename", "")).strip()
            aliases = self._document_aliases(document)
            if any(alias in lowered for alias in aliases):
                matches.append(filename)
        return matches
    def _search_doc_attempt_counts(self, s: AgentState) -> dict[str, int]:
        counts: dict[str, int] = {}
        for evidence in s.evidence:
            if evidence.source_tool != "search_docs":
                continue
            try:
                payload = json.loads(evidence.tool_input)
            except json.JSONDecodeError:
                continue
            filters = payload.get("document_filters", [])
            if not isinstance(filters, list):
                continue
            for filename in filters:
                key = str(filename).strip().lower()
                if key:
                    counts[key] = counts.get(key, 0) + 1
        return counts
    def _document_filter_for_active_subgoal(self, s: AgentState, document_catalog: list[dict], routing_text: str = "") -> tuple[list[str], str]:
        routing_matches = self._document_filters_for_text(routing_text, document_catalog)
        if len(routing_matches) == 1:
            return routing_matches, routing_text
        candidates: list[tuple[str, str]] = []
        for subgoal in s.subgoals:
            if subgoal.status == "done":
                continue
            matches = self._document_filters_for_text(subgoal.description, document_catalog)
            if len(matches) == 1:
                candidates.append((matches[0], subgoal.description))
        if not candidates:
            label_matches = self._document_filters_for_text(self._label(s), document_catalog)
            return (label_matches, self._label(s)) if len(label_matches) == 1 else ([], "")
        attempt_counts = self._search_doc_attempt_counts(s)
        filename, target_text = min(candidates, key=lambda item: attempt_counts.get(item[0].lower(), 0))
        return [filename], target_text
    def _document_aliases(self, document: dict) -> set[str]:
        filename = str(document.get("filename", "")).strip()
        subject_hint = str(document.get("subject_hint", "")).strip().lower()
        stem_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", filename.lower())
            if len(token) > 1 and token not in {"pdf", "txt", "md", "ar", "annual", "report"}
        }
        return {subject_hint, *stem_tokens} - {""}
    def _targeted_search_query(self, raw_query: str, s: AgentState, filters: list[str], document_catalog: list[dict], routing_text: str = "") -> str:
        label = (routing_text or self._label(s)).strip()
        if not filters or not label:
            return raw_query
        selected_filename = filters[0].lower()
        selected_aliases: set[str] = set()
        other_aliases: set[str] = set()
        for document in document_catalog:
            if not isinstance(document, dict):
                continue
            aliases = self._document_aliases(document)
            if str(document.get("filename", "")).strip().lower() == selected_filename:
                selected_aliases.update(aliases)
            else:
                other_aliases.update(aliases)
        lowered_query = raw_query.lower()
        if other_aliases.intersection(set(re.findall(r"[a-z0-9]+", lowered_query))):
            return label
        subject = sorted(selected_aliases, key=len, reverse=True)[0] if selected_aliases else ""
        if subject and subject not in lowered_query:
            return f"{subject} {raw_query}".strip()
        return raw_query
    def build_tool_input(self, s: AgentState, a: str, action_rationale: str = "") -> object:
        if a == "query_data": return self.llm.generate_json(build_query_data_input_prompt(s.normalized_question, s.plan_summary, [e.to_dict() for e in s.evidence[-3:]], get_db_schema(self.settings.sqlite_db_path), get_db_metadata(self.settings.sqlite_db_path, sample_rows=3)), model_id=self.settings.gemini_fast_model)["tool_input"]
        document_catalog = get_doc_index_metadata(self.settings.doc_index_path).get("documents", []) if a == "search_docs" else None
        built = self.llm.generate_json(build_tool_input_prompt(a, s.normalized_question, s.plan_summary, [e.to_dict() for e in s.evidence[-3:]], self.tool_descriptions, document_catalog=document_catalog if isinstance(document_catalog, list) else None), model_id=self.settings.gemini_fast_model)
        raw = str(built.get("tool_input", "")).strip()
        if a != "search_docs":
            return raw
        filters = built.get("document_filters", [])
        normalized_filters = [str(item).strip() for item in filters if str(item).strip()] if isinstance(filters, list) else []
        if isinstance(document_catalog, list):
            targeted_filters, target_text = self._document_filter_for_active_subgoal(s, document_catalog, action_rationale)
            if targeted_filters:
                normalized_filters = targeted_filters
                raw = self._targeted_search_query(raw, s, normalized_filters, document_catalog, target_text)
        return {"query": raw, "document_filters": normalized_filters}
    def reformulate_tool_input(self, s: AgentState, a: str, previous: object, error: Exception) -> str: return self.llm.generate_json(build_reformulate_tool_input_prompt(a, self._format_tool_input(previous), error, s.normalized_question), model_id=self.settings.gemini_fast_model).get("tool_input", self._format_tool_input(previous))
    def _label(self, s: AgentState) -> str: return next((g.description for g in s.subgoals if g.status != "done"), s.subgoals[0].description if s.subgoals else "Answer the question")
    def _store(self, s: AgentState, a: str, tool_input: object, r: object) -> None:
        formatted_tool_input = self._format_tool_input(tool_input)
        if a == "search_docs":
            for i, item in enumerate(r, 1): s.evidence.append(EvidenceItem(evidence_id=f"{a}-{s.steps_used}-{i}", source_tool=a, source_reference=f'{item["filename"]} p.{item["page_number"]}', summary=item["content"][:240], related_subgoal=self._label(s), tool_input=formatted_tool_input, normalized_claim=normalize_claim(item["content"]), confidence=confidence_from_score(item.get("score", 0.0)), raw_result=json.dumps(item, ensure_ascii=True), usability_flag="pending_llm_review", usable=False))
        elif a == "query_data":
            rows, cols, preview = r.get("rows", []), r.get("columns", []), r.get("rows", [])[:3]; s.evidence.append(EvidenceItem(evidence_id=f"{a}-{s.steps_used}-1", source_tool=a, source_reference=f'{self.settings.sqlite_db_path.name} | {extract_table_references_from_sql(str(tool_input))}', summary=f"Columns={cols}; preview_rows={preview}", related_subgoal=self._label(s), tool_input=formatted_tool_input, normalized_claim=normalize_claim(f"{cols} {preview}"), confidence=0.9 if rows else 0.2, raw_result=json.dumps(r, ensure_ascii=True), usability_flag="usable" if rows else "weak", usable=bool(rows)))
        else:
            for i, item in enumerate(r, 1): date = item.get("published_date"); s.evidence.append(EvidenceItem(evidence_id=f"{a}-{s.steps_used}-{i}", source_tool=a, source_reference=item["url"], summary=f"Web source retrieved from {item['url']}. Published date: {date or 'unknown'}. Snippet: {item['snippet'][:240]}", related_subgoal=self._label(s), tool_input=formatted_tool_input, normalized_claim=normalize_claim(f"Web source {item['url']} {date or 'unknown date'} {item['snippet']}"), confidence=0.7 if date else 0.5, raw_result=json.dumps(item, ensure_ascii=True)))
    def _progress(self, s: AgentState, a: str, r: object) -> None:
        if a != "search_docs": return
        rs = r if isinstance(r, list) else []; sig = "|".join(f'{i.get("filename")}:{i.get("page_number")}:{i.get("chunk_id")}' for i in rs[:3])
        if sig: s.recent_search_signatures = (s.recent_search_signatures + [sig])[-4:]
    def run_tool(self, s: AgentState, a: str, tool_input: object, rationale: str, retry_count: int = 0, *, step_number: int | None = None, count_step: bool = True) -> str:
        before = state_snapshot(s)
        step = step_number if step_number is not None else s.steps_used + 1
        if count_step:
            s.steps_used = max(s.steps_used, step)
        if a == "search_docs":
            query = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
            filters = tool_input.get("document_filters", []) if isinstance(tool_input, dict) else []
            results = [i.to_dict() for i in search_docs(self.settings.doc_index_path, query, filename_filter=filters)]
        else:
            results = query_data(self.settings.sqlite_db_path, str(tool_input)).to_dict() if a == "query_data" else [i.to_dict() for i in web_search(str(tool_input), tavily_api_key=self.settings.tavily_api_key)]
        s.local_doc_attempted, s.web_fallback_used = s.local_doc_attempted or a == "search_docs", s.web_fallback_used or a == "web_search"; summary = summarize_tool_result(a, results); self._store(s, a, tool_input, results); self._progress(s, a, results); s.trace.append(TraceStep(step_number=step, action=a, rationale=rationale, tool_input=self._format_tool_input(tool_input), result_summary=summary, status="ok", retry_count=retry_count, state_before=before, state_after=state_snapshot(s))); return summary
    def log_tool_error(self, s: AgentState, a: str, rationale: str, tool_input: object, error: Exception, retry_count: int = 0, *, step_number: int | None = None) -> None: s.trace.append(TraceStep(step_number=step_number if step_number is not None else s.steps_used, action=a, rationale=rationale, tool_input=self._format_tool_input(tool_input), result_summary=str(error), status="error", retry_count=retry_count, state_before=state_snapshot(s), state_after=state_snapshot(s)))
    def run_tool_with_retry(self, s: AgentState, a: str, rationale: str) -> str:
        tool_input = self.build_tool_input(s, a, rationale)
        step_number = s.steps_used + 1
        exc = None
        try: return self.run_tool(s, a, tool_input, rationale, step_number=step_number, count_step=True)
        except Exception as err: exc = err; self.log_tool_error(s, a, rationale, tool_input, err, step_number=step_number)
        if s.steps_used >= self.max_tool_calls: raise
        return self.run_tool(s, a, self.reformulate_tool_input(s, a, tool_input, exc), f"{rationale} | retry after error", retry_count=1, step_number=step_number, count_step=False)
    def check_sufficiency(self, s: AgentState, a: str, summary: str) -> dict: return self.llm.generate_json(build_sufficiency_with_evidence_review_prompt(s.normalized_question, a, summary, [g.to_dict() for g in s.subgoals], [e.to_dict() for e in s.evidence[-6:]], self._meta()), model_id=self.settings.gemini_fast_model)
    def apply_evidence_updates(self, s: AgentState, updates: list[dict]) -> None:
        by_id = {e.evidence_id: e for e in s.evidence}
        for update in updates or []:
            if not isinstance(update, dict):
                continue
            evidence = by_id.get(str(update.get("evidence_id", "")))
            if evidence is None:
                continue
            evidence.usable = bool(update.get("usable", False))
            evidence.usability_flag = str(update.get("usability_flag") or ("llm_usable" if evidence.usable else "llm_rejected"))
            related_subgoal = str(update.get("related_subgoal", "")).strip()
            if related_subgoal:
                evidence.related_subgoal = related_subgoal
                #here starts the agent loop 
    def apply_subgoal_updates(self, s: AgentState, updates: list[dict]) -> None:
        by = {g.description: g for g in s.subgoals}
        for u in updates or []:
            if u.get("description") in by: by[u["description"]].status, by[u["description"]].notes = u.get("status", by[u["description"]].status), u.get("notes", by[u["description"]].notes)
    def should_answer_early(self, s: AgentState) -> bool: return bool(s.evidence) and not (needs_commentary(s.normalized_question) and self._open(s)) and (all(g.status == "done" for g in s.subgoals) or (self._is_simple_structured_question(s.normalized_question) and any(e.source_tool == "query_data" and e.usable for e in s.evidence)))
    def compose_answer(self, s: AgentState, *, direct_answer: bool) -> str:
        if direct_answer: return self.llm.generate_text(build_direct_answer_prompt(s.normalized_question), model_id=self.settings.gemini_fast_model)
        composed = self.llm.generate_json(build_compose_answer_prompt(s.normalized_question, s.plan_summary, [e.to_dict() for e in s.evidence if e.usable]), model_id=self.settings.gemini_pro_model); used = set(composed.get("used_evidence_ids", [])); s.citations = [e.source_reference for e in s.evidence if e.evidence_id in used] or sorted({e.source_reference for e in s.evidence if e.usable}); return self._append_structured_tables(composed.get("answer", ""), used, s.evidence)
    def finalize_after_cap(self, s: AgentState) -> None:
        usable = [e.to_dict() for e in s.evidence if e.usable]
        if not usable:
            s.status, s.final_answer = "refused", "Stopped after reaching the hard limit of 8 tool calls without enough usable evidence to answer."
            return
        composed = self.llm.generate_json(build_cap_final_answer_prompt(s.normalized_question, s.plan_summary, [g.to_dict() for g in s.subgoals], usable), model_id=self.settings.gemini_pro_model)
        outcome = str(composed.get("outcome", "refuse")).strip().lower()
        used = set(composed.get("used_evidence_ids", []))
        if outcome in {"answer", "partial"} and str(composed.get("answer", "")).strip():
            s.status = "answered"
            s.citations = [e.source_reference for e in s.evidence if e.evidence_id in used] or sorted({e.source_reference for e in s.evidence if e.usable})
            s.final_answer = self._append_structured_tables(str(composed.get("answer", "")), used, s.evidence)
            return
        s.status, s.final_answer = "refused", str(composed.get("answer", "")).strip() or "Stopped after reaching the hard limit of 8 tool calls before enough relevant evidence was gathered."
    def finalize(self, s: AgentState) -> AgentRunResult: return AgentRunResult(question=s.question, normalized_question=s.normalized_question or s.question, plan_summary=s.plan_summary, final_answer=s.final_answer, citations=s.citations or sorted({e.source_reference for e in s.evidence}), citation_map=[{"evidence_id": e.evidence_id, "source_reference": e.source_reference, "related_subgoal": e.related_subgoal, "usable": e.usable} for e in s.evidence], steps_used=s.steps_used, status=s.status, trace=[t.to_dict() for t in s.trace], subgoals=[g.to_dict() for g in s.subgoals], evidence=[e.to_dict() for e in s.evidence])
    def run(self, question: str):
        s = self.initialize_state(question); u = self.understand_question(question); self.hydrate_understanding(s, u)
        if u.get("mode") == "refusal": s.status = "refused"; s.plan_summary = u.get("reason", "Question is out of scope for the available tools."); s.final_answer = u.get("reason", "I cannot answer that with the available tools."); return self.finalize(s)
        p = self.plan(s.normalized_question, u); self.hydrate_plan(s, p)
        if u.get("mode") == "no_tool": s.status, s.final_answer = "answered", self.compose_answer(s, direct_answer=True); return self.finalize(s)
        while s.steps_used < self.max_tool_calls:
            d = {"action": s.forced_next_action, "rationale": s.forced_next_rationale, "refusal_reason": ""} if s.forced_next_action else self.choose_next_action(s); s.forced_next_action, s.forced_next_rationale = "", ""; a, r = d.get("action", "answer"), d.get("rationale", "")
            if a in {"answer", "refuse"}: s.status = "answered" if a == "answer" else "refused"; s.final_answer = s.final_answer or d.get("refusal_reason", "Insufficient evidence to answer safely."); break
            try: summary = self.run_tool_with_retry(s, a, r)
            except Exception: 
                if s.steps_used >= self.max_tool_calls: s.status, s.final_answer = "refused", "Stopped after repeated tool failures and reaching the 8-call limit."; break
                continue
            suff = self.check_sufficiency(s, a, summary); self.apply_evidence_updates(s, suff.get("evidence_updates", [])); self.apply_subgoal_updates(s, suff.get("subgoals", [])); early = self.should_answer_early(s); outcome = "continue" if suff.get("outcome", "continue") == "sufficient" and not early else suff.get("outcome", "continue")
            if outcome == "continue" and suff.get("next_action") in {"search_docs", "query_data", "web_search"}:
                next_action = suff.get("next_action", "")
                s.forced_next_action, s.forced_next_rationale = next_action, suff.get("reason", "")
            if outcome == "sufficient" or early: s.status = "answered"; break
            if outcome == "refuse": s.status, s.final_answer = "refused", suff.get("reason", "I could not gather enough grounded evidence to answer."); break
        if s.steps_used >= self.max_tool_calls and s.status == "running": self.finalize_after_cap(s)
        if s.status == "answered" and not s.final_answer: s.final_answer = self.compose_answer(s, direct_answer=False)
        return self.finalize(s)
#loops ending here it uses agent support to run the loop
