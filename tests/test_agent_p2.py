from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_p0.agent import AgentRunner
from agentic_rag_p0.config import Settings
from agentic_rag_p0.data_tool import build_sqlite_db
from agentic_rag_p0.document_tool import build_doc_index


class FakeLLMClient:
    def __init__(self) -> None:
        self.sql_attempts = 0

    def generate_text(self, prompt: str, **_: object) -> str:
        if "without calling tools" in prompt:
            return "The answer is 4."
        raise AssertionError(f"Unexpected text prompt: {prompt}")

    def generate_json(self, prompt: str, **_: object) -> dict:
        question = self._extract_question(prompt)

        if "question-understanding stage" in prompt:
            if "recommend" in question.lower():
                return {
                    "normalized_question": question,
                    "entities": [],
                    "timeframe": "",
                    "recent_intent": False,
                    "mode": "refusal",
                    "reason": "Recommendation requests are out of scope.",
                }
            if "2+2" in question:
                return {
                    "normalized_question": question,
                    "entities": [],
                    "timeframe": "",
                    "recent_intent": False,
                    "mode": "no_tool",
                    "reason": "",
                }
            if "Compare Alpha revenue" in question:
                mode = "multi-tool"
            else:
                mode = "single-tool"
            return {
                "normalized_question": question,
                "entities": ["Alpha"],
                "timeframe": "FY24",
                "recent_intent": False,
                "mode": mode,
                "reason": "",
            }

        if "planning stage" in prompt:
            if "Compare Alpha revenue" in question:
                return {
                    "plan_summary": "Get the revenue from structured data, then fetch management commentary from docs.",
                    "answer_requirements": ["revenue", "commentary"],
                    "subgoals": [
                        {"description": "Find Alpha revenue", "status": "pending", "notes": ""},
                        {"description": "Find Alpha commentary", "status": "pending", "notes": ""},
                    ],
                    "likely_tools": ["query_data", "search_docs"],
                    "risks": [],
                }
            return {
                "plan_summary": "Search the local docs for the relevant commentary.",
                "answer_requirements": ["commentary"],
                "subgoals": [
                    {"description": "Find Alpha commentary", "status": "pending", "notes": ""}
                ],
                "likely_tools": ["search_docs"],
                "risks": [],
            }

        if "action selector" in prompt:
            steps = self._extract_steps_used(prompt)
            if "Compare Alpha revenue" in question:
                return {
                    "action": "query_data" if steps == 0 else "search_docs" if steps == 1 else "answer",
                    "rationale": "Need structured value first." if steps == 0 else "Need commentary next." if steps == 1 else "Evidence is sufficient.",
                    "refusal_reason": "",
                }
            if "Need repaired SQL" in question:
                return {
                    "action": "query_data" if steps < 2 else "answer",
                    "rationale": "Need the structured value.",
                    "refusal_reason": "",
                }
            return {
                "action": "search_docs" if steps == 0 else "answer",
                "rationale": "Docs contain the answer.",
                "refusal_reason": "",
            }

        if "Generate one read-only SQLite query" in prompt:
            if "Need repaired SQL" in question and self.sql_attempts == 0:
                self.sql_attempts += 1
                return {"tool_input": "SELECT missing_column FROM financials"}
            return {"tool_input": "SELECT company, revenue FROM financials WHERE company = 'Alpha' AND year = 2024"}

        if "Generate the input for the tool `search_docs`" in prompt:
            if "Compare Alpha revenue" in question:
                return {"tool_input": "Alpha CEO commentary automation"}
            return {"tool_input": "Alpha margins automation"}

        if "Repair or tighten the tool input" in prompt:
            return {"tool_input": "SELECT company, revenue FROM financials WHERE company = 'Alpha' AND year = 2024"}

        if "sufficiency checker" in prompt:
            if "Compare Alpha revenue" in question:
                if "Last action: query_data" in prompt:
                    return {
                        "outcome": "continue",
                        "reason": "Need commentary from docs.",
                        "subgoals": [
                            {"description": "Find Alpha revenue", "status": "done", "notes": "Revenue found in data."},
                            {"description": "Find Alpha commentary", "status": "partial", "notes": "Still need commentary."},
                        ],
                    }
                return {
                    "outcome": "sufficient",
                    "reason": "Both revenue and commentary are covered.",
                    "subgoals": [
                        {"description": "Find Alpha revenue", "status": "done", "notes": "Revenue found in data."},
                        {"description": "Find Alpha commentary", "status": "done", "notes": "Commentary found in docs."},
                    ],
                }
            return {
                "outcome": "sufficient",
                "reason": "Enough evidence collected.",
                "subgoals": [
                    {"description": "Find Alpha commentary", "status": "done", "notes": "Commentary found."}
                ],
            }

        if "final answer composer" in prompt:
            evidence = json.loads(prompt.split("Evidence:\n", 1)[1])
            evidence_ids = [item["evidence_id"] for item in evidence]
            if "Compare Alpha revenue" in question:
                return {
                    "answer": "Alpha reported revenue of 100, and management attributed margin strength to automation.",
                    "used_evidence_ids": evidence_ids,
                }
            if "Need repaired SQL" in question:
                return {
                    "answer": "Alpha revenue is 100.",
                    "used_evidence_ids": evidence_ids,
                }
            return {
                "answer": "Alpha said margins improved due to automation.",
                "used_evidence_ids": evidence_ids,
            }

        raise AssertionError(f"Unexpected JSON prompt: {prompt}")

    @staticmethod
    def _extract_question(prompt: str) -> str:
        marker = "Question:"
        if marker not in prompt:
            return ""
        return prompt.rsplit(marker, 1)[1].strip().splitlines()[0].strip()

    @staticmethod
    def _extract_steps_used(prompt: str) -> int:
        marker = "Steps used:"
        if marker not in prompt:
            return 0
        return int(prompt.rsplit(marker, 1)[1].strip().splitlines()[0])


class AgentP2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        docs_dir = root / "docs"
        structured_dir = root / "structured"
        docs_dir.mkdir()
        structured_dir.mkdir()
        (docs_dir / "alpha_report.txt").write_text(
            "Alpha management said margins improved due to automation and delivery discipline.",
            encoding="utf-8",
        )
        (structured_dir / "financials.csv").write_text(
            "company,year,revenue\nAlpha,2024,100\nBeta,2024,80\n",
            encoding="utf-8",
        )
        doc_index = root / "artifacts" / "docs_index.json"
        sqlite_db = root / "artifacts" / "structured.db"
        build_doc_index(docs_dir, doc_index)
        build_sqlite_db(structured_dir, sqlite_db)
        self.settings = Settings(
            docs_dir=docs_dir,
            structured_dir=structured_dir,
            doc_index_path=doc_index,
            sqlite_db_path=sqlite_db,
            llm_cache_path=root / "artifacts" / "llm_cache.json",
            tavily_api_key=None,
            vertex_project_id="test-project",
            vertex_location="us-central1",
            gemini_fast_model="fake-fast",
            gemini_pro_model="fake-pro",
            continuity_enabled=False,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_single_tool_doc_answer(self) -> None:
        runner = AgentRunner(self.settings, llm_client=FakeLLMClient())
        result = runner.run("What does Alpha report say about margins?")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.steps_used, 1)
        self.assertIn("automation", result.final_answer.lower())
        self.assertTrue(any("alpha_report.txt" in citation for citation in result.citations))

    def test_multi_tool_answer_with_both_citations(self) -> None:
        runner = AgentRunner(self.settings, llm_client=FakeLLMClient())
        result = runner.run("Compare Alpha revenue and CEO commentary")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.steps_used, 2)
        self.assertTrue(any("structured.db" in citation for citation in result.citations))
        self.assertTrue(any("alpha_report.txt" in citation for citation in result.citations))

    def test_query_data_used_in_final_answer_renders_markdown_table(self) -> None:
        runner = AgentRunner(self.settings, llm_client=FakeLLMClient())
        result = runner.run("Compare Alpha revenue and CEO commentary")
        self.assertIn("| company | revenue |", result.final_answer)
        self.assertIn("| Alpha | 100 |", result.final_answer)

    def test_retry_path_logs_error_then_succeeds(self) -> None:
        runner = AgentRunner(self.settings, llm_client=FakeLLMClient())
        result = runner.run("Need repaired SQL for Alpha revenue")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.steps_used, 1)
        self.assertEqual(result.trace[0]["status"], "error")
        self.assertEqual(result.trace[0]["step_number"], 1)
        self.assertEqual(result.trace[1]["status"], "ok")
        self.assertEqual(result.trace[1]["step_number"], 1)
        self.assertEqual(result.trace[1]["retry_count"], 1)

    def test_no_tool_direct_answer(self) -> None:
        runner = AgentRunner(self.settings, llm_client=FakeLLMClient())
        result = runner.run("What is 2+2?")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.steps_used, 0)
        self.assertEqual(result.final_answer, "The answer is 4.")

    def test_refusal_path(self) -> None:
        runner = AgentRunner(self.settings, llm_client=FakeLLMClient())
        result = runner.run("Recommend which company I should buy")
        self.assertEqual(result.status, "refused")
        self.assertIn("out of scope", result.final_answer.lower())
