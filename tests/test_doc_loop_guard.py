from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_rag_p0.agent import AgentRunner
from agentic_rag_p0.config import Settings
from agentic_rag_p0.document_tool import build_doc_index


class RepeatingDocLLM:
    def generate_text(self, prompt: str, **_: object) -> str:
        return "unused"

    def generate_json(self, prompt: str, **_: object) -> dict:
        if "question-understanding stage" in prompt:
            return {
                "normalized_question": "What reason did Wipro give for its margin improvement in FY25?",
                "entities": ["Wipro", "FY25"],
                "timeframe": "FY25",
                "recent_intent": False,
                "mode": "single-tool",
                "reason": "",
            }
        if "planning stage" in prompt:
            return {
                "plan_summary": "Search docs for commentary.",
                "answer_requirements": ["reason"],
                "subgoals": [
                    {
                        "description": "Retrieve commentary from the Wipro document.",
                        "status": "pending",
                        "notes": "",
                    }
                ],
                "likely_tools": ["search_docs"],
                "risks": [],
            }
        if "action selector" in prompt:
            return {"action": "search_docs", "rationale": "Need commentary.", "refusal_reason": ""}
        if "Generate the input for the tool `search_docs`" in prompt:
            return {"tool_input": "Wipro margin improvement reasons FY25"}
        if "sufficiency checker" in prompt:
            return {
                "outcome": "continue",
                "reason": "Need more.",
                "subgoals": [
                    {
                        "description": "Retrieve commentary from the Wipro document.",
                        "status": "partial",
                        "notes": "Still missing commentary.",
                    }
                ],
            }
        raise AssertionError(prompt)


class DocLoopGuardTests(unittest.TestCase):
    def test_repeated_same_doc_results_refuse_early(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "wipro.pdf.txt").write_text(
                "VALUE CREATION FOR STAKEHOLDERS RISK MANAGEMENT STATUTORY REPORTS AND FINANCIAL STATEMENTS "
                "Notes to the Consolidated Financial Statements Subsidiaries Country of Incorporation Wipro...",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key="test-key",
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )
            runner = AgentRunner(settings, llm_client=RepeatingDocLLM())
            result = runner.run("What reason did Wipro give for its margin improvement in FY25?")
            self.assertEqual(result.status, "refused")
            self.assertLess(result.steps_used, 8)
            self.assertIn("repeated", result.final_answer.lower())


class WebFirstOnLowCoverageLLM:
    def generate_text(self, prompt: str, **_: object) -> str:
        return "unused"

    def generate_json(self, prompt: str, **_: object) -> dict:
        if "question-understanding stage" in prompt:
            return {
                "normalized_question": "What reason did TCS give for its margin improvement in FY24?",
                "entities": ["TCS", "FY24"],
                "timeframe": "FY24",
                "recent_intent": True,
                "mode": "single-tool",
                "reason": "",
            }
        if "planning stage" in prompt:
            return {
                "plan_summary": "Need commentary.",
                "answer_requirements": ["reason"],
                "subgoals": [{"description": "Find TCS FY24 commentary.", "status": "pending", "notes": ""}],
                "likely_tools": ["search_docs", "web_search"],
                "risks": [],
            }
        if "sufficiency checker" in prompt:
            return {
                "outcome": "sufficient",
                "reason": "Web results sufficient.",
                "subgoals": [{"description": "Find TCS FY24 commentary.", "status": "done", "notes": ""}],
            }
        if "final answer composer" in prompt:
            return {"answer": "Web answer.", "used_evidence_ids": ["web_search-1-1"]}
        raise AssertionError(prompt)


class CoverageRoutingTests(unittest.TestCase):
    def test_low_local_coverage_goes_web_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "wipro.pdf.txt").write_text(
                "Wipro FY25 margin commentary and performance outlook.",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key=None,
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )
            runner = AgentRunner(settings, llm_client=RedirectToWebLLM())
            state = runner.service.initialize_state("What reason did TCS give for its margin improvement in FY24?")
            state.normalized_question = "What reason did TCS give for its margin improvement in FY24?"
            state.subgoals = []
            decision = runner.service.choose_next_action(state)
            self.assertEqual(decision["action"], "web_search")


class RedirectToWebLLM:
    def generate_text(self, prompt: str, **_: object) -> str:
        return "unused"

    def generate_json(self, prompt: str, **_: object) -> dict:
        if "question-understanding stage" in prompt:
            return {
                "normalized_question": "What reason did TCS give for its margin improvement in FY24?",
                "entities": ["TCS", "FY24"],
                "timeframe": "FY24",
                "recent_intent": True,
                "mode": "single-tool",
                "reason": "",
            }
        if "planning stage" in prompt:
            return {
                "plan_summary": "Try local docs, then web if needed.",
                "answer_requirements": ["reason"],
                "subgoals": [{"description": "Find TCS FY24 commentary.", "status": "pending", "notes": ""}],
                "likely_tools": ["search_docs", "web_search"],
                "risks": [],
            }
        if "action selector" in prompt:
            return {"action": "search_docs", "rationale": "Try local docs first.", "refusal_reason": ""}
        if "Generate the input for the tool `search_docs`" in prompt:
            return {"tool_input": "TCS FY24 margin improvement reasons"}
        if "Generate the input for the tool `web_search`" in prompt:
            return {"tool_input": "TCS FY24 margin reasons"}
        if "sufficiency checker" in prompt:
            if '"source_tool": "web_search"' in prompt:
                return {
                    "outcome": "sufficient",
                    "reason": "Web results sufficient.",
                    "subgoals": [{"description": "Find TCS FY24 commentary.", "status": "done", "notes": ""}],
                }
            return {
                "outcome": "continue",
                "reason": "Local docs insufficient.",
                "subgoals": [{"description": "Find TCS FY24 commentary.", "status": "partial", "notes": ""}],
            }
        if "final answer composer" in prompt:
            return {"answer": "Web answer.", "used_evidence_ids": ["web_search-1-1"]}
        raise AssertionError(prompt)


class RedirectAccountingTests(unittest.TestCase):
    def test_sufficiency_can_direct_next_tool_without_action_selector_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "tcs.pdf.txt").write_text(
                "TCS FY25 comparison text mentioning FY24 but without the reason.",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key="test-key",
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )

            class NextActionFromSufficiencyLLM(RedirectToWebLLM):
                def __init__(self) -> None:
                    self.action_calls = 0

                def generate_json(self, prompt: str, **_: object) -> dict:
                    if "action selector" in prompt:
                        self.action_calls += 1
                    if "sufficiency checker" in prompt:
                        if '"source_tool": "web_search"' in prompt:
                            return {
                                "outcome": "sufficient",
                                "reason": "Web results sufficient.",
                                "next_action": "",
                                "subgoals": [{"description": "Find TCS FY24 commentary.", "status": "done", "notes": ""}],
                            }
                        return {
                            "outcome": "continue",
                            "reason": "Use web search next.",
                            "next_action": "web_search",
                            "subgoals": [{"description": "Find TCS FY24 commentary.", "status": "partial", "notes": ""}],
                        }
                    return super().generate_json(prompt, **_)

            llm = NextActionFromSufficiencyLLM()
            runner = AgentRunner(settings, llm_client=llm)
            fake_web_results = [
                {
                    "title": "TCS FY24 margin",
                    "snippet": "Published analysis says utilization and realization helped margins.",
                    "url": "https://example.com/tcs-fy24",
                    "published_date": "2025-04-01",
                }
            ]
            with patch("agentic_rag_p0.agent_service.web_search", return_value=[type("R", (), {"to_dict": lambda self: fake_web_results[0]})()]):
                result = runner.run("What reason did TCS give for its margin improvement in FY24?")

            self.assertEqual(result.status, "answered")
            self.assertEqual(llm.action_calls, 1)
            self.assertEqual(result.steps_used, 2)

    def test_redirected_web_search_increases_step_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "tcs.pdf.txt").write_text(
                "TCS FY25 margin commentary only. No FY24 explanation here.",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key=None,
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )
            runner = AgentRunner(settings, llm_client=RedirectToWebLLM())
            fake_web_results = [
                {
                    "title": "TCS FY24 margin",
                    "snippet": "Published analysis says utilization and realization helped margins.",
                    "url": "https://example.com/tcs-fy24",
                    "published_date": "2025-04-01",
                }
            ]
            with patch("agentic_rag_p0.agent_service.web_search", return_value=[type("R", (), {"to_dict": lambda self: fake_web_results[0]})()]):
                result = runner.run("What reason did TCS give for its margin improvement in FY24?")

            self.assertEqual(result.status, "answered")
            self.assertEqual(result.steps_used, 2)

    def test_boilerplate_doc_hits_are_marked_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "tcs.pdf.txt").write_text(
                "Standalone Financial Statements 2024-25 Notes forming part of Standalone Financial Statements "
                "List of subsidiaries of the Company is as follows: Tata Consultancy Services Switzerland Ltd "
                "Tata Consultancy Services France Tata Consultancy Services Saudi Arabia.",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key="test-key",
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )
            runner = AgentRunner(settings, llm_client=RedirectToWebLLM())
            usable, flag = runner.service._doc_result_usability(
                "What reason did TCS give for its margin improvement in FY24?",
                {
                    "filename": "tcs.pdf",
                    "page_number": 302,
                    "content": (
                        "Standalone Financial Statements 2024-25 Notes forming part of Standalone "
                        "Financial Statements List of subsidiaries of the Company is as follows."
                    ),
                    "score": 18.0,
                },
            )
            self.assertFalse(usable)
            self.assertEqual(flag, "boilerplate")

    def test_wrong_subject_doc_hits_are_marked_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "wipro.pdf.txt").write_text(
                "Wipro operating margin improved because utilization increased.",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key="test-key",
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )
            runner = AgentRunner(settings, llm_client=RedirectToWebLLM())
            usable, flag = runner.service._doc_result_usability(
                "What reason did TCS give for its margin improvement in FY24?",
                {
                    "filename": "wipro.pdf",
                    "page_number": 27,
                    "content": "Wipro operating margin improved because utilization increased.",
                    "score": 18.0,
                },
                "Retrieve relevant text passages from the TCS document that explain the company's margin improvement in Fiscal Year 2024.",
            )
            self.assertFalse(usable)
            self.assertEqual(flag, "wrong_subject")

    def test_partial_commentary_cannot_finalize_before_web(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "tcs.pdf.txt").write_text(
                "TCS FY25 margin comparison text mentioning FY24 value but no FY24 reason.",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key="test-key",
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )

            class AnswerTooEarlyLLM(RedirectToWebLLM):
                def generate_json(self, prompt: str, **_: object) -> dict:
                    if "action selector" in prompt and '"source_tool": "search_docs"' in prompt:
                        return {"action": "answer", "rationale": "Answer now.", "refusal_reason": ""}
                    return super().generate_json(prompt, **_)

            runner = AgentRunner(settings, llm_client=AnswerTooEarlyLLM())
            fake_web_results = [
                {
                    "title": "TCS FY24 margin",
                    "snippet": "Published analysis says utilization and realization helped margins.",
                    "url": "https://example.com/tcs-fy24",
                    "published_date": "2025-04-01",
                }
            ]
            with patch("agentic_rag_p0.agent_service.web_search", return_value=[type("R", (), {"to_dict": lambda self: fake_web_results[0]})()]):
                result = runner.run("What reason did TCS give for its margin improvement in FY24?")

            self.assertEqual(result.status, "answered")
            self.assertEqual(result.steps_used, 2)
            self.assertTrue(any(item["source_tool"] == "web_search" for item in result.evidence))

    def test_after_web_fallback_commentary_does_not_return_to_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "tcs.pdf.txt").write_text(
                "TCS FY25 margin comparison text mentioning FY24 value but no FY24 reason.",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key="test-key",
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )
            runner = AgentRunner(settings, llm_client=RedirectToWebLLM())
            state = runner.service.initialize_state("What reason did TCS give for its margin improvement in FY24?")
            state.normalized_question = "What reason did TCS give for its margin improvement in FY24?"
            state.local_doc_attempted = True
            state.web_fallback_used = True
            from agentic_rag_p0.models import Subgoal
            state.subgoals = [Subgoal(description="Find TCS FY24 commentary.", status="partial", notes="")]
            decision = runner.service.choose_next_action(state)
            self.assertNotEqual(decision["action"], "answer")

    def test_normal_web_search_still_counts_as_tool_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "wipro.pdf.txt").write_text(
                "Wipro FY25 margin commentary only.",
                encoding="utf-8",
            )
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key=None,
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )
            runner = AgentRunner(settings, llm_client=WebFirstOnLowCoverageLLM())
            fake_web_results = [
                {
                    "title": "TCS FY24 margin",
                    "snippet": "Published analysis says utilization and realization helped margins.",
                    "url": "https://example.com/tcs-fy24",
                    "published_date": "2025-04-01",
                }
            ]
            with patch("agentic_rag_p0.agent_service.web_search", return_value=[type("R", (), {"to_dict": lambda self: fake_web_results[0]})()]):
                result = runner.run("What reason did TCS give for its margin improvement in FY24?")

            self.assertEqual(result.status, "answered")
            self.assertEqual(result.steps_used, 1)

    def test_empty_web_query_does_not_use_regex_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "tcs.pdf.txt").write_text("TCS FY25 commentary.", encoding="utf-8")
            index_path = root / "artifacts" / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            settings = Settings(
                docs_dir=docs_dir,
                structured_dir=root / "structured",
                doc_index_path=index_path,
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key="test-key",
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )

            class EmptyWebInputLLM(RedirectToWebLLM):
                def generate_json(self, prompt: str, **_: object) -> dict:
                    if "Generate the input for the tool `web_search`" in prompt:
                        return {"tool_input": ""}
                    return super().generate_json(prompt, **_)

            runner = AgentRunner(settings, llm_client=EmptyWebInputLLM())
            state = runner.service.initialize_state("What reason did TCS give for its margin improvement in FY24?")
            state.normalized_question = "What reason did TCS give for its margin improvement in FY24?"
            state.plan_summary = "Need commentary."
            built = runner.service.build_tool_input(state, "web_search")
            self.assertEqual(built, "")
