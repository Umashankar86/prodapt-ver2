from __future__ import annotations

import unittest

from agentic_rag_p0.llm import GeminiClient


class LlmJsonRecoveryTests(unittest.TestCase):
    def test_recovers_action_selector_markdown_fallback(self) -> None:
        raw_text = """
The current evidence confirms that TCS's operating margin improved in FY2024.

**Action:** search_docs
**Rationale:** The current evidence confirms the margin improvement in FY2024 but does not explicitly state the reasons for it.
"""
        recovered = GeminiClient._extract_json(raw_text)
        self.assertEqual(recovered["action"], "search_docs")
        self.assertIn("does not explicitly state the reasons", recovered["rationale"])
        self.assertEqual(recovered["refusal_reason"], "")


if __name__ == "__main__":
    unittest.main()
