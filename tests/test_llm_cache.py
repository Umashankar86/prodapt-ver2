from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_rag_p0.config import Settings
from agentic_rag_p0.llm import GeminiClient


class _FakePart:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.parts = [_FakePart(text)]


class _FakeCandidate:
    def __init__(self, text: str) -> None:
        self.content = _FakeContent(text)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.candidates = [_FakeCandidate(text)]


class _FakeModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, **_: object) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse("ok")


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


class _FakeThinkingConfig:
    def __init__(self, **_: object) -> None:
        pass


class _FakeGenerateContentConfig:
    def __init__(self, **_: object) -> None:
        pass


class _FakeTypes:
    ThinkingConfig = _FakeThinkingConfig
    GenerateContentConfig = _FakeGenerateContentConfig


class LlmCacheTests(unittest.TestCase):
    def test_llm_cache_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                docs_dir=root / "docs",
                structured_dir=root / "structured",
                doc_index_path=root / "artifacts" / "docs_index.json",
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key=None,
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
            )
            client = GeminiClient(settings)
            client._client = _FakeClient()
            client._types = _FakeTypes()

            self.assertFalse(settings.llm_cache_enabled)
            self.assertEqual(client.generate_text("prompt"), "ok")
            self.assertEqual(client.generate_text("prompt"), "ok")
            self.assertEqual(client._client.models.calls, 2)
            self.assertFalse(settings.llm_cache_path.exists())

    def test_llm_cache_only_used_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                docs_dir=root / "docs",
                structured_dir=root / "structured",
                doc_index_path=root / "artifacts" / "docs_index.json",
                sqlite_db_path=root / "artifacts" / "structured.db",
                llm_cache_path=root / "artifacts" / "llm_cache.json",
                tavily_api_key=None,
                vertex_project_id="test-project",
                vertex_location="us-central1",
                gemini_fast_model="fake-fast",
                gemini_pro_model="fake-pro",
                continuity_enabled=False,
                llm_cache_enabled=True,
            )
            client = GeminiClient(settings)
            client._client = _FakeClient()
            client._types = _FakeTypes()

            self.assertEqual(client.generate_text("prompt"), "ok")
            self.assertEqual(client.generate_text("prompt"), "ok")
            self.assertEqual(client._client.models.calls, 1)
            self.assertTrue(settings.llm_cache_path.exists())

