from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

from agentic_rag_p0.data_tool import build_sqlite_db, query_data
from agentic_rag_p0.document_tool import build_doc_index, search_docs, upgrade_doc_metadata
from agentic_rag_p0.web_tool import web_search


class ToolTests(unittest.TestCase):
    def test_web_search_reranks_toward_stronger_sources(self) -> None:
        payload = {
            "results": [
                {
                    "title": "TCS margin update - YouTube",
                    "content": "Quick video about margin improvement reasons in 2024.",
                    "url": "https://www.youtube.com/watch?v=demo",
                    "published_date": None,
                },
                {
                    "title": "Q4 earnings transcript",
                    "content": "Management said margins improved because currency tailwinds offset pressure.",
                    "url": "https://example.com/investors/earnings-transcript.pdf",
                    "published_date": "2024-04-12",
                },
            ]
        }

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(payload).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            results = web_search("TCS margin improvement reasons 2024", tavily_api_key="test-key", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIn("transcript", results[0].url)

    def test_search_docs_returns_relevant_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "report.txt").write_text("Margins improved because automation reduced costs.", encoding="utf-8")
            index_path = root / "docs_index.json"
            build_doc_index(docs_dir, index_path)
            results = search_docs(index_path, "automation margins")
            self.assertTrue(results)
            self.assertIn("automation", results[0].content.lower())

    def test_search_docs_filename_filter_keeps_faiss_indices_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "a.txt").write_text("alpha margin commentary", encoding="utf-8")
            (docs_dir / "b.txt").write_text("beta margin commentary", encoding="utf-8")
            index_path = root / "docs_index.json"
            build_doc_index(docs_dir, index_path)

            class FakeIndex:
                def search(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
                    import numpy as np

                    return np.array([[0.9, 0.7]], dtype="float32"), np.array([[1, 0]], dtype="int64")

            class FakeFaiss:
                @staticmethod
                def read_index(_path: str) -> FakeIndex:
                    return FakeIndex()

            index_path.with_suffix(".faiss").write_bytes(b"fake")
            with patch("agentic_rag_p0.document_tool._load_faiss", return_value=FakeFaiss()):
                results = search_docs(index_path, "beta margin", filename_filter="b.txt")

            self.assertTrue(results)
            self.assertEqual(results[0].filename, "b.txt")

    def test_search_docs_lexical_rescue_surfaces_margin_commentary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "tcs.txt").write_text(
                "Business Responsibility & Sustainability Report water withdrawal and community indicators.\n"
                "Analysis of revenue growth and margin performance. EBIT margins were 24.3% in FY 2025. "
                "Tailwinds such as improved utilization, productivity, realization, and favorable currency movements contributed positively to margins.",
                encoding="utf-8",
            )
            index_path = root / "docs_index.json"
            build_doc_index(docs_dir, index_path)

            class FakeIndex:
                def search(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
                    import numpy as np

                    return np.array([[0.9]], dtype="float32"), np.array([[0]], dtype="int64")

            class FakeFaiss:
                @staticmethod
                def read_index(_path: str) -> FakeIndex:
                    return FakeIndex()

            index_path.with_suffix(".faiss").write_bytes(b"fake")
            with patch("agentic_rag_p0.document_tool._load_faiss", return_value=FakeFaiss()):
                results = search_docs(index_path, "TCS FY25 margin improvement reasons", top_k=3)

            self.assertTrue(results)
            self.assertIn("utilization", results[0].content.lower())

    def test_build_doc_index_writes_rich_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "wipro_report.txt").write_text(
                "Performance and outlook. In 2025 Wipro said operating margin improved due to utilization and overhead optimization.",
                encoding="utf-8",
            )
            index_path = root / "docs_index.json"
            result = build_doc_index(docs_dir, index_path)

            self.assertEqual(result["schema_version"], 2)
            metadata_path = Path(result["metadata_path"])
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["version"], 2)
            self.assertIn("documents", metadata)
            self.assertIn("chunks", metadata)
            self.assertTrue(metadata["documents"])
            self.assertEqual(metadata["documents"][0]["subject_hint"], "WIPRO")
            self.assertIn("2025", metadata["documents"][0]["temporal_markers"])
            self.assertIn("store_path", metadata["documents"][0])
            self.assertTrue(Path(metadata["documents"][0]["store_path"]).exists())
            self.assertIn("page_store_path", metadata["documents"][0])
            page_store_path = Path(metadata["documents"][0]["page_store_path"])
            self.assertTrue(page_store_path.exists())
            page_store = json.loads(page_store_path.read_text(encoding="utf-8"))
            self.assertEqual(page_store["filename"], "wipro_report.txt")
            self.assertEqual(page_store["pages"][0]["page_number"], 1)

    def test_upgrade_doc_metadata_enriches_old_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = root / "docs_index.json"
            old_payload = {
                "backend": "numpy",
                "vector_dim": 1024,
                "index_binary": str(root / "docs_index.faiss"),
                "vector_fallback": str(root / "docs_index.npy"),
                "chunks": [
                    {
                        "chunk_id": "tcs-p1-c1",
                        "filename": "tcs.pdf",
                        "page_number": 1,
                        "content": "TCS operating margin improved in FY24 due to utilization and realization.",
                        "token_count": 11,
                    }
                ],
            }
            index_path.write_text(json.dumps(old_payload, indent=2), encoding="utf-8")

            result = upgrade_doc_metadata(index_path)

            self.assertEqual(result["schema_version"], 2)
            upgraded = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(upgraded["schema_version"], 2)
            self.assertIn("metadata_path", upgraded)
            self.assertTrue(upgraded["documents"])
            self.assertEqual(upgraded["documents"][0]["subject_hint"], "TCS")
            self.assertIn("page_store_path", upgraded["documents"][0])
            self.assertTrue(Path(upgraded["documents"][0]["page_store_path"]).exists())

    def test_query_data_blocks_multi_statement_sql(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            structured_dir = root / "structured"
            structured_dir.mkdir()
            (structured_dir / "financials.csv").write_text("company,year,revenue\nAlpha,2024,100\n", encoding="utf-8")
            db_path = root / "structured.db"
            build_sqlite_db(structured_dir, db_path)
            with self.assertRaises(ValueError):
                query_data(db_path, "SELECT * FROM financials; DROP TABLE financials")

    def test_query_data_truncates_large_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            structured_dir = root / "structured"
            structured_dir.mkdir()
            rows = "\n".join(f"Alpha,{year},{year}" for year in range(1, 260))
            (structured_dir / "financials.csv").write_text(f"company,year,revenue\n{rows}\n", encoding="utf-8")
            db_path = root / "structured.db"
            build_sqlite_db(structured_dir, db_path)
            result = query_data(db_path, "SELECT * FROM financials")
            self.assertEqual(result.row_count, 200)
            self.assertTrue(result.truncated)
