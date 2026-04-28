from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from .models import DocChunk, DocSearchResult

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
SUPPORTED_DOC_SUFFIXES = {".pdf", ".txt", ".md"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "this",
    "their", "its", "into", "than", "then", "also", "have", "had", "not", "but", "if",
    "you", "your", "they", "them", "we", "our", "can", "may", "all", "any", "more",
}
COMMENTARY_HINTS = [
    "because",
    "due to",
    "as a result",
    "therefore",
    "resulted in",
    "caused by",
    "driven by",
    "led to",
    "contributed to",
    "overview",
    "summary",
    "background",
    "commentary",
    "analysis",
]
BOILERPLATE_HINTS = [
    "table of contents",
    "all rights reserved",
    "terms and conditions",
    "privacy policy",
    "appendix",
    "glossary",
    "index",
    "references",
    "revision history",
    "copyright",
    "page intentionally left blank",
]
EXPLANATION_QUERY_TERMS = {"reason", "why", "explain", "driver", "drivers", "factor", "factors", "influence", "influenced", "cause", "caused", "drove", "drive"}
FILENAME_NOISE_TOKENS = STOPWORDS | {"pdf", "txt", "md", "doc", "docx", "page", "pages", "part", "section", "chapter", "appendix", "final", "draft", "copy", "v1", "v2", "v3"}
EMBEDDING_MODEL_NAME = os.environ.get("AGENTIC_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_HASH_ID = f"sentence-transformers:{EMBEDDING_MODEL_NAME}"
SUMMARY_PAGE_COUNT = 3
SUMMARY_CHAR_BUDGET = 6000
SUMMARY_LINE_COUNT = 4
SECTION_HINTS = {
    "explanatory_text": ["overview", "summary", "background", "analysis", "discussion", "findings", "key points"],
    "tabular_reference": ["table", "figure", "appendix", "references", "glossary", "index"],
    "policy_governance": ["policy", "procedure", "governance", "compliance", "approval", "committee"],
    "risk_disclosure": ["risk", "risks", "hazard", "limitation", "constraint", "uncertainty"],
}


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _index_file_path(index_path: Path) -> Path:
    return index_path.with_suffix(".faiss")


def _vector_file_path(index_path: Path) -> Path:
    return index_path.with_suffix(".npy")


def _store_dir_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}_stores")


def _page_store_dir_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}_page_stores")


def _store_slug(filename: str) -> str:
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "document"


def _sanitize_summary_text(summary: str) -> str:
    lines = [_normalize_whitespace(line) for line in summary.splitlines() if _normalize_whitespace(line)]
    cleaned = [re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip() for line in lines]
    cleaned = [line for line in cleaned if line]
    if not cleaned:
        return ""
    if len(cleaned) >= SUMMARY_LINE_COUNT:
        return "\n".join(cleaned[:SUMMARY_LINE_COUNT])
    while len(cleaned) < SUMMARY_LINE_COUNT:
        cleaned.append(cleaned[-1])
    return "\n".join(cleaned[:SUMMARY_LINE_COUNT])


def _summary_excerpt(filename: str, pages: list[tuple[int | None, str]], page_limit: int = SUMMARY_PAGE_COUNT) -> str:
    parts: list[str] = []
    for page_number, page_text in pages[:page_limit]:
        if not page_text:
            continue
        label = f"Page {page_number}" if page_number is not None else "Document text"
        parts.append(f"{label}: {page_text[:1800]}")
    excerpt = "\n\n".join(parts)
    return excerpt[:SUMMARY_CHAR_BUDGET]


def _fallback_document_summary(filename: str, pages: list[tuple[int | None, str]]) -> str:
    full_text = " ".join(page_text for _, page_text in pages[:SUMMARY_PAGE_COUNT] if page_text)
    years = _extract_temporal_markers(full_text)
    metrics = _extract_metrics(full_text)[:4]
    section_type = _detect_section_type(full_text) if full_text else "document_text"
    section_title = _infer_section_title(full_text, section_type) if full_text else "Document Text"
    subject_hints = _extract_subject_hints(filename)
    line_1 = f"{filename} focuses on {' / '.join(subject_hints) or Path(filename).stem}."
    line_2 = f"Opening pages center on {section_title.lower()} and {section_type.replace('_', ' ')} content."
    line_3 = f"Likely time coverage: {', '.join(years[:4]) if years else 'not clear from the opening pages'}."
    line_4 = f"Top early-page topics: {', '.join(metrics) if metrics else 'general corporate and financial commentary'}."
    return _sanitize_summary_text("\n".join([line_1, line_2, line_3, line_4]))


@lru_cache(maxsize=1)
def _summary_llm_client():
    if not os.environ.get("VERTEX_PROJECT_ID"):
        return None
    try:
        from .config import load_settings
        from .llm import GeminiClient
        return GeminiClient(load_settings())
    except Exception:
        return None


def _llm_document_summary(filename: str, pages: list[tuple[int | None, str]]) -> str:
    client = _summary_llm_client()
    if client is None:
        return ""
    excerpt = _summary_excerpt(filename, pages)
    if not excerpt:
        return ""
    prompt = f"""
You are summarizing a local document while a retrieval index is being built.
Write exactly 4 short lines.
- No bullets.
- No markdown.
- Keep each line under 140 characters.
- Focus on scope, company/entity, timeframe, and main topics from the opening pages only.

Document: {filename}
Opening pages:
{excerpt}
"""
    try:
        return _sanitize_summary_text(client.generate_text(prompt, call_kind="doc_index_summary"))
    except Exception:
        return ""


def _generate_document_summary(
    filename: str,
    pages: list[tuple[int | None, str]],
    summary_generator: Callable[[str, str], str] | None = None,
) -> str:
    excerpt = _summary_excerpt(filename, pages)
    generated = ""
    if summary_generator is not None and excerpt:
        try:
            generated = summary_generator(filename, excerpt)
        except Exception:
            generated = ""
    if not generated:
        generated = _llm_document_summary(filename, pages)
    if not generated:
        generated = _fallback_document_summary(filename, pages)
    return _sanitize_summary_text(generated) or _fallback_document_summary(filename, pages)


@lru_cache(maxsize=1)
def _embedding_model():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required for learned document embeddings. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _embedding_dim() -> int:
    dim = _embedding_model().get_sentence_embedding_dimension()
    if not dim:
        raise RuntimeError(f"Could not determine embedding dimension for {EMBEDDING_MODEL_NAME}")
    return int(dim)


def _embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, _embedding_dim()), dtype="float32")
    vectors = _embedding_model().encode(
        [text or " " for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    array = np.asarray(vectors, dtype="float32")
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    return array


def _embed_text(text: str) -> np.ndarray:
    return _embed_texts([text])[0]


def _load_faiss():
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError:
        return None
    return faiss


def _chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def _legacy_metadata_file_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}_metadata.json")


def _summary_file_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}_summary.txt")


def _detect_section_type(text: str) -> str:
    lowered = text.lower()
    for section_type, hints in SECTION_HINTS.items():
        if any(hint in lowered for hint in hints):
            return section_type
    return "document_text"


def _extract_temporal_markers(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", text)))


def _extract_subject_hints(filename: str) -> list[str]:
    stem_tokens = [
        token.lower()
        for token in re.split(r"[^A-Za-z0-9]+", Path(filename).stem)
        if token and token.lower() not in FILENAME_NOISE_TOKENS and not token.isdigit()
    ]
    return [token.upper() for token in stem_tokens[:2]]


def _extract_metrics(text: str) -> list[str]:
    counts = Counter(token for token in _tokenize(text) if token not in STOPWORDS and len(token) > 3 and not token.isdigit())
    return [token for token, _ in counts.most_common(8)]


def _profile_tokenize(text: str) -> list[str]:
    return [
        token
        for token in _tokenize(text)
        if token not in STOPWORDS and (len(token) > 2 or token.isdigit())
    ]


def _page_profile_texts(pages: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    tokenized_pages = [_profile_tokenize(page_text) for _, page_text in pages]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_pages:
        doc_freq.update(set(tokens))
    total_pages = max(1, len(tokenized_pages))
    profiles: list[tuple[int | None, str]] = []
    for (page_number, page_text), tokens in zip(pages, tokenized_pages, strict=False):
        if not tokens:
            continue
        counts = Counter(tokens)
        scored_terms = sorted(
            counts,
            key=lambda term: (
                counts[term] * (math.log((1 + total_pages) / (1 + doc_freq[term])) + 1.0),
                counts[term],
                term,
            ),
            reverse=True,
        )
        top_terms = scored_terms[:40]
        repeated_terms = []
        for term in top_terms:
            repeated_terms.extend([term] * min(3, counts[term]))
        section_type = _detect_section_type(page_text)
        section_title = _infer_section_title(page_text, section_type)
        years = _extract_temporal_markers(page_text)
        metrics = _extract_metrics(page_text)
        profile_parts = [section_type, section_title, *years, *metrics, *repeated_terms]
        profile_text = " ".join(part for part in profile_parts if part)
        if profile_text:
            profiles.append((page_number, profile_text))
    return profiles


def _commentary_score(text: str) -> float:
    lowered = text.lower()
    return round(sum(1.0 for hint in COMMENTARY_HINTS if hint in lowered) / max(1, len(COMMENTARY_HINTS)), 4)


def _boilerplate_score(text: str) -> float:
    lowered = text.lower()
    return round(sum(1.0 for hint in BOILERPLATE_HINTS if hint in lowered) / max(1, len(BOILERPLATE_HINTS)), 4)


def _infer_section_title(text: str, section_type: str) -> str:
    lowered = text.lower()
    for title in COMMENTARY_HINTS + BOILERPLATE_HINTS:
        if title in lowered:
            return title.title()
    return section_type.replace("_", " ").title()


def _build_enriched_chunk(
    *,
    filename: str,
    chunk_id: str,
    page_number: int | None,
    chunk_text: str,
    token_count: int,
    chunk_index: int,
) -> DocChunk:
    section_type = _detect_section_type(chunk_text)
    temporal_markers = _extract_temporal_markers(chunk_text)
    subject_hints = _extract_subject_hints(filename)
    metrics_mentioned = _extract_metrics(chunk_text)
    commentary_score = _commentary_score(chunk_text)
    boilerplate_score = _boilerplate_score(chunk_text)
    return DocChunk(
        chunk_id=chunk_id,
        filename=filename,
        page_number=page_number,
        content=chunk_text,
        token_count=token_count,
        doc_id=Path(filename).stem,
        chunk_index=chunk_index,
        section_title=_infer_section_title(chunk_text, section_type),
        section_type=section_type,
        char_start=0,
        char_end=len(chunk_text),
        companies_mentioned=subject_hints,
        years_mentioned=temporal_markers,
        metrics_mentioned=metrics_mentioned,
        commentary_score=commentary_score,
        boilerplate_score=boilerplate_score,
        contains_financial_statement=section_type == "tabular_reference",
        contains_management_commentary=section_type == "explanatory_text",
        contains_forward_looking="outlook" in chunk_text.lower() or "guidance" in chunk_text.lower(),
    )


def _build_document_corpus(chunks: list[DocChunk]) -> list[str]:
    return sorted({chunk.filename for chunk in chunks})


def _corpus_summary_excerpt(pages_by_filename: dict[str, list[tuple[int | None, str]]]) -> str:
    excerpts: list[str] = []
    for filename in sorted(pages_by_filename):
        excerpts.append(f"Document: {filename}\n{_summary_excerpt(filename, pages_by_filename[filename])}")
    return "\n\n".join(excerpts)[: SUMMARY_CHAR_BUDGET * 3]


def _fallback_corpus_summary(pages_by_filename: dict[str, list[tuple[int | None, str]]]) -> str:
    filenames = sorted(pages_by_filename)
    opening_text = " ".join(
        page_text
        for filename in filenames
        for _, page_text in pages_by_filename[filename][:SUMMARY_PAGE_COUNT]
        if page_text
    )
    years = _extract_temporal_markers(opening_text)
    line_1 = f"Corpus files: {', '.join(filenames)}."
    line_2 = "These are local annual or integrated financial reports for the covered companies."
    line_3 = (
        f"Opening pages mostly reference {', '.join(years[:4])} and recent business performance themes."
        if years
        else "Opening pages mainly cover recent business performance, strategy, governance, and outlook."
    )
    line_4 = "Pick the PDF matching the company in the question before searching inside the corpus."
    return _sanitize_summary_text("\n".join([line_1, line_2, line_3, line_4]))


def _llm_corpus_summary(pages_by_filename: dict[str, list[tuple[int | None, str]]]) -> str:
    client = _summary_llm_client()
    if client is None:
        return ""
    excerpt = _corpus_summary_excerpt(pages_by_filename)
    if not excerpt:
        return ""
    prompt = f"""
You are summarizing a local PDF corpus while a retrieval index is being built.
Write exactly 4 short lines.
- No bullets.
- No markdown.
- Keep each line under 140 characters.
- Mention the PDF names present in the corpus.
- Say what kind of reports these are, what period the opening pages point to, and the main topics they cover.

Opening pages from the corpus:
{excerpt}
"""
    try:
        return _sanitize_summary_text(client.generate_text(prompt, call_kind="doc_index_summary"))
    except Exception:
        return ""


def _generate_corpus_summary(
    pages_by_filename: dict[str, list[tuple[int | None, str]]],
    summary_generator: Callable[[str, str], str] | None = None,
) -> str:
    excerpt = _corpus_summary_excerpt(pages_by_filename)
    generated = ""
    if summary_generator is not None and excerpt:
        try:
            generated = summary_generator("corpus", excerpt)
        except Exception:
            generated = ""
    if not generated:
        generated = _llm_corpus_summary(pages_by_filename)
    if not generated:
        generated = _fallback_corpus_summary(pages_by_filename)
    return _sanitize_summary_text(generated) or _fallback_corpus_summary(pages_by_filename)


def _build_index_documents(
    filenames: list[str],
    document_store_paths: dict[str, str],
    page_topic_store_paths: dict[str, str],
) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for filename in sorted(filenames):
        document: dict[str, object] = {"filename": filename}
        if filename in document_store_paths:
            document["store_path"] = document_store_paths[filename]
        if filename in page_topic_store_paths:
            document["page_store_path"] = page_topic_store_paths[filename]
        documents.append(document)
    return documents


def _build_index_stats(chunks: list[DocChunk]) -> dict[str, object]:
    section_counts = Counter(chunk.section_type for chunk in chunks)
    document_count = len(_build_document_corpus(chunks))
    return {
        "document_count": document_count,
        "chunk_count": len(chunks),
        "section_type_counts": dict(section_counts),
    }


def upgrade_doc_metadata(
    index_path: Path,
    summary_generator: Callable[[str, str], str] | None = None,
) -> dict[str, object]:
    if not index_path.exists():
        raise FileNotFoundError(f"Document index not found: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = [DocChunk(**raw_chunk) for raw_chunk in payload.get("chunks", [])]
    enriched_chunks: list[DocChunk] = []
    for index, chunk in enumerate(chunks, start=1):
        if chunk.doc_id and chunk.section_type and chunk.years_mentioned is not None:
            enriched_chunks.append(chunk)
            continue
        enriched_chunks.append(
            _build_enriched_chunk(
                filename=chunk.filename,
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                chunk_text=chunk.content,
                token_count=chunk.token_count,
                chunk_index=chunk.chunk_index or index,
            )
        )
    legacy_metadata_path = _legacy_metadata_file_path(index_path)
    summary_path = _summary_file_path(index_path)
    store_dir = _store_dir_path(index_path)
    page_store_dir = _page_store_dir_path(index_path)
    store_dir.mkdir(parents=True, exist_ok=True)
    page_store_dir.mkdir(parents=True, exist_ok=True)
    chunks_by_filename: dict[str, list[DocChunk]] = {}
    pages_by_filename: dict[str, dict[int | None, list[str]]] = {}
    for chunk in enriched_chunks:
        chunks_by_filename.setdefault(chunk.filename, []).append(chunk)
        pages_by_filename.setdefault(chunk.filename, {}).setdefault(chunk.page_number, []).append(chunk.content)
    normalized_pages_by_filename = {
        filename: [
            (page_number, " ".join(page_chunks[page_number]))
            for page_number in sorted(page_chunks, key=lambda value: value or 0)
        ]
        for filename, page_chunks in pages_by_filename.items()
    }
    corpus_summary = _generate_corpus_summary(
        normalized_pages_by_filename,
        summary_generator=summary_generator,
    )
    summary_path.write_text(corpus_summary, encoding="utf-8")
    if legacy_metadata_path.exists():
        legacy_metadata_path.unlink()
    document_store_paths: dict[str, str] = {}
    page_topic_store_paths: dict[str, str] = {}
    for filename, doc_chunks in chunks_by_filename.items():
        store_index_path = store_dir / f"{_store_slug(filename)}.json"
        _write_store_payload(store_index_path, doc_chunks)
        document_store_paths[filename] = str(store_index_path)
    for filename, page_chunks in pages_by_filename.items():
        page_store_path = page_store_dir / f"{_store_slug(filename)}-pages.json"
        doc_pages = normalized_pages_by_filename[filename]
        _write_page_topic_store(page_store_path, filename, doc_pages)
        page_topic_store_paths[filename] = str(page_store_path)
    index_documents = _build_index_documents(
        list(chunks_by_filename.keys()),
        document_store_paths,
        page_topic_store_paths,
    )
    payload["schema_version"] = 2
    payload.pop("metadata_path", None)
    payload["summary_path"] = str(summary_path)
    payload["document_corpus"] = sorted(chunks_by_filename.keys())
    payload["corpus_summary"] = corpus_summary
    payload["documents"] = index_documents
    payload["stats"] = _build_index_stats(enriched_chunks)
    payload["chunks"] = [chunk.to_dict() for chunk in enriched_chunks]
    payload["store_dir"] = str(store_dir)
    payload["page_store_dir"] = str(page_store_dir)
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "schema_version": 2,
        "document_count": len(index_documents),
        "chunk_count": len(enriched_chunks),
        "page_store_dir": str(page_store_dir),
    }


def _read_pdf_pages(path: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        pages = [_normalize_whitespace(page) for page in result.stdout.split("\f")]
        return [page for page in pages if page]
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            from pypdf import PdfReader  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PDF parsing requires `pdftotext` or `pypdf`. Install one of them before indexing PDFs."
            ) from exc
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(_normalize_whitespace(page.extract_text() or ""))
        return [page for page in pages if page]


def _read_document_pages(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return [(index + 1, page_text) for index, page_text in enumerate(_read_pdf_pages(path))]
    if suffix in {".txt", ".md"}:
        text = _normalize_whitespace(path.read_text(encoding="utf-8"))
        return [(1, text)] if text else []
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _write_store_payload(store_index_path: Path, chunks: list[DocChunk]) -> None:
    vectors = _embed_texts([chunk.content for chunk in chunks])
    vector_dim = int(vectors.shape[1]) if vectors.ndim == 2 else _embedding_dim()
    faiss = _load_faiss()
    backend = "faiss" if faiss is not None else "numpy"
    if chunks and faiss is not None:
        index = faiss.IndexFlatIP(vector_dim)
        index.add(vectors)
        faiss.write_index(index, str(_index_file_path(store_index_path)))
    else:
        np.save(_vector_file_path(store_index_path), vectors)
    store_payload = {
        "schema_version": 2,
        "backend": backend,
        "vector_dim": vector_dim,
        "embedding_hash": EMBEDDING_HASH_ID,
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    store_index_path.write_text(json.dumps(store_payload, indent=2), encoding="utf-8")


def _write_page_topic_store(
    page_store_path: Path,
    filename: str,
    pages: list[tuple[int | None, str]],
) -> None:
    profiles = _page_profile_texts(pages)
    vectors = _embed_texts([profile_text for _, profile_text in profiles])
    vector_dim = int(vectors.shape[1]) if vectors.ndim == 2 else _embedding_dim()
    faiss = _load_faiss()
    backend = "faiss" if faiss is not None else "numpy"
    if profiles and faiss is not None:
        index = faiss.IndexFlatIP(vector_dim)
        index.add(vectors)
        faiss.write_index(index, str(_index_file_path(page_store_path)))
    else:
        np.save(_vector_file_path(page_store_path), vectors)
    payload = {
        "schema_version": 2,
        "backend": backend,
        "vector_dim": vector_dim,
        "embedding_hash": EMBEDDING_HASH_ID,
        "filename": filename,
        "pages": [
            {"row": row, "filename": filename, "page_number": page_number}
            for row, (page_number, _) in enumerate(profiles)
        ],
    }
    page_store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_doc_index(
    docs_dir: Path,
    index_path: Path,
    summary_generator: Callable[[str, str], str] | None = None,
) -> dict:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    chunks: list[DocChunk] = []
    chunks_by_filename: dict[str, list[DocChunk]] = {}
    pages_by_filename: dict[str, list[tuple[int | None, str]]] = {}
    for doc_path in sorted(docs_dir.rglob("*")):
        if not doc_path.is_file() or doc_path.suffix.lower() not in SUPPORTED_DOC_SUFFIXES:
            continue
        doc_pages = _read_document_pages(doc_path)
        pages_by_filename[doc_path.name] = doc_pages
        for page_number, page_text in doc_pages:
            for local_idx, chunk_text in enumerate(_chunk_text(page_text)):
                tokens = _tokenize(chunk_text)
                if not tokens:
                    continue
                chunk_id = f"{doc_path.stem}-p{page_number or 1}-c{local_idx + 1}"
                chunk = _build_enriched_chunk(
                    filename=doc_path.name,
                    chunk_id=chunk_id,
                    page_number=page_number,
                    chunk_text=chunk_text,
                    token_count=len(tokens),
                    chunk_index=local_idx + 1,
                )
                chunks.append(chunk)
                chunks_by_filename.setdefault(doc_path.name, []).append(chunk)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = _embed_texts([chunk.content for chunk in chunks])
    vector_dim = int(vectors.shape[1]) if vectors.ndim == 2 else _embedding_dim()
    faiss = _load_faiss()
    backend = "faiss" if faiss is not None else "numpy"
    if chunks and faiss is not None:
        index = faiss.IndexFlatIP(vector_dim)
        index.add(vectors)
        faiss.write_index(index, str(_index_file_path(index_path)))
    else:
        np.save(_vector_file_path(index_path), vectors)
    legacy_metadata_path = _legacy_metadata_file_path(index_path)
    summary_path = _summary_file_path(index_path)
    corpus_summary = _generate_corpus_summary(
        pages_by_filename,
        summary_generator=summary_generator,
    )
    summary_path.write_text(corpus_summary, encoding="utf-8")
    if legacy_metadata_path.exists():
        legacy_metadata_path.unlink()
    store_dir = _store_dir_path(index_path)
    store_dir.mkdir(parents=True, exist_ok=True)
    page_store_dir = _page_store_dir_path(index_path)
    page_store_dir.mkdir(parents=True, exist_ok=True)
    document_store_paths: dict[str, str] = {}
    page_topic_store_paths: dict[str, str] = {}
    for filename, doc_chunks in chunks_by_filename.items():
        store_index_path = store_dir / f"{_store_slug(filename)}.json"
        _write_store_payload(store_index_path, doc_chunks)
        document_store_paths[filename] = str(store_index_path)
    for filename, doc_pages in pages_by_filename.items():
        page_store_path = page_store_dir / f"{_store_slug(filename)}-pages.json"
        _write_page_topic_store(page_store_path, filename, doc_pages)
        page_topic_store_paths[filename] = str(page_store_path)
    index_documents = _build_index_documents(
        list(chunks_by_filename.keys()),
        document_store_paths,
        page_topic_store_paths,
    )
    payload = {
        "schema_version": 2,
        "backend": backend,
        "vector_dim": vector_dim,
        "embedding_hash": EMBEDDING_HASH_ID,
        "chunks": [chunk.to_dict() for chunk in chunks],
        "documents": index_documents,
        "document_corpus": sorted(chunks_by_filename.keys()),
        "corpus_summary": corpus_summary,
        "stats": _build_index_stats(chunks),
        "index_binary": str(_index_file_path(index_path)),
        "vector_fallback": str(_vector_file_path(index_path)),
        "summary_path": str(summary_path),
        "store_dir": str(store_dir),
        "page_store_dir": str(page_store_dir),
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "chunk_count": len(chunks),
        "document_count": len(index_documents),
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "backend": backend,
        "schema_version": 2,
        "page_store_dir": str(page_store_dir),
    }


def _idf(chunks: list[DocChunk]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for chunk in chunks:
        doc_freq.update(set(_tokenize(chunk.content)))
    total_docs = max(1, len(chunks))
    return {
        token: math.log((1 + total_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def _score_chunk(query_tokens: list[str], chunk: DocChunk, idf_map: dict[str, float]) -> float:
    chunk_counts = Counter(_tokenize(chunk.content))
    if not chunk_counts:
        return 0.0
    length_norm = 1.0 / math.sqrt(max(1, chunk.token_count))
    score = 0.0
    for token in query_tokens:
        if token not in chunk_counts:
            continue
        score += chunk_counts[token] * idf_map.get(token, 1.0)
    score *= length_norm
    lowered = chunk.content.lower()
    query_set = set(query_tokens)
    matched_terms = sum(1 for token in query_set if token in chunk_counts)
    if query_set:
        score *= 1.0 + (matched_terms / max(1, len(query_set))) * 0.35

    commentary_bonus = sum(0.35 for hint in COMMENTARY_HINTS if hint in lowered)
    boilerplate_penalty = sum(0.45 for hint in BOILERPLATE_HINTS if hint in lowered)
    if query_set.intersection(EXPLANATION_QUERY_TERMS):
        score += commentary_bonus + chunk.commentary_score * 2.5
        score -= boilerplate_penalty + chunk.boilerplate_score * 2.0
        if chunk.contains_management_commentary:
            score += 0.8
        if chunk.contains_financial_statement:
            score -= 0.35
    else:
        score += commentary_bonus * 0.5
        score -= boilerplate_penalty * 0.5
    if score < 0:
        return 0.0
    return score


def _page_matches(chunk: DocChunk, allowed_pages_by_filename: dict[str, set[int | None]]) -> bool:
    allowed_pages = allowed_pages_by_filename.get(chunk.filename.lower())
    return allowed_pages is None or chunk.page_number in allowed_pages


def _retrieve_page_refs(
    page_store_path: Path,
    query_vector: np.ndarray,
    top_k: int,
) -> list[tuple[str, int | None, float]]:
    if not page_store_path.exists():
        return []
    payload = json.loads(page_store_path.read_text(encoding="utf-8"))
    pages = payload.get("pages", [])
    if not isinstance(pages, list) or not pages:
        return []
    if payload.get("embedding_hash") != EMBEDDING_HASH_ID:
        return []
    search_k = min(max(top_k, 1), len(pages))
    faiss = _load_faiss()
    if faiss is not None and _index_file_path(page_store_path).exists():
        index = faiss.read_index(str(_index_file_path(page_store_path)))
        scores, indices = index.search(np.expand_dims(query_vector, axis=0), search_k)
        refs: list[tuple[str, int | None, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < 0 or int(idx) >= len(pages):
                continue
            if float(score) <= 0:
                continue
            page = pages[int(idx)]
            if not isinstance(page, dict):
                continue
            refs.append((str(page.get("filename", payload.get("filename", ""))), page.get("page_number"), float(score)))
        return refs

    vector_path = _vector_file_path(page_store_path)
    if not vector_path.exists():
        return []
    vectors = np.load(vector_path)
    if len(vectors) == 0:
        return []
    scores = vectors @ query_vector
    best_indices = np.argsort(scores)[::-1][:search_k]
    refs = []
    for idx in best_indices:
        if float(scores[int(idx)]) <= 0:
            continue
        page = pages[int(idx)]
        if not isinstance(page, dict):
            continue
        refs.append((str(page.get("filename", payload.get("filename", ""))), page.get("page_number"), float(scores[int(idx)])))
    return refs


def _selected_document_paths(
    payload: dict,
    lowered_filters: set[str],
) -> tuple[list[Path], list[Path]]:
    store_paths: list[Path] = []
    page_store_paths: list[Path] = []
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        return store_paths, page_store_paths
    for document in documents:
        if not isinstance(document, dict):
            continue
        filename = str(document.get("filename", "")).lower()
        if lowered_filters and filename not in lowered_filters:
            continue
        store_path = document.get("store_path")
        page_store_path = document.get("page_store_path")
        if isinstance(store_path, str):
            store_paths.append(Path(store_path))
        if isinstance(page_store_path, str):
            page_store_paths.append(Path(page_store_path))
    return store_paths, page_store_paths


def search_docs(
    index_path: Path,
    query: str,
    top_k: int = 3,
    filename_filter: str | list[str] | None = None,
) -> list[DocSearchResult]:
    if not index_path.exists():
        raise FileNotFoundError(f"Document index not found: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = [DocChunk(**raw_chunk) for raw_chunk in payload.get("chunks", [])]
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    filters = [filename_filter] if isinstance(filename_filter, str) else list(filename_filter or [])
    lowered_filters = {item.lower() for item in filters if item}
    filtered_chunks = chunks
    selected_store_paths: list[Path] = []
    selected_page_store_paths: list[Path] = []
    if lowered_filters:
        filtered_chunks = [chunk for chunk in chunks if chunk.filename.lower() in lowered_filters]
    selected_store_paths, selected_page_store_paths = _selected_document_paths(payload, lowered_filters)
    idf_map = _idf(filtered_chunks)
    query_vector = _embed_text(query)
    allowed_pages_by_filename: dict[str, set[int | None]] = {}
    if selected_page_store_paths:
        for page_store_path in selected_page_store_paths:
            for filename, page_number, _score in _retrieve_page_refs(page_store_path, query_vector, top_k=max(top_k * 4, 12)):
                key = filename.lower()
                if not key:
                    continue
                allowed_pages_by_filename.setdefault(key, set()).add(page_number)
    elif not lowered_filters:
        _, page_store_paths = _selected_document_paths(payload, set())
        for page_store_path in page_store_paths:
            for filename, page_number, _score in _retrieve_page_refs(page_store_path, query_vector, top_k=max(top_k * 2, 6)):
                key = filename.lower()
                if not key:
                    continue
                allowed_pages_by_filename.setdefault(key, set()).add(page_number)
        if allowed_pages_by_filename:
            selected_store_paths, _ = _selected_document_paths(payload, set(allowed_pages_by_filename))
    if allowed_pages_by_filename:
        page_filtered = [chunk for chunk in filtered_chunks if _page_matches(chunk, allowed_pages_by_filename)]
        if page_filtered:
            filtered_chunks = page_filtered
            idf_map = _idf(filtered_chunks)
    semantic_candidates: list[tuple[DocChunk, float]] = []
    if selected_store_paths:
        for store_path in selected_store_paths:
            if not store_path.exists():
                continue
            store_payload = json.loads(store_path.read_text(encoding="utf-8"))
            store_chunks = [DocChunk(**raw_chunk) for raw_chunk in store_payload.get("chunks", [])]
            semantic_candidates.extend(
                _retrieve_candidates(
                    index_path=store_path,
                    payload=store_payload,
                    chunks=store_chunks,
                    query_vector=query_vector,
                    top_k=max(top_k * 4, 16),
                    allowed_pages_by_filename=allowed_pages_by_filename,
                )
            )
    else:
        semantic_candidates = _retrieve_candidates(
            index_path=index_path,
            payload=payload,
            chunks=chunks,
            query_vector=query_vector,
            top_k=max(top_k * 8, 24),
            allowed_pages_by_filename=allowed_pages_by_filename,
        )
    lexical_candidates = sorted(
        (
            (chunk, 0.0)
            for chunk in filtered_chunks
            if _score_chunk(query_tokens, chunk, idf_map) > 0
        ),
        key=lambda item: _score_chunk(query_tokens, item[0], idf_map),
        reverse=True,
    )[: max(top_k * 12, 36)]
    candidates = semantic_candidates
    if lowered_filters:
        candidates = [
            (chunk, semantic_score)
            for chunk, semantic_score in semantic_candidates
            if chunk.filename.lower() in lowered_filters
        ]
        lexical_candidates = [
            (chunk, semantic_score)
            for chunk, semantic_score in lexical_candidates
            if chunk.filename.lower() in lowered_filters
        ]
    merged: dict[str, tuple[DocChunk, float]] = {}
    for chunk, semantic_score in candidates:
        merged[chunk.chunk_id] = (chunk, semantic_score)
    for chunk, semantic_score in lexical_candidates:
        existing = merged.get(chunk.chunk_id)
        if existing is None or semantic_score > existing[1]:
            merged[chunk.chunk_id] = (chunk, semantic_score)
    candidates = list(merged.values())
    if not candidates and lowered_filters:
        candidates = [(chunk, 0.0) for chunk in filtered_chunks]
    ranked = sorted(
        (
            DocSearchResult(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                content=chunk.content,
                score=round(_score_chunk(query_tokens, chunk, idf_map) + semantic_score, 4),
            )
            for chunk, semantic_score in candidates
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return [item for item in ranked[:top_k] if item.score > 0]


def _retrieve_candidates(
    index_path: Path,
    payload: dict,
    chunks: list[DocChunk],
    query_vector: np.ndarray,
    top_k: int,
    allowed_pages_by_filename: dict[str, set[int | None]] | None = None,
) -> list[tuple[DocChunk, float]]:
    if not chunks:
        return []
    allowed_pages = allowed_pages_by_filename or {}
    search_k = len(chunks) if allowed_pages else min(top_k, len(chunks))
    stored_embedding_hash = payload.get("embedding_hash")
    faiss = _load_faiss()
    if stored_embedding_hash == EMBEDDING_HASH_ID and faiss is not None and _index_file_path(index_path).exists():
        index = faiss.read_index(str(_index_file_path(index_path)))
        scores, indices = index.search(np.expand_dims(query_vector, axis=0), search_k)
        candidates: list[tuple[DocChunk, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < 0 or int(idx) >= len(chunks):
                continue
            chunk = chunks[int(idx)]
            if allowed_pages and not _page_matches(chunk, allowed_pages):
                continue
            candidates.append((chunk, float(score)))
            if len(candidates) >= top_k:
                break
        return candidates

    vector_path = _vector_file_path(index_path)
    if not vector_path.exists() or stored_embedding_hash != EMBEDDING_HASH_ID:
        vectors = _embed_texts([chunk.content for chunk in chunks])
        np.save(vector_path, vectors)
    else:
        vectors = np.load(vector_path)
    if len(vectors) == 0:
        return []
    scores = vectors @ query_vector
    best_indices = np.argsort(scores)[::-1][:search_k]
    candidates = []
    for idx in best_indices:
        chunk = chunks[int(idx)]
        if allowed_pages and not _page_matches(chunk, allowed_pages):
            continue
        candidates.append((chunk, float(scores[int(idx)])))
        if len(candidates) >= top_k:
            break
    return candidates


def get_doc_index_metadata(index_path: Path, sample_chunks: int = 5) -> dict[str, object]:
    if not index_path.exists():
        raise FileNotFoundError(f"Document index not found: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    documents_raw = payload.get("documents", [])
    compact_documents = [
        {"filename": str(document.get("filename", "")).strip()}
        for document in documents_raw
        if isinstance(document, dict) and str(document.get("filename", "")).strip()
    ] if isinstance(documents_raw, list) else []
    document_corpus = [document["filename"] for document in compact_documents]
    summary_text = str(payload.get("corpus_summary", "")).strip()
    summary_path_raw = payload.get("summary_path")
    if not summary_text and isinstance(summary_path_raw, str):
        summary_path = Path(summary_path_raw)
        if summary_path.exists():
            summary_text = summary_path.read_text(encoding="utf-8").strip()
    if document_corpus:
        return {
            "document_count": len(compact_documents),
            "documents": compact_documents,
            "document_corpus": document_corpus,
            "corpus_summary": summary_text,
            "stats": {"document_count": len(compact_documents)},
            "schema_version": int(payload.get("schema_version", 2)),
            "summary_path": str(summary_path_raw) if isinstance(summary_path_raw, str) else "",
        }
    chunks = [DocChunk(**raw_chunk) for raw_chunk in payload.get("chunks", [])]
    by_file: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        entry = by_file.setdefault(
            chunk.filename,
            {
                "filename": chunk.filename,
                "chunk_count": 0,
                "pages": set(),
                "sample_chunks": [],
                "tokens": [],
                "matched_keywords": set(),
                "temporal_markers": set(),
            },
        )
        entry["chunk_count"] = int(entry["chunk_count"]) + 1
        if chunk.page_number is not None:
            cast_pages = entry["pages"]
            assert isinstance(cast_pages, set)
            cast_pages.add(chunk.page_number)
        sample_list = entry["sample_chunks"]
        assert isinstance(sample_list, list)
        if len(sample_list) < sample_chunks:
            sample_list.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "content_preview": chunk.content[:180],
                }
            )
        token_list = entry["tokens"]
        assert isinstance(token_list, list)
        chunk_tokens = [token for token in _tokenize(chunk.content) if token not in STOPWORDS and len(token) > 2]
        token_list.extend(chunk_tokens[:80])
        matched_keywords = entry["matched_keywords"]
        assert isinstance(matched_keywords, set)
        for keyword in chunk.metrics_mentioned or []:
            matched_keywords.add(keyword)
        temporal_markers = entry["temporal_markers"]
        assert isinstance(temporal_markers, set)
        for marker in re.findall(r"\b(?:19|20)\d{2}\b", chunk.content):
            temporal_markers.add(marker)
    documents = []
    document_corpus: list[str] = []
    for entry in by_file.values():
        filename = str(entry["filename"])
        documents.append({"filename": filename})
        document_corpus.append(filename)
    documents = sorted(documents, key=lambda item: item["filename"])
    document_corpus = sorted(document_corpus)
    corpus_summary = _sanitize_summary_text(
        "\n".join(
            [
                f"Corpus files: {', '.join(document_corpus)}." if document_corpus else "Corpus files are not available.",
                "These are local annual or integrated financial reports for the indexed companies.",
                "Use the PDF name to route company-specific document searches.",
                "Open the matching file when the question asks for company commentary or report context.",
            ]
        )
    )
    return {
        "document_count": len(documents),
        "documents": documents,
        "document_corpus": document_corpus,
        "corpus_summary": corpus_summary,
        "stats": {"document_count": len(documents)},
        "schema_version": int(payload.get("schema_version", 1)),
    }
