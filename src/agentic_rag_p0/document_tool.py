from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

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
VECTOR_DIM = 1024
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


def _store_slug(filename: str) -> str:
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "document"


def _hash_token(token: str) -> tuple[int, float]:
    bucket = hash(token) % VECTOR_DIM
    sign = -1.0 if (hash(f"{token}:sign") % 2) else 1.0
    return bucket, sign


def _embed_text(text: str) -> np.ndarray:
    vector = np.zeros(VECTOR_DIM, dtype="float32")
    counts = Counter(token for token in _tokenize(text) if token not in STOPWORDS)
    for token, count in counts.items():
        bucket, sign = _hash_token(token)
        vector[bucket] += float(count) * sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


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


def _metadata_file_path(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}_metadata.json")


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


def _build_metadata_documents(chunks: list[DocChunk]) -> list[dict[str, object]]:
    by_doc: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        entry = by_doc.setdefault(
            chunk.doc_id or Path(chunk.filename).stem,
            {
                "doc_id": chunk.doc_id or Path(chunk.filename).stem,
                "filename": chunk.filename,
                "subject_hint": "",
                "source_kind": "report" if any(token in chunk.filename.lower() for token in ["report", "statement", "presentation", "filing"]) else "document",
                "temporal_markers": set(),
                "section_types": Counter(),
                "keyword_coverage": set(),
                "metrics_mentioned": set(),
                "page_numbers": set(),
                "chunk_count": 0,
                "sample_chunks": [],
                "avg_commentary_score": 0.0,
                "avg_boilerplate_score": 0.0,
            },
        )
        entry["chunk_count"] = int(entry["chunk_count"]) + 1
        cast_pages = entry["page_numbers"]
        assert isinstance(cast_pages, set)
        if chunk.page_number is not None:
            cast_pages.add(chunk.page_number)
        temporal_markers = entry["temporal_markers"]
        assert isinstance(temporal_markers, set)
        for marker in chunk.years_mentioned or []:
            temporal_markers.add(marker)
        section_types = entry["section_types"]
        assert isinstance(section_types, Counter)
        section_types.update([chunk.section_type])
        metrics = entry["metrics_mentioned"]
        assert isinstance(metrics, set)
        for metric in chunk.metrics_mentioned or []:
            metrics.add(metric)
        coverage = entry["keyword_coverage"]
        assert isinstance(coverage, set)
        for keyword in chunk.metrics_mentioned or []:
            coverage.add(keyword)
        sample_chunks = entry["sample_chunks"]
        assert isinstance(sample_chunks, list)
        if len(sample_chunks) < 5:
            sample_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "section_type": chunk.section_type,
                    "content_preview": chunk.content[:180],
                }
            )
        entry["avg_commentary_score"] = float(entry["avg_commentary_score"]) + chunk.commentary_score
        entry["avg_boilerplate_score"] = float(entry["avg_boilerplate_score"]) + chunk.boilerplate_score
        if not entry["subject_hint"] and chunk.companies_mentioned:
            entry["subject_hint"] = chunk.companies_mentioned[0]

    documents: list[dict[str, object]] = []
    for entry in by_doc.values():
        chunk_count = max(1, int(entry["chunk_count"]))
        section_types = entry["section_types"]
        assert isinstance(section_types, Counter)
        documents.append(
            {
                "doc_id": entry["doc_id"],
                "filename": entry["filename"],
                "subject_hint": entry["subject_hint"],
                "source_kind": entry["source_kind"],
                "likely_type": entry["source_kind"],
                "temporal_markers": sorted(entry["temporal_markers"]) if isinstance(entry["temporal_markers"], set) else [],
                "section_types": dict(section_types.most_common()),
                "keyword_coverage": sorted(entry["keyword_coverage"]) if isinstance(entry["keyword_coverage"], set) else [],
                "metrics_mentioned": sorted(entry["metrics_mentioned"]) if isinstance(entry["metrics_mentioned"], set) else [],
                "chunk_count": chunk_count,
                "page_count_estimate": len(entry["page_numbers"]) if isinstance(entry["page_numbers"], set) else 0,
                "page_numbers": sorted(entry["page_numbers"]) if isinstance(entry["page_numbers"], set) else [],
                "avg_commentary_score": round(float(entry["avg_commentary_score"]) / chunk_count, 4),
                "avg_boilerplate_score": round(float(entry["avg_boilerplate_score"]) / chunk_count, 4),
                "sample_chunks": entry["sample_chunks"],
                "store_id": _store_slug(str(entry["filename"])),
            }
        )
    return sorted(documents, key=lambda item: str(item["filename"]))


def _build_metadata_payload(chunks: list[DocChunk], backend: str, index_path: Path) -> dict[str, object]:
    documents = _build_metadata_documents(chunks)
    section_counts = Counter(chunk.section_type for chunk in chunks)
    subject_hint_counts = Counter(
        subject_hint
        for chunk in chunks
        for subject_hint in (chunk.companies_mentioned or [])
    )
    return {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "embedding_dim": VECTOR_DIM,
        "stats": {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "section_type_counts": dict(section_counts),
            "subject_hint_counts": dict(subject_hint_counts),
        },
        "documents": documents,
        "chunks": [chunk.to_dict() for chunk in chunks],
        "manifest_path": str(index_path),
    }


def upgrade_doc_metadata(index_path: Path) -> dict[str, object]:
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
    metadata_payload = _build_metadata_payload(enriched_chunks, str(payload.get("backend", "numpy")), index_path)
    metadata_path = _metadata_file_path(index_path)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    payload["schema_version"] = 2
    payload["metadata_path"] = str(metadata_path)
    payload["documents"] = metadata_payload["documents"]
    payload["stats"] = metadata_payload["stats"]
    payload["chunks"] = [chunk.to_dict() for chunk in enriched_chunks]
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "schema_version": 2,
        "document_count": len(metadata_payload["documents"]) if isinstance(metadata_payload["documents"], list) else 0,
        "chunk_count": len(enriched_chunks),
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
    vectors = np.vstack([_embed_text(chunk.content) for chunk in chunks]).astype("float32") if chunks else np.zeros((0, VECTOR_DIM), dtype="float32")
    faiss = _load_faiss()
    backend = "faiss" if faiss is not None else "numpy"
    if chunks and faiss is not None:
        index = faiss.IndexFlatIP(VECTOR_DIM)
        index.add(vectors)
        faiss.write_index(index, str(_index_file_path(store_index_path)))
    else:
        np.save(_vector_file_path(store_index_path), vectors)
    store_payload = {
        "schema_version": 2,
        "backend": backend,
        "vector_dim": VECTOR_DIM,
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    store_index_path.write_text(json.dumps(store_payload, indent=2), encoding="utf-8")


def build_doc_index(docs_dir: Path, index_path: Path) -> dict:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    chunks: list[DocChunk] = []
    chunks_by_filename: dict[str, list[DocChunk]] = {}
    for doc_path in sorted(docs_dir.rglob("*")):
        if not doc_path.is_file() or doc_path.suffix.lower() not in SUPPORTED_DOC_SUFFIXES:
            continue
        for page_number, page_text in _read_document_pages(doc_path):
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
    vectors = np.vstack([_embed_text(chunk.content) for chunk in chunks]).astype("float32") if chunks else np.zeros((0, VECTOR_DIM), dtype="float32")
    faiss = _load_faiss()
    backend = "faiss" if faiss is not None else "numpy"
    if chunks and faiss is not None:
        index = faiss.IndexFlatIP(VECTOR_DIM)
        index.add(vectors)
        faiss.write_index(index, str(_index_file_path(index_path)))
    else:
        np.save(_vector_file_path(index_path), vectors)
    metadata_path = _metadata_file_path(index_path)
    metadata_payload = _build_metadata_payload(chunks, backend, index_path)
    store_dir = _store_dir_path(index_path)
    store_dir.mkdir(parents=True, exist_ok=True)
    document_store_paths: dict[str, str] = {}
    for filename, doc_chunks in chunks_by_filename.items():
        store_index_path = store_dir / f"{_store_slug(filename)}.json"
        _write_store_payload(store_index_path, doc_chunks)
        document_store_paths[filename] = str(store_index_path)
    for document in metadata_payload.get("documents", []):
        if isinstance(document, dict):
            filename = str(document.get("filename", ""))
            if filename in document_store_paths:
                document["store_path"] = document_store_paths[filename]
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    payload = {
        "schema_version": 2,
        "backend": backend,
        "vector_dim": VECTOR_DIM,
        "chunks": [chunk.to_dict() for chunk in chunks],
        "documents": metadata_payload["documents"],
        "stats": metadata_payload["stats"],
        "index_binary": str(_index_file_path(index_path)),
        "vector_fallback": str(_vector_file_path(index_path)),
        "metadata_path": str(metadata_path),
        "store_dir": str(store_dir),
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "chunk_count": len(chunks),
        "document_count": len(metadata_payload["documents"]) if isinstance(metadata_payload["documents"], list) else 0,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "backend": backend,
        "schema_version": 2,
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
    if lowered_filters:
        filtered_chunks = [chunk for chunk in chunks if chunk.filename.lower() in lowered_filters]
        documents = payload.get("documents", [])
        if isinstance(documents, list):
            for document in documents:
                if not isinstance(document, dict):
                    continue
                filename = str(document.get("filename", "")).lower()
                store_path = document.get("store_path")
                if filename in lowered_filters and isinstance(store_path, str):
                    selected_store_paths.append(Path(store_path))
    idf_map = _idf(filtered_chunks)
    query_vector = _embed_text(query)
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
                )
            )
    else:
        semantic_candidates = _retrieve_candidates(
            index_path=index_path,
            payload=payload,
            chunks=chunks,
            query_vector=query_vector,
            top_k=max(top_k * 8, 24),
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
) -> list[tuple[DocChunk, float]]:
    if not chunks:
        return []
    faiss = _load_faiss()
    if faiss is not None and _index_file_path(index_path).exists():
        index = faiss.read_index(str(_index_file_path(index_path)))
        scores, indices = index.search(np.expand_dims(query_vector, axis=0), min(top_k, len(chunks)))
        candidates: list[tuple[DocChunk, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < 0:
                continue
            candidates.append((chunks[int(idx)], float(score)))
        return candidates

    vector_path = _vector_file_path(index_path)
    if not vector_path.exists():
        vectors = np.vstack([_embed_text(chunk.content) for chunk in chunks]).astype("float32")
        np.save(vector_path, vectors)
    else:
        vectors = np.load(vector_path)
    if len(vectors) == 0:
        return []
    scores = vectors @ query_vector
    best_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]
    return [(chunks[int(idx)], float(scores[int(idx)])) for idx in best_indices]


def get_doc_index_metadata(index_path: Path, sample_chunks: int = 5) -> dict[str, object]:
    if not index_path.exists():
        raise FileNotFoundError(f"Document index not found: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    metadata_path_raw = payload.get("metadata_path")
    if isinstance(metadata_path_raw, str):
        metadata_path = Path(metadata_path_raw)
        if metadata_path.exists():
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            documents = metadata_payload.get("documents", [])
            stats = metadata_payload.get("stats", {})
            if isinstance(documents, list):
                return {
                    "document_count": int(stats.get("document_count", len(documents))) if isinstance(stats, dict) else len(documents),
                    "documents": documents,
                    "stats": stats,
                    "schema_version": metadata_payload.get("version", 2),
                    "metadata_path": str(metadata_path),
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
    for entry in by_file.values():
        pages = sorted(entry["pages"]) if isinstance(entry["pages"], set) else []
        token_counter = Counter(entry["tokens"]) if isinstance(entry["tokens"], list) else Counter()
        top_terms = [term for term, _ in token_counter.most_common(12)]
        matched_keywords = sorted(entry["matched_keywords"]) if isinstance(entry["matched_keywords"], set) else []
        filename_lower = str(entry["filename"]).lower()
        likely_type = "report" if any(token in filename_lower for token in ["report", "statement", "presentation", "filing"]) else "document"
        subject_hint = " ".join(_extract_subject_hints(str(entry["filename"])))
        temporal_markers = sorted(entry["temporal_markers"]) if isinstance(entry["temporal_markers"], set) else []
        documents.append(
            {
                "filename": entry["filename"],
                "subject_hint": subject_hint,
                "likely_type": likely_type,
                "temporal_markers": temporal_markers,
                "chunk_count": entry["chunk_count"],
                "page_count_estimate": len(pages),
                "keyword_coverage": matched_keywords,
                "top_terms": top_terms,
                "sample_chunks": entry["sample_chunks"],
            }
        )
    return {"document_count": len(documents), "documents": sorted(documents, key=lambda item: item["filename"])}
