from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .models import WebResult

WEAK_SOURCE_MARKERS = {
    "youtube", "youtu.be", "reddit", "twitter", "x.com", "facebook", "instagram", "linkedin",
    "tiktok", "scribd", "pinterest", "quora", "msn",
}
PRIMARY_SOURCE_MARKERS = {
    "investor", "investors", "ir.", "annual-report", "annual_report", "earnings", "results",
    "release", "transcript", "presentation", "filing", "sec", "nse", "bse", ".pdf",
}


def _tokenize_query(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text) if len(token) > 2]


def _score_result(query: str, item: dict) -> float:
    title = str(item.get("title", ""))
    snippet = str(item.get("content", ""))
    url = str(item.get("url", ""))
    haystack = f"{title} {snippet} {url}".lower()
    query_tokens = set(_tokenize_query(query))
    overlap = len(query_tokens.intersection(set(_tokenize_query(haystack))))
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    score = overlap * 2.0 + (0.25 if item.get("published_date") else 0.0)
    if path.endswith(".pdf") or any(marker in path for marker in ("transcript", "release", "filing", "presentation")):
        score += 6.0
    if any(marker in host or marker in path for marker in PRIMARY_SOURCE_MARKERS):
        score += 5.0
    if any(marker in host or marker in path for marker in WEAK_SOURCE_MARKERS):
        score -= 4.0
    if len(snippet.strip()) < 40:
        score -= 0.75
    if overlap == 0:
        score -= 2.0
    return score


def web_search(query: str, tavily_api_key: str | None, top_k: int = 3) -> list[WebResult]:
    trimmed_query = " ".join(query.split())
    if not trimmed_query:
        return []
    if len(trimmed_query.split()) > 10:
        raise ValueError("web_search expects a query under 10 words.")
    if not tavily_api_key:
        raise RuntimeError("Set TAVILY_API_KEY in .env to enable live web search.")

    payload = json.dumps(
        {
            "api_key": tavily_api_key,
            "query": trimmed_query,
            "max_results": max(top_k * 3, 6),
            "search_depth": "basic",
            "include_answer": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url="https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"web_search failed: {exc}") from exc

    results = []
    ranked = sorted(raw.get("results", []), key=lambda item: _score_result(trimmed_query, item), reverse=True)
    for item in ranked[:top_k]:
        results.append(
            WebResult(
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                url=item.get("url", ""),
                published_date=item.get("published_date"),
            )
        )
    return results
