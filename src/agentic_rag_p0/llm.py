from __future__ import annotations

import json
import re

from .cache import JsonCache, JsonLog
from .config import Settings


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._types = None
        self._cache = JsonCache(settings.llm_cache_path) if settings.llm_cache_enabled else None
        self._log = JsonLog(settings.llm_log_path)

    def _ensure_client(self) -> None:
        if self._client is not None and self._types is not None:
            return
        if not self.settings.vertex_project_id:
            raise RuntimeError("Set VERTEX_PROJECT_ID in .env before running the P1 agent.")
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The `google-genai` package is required for the P1 agent. Install it with `pip install -r requirements.txt`."
            ) from exc

        self._client = genai.Client(
            vertexai=True,
            project=self.settings.vertex_project_id,
            location=self.settings.vertex_location,
        )
        self._types = types

    def generate_text(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        include_thoughts: bool = False,
    ) -> str:
        self._ensure_client()
        assert self._client is not None
        assert self._types is not None

        config = self._types.GenerateContentConfig(
            thinking_config=self._types.ThinkingConfig(include_thoughts=include_thoughts)
        )
        cache_key = None
        if self._cache is not None:
            cache_key = self._cache.build_key(model_id or self.settings.gemini_fast_model, prompt)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._log_response(prompt, model_id or self.settings.gemini_fast_model, cached, include_thoughts, cached=True)
                return cached
        response = self._client.models.generate_content(
            model=model_id or self.settings.gemini_fast_model,
            contents=prompt,
            config=config,
        )
        text = self._extract_text(response)
        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, text)
        self._log_response(prompt, model_id or self.settings.gemini_fast_model, text, include_thoughts, cached=False)
        return text

    def generate_json(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        include_thoughts: bool = False,
    ) -> dict:
        raw_text = self.generate_text(
            prompt,
            model_id=model_id,
            include_thoughts=include_thoughts,
        )
        return self._extract_json(raw_text)

    @staticmethod
    def _extract_text(response: object) -> str:
        text_parts: list[str] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
        if text_parts:
            return "".join(text_parts).strip()
        fallback_text = getattr(response, "text", None)
        if fallback_text:
            return str(fallback_text).strip()
        raise RuntimeError("Gemini response did not contain any text output.")

    @staticmethod
    def _extract_json(raw_text: str) -> dict:
        stripped = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
        candidate = fenced_match.group(1) if fenced_match else stripped
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(candidate[start : end + 1])
        recovered = GeminiClient._recover_structured_fallback(stripped)
        if recovered is not None:
            return recovered
        raise RuntimeError(f"Expected JSON object from Gemini, got: {raw_text}")

    @staticmethod
    def _recover_structured_fallback(raw_text: str) -> dict | None:
        action_match = re.search(r"\*\*Action:\*\*\s*([a-zA-Z_]+)", raw_text)
        if action_match:
            rationale_match = re.search(
                r"\*\*Rationale:\*\*\s*(.+?)(?:\n\s*\*\*[A-Za-z ]+:\*\*|$)",
                raw_text,
                re.DOTALL,
            )
            refusal_match = re.search(
                r"\*\*Refusal(?: Reason)?:\*\*\s*(.+?)(?:\n\s*\*\*[A-Za-z ]+:\*\*|$)",
                raw_text,
                re.DOTALL,
            )
            return {
                "action": action_match.group(1).strip(),
                "rationale": rationale_match.group(1).strip() if rationale_match else "",
                "refusal_reason": refusal_match.group(1).strip() if refusal_match else "",
            }

        outcome_match = re.search(r"\*\*Outcome:\*\*\s*([a-zA-Z_]+)", raw_text)
        if outcome_match:
            reason_match = re.search(
                r"\*\*Reason:\*\*\s*(.+?)(?:\n\s*\*\*[A-Za-z ]+:\*\*|$)",
                raw_text,
                re.DOTALL,
            )
            return {
                "outcome": outcome_match.group(1).strip(),
                "reason": reason_match.group(1).strip() if reason_match else "",
                "subgoals": [],
            }

        tool_input_match = re.search(
            r"(?:\*\*Tool Input:\*\*|\*\*SQL Query:\*\*|\*\*Query:\*\*)\s*(.+)",
            raw_text,
            re.DOTALL,
        )
        if tool_input_match:
            return {"tool_input": tool_input_match.group(1).strip()}
        return None

    def _log_response(
        self,
        prompt: str,
        model_id: str,
        response_text: str,
        include_thoughts: bool,
        *,
        cached: bool,
    ) -> None:
        self._log.append(
            {
                "model_id": model_id,
                "include_thoughts": include_thoughts,
                "cached": cached,
                "prompt": prompt,
                "response_text": response_text,
            }
        )
