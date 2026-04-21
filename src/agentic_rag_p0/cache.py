from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class JsonCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._data is not None:
            return self._data
        if not self.path.exists():
            self._data = {}
            return self._data
        self._data = json.loads(self.path.read_text(encoding="utf-8"))
        return self._data

    def get(self, key: str) -> str | None:
        return self._load().get(key)

    def set(self, key: str, value: str) -> None:
        data = self._load()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

    @staticmethod
    def build_key(prefix: str, payload: str) -> str:
        digest = hashlib.sha256(f"{prefix}:{payload}".encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"


class JsonLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict[str, object]) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
        else:
            data = []
        item = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        data.append(item)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
