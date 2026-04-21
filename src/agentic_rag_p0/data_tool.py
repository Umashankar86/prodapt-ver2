from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

from .models import QueryResult

READ_ONLY_SQL_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
DISALLOWED_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)
MULTI_STATEMENT_SQL_RE = re.compile(r";\s*\S+")
MAX_SQL_ROWS = 200


def _sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    return sanitized.strip("_") or "table_name"


def _infer_sqlite_type(values: list[str]) -> str:
    non_empty = [value for value in values if value not in {"", None}]
    if not non_empty:
        return "TEXT"
    try:
        for value in non_empty:
            int(value)
        return "INTEGER"
    except ValueError:
        pass
    try:
        for value in non_empty:
            float(value)
        return "REAL"
    except ValueError:
        return "TEXT"


def build_sqlite_db(structured_dir: Path, db_path: Path) -> dict:
    if not structured_dir.exists():
        raise FileNotFoundError(f"Structured directory not found: {structured_dir}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    table_summaries: list[dict[str, object]] = []
    connection = sqlite3.connect(db_path)
    try:
        for csv_path in sorted(structured_dir.glob("*.csv")):
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                if not reader.fieldnames:
                    continue

            table_name = _sanitize_identifier(csv_path.stem)
            column_types = {}
            for field in reader.fieldnames:
                sample_values = [row.get(field, "") for row in rows[:50]]
                column_types[field] = _infer_sqlite_type(sample_values)

            columns_sql = ", ".join(
                f'"{field}" {column_types[field]}' for field in reader.fieldnames
            )
            connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            connection.execute(f'CREATE TABLE "{table_name}" ({columns_sql})')

            placeholders = ", ".join("?" for _ in reader.fieldnames)
            quoted_fields = ", ".join(f'"{field}"' for field in reader.fieldnames)
            insert_sql = (
                f'INSERT INTO "{table_name}" ({quoted_fields}) '
                f"VALUES ({placeholders})"
            )
            connection.executemany(
                insert_sql,
                [[row.get(field, None) for field in reader.fieldnames] for row in rows],
            )
            table_summaries.append(
                {
                    "table_name": table_name,
                    "row_count": len(rows),
                    "columns": reader.fieldnames,
                }
            )
        connection.commit()
    finally:
        connection.close()

    return {"db_path": str(db_path), "tables": table_summaries}


def _validate_read_only_sql(sql: str) -> None:
    if not READ_ONLY_SQL_RE.search(sql):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")
    if DISALLOWED_SQL_RE.search(sql):
        raise ValueError("Query contains a disallowed SQL keyword.")
    if MULTI_STATEMENT_SQL_RE.search(sql.strip()):
        raise ValueError("Only a single read-only SQL statement is allowed.")


def query_data(db_path: Path, sql: str) -> QueryResult:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    _validate_read_only_sql(sql)
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description or []]
        fetched_rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
        truncated = len(fetched_rows) > MAX_SQL_ROWS
        rows = [list(row) for row in fetched_rows[:MAX_SQL_ROWS]]
        return QueryResult(columns=columns, rows=rows, row_count=len(rows), truncated=truncated)
    finally:
        connection.close()


def get_db_schema(db_path: Path) -> list[dict[str, object]]:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        schema = []
        for (table_name,) in tables:
            columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            schema.append(
                {
                    "table_name": table_name,
                    "columns": [
                        {"name": column[1], "type": column[2]} for column in columns
                    ],
                }
            )
        return schema
    finally:
        connection.close()


def get_db_metadata(db_path: Path, sample_rows: int = 3) -> list[dict[str, object]]:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        metadata = []
        for (table_name,) in tables:
            columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            row_count = connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            sample = connection.execute(f'SELECT * FROM "{table_name}" LIMIT {sample_rows}').fetchall()
            metadata.append(
                {
                    "table_name": table_name,
                    "row_count": row_count,
                    "columns": [{"name": column[1], "type": column[2]} for column in columns],
                    "sample_rows": [list(row) for row in sample],
                }
            )
        return metadata
    finally:
        connection.close()
