import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List

logger = logging.getLogger(__name__)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str, tolerant: bool = False) -> List[Dict[str, Any]]:
    """Load JSONL file. If tolerant=True, skip corrupt trailing lines (e.g. from crash)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if tolerant:
                    logger.warning("Skipping corrupt line %d in %s", lineno, path)
                    continue
                raise
    return rows


def iter_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    """Streaming JSONL reader for large files."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    """Append a single row. Builds line in memory first, then flushes."""
    ensure_parent(path)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def open_jsonl_writer(path: str):
    """Context-managed buffered JSONL writer — avoids repeated open/close per row."""
    ensure_parent(path)
    f = open(path, "a", encoding="utf-8")
    try:
        def write_row(row: Dict[str, Any]) -> None:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        yield write_row
    finally:
        f.flush()
        os.fsync(f.fileno())
        f.close()
