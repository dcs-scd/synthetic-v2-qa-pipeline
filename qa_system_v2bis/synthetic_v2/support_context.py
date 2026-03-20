from typing import Dict, Any, Optional

from .io_utils import load_jsonl


def safe_text(x) -> str:
    return "" if x is None else str(x)


def truncate(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def build_support_index(records):
    out = {}
    for row in records:
        rid = row.get("record_id")
        if rid:
            out[rid] = row
    return out


def load_support_index(records_jsonl_path: str):
    rows = load_jsonl(records_jsonl_path)
    return build_support_index(rows)


def summarize_support_record(
    support_record: Dict[str, Any],
    max_question_len: int = 220,
    max_answer_len: int = 420,
) -> Dict[str, Any]:
    return {
        "record_id": support_record.get("record_id"),
        "source": support_record.get("source"),
        "model_name": support_record.get("model_name"),
        "class_id": support_record.get("class_id"),
        "question": truncate(safe_text(support_record.get("question")), max_question_len),
        "answer_snippet": truncate(safe_text(support_record.get("answer")), max_answer_len),
    }


def render_support_block(support_record: Optional[Dict[str, Any]]) -> str:
    if not support_record:
        return ""

    s = summarize_support_record(support_record)

    return (
        "LOCAL SUPPORT EXEMPLAR (for semantic neighborhood only; do not copy wording literally):\n"
        f"- record_id: {s['record_id']}\n"
        f"- source: {s['source']}\n"
        f"- question: {s['question']}\n"
        f"- answer_snippet: {s['answer_snippet']}\n"
    )