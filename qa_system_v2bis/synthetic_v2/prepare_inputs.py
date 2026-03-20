import argparse
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import load_json, load_jsonl, save_json, save_jsonl
from .text_utils import (
    normalize_model_name,
    extract_model_from_tags,
    extract_level_from_tags,
    normalize_question,
    normalize_answer,
    safe_text,
)


def first_message_by_role(messages: List[Dict[str, Any]], role: str) -> Optional[str]:
    for msg in messages or []:
        if msg.get("role") == role:
            content = safe_text(msg.get("content"))
            if content:
                return content
    return None


def extract_question_answer(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Priority:
    1. explicit question / answer fields
    2. messages[user] / messages[assistant]
    """
    q = normalize_question(row.get("question"))
    a = normalize_answer(row.get("answer"))

    if q and a:
        return q, a

    messages = row.get("messages") or []
    if not q:
        q = normalize_question(first_message_by_role(messages, "user"))
    if not a:
        a = normalize_answer(first_message_by_role(messages, "assistant"))

    return q, a


def extract_model_name(row: Dict[str, Any]) -> Optional[str]:
    for key in ["model_name", "model"]:
        if key in row:
            m = normalize_model_name(row.get(key))
            if m:
                return m

    tags = row.get("tags") or []
    return extract_model_from_tags(tags)


def extract_level(row: Dict[str, Any]) -> Optional[str]:
    if row.get("level"):
        return safe_text(row.get("level"))
    return extract_level_from_tags(row.get("tags") or [])


def build_corpus_index(corpus_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for row in corpus_rows:
        rid = safe_text(row.get("id"))
        if rid:
            idx[rid] = row
    return idx


def normalize_corpus_row(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    issues = []

    source_id = safe_text(row.get("id")) or safe_text(row.get("global_id")) or ""
    model_name = extract_model_name(row)
    question, answer = extract_question_answer(row)

    if not model_name:
        issues.append("MISSING_MODEL_NAME")
    if not question:
        issues.append("MISSING_QUESTION")
    if not answer:
        issues.append("MISSING_ANSWER")

    if issues:
        return None, issues

    norm_row = {
        "record_id": f"corpus::{source_id}" if source_id else None,
        "source": "corpus",
        "source_id": source_id,
        "model_name": model_name,
        "question": question,
        "answer": answer,
        "level": extract_level(row),
        "global_id": safe_text(row.get("global_id")),
        "tags": row.get("tags") or [],
        "raw_ref": {
            "id": row.get("id"),
            "idx": row.get("idx"),
        },
    }
    return norm_row, issues


def normalize_seed_row(seed: Dict[str, Any], corpus_index: Dict[str, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Seed priority:
    1. use seed_q / seed_a if already present
    2. otherwise join by seed_id -> corpus.id
    """
    issues = []

    seed_id = safe_text(seed.get("seed_id"))
    model_name = normalize_model_name(seed.get("model_name"))
    level = safe_text(seed.get("level"))
    tier = safe_text(seed.get("tier"))
    seed_type = safe_text(seed.get("seed_type"))

    seed_q = normalize_question(seed.get("seed_q"))
    seed_a = normalize_answer(seed.get("seed_a"))

    join_source = None
    if not seed_q or not seed_a:
        if seed_id and seed_id in corpus_index:
            joined = corpus_index[seed_id]
            jq, ja = extract_question_answer(joined)
            seed_q = seed_q or jq
            seed_a = seed_a or ja
            join_source = seed_id
        else:
            issues.append("SEED_JOIN_MISS")

    if not model_name:
        issues.append("MISSING_MODEL_NAME")
    if not seed_q:
        issues.append("MISSING_QUESTION")
    if not seed_a:
        issues.append("MISSING_ANSWER")

    if issues:
        return None, issues

    norm_seed = {
        "record_id": f"seed::{seed_id}::{seed_type}" if seed_id and seed_type else f"seed::{seed_id}" if seed_id else f"seed::_anon_{hash((model_name, seed_q))}",
        "source": "seed",
        "source_id": seed_id,
        "seed_id": seed_id,
        "model_name": model_name,
        "question": seed_q,
        "answer": seed_a,
        "seed_q": seed_q,
        "seed_a": seed_a,
        "class_id": seed.get("class_id"),
        "level": level,
        "tier": tier,
        "seed_type": seed_type,
        "raw_ref": {
            "joined_from_corpus_id": join_source,
        },
    }
    return norm_seed, issues


def normalize_synth_row(row: Dict[str, Any], source: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    issues = []

    source_id = safe_text(row.get("seed_id")) or safe_text(row.get("id")) or ""
    model_name = extract_model_name(row) or normalize_model_name(row.get("model_name"))
    question = normalize_question(row.get("question"))
    answer = normalize_answer(row.get("answer"))

    if not model_name:
        issues.append("MISSING_MODEL_NAME")
    if not question:
        issues.append("MISSING_QUESTION")
    if not answer:
        issues.append("MISSING_ANSWER")

    if issues:
        return None, issues

    norm_row = {
        "record_id": f"{source}::{source_id}" if source_id else None,
        "source": source,
        "source_id": source_id,
        "model_name": model_name,
        "question": question,
        "answer": answer,
        "seed_id": safe_text(row.get("seed_id")),
        "class_id": row.get("class_id"),
        "level": safe_text(row.get("level")),
        "tier": safe_text(row.get("tier")),
        "raw_ref": {
            "reason": row.get("reason"),
        }
    }
    return norm_row, issues


def count_duplicates(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter()
    for r in rows:
        rid = r.get("record_id")
        if rid:
            c[rid] += 1
    return {rid: n for rid, n in c.items() if n > 1}


def build_consistency_report(
    corpus_rows_raw: List[Dict[str, Any]],
    seeds_raw: List[Dict[str, Any]],
    normalized_corpus: List[Dict[str, Any]],
    normalized_seeds: List[Dict[str, Any]],
    corpus_issues: Counter,
    seed_issues: Counter,
    extra_sources_summary: Dict[str, Dict[str, int]],
    duplicate_ids: Dict[str, int],
) -> Dict[str, Any]:
    by_model = Counter()
    for row in normalized_corpus + normalized_seeds:
        if row.get("model_name"):
            by_model[row["model_name"]] += 1

    return {
        "counts": {
            "corpus_raw": len(corpus_rows_raw),
            "seeds_raw": len(seeds_raw),
            "corpus_normalized_ok": len(normalized_corpus),
            "seeds_normalized_ok": len(normalized_seeds),
            "all_records_total": len(normalized_corpus) + len(normalized_seeds) + sum(
                v.get("normalized_ok", 0) for v in extra_sources_summary.values()
            ),
        },
        "issues": {
            "corpus": dict(corpus_issues),
            "seeds": dict(seed_issues),
            "extra_sources": extra_sources_summary,
        },
        "duplicates": {
            "duplicate_record_ids": duplicate_ids,
            "duplicate_record_id_count": len(duplicate_ids),
        },
        "distribution": {
            "records_by_model": dict(by_model)
        }
    }


def normalize_extra_source(
    path: Optional[str],
    source_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not path:
        return [], {"raw": 0, "normalized_ok": 0}

    rows = load_jsonl(path)
    norm_rows = []
    issues = Counter()

    for row in rows:
        norm_row, row_issues = normalize_synth_row(row, source=source_name)
        if row_issues:
            for x in row_issues:
                issues[x] += 1
        if norm_row:
            norm_rows.append(norm_row)

    summary = {
        "raw": len(rows),
        "normalized_ok": len(norm_rows),
        "issues": dict(issues),
    }
    return norm_rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-plan", required=True, help="Path to synth_plan.json")
    parser.add_argument("--corpus-jsonl", required=True, help="Path to corpus_combined.jsonl")
    parser.add_argument("--accepted-synth-jsonl", default=None, help="Optional accepted synthetic JSONL")
    parser.add_argument("--rejected-synth-jsonl", default=None, help="Optional rejected synthetic JSONL")
    parser.add_argument("--out-seeds", required=True, help="Output JSONL for seeds_with_text")
    parser.add_argument("--out-all-records", required=True, help="Output JSONL for all_records")
    parser.add_argument("--out-report", required=True, help="Output JSON for consistency report")
    args = parser.parse_args()

    synth_plan = load_json(args.synth_plan)
    corpus_rows_raw = load_jsonl(args.corpus_jsonl)

    corpus_index = build_corpus_index(corpus_rows_raw)

    normalized_corpus = []
    normalized_seeds = []
    corpus_issues = Counter()
    seed_issues = Counter()

    # Normalize corpus
    for row in corpus_rows_raw:
        norm_row, row_issues = normalize_corpus_row(row)
        if row_issues:
            for x in row_issues:
                corpus_issues[x] += 1
        if norm_row:
            normalized_corpus.append(norm_row)

    # Normalize seeds
    for seed in synth_plan:
        norm_seed, row_issues = normalize_seed_row(seed, corpus_index)
        if row_issues:
            for x in row_issues:
                seed_issues[x] += 1
        if norm_seed:
            normalized_seeds.append(norm_seed)

    # Optional extra sources
    accepted_synth_rows, accepted_summary = normalize_extra_source(
        args.accepted_synth_jsonl, "accepted_synth"
    )
    rejected_synth_rows, rejected_summary = normalize_extra_source(
        args.rejected_synth_jsonl, "rejected_synth"
    )

    all_records = normalized_corpus + normalized_seeds + accepted_synth_rows + rejected_synth_rows

    duplicate_ids = count_duplicates(all_records)

    report = build_consistency_report(
        corpus_rows_raw=corpus_rows_raw,
        seeds_raw=synth_plan,
        normalized_corpus=normalized_corpus,
        normalized_seeds=normalized_seeds,
        corpus_issues=corpus_issues,
        seed_issues=seed_issues,
        extra_sources_summary={
            "accepted_synth": accepted_summary,
            "rejected_synth": rejected_summary,
        },
        duplicate_ids=duplicate_ids,
    )

    save_jsonl(normalized_seeds, args.out_seeds)
    save_jsonl(all_records, args.out_all_records)
    save_json(report, args.out_report)


if __name__ == "__main__":
    main()
