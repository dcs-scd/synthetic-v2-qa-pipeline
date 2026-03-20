import argparse
import json
import logging
import re as _re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Set

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .io_utils import load_json, load_jsonl, append_jsonl, save_json
from .prompt_builder import build_prompt, get_model_profile
from .gate1_embedding import run_gate1_embedding, AlwaysPassEmbeddingIndex
from .gate2_model_validation import validate_generated_row
from .gate3_dedup import InMemoryDedupIndex, run_gate3_dedup
from .telemetry import make_event, record_attempt, summarize_telemetry, TelemetrySummary
from .text_utils import normalize_model_name
from .batch_scheduler import QuotaTracker, load_quotas
from .support_context import build_support_index


logger = logging.getLogger(__name__)

_SENSITIVE_RE = _re.compile(r'(sk-|key-|token[=:])\S+', _re.IGNORECASE)
MAX_LLM_RESPONSE_SIZE = 50_000
MAX_LLM_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0  # seconds
LLM_RETRY_BACKOFF_FACTOR = 2.0
PROGRESS_REPORT_INTERVAL = 500  # seeds between periodic summary logs


def _safe_error_str(e: Exception, max_len: int = 200) -> str:
    msg = str(e)[:max_len]
    return _SENSITIVE_RE.sub('[REDACTED]', msg)


# -------------------------------------------------------------------
# LLM client smoke-test adapters
# -------------------------------------------------------------------

class EchoSeedLLMClient:
    """
    Smoke-test LLM adapter.
    Returns the seed's own Q/A as JSON.
    Useful for pipeline testing without calling a real model.
    """
    def generate(self, prompt: str, seed_row: Dict[str, Any]) -> str:
        q = seed_row.get("seed_q") or seed_row.get("question") or ""
        a = seed_row.get("seed_a") or seed_row.get("answer") or ""
        return json.dumps({"question": q, "answer": a}, ensure_ascii=False)


class StaticJSONLLMClient:
    """
    Another smoke-test adapter.
    Returns the same canned JSON object every time.
    """
    def __init__(self, question: str, answer: str):
        self.obj = {"question": question, "answer": answer}

    def generate(self, prompt: str, seed_row: Dict[str, Any]) -> str:
        return json.dumps(self.obj, ensure_ascii=False)


# -------------------------------------------------------------------
# JSON parsing
# -------------------------------------------------------------------

def parse_llm_json(raw: str) -> Dict[str, Any]:
    if len(raw) > MAX_LLM_RESPONSE_SIZE:
        return {"ok": False, "error": "response_too_large", "raw": raw[:1000]}
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {"ok": False, "error": f"json_parse_error: {e}", "raw": raw[:1000]}

    if not isinstance(obj, dict):
        return {"ok": False, "error": "response_not_json_object", "raw": raw[:1000]}

    if "skip" in obj:
        return {"ok": False, "error": f"model_returned_skip:{obj.get('skip')}", "raw": raw[:1000]}

    if "question" not in obj or "answer" not in obj:
        return {"ok": False, "error": "missing_question_or_answer", "raw": raw[:1000]}

    if not isinstance(obj["question"], str) or not isinstance(obj["answer"], str):
        return {"ok": False, "error": "question_or_answer_not_string", "raw": raw[:1000]}

    return {"ok": True, "data": obj}


# -------------------------------------------------------------------
# Result shaping
# -------------------------------------------------------------------

def accept_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "accepted",
        "record": record
    }


def reject_record(seed_row: Dict[str, Any], reason: str, details: Dict[str, Any]) -> Dict[str, Any]:
    rec = {
        "seed_id": seed_row.get("seed_id"),
        "model_name": seed_row.get("model_name"),
        "class_id": seed_row.get("class_id"),
        "level": seed_row.get("level"),
        "tier": seed_row.get("tier"),
        "route_mode": seed_row.get("route_mode"),
        "extension_family": seed_row.get("extension_family"),
        "reason": reason,
        "details": details or {}
    }
    return {
        "status": "rejected",
        "record": rec
    }


# -------------------------------------------------------------------
# LLM retry wrapper
# -------------------------------------------------------------------

def _call_llm_with_retry(llm_client, prompt: str, seed_row: Dict[str, Any]) -> str:
    """Call LLM with exponential backoff retry on transient errors."""
    last_error = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            return llm_client.generate(prompt, seed_row=seed_row)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            is_transient = any(s in err_str for s in [
                "429", "rate limit", "too many requests",
                "500", "502", "503", "504",
                "timeout", "timed out", "connection",
            ])
            if not is_transient or attempt == MAX_LLM_RETRIES - 1:
                raise
            delay = LLM_RETRY_BASE_DELAY * (LLM_RETRY_BACKOFF_FACTOR ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_LLM_RETRIES, delay, _safe_error_str(e)
            )
            time.sleep(delay)
    raise last_error  # unreachable but satisfies type checker


# -------------------------------------------------------------------
# Seed processing
# -------------------------------------------------------------------

def process_seed(
    seed_row: Dict[str, Any],
    model_profiles: Dict[str, Any],
    llm_client,
    embedding_index,
    support_index=None,
) -> Dict[str, Any]:
    route_mode = seed_row.get("route_mode")
    if route_mode == "skip":
        return reject_record(seed_row, "SEED_ROUTED_SKIP", seed_row.get("route_diagnostics", {}))

    model_name = normalize_model_name(seed_row.get("model_name"))
    profile = get_model_profile(model_profiles, model_name)
    if not profile:
        return reject_record(seed_row, "MISSING_MODEL_PROFILE", {"model_name": model_name})

    try:
        prompt = build_prompt(seed_row, profile, support_index=support_index)
    except Exception as e:
        return reject_record(seed_row, "PROMPT_BUILD_ERROR", {"error": _safe_error_str(e)})

    # LLM call (with retry)
    try:
        raw = _call_llm_with_retry(llm_client, prompt, seed_row)
    except Exception as e:
        return reject_record(seed_row, "LLM_CALL_ERROR", {"error": _safe_error_str(e)})

    parsed = parse_llm_json(raw)
    if not parsed["ok"]:
        return reject_record(seed_row, "BAD_JSON", parsed)

    qa = parsed["data"]

    # Gate 1
    gate1 = run_gate1_embedding(
        question=qa["question"],
        class_id=seed_row.get("class_id"),
        embedding_index=embedding_index
    )
    if not gate1["ok"]:
        return reject_record(seed_row, gate1["reason"], gate1["details"])

    # Gate 2
    gate2 = validate_generated_row(
        generated_row=qa,
        routed_seed_row=seed_row,
        model_profiles=model_profiles
    )
    if not gate2["ok"]:
        return reject_record(seed_row, gate2["reason"], gate2["details"])

    # Gate 3 (dedup) runs in the main thread to avoid TOCTOU races
    accepted = {
        "seed_id": seed_row.get("seed_id"),
        "model_name": model_name,
        "class_id": seed_row.get("class_id"),
        "level": seed_row.get("level"),
        "tier": seed_row.get("tier"),
        "route_mode": seed_row.get("route_mode"),
        "extension_family": seed_row.get("extension_family"),
        "question": qa["question"],
        "answer": qa["answer"],
        "gate1": gate1,
        "gate2": gate2,
    }

    return {"status": "pending_gate3", "record": accepted}


# -------------------------------------------------------------------
# Batch generation
# -------------------------------------------------------------------

def generate_synthetic(
    routed_seeds: List[Dict[str, Any]],
    model_profiles: Dict[str, Any],
    llm_client,
    embedding_index,
    dedup_index: InMemoryDedupIndex,
    accepted_path: str,
    rejected_path: str,
    telemetry_path: Optional[str] = None,
    support_index=None,
    near_dup_threshold: float = 0.985,
    limit: Optional[int] = None,
    max_workers: int = 8,
    processed_seed_ids: Optional[Set[str]] = None,
    quota_tracker: Optional[QuotaTracker] = None,
) -> Dict[str, Any]:
    from .io_utils import open_jsonl_writer

    summary = TelemetrySummary()
    rows = routed_seeds[:limit] if limit is not None else routed_seeds
    if processed_seed_ids:
        rows = [r for r in rows if r.get("seed_id") not in processed_seed_ids]
    if quota_tracker:
        filtered = []
        for r in rows:
            ok, reason = quota_tracker.should_process(r)
            if ok:
                filtered.append(r)
            else:
                quota_tracker.record_skip(reason)
        rows = filtered

    def _process_through_gate2(seed_row):
        # Process through gates 1 and 2 in thread pool (thread-safe, no shared state)
        result = process_seed(
            seed_row=seed_row,
            model_profiles=model_profiles,
            llm_client=llm_client,
            embedding_index=embedding_index,
            support_index=support_index,
        )
        return seed_row, result

    with open_jsonl_writer(accepted_path) as write_accepted, \
         open_jsonl_writer(rejected_path) as write_rejected:

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_through_gate2, seed): seed for seed in rows}

            total = len(rows)
            accepted_count = 0
            rejected_count = 0

            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=total, desc="Generating", unit="seed")

            for i, future in enumerate(iterator, 1):
                seed_row, result = future.result()

                if result["status"] == "rejected":
                    write_rejected(result["record"])
                    rejected_count += 1
                    event = make_event(
                        seed_row,
                        status="rejected",
                        reason=result["record"]["reason"],
                        details=result["record"].get("details", {})
                    )
                elif result["status"] == "pending_gate3":
                    # Gate 3 runs in main thread — no TOCTOU race on dedup_index
                    gate3 = run_gate3_dedup(
                        question=result["record"]["question"],
                        answer=result["record"]["answer"],
                        dedup_index=dedup_index,
                        near_dup_threshold=near_dup_threshold,
                    )
                    if not gate3["ok"]:
                        rej = reject_record(seed_row, gate3["reason"], gate3["details"])
                        write_rejected(rej["record"])
                        rejected_count += 1
                        event = make_event(
                            seed_row,
                            status="rejected",
                            reason=gate3["reason"],
                            details=gate3["details"],
                        )
                    else:
                        result["record"]["gate3"] = gate3
                        dedup_index.add(result["record"])
                        write_accepted(result["record"])
                        accepted_count += 1
                        if quota_tracker:
                            quota_tracker.record_accepted(seed_row)
                        event = make_event(
                            seed_row,
                            status="accepted",
                            reason="ACCEPTED",
                            details={
                                "gate1": result["record"]["gate1"],
                                "gate2": result["record"]["gate2"],
                                "gate3": gate3,
                            },
                        )
                else:
                    continue

                summary.record(event)
                if telemetry_path:
                    record_attempt(telemetry_path, event)

                # Periodic summary every 500 seeds
                if i % PROGRESS_REPORT_INTERVAL == 0:
                    rate = accepted_count / max(i, 1) * 100
                    logger.info(
                        "Progress: %d/%d processed | %d accepted (%.1f%%) | %d rejected",
                        i, total, accepted_count, rate, rejected_count
                    )

    return summary.to_dict()


# -------------------------------------------------------------------
# CLI helpers
# -------------------------------------------------------------------

def build_llm_client(mode: str):
    """
    Built-in smoke-test modes only.
    Production should inject a real LLM adapter programmatically.
    """
    if mode == "echo_seed":
        return EchoSeedLLMClient()
    elif mode == "static_demo":
        return StaticJSONLLMClient(
            question="How does the model work?",
            answer="This is a static smoke-test answer."
        )
    else:
        raise ValueError(
            f"Unknown --llm-mode={mode!r}. "
            "Built-in options: echo_seed, static_demo. "
            "For production, use the library API and pass a real client object."
        )


def build_embedding_index(mode: str):
    """
    Built-in smoke-test modes only.
    Production should inject a real embedding adapter programmatically.
    """
    if mode == "always_pass":
        return AlwaysPassEmbeddingIndex()
    else:
        raise ValueError(
            f"Unknown --embedding-mode={mode!r}. "
            "Built-in option: always_pass. "
            "For production, use the library API and pass a real embedding adapter."
        )


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed-seeds", required=True, help="Path to seed_routes.jsonl")
    parser.add_argument("--profiles", required=True, help="Path to model_profiles_merged.json")
    parser.add_argument("--accepted", required=True, help="Output accepted.jsonl")
    parser.add_argument("--rejected", required=True, help="Output rejected.jsonl")
    parser.add_argument("--telemetry", default=None, help="Optional telemetry.jsonl path")
    parser.add_argument("--summary-out", required=True, help="Output JSON summary path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of seeds to process")
    parser.add_argument("--near-dup-threshold", type=float, default=0.985)

    parser.add_argument("--max-workers", type=int, default=8, help="Max concurrent LLM calls")

    parser.add_argument("--quotas", default=None, help="Optional quota config JSON path")
    parser.add_argument("--records-jsonl", default=None, help="Optional all_records.jsonl for support exemplar lookup")

    # smoke-test adapters
    parser.add_argument("--llm-mode", default="echo_seed", help="echo_seed | static_demo")
    parser.add_argument("--embedding-mode", default="always_pass", help="always_pass")
    args = parser.parse_args()

    routed_seeds = load_jsonl(args.routed_seeds)
    model_profiles = load_json(args.profiles)

    support_index = None
    if args.records_jsonl:
        support_index = build_support_index(load_jsonl(args.records_jsonl))

    llm_client = build_llm_client(args.llm_mode)
    embedding_index = build_embedding_index(args.embedding_mode)
    dedup_index = InMemoryDedupIndex()

    processed_seed_ids: Set[str] = set()
    existing: List[Dict[str, Any]] = []
    from pathlib import Path
    if Path(args.accepted).exists():
        existing = load_jsonl(args.accepted, tolerant=True)
        for row in existing:
            dedup_index.add(row)
            sid = row.get("seed_id")
            if sid:
                processed_seed_ids.add(sid)
    if Path(args.rejected).exists():
        for row in load_jsonl(args.rejected, tolerant=True):
            sid = row.get("seed_id")
            if sid:
                processed_seed_ids.add(sid)
    if processed_seed_ids:
        print(f"Resuming: skipping {len(processed_seed_ids)} already-processed seeds")

    quotas = load_quotas(args.quotas)
    quota_tracker = QuotaTracker(quotas)
    if existing:
        quota_tracker.seed_from_existing(existing)

    summary = generate_synthetic(
        routed_seeds=routed_seeds,
        model_profiles=model_profiles,
        llm_client=llm_client,
        embedding_index=embedding_index,
        dedup_index=dedup_index,
        accepted_path=args.accepted,
        rejected_path=args.rejected,
        telemetry_path=args.telemetry,
        support_index=support_index,
        near_dup_threshold=args.near_dup_threshold,
        limit=args.limit,
        max_workers=args.max_workers,
        processed_seed_ids=processed_seed_ids,
        quota_tracker=quota_tracker,
    )

    save_json(summary, args.summary_out)


if __name__ == "__main__":
    main()
