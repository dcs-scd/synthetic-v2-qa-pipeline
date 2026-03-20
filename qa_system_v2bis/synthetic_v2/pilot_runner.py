"""Tranche-based pilot runner for synthetic QA generation.

Reads planned_tasks.jsonl, executes in tranches via batch APIs
(OpenAI and xAI) for 50% cost reduction, enforces per-tranche
quota targets, and applies stop conditions.

Nano limit: 1,000 requests per OpenAI batch (auto-chunked).
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .io_utils import load_json, load_jsonl, save_json, save_jsonl
from .prompt_builder import build_prompt, get_model_profile
from .run_synthetic_v2 import parse_llm_json
from .gate2_model_validation import validate_generated_row
from .gate3_dedup import InMemoryDedupIndex, run_gate3_dedup
from .support_context import build_support_index
from .text_utils import normalize_model_name

logger = logging.getLogger(__name__)


# ── Stop-condition thresholds ──────────────────────────────────────
STOP_BAD_JSON_RATE = 0.05        # > 5% BAD_JSON → stop
STOP_LLM_ERROR_RATE = 0.05      # > 5% LLM_CALL_ERROR → stop
STOP_DUP_RATE = 0.25            # > 25% QUESTION_TEMPLATE_DUP or NEAR_DUP → stop
STOP_MIN_ACCEPTANCE_RATE = 0.75  # < 75% acceptance → stop

# ── Batch API defaults ─────────────────────────────────────────────
NANO_MODEL = "gpt-5.4-nano"
NANO_TEMPERATURE = 0.4
NANO_MAX_TOKENS = 1400
GROK_MODEL = "grok-4-1-fast-reasoning"
GROK_TEMPERATURE = 0.4
GROK_MAX_TOKENS = 1200
GROK_ADD_CHUNK_SIZE = 20
BATCH_POLL_INTERVAL = 15.0
BATCH_MAX_WAIT = 7200


def index_routed_seeds_by_id(routed_seeds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index seeds by seed_id for O(1) lookup."""
    index: Dict[str, Dict[str, Any]] = {}
    for seed in routed_seeds:
        sid = seed.get("seed_id")
        if sid:
            index[sid] = seed
    return index


def load_seen_task_keys(paths: List[str]) -> Set[str]:
    """Load already-done task keys from prior accepted/rejected JSONL files."""
    seen: Set[str] = set()
    for path in paths:
        if not Path(path).exists():
            continue
        for row in load_jsonl(path, tolerant=True):
            tk = row.get("task_key")
            if tk:
                seen.add(tk)
    return seen


def attach_task_to_seed(seed_row: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """Merge task metadata into a copy of the seed row."""
    merged = dict(seed_row)
    for key in (
        "task_key", "transform_type", "provider_hint", "support_record_id",
        "template_id", "support_relation", "support_source", "support_score",
        "model_bucket", "priority", "tier_route_key",
    ):
        if key in task:
            merged[key] = task[key]
    if "route_mode" in task:
        merged["route_mode"] = task["route_mode"]
    return merged


def select_tasks_for_tranche(
    tasks: List[Dict[str, Any]],
    tranche_size: int,
    remaining_targets: Optional[Dict[str, int]],
    already_scheduled: Set[str],
) -> List[Dict[str, Any]]:
    """Pick the next tranche of tasks respecting quotas and dedup."""
    selected: List[Dict[str, Any]] = []
    mode_counts: Counter = Counter()

    for task in tasks:
        if len(selected) >= tranche_size:
            break
        tk = task.get("task_key")
        if tk and tk in already_scheduled:
            continue
        if remaining_targets:
            mode = task.get("route_mode")
            if mode and mode in remaining_targets:
                if mode_counts[mode] >= remaining_targets[mode]:
                    continue
        selected.append(task)
        mode_counts[task.get("route_mode")] += 1

    return selected


def _check_stop_conditions(tranche_summary: Dict[str, Any], tranche_size: int) -> Optional[str]:
    """Check whether a tranche's results trigger a stop condition."""
    by_reason = tranche_summary.get("by_reason", {})
    by_status = tranche_summary.get("by_status", {})
    total = tranche_summary.get("total_events", 0)
    if total == 0:
        return "EMPTY_TRANCHE"

    bad_json = by_reason.get("BAD_JSON", 0)
    llm_error = by_reason.get("LLM_CALL_ERROR", 0) + by_reason.get("NO_RESPONSE", 0)
    q_dup = by_reason.get("QUESTION_TEMPLATE_DUP", 0)
    near_dup = by_reason.get("NEAR_DUP_QUESTION", 0)
    accepted = by_status.get("accepted", 0)

    if bad_json / total > STOP_BAD_JSON_RATE:
        return f"BAD_JSON_RATE={bad_json}/{total}={bad_json/total:.2%}"
    if llm_error / total > STOP_LLM_ERROR_RATE:
        return f"LLM_ERROR_RATE={llm_error}/{total}={llm_error/total:.2%}"
    dup_total = q_dup + near_dup
    if dup_total / total > STOP_DUP_RATE:
        return f"DUP_RATE={dup_total}/{total}={dup_total/total:.2%}"
    acceptance_rate = accepted / total
    if acceptance_rate < STOP_MIN_ACCEPTANCE_RATE:
        return f"LOW_ACCEPTANCE={accepted}/{total}={acceptance_rate:.2%}"
    return None


# ── Batch submission helpers ──────────────────────────────────────


def _build_prompts_for_tasks(
    tasks: List[Dict[str, Any]],
    seed_index: Dict[str, Dict[str, Any]],
    profiles: Dict[str, Any],
    support_index: Optional[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Optional[str]]]:
    """Build prompts for all tasks. Returns list of (merged_seed, prompt|None)."""
    results = []
    for task in tasks:
        sid = task.get("seed_id")
        base_seed = seed_index.get(sid, {})
        merged = attach_task_to_seed(base_seed, task)
        mn = normalize_model_name(merged.get("model_name"))
        profile = get_model_profile(profiles, mn)
        if not profile:
            results.append((merged, None))
            continue
        try:
            prompt = build_prompt(merged, profile, support_index=support_index)
            results.append((merged, prompt))
        except Exception as e:
            logger.warning("Failed to build prompt for %s: %s", merged.get("task_key"), e)
            results.append((merged, None))
    return results


def _submit_nano_batch(
    prompts_with_indices: List[Tuple[int, str]],
    tranche_dir: str,
) -> List[str]:
    """Submit nano prompts via OpenAI batch API (chunked at 1000). Returns batch_ids."""
    from openai import OpenAI
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from batch_api_utils import (
        build_openai_batch_request, submit_openai_batch_chunked,
    )

    client = OpenAI(api_key=os.environ.get("OAI_KEY") or os.environ.get("OPENAI_API_KEY"))

    requests = []
    for idx, prompt in prompts_with_indices:
        req = build_openai_batch_request(
            custom_id=f"req-{idx}",
            prompt=prompt,
            model=NANO_MODEL,
            temperature=NANO_TEMPERATURE,
            max_tokens=NANO_MAX_TOKENS,
        )
        requests.append(req)

    logger.info("Submitting %d nano requests via batch API (chunked at 1000)", len(requests))
    batch_ids = submit_openai_batch_chunked(
        client=client,
        requests=requests,
        temp_dir=os.path.join(tranche_dir, "batch_tmp"),
        description=f"Pilot tranche nano batch",
    )
    return batch_ids


def _poll_and_collect_nano(batch_ids: List[str]) -> Dict[int, str]:
    """Poll all nano batches and collect results. Returns {original_idx: raw_text}."""
    from openai import OpenAI
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from batch_api_utils import poll_and_collect_all_chunks

    client = OpenAI(api_key=os.environ.get("OAI_KEY") or os.environ.get("OPENAI_API_KEY"))
    results_by_index, summary = poll_and_collect_all_chunks(
        client=client,
        batch_ids=batch_ids,
        poll_interval=BATCH_POLL_INTERVAL,
        timeout=BATCH_MAX_WAIT,
    )
    logger.info(
        "Nano batch collection: %d/%d results (%.1f%%)",
        summary["total_collected"], summary["total_expected"],
        summary["collection_rate"] * 100,
    )
    return results_by_index


def _submit_grok_batch(
    prompts_with_indices: List[Tuple[int, str]],
) -> str:
    """Submit grok prompts via xAI batch API. Returns batch_id."""
    from xai_sdk import Client as XAIClient

    client = XAIClient(api_key=os.environ.get("XAI_KEY") or os.environ.get("XAI_API_KEY"))

    batch = client.batch.create()
    batch_id = batch.id
    logger.info("Created Grok batch %s for %d requests", batch_id, len(prompts_with_indices))

    # Add requests in chunks
    for i in range(0, len(prompts_with_indices), GROK_ADD_CHUNK_SIZE):
        chunk = prompts_with_indices[i:i + GROK_ADD_CHUNK_SIZE]
        for idx, prompt in chunk:
            client.batch.add_request(
                batch_id=batch_id,
                batch_request_id=str(idx),
                model=GROK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GROK_TEMPERATURE,
                max_tokens=GROK_MAX_TOKENS,
            )
        time.sleep(0.5)

    client.batch.start(batch_id)
    logger.info("Started Grok batch %s", batch_id)
    return batch_id


def _poll_and_collect_grok(batch_id: str) -> Dict[int, str]:
    """Poll grok batch and collect results. Returns {original_idx: raw_text}."""
    from xai_sdk import Client as XAIClient

    client = XAIClient(api_key=os.environ.get("XAI_KEY") or os.environ.get("XAI_API_KEY"))

    start = time.time()
    while time.time() - start < BATCH_MAX_WAIT:
        status = client.batch.get(batch_id)
        if status.status in ("completed", "failed", "expired"):
            break
        logger.info("Grok batch %s: status=%s", batch_id, status.status)
        time.sleep(BATCH_POLL_INTERVAL)

    if status.status != "completed":
        logger.error("Grok batch %s finished with status %s", batch_id, status.status)
        return {}

    results = {}
    pt = None
    while True:
        page = client.batch.list_batch_results(batch_id=batch_id, limit=500, pagination_token=pt)
        for r in page.succeeded:
            try:
                idx = int(r.batch_request_id)
                results[idx] = r.response.content if r.response else ""
            except (ValueError, AttributeError):
                pass
        pt = page.pagination_token
        if not pt:
            break

    logger.info("Grok batch collected %d results", len(results))
    return results


def _gate_results(
    all_results: Dict[int, str],
    task_seed_pairs: List[Tuple[Dict[str, Any], Optional[str]]],
    profiles: Dict[str, Any],
    dedup_index: InMemoryDedupIndex,
    tranche_dir: str,
) -> Dict[str, Any]:
    """Run gates on raw LLM results. Write accepted/rejected. Return summary."""
    accepted = []
    rejected = []
    by_reason: Counter = Counter()
    by_status: Counter = Counter()

    for idx, (merged_seed, prompt) in enumerate(task_seed_pairs):
        task_key = merged_seed.get("task_key", f"idx-{idx}")
        provider = merged_seed.get("provider_hint", "unknown")
        model_name = merged_seed.get("model_name", "")

        raw = all_results.get(idx)
        if raw is None or raw == "":
            by_reason["NO_RESPONSE"] += 1
            by_status["rejected"] += 1
            rejected.append({
                "task_key": task_key, "model_name": model_name,
                "reason": "NO_RESPONSE", "provider": provider,
            })
            continue

        # Parse JSON
        parsed = parse_llm_json(raw)
        if not parsed.get("ok"):
            reason = "BAD_JSON"
            by_reason[reason] += 1
            by_status["rejected"] += 1
            rejected.append({
                "task_key": task_key, "model_name": model_name,
                "reason": reason, "provider": provider,
                "error": parsed.get("error", ""),
            })
            continue

        data = parsed["data"]
        gen_q = data.get("question", "")
        gen_a = data.get("answer", "")

        # Gate 2: model validation
        mn_norm = normalize_model_name(model_name)
        profile = get_model_profile(profiles, mn_norm)
        if profile:
            g2 = validate_generated_row(
                question=gen_q, answer=gen_a,
                model_name=mn_norm, profile=profile,
                route_mode=merged_seed.get("route_mode"),
                extension_family=merged_seed.get("extension_family"),
            )
            if not g2.get("pass"):
                reason = g2.get("reason", "GATE2_FAIL")
                by_reason[reason] += 1
                by_status["rejected"] += 1
                rejected.append({
                    "task_key": task_key, "model_name": model_name,
                    "reason": reason, "provider": provider,
                })
                continue

        # Gate 3: dedup
        g3 = run_gate3_dedup(gen_q, dedup_index)
        if not g3.get("pass"):
            reason = g3.get("reason", "NEAR_DUP_QUESTION")
            by_reason[reason] += 1
            by_status["rejected"] += 1
            rejected.append({
                "task_key": task_key, "model_name": model_name,
                "reason": reason, "provider": provider,
            })
            continue

        # Accepted
        dedup_index.add({"question": gen_q})
        by_status["accepted"] += 1
        accepted.append({
            "task_key": task_key,
            "seed_id": merged_seed.get("seed_id"),
            "model_name": model_name,
            "route_mode": merged_seed.get("route_mode"),
            "transform_type": merged_seed.get("transform_type"),
            "provider": provider,
            "question": gen_q,
            "answer": gen_a,
        })

    # Write results
    save_jsonl(accepted, os.path.join(tranche_dir, "accepted.jsonl"))
    save_jsonl(rejected, os.path.join(tranche_dir, "rejected.jsonl"))

    summary = {
        "total_events": len(task_seed_pairs),
        "by_status": dict(by_status),
        "by_reason": dict(by_reason),
    }
    return summary


def run_tranche(
    tasks: List[Dict[str, Any]],
    seed_index: Dict[str, Dict[str, Any]],
    profiles: Dict[str, Any],
    support_index: Optional[Dict[str, Any]],
    out_dir: str,
    tranche_num: int,
    dedup_index: Optional[InMemoryDedupIndex] = None,
) -> Dict[str, Any]:
    """Execute one tranche via batch APIs and return summary.

    Flow:
    1. Build prompts for all tasks
    2. Split by provider (nano vs grok)
    3. Submit nano via OpenAI batch API (chunked at 1000)
    4. Submit grok via xAI batch API
    5. Poll both, collect results
    6. Run gates on collected results
    """
    tranche_dir = os.path.join(out_dir, f"tranche_{tranche_num:03d}")
    os.makedirs(tranche_dir, exist_ok=True)

    # 1. Build prompts
    logger.info("Building prompts for %d tasks...", len(tasks))
    task_seed_pairs = _build_prompts_for_tasks(tasks, seed_index, profiles, support_index)

    # 2. Split by provider
    nano_prompts: List[Tuple[int, str]] = []
    grok_prompts: List[Tuple[int, str]] = []
    no_prompt_count = 0

    for idx, (merged_seed, prompt) in enumerate(task_seed_pairs):
        if prompt is None:
            no_prompt_count += 1
            continue
        provider = merged_seed.get("provider_hint", "gpt-5.4-nano")
        if "grok" in provider:
            grok_prompts.append((idx, prompt))
        else:
            nano_prompts.append((idx, prompt))

    logger.info(
        "Provider split: %d nano, %d grok, %d no-prompt",
        len(nano_prompts), len(grok_prompts), no_prompt_count,
    )

    # 3+4. Submit batches (parallel if both providers present)
    all_results: Dict[int, str] = {}
    batch_meta: Dict[str, Any] = {}

    if nano_prompts:
        nano_batch_ids = _submit_nano_batch(nano_prompts, tranche_dir)
        batch_meta["nano_batch_ids"] = nano_batch_ids
        batch_meta["nano_count"] = len(nano_prompts)

    if grok_prompts:
        grok_batch_id = _submit_grok_batch(grok_prompts)
        batch_meta["grok_batch_id"] = grok_batch_id
        batch_meta["grok_count"] = len(grok_prompts)

    # Save batch metadata for resume/debugging
    save_json(batch_meta, os.path.join(tranche_dir, "batch_meta.json"))

    # 5. Poll and collect
    if nano_prompts:
        nano_results = _poll_and_collect_nano(nano_batch_ids)
        # Map back: nano_results keys are req-{idx} indices, need to map to our idx
        for local_idx, (original_idx, _) in enumerate(nano_prompts):
            raw = nano_results.get(local_idx)
            if raw is not None:
                all_results[original_idx] = raw

    if grok_prompts:
        grok_results = _poll_and_collect_grok(grok_batch_id)
        for original_idx, raw in grok_results.items():
            all_results[original_idx] = raw

    logger.info(
        "Collected %d/%d results (%.1f%%)",
        len(all_results), len(nano_prompts) + len(grok_prompts),
        100 * len(all_results) / max(1, len(nano_prompts) + len(grok_prompts)),
    )

    # 6. Gate results
    if dedup_index is None:
        dedup_index = InMemoryDedupIndex()

    summary = _gate_results(all_results, task_seed_pairs, profiles, dedup_index, tranche_dir)
    summary["tranche_num"] = tranche_num
    summary["tranche_dir"] = tranche_dir
    summary["task_count"] = len(tasks)
    summary["batch_meta"] = batch_meta

    save_json(summary, os.path.join(tranche_dir, "summary.json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tranche-based pilot runner (batch API mode)"
    )
    parser.add_argument("--planned-tasks", required=True)
    parser.add_argument("--routed-seeds", required=True)
    parser.add_argument("--records", default=None)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tranche-size", type=int, default=1000,
                        help="Tasks per tranche (default 1000, nano batch limit)")
    parser.add_argument("--target-total", type=int, default=10000)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # 1. Load planned tasks
    logger.info("Loading planned tasks from %s", args.planned_tasks)
    all_tasks = load_jsonl(args.planned_tasks)
    logger.info("Loaded %d planned tasks", len(all_tasks))

    # 2. Load routed seeds and build index
    routed_seeds = load_jsonl(args.routed_seeds)
    seed_index = index_routed_seeds_by_id(routed_seeds)
    logger.info("Indexed %d seeds", len(seed_index))

    # 3. Load model profiles
    profiles = load_json(args.profiles)

    # 4. Load support index
    support_index = None
    if args.records:
        records = load_jsonl(args.records)
        support_index = build_support_index(records)
        logger.info("Built support index from %d records", len(records))

    # 5. Resume
    already_scheduled: Set[str] = set()
    total_accepted = 0
    total_rejected = 0
    dedup_index = InMemoryDedupIndex()

    if args.resume_from and Path(args.resume_from).exists():
        logger.info("Resuming from %s", args.resume_from)
        resume_dir = Path(args.resume_from)
        prior_paths = []
        for tranche_subdir in sorted(resume_dir.glob("tranche_*")):
            for fname in ("accepted.jsonl", "rejected.jsonl"):
                p = tranche_subdir / fname
                if p.exists():
                    prior_paths.append(str(p))

        already_scheduled = load_seen_task_keys(prior_paths)
        logger.info("Loaded %d prior task keys", len(already_scheduled))

        for tranche_subdir in sorted(resume_dir.glob("tranche_*")):
            acc_path = tranche_subdir / "accepted.jsonl"
            if acc_path.exists():
                for row in load_jsonl(str(acc_path), tolerant=True):
                    dedup_index.add(row)
                    total_accepted += 1
            rej_path = tranche_subdir / "rejected.jsonl"
            if rej_path.exists():
                total_rejected += len(load_jsonl(str(rej_path), tolerant=True))

        logger.info("Resume: %d accepted, %d rejected", total_accepted, total_rejected)

    # 6. Tranche loop
    os.makedirs(args.out_dir, exist_ok=True)
    tranche_num = 0
    tranche_summaries: List[Dict[str, Any]] = []
    stop_reason: Optional[str] = None

    while total_accepted < args.target_total:
        remaining = args.target_total - total_accepted
        effective_size = min(args.tranche_size, remaining * 2)

        tranche_tasks = select_tasks_for_tranche(
            tasks=all_tasks,
            tranche_size=effective_size,
            remaining_targets=None,
            already_scheduled=already_scheduled,
        )

        if not tranche_tasks:
            logger.warning("Exhausted task pool")
            stop_reason = "TASK_POOL_EXHAUSTED"
            break

        logger.info(
            "Tranche %d: %d tasks (remaining: %d)",
            tranche_num, len(tranche_tasks), remaining,
        )

        for t in tranche_tasks:
            tk = t.get("task_key")
            if tk:
                already_scheduled.add(tk)

        t0 = time.time()
        summary = run_tranche(
            tasks=tranche_tasks,
            seed_index=seed_index,
            profiles=profiles,
            support_index=support_index,
            out_dir=args.out_dir,
            tranche_num=tranche_num,
            dedup_index=dedup_index,
        )
        elapsed = time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)

        tranche_accepted = summary.get("by_status", {}).get("accepted", 0)
        tranche_rejected = summary.get("total_events", 0) - tranche_accepted
        total_accepted += tranche_accepted
        total_rejected += tranche_rejected

        logger.info(
            "Tranche %d: %d accepted, %d rejected (%.1fs). Cumulative: %d/%d",
            tranche_num, tranche_accepted, tranche_rejected, elapsed,
            total_accepted, args.target_total,
        )

        by_reason = summary.get("by_reason", {})
        if by_reason:
            logger.info("  Rejections: %s", ", ".join(
                f"{k}={v}" for k, v in sorted(by_reason.items(), key=lambda x: -x[1])
            ))

        tranche_summaries.append(summary)

        stop_reason = _check_stop_conditions(summary, len(tranche_tasks))
        if stop_reason:
            logger.warning("Stop: %s", stop_reason)
            break

        tranche_num += 1

    # 7. Final summary
    final_summary = {
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "target_total": args.target_total,
        "tranches_completed": len(tranche_summaries),
        "stop_reason": stop_reason,
        "tranche_summaries": tranche_summaries,
    }
    save_json(final_summary, os.path.join(args.out_dir, "final_summary.json"))
    logger.info(
        "Done: %d/%d accepted (%d tranches, stop=%s)",
        total_accepted, args.target_total, len(tranche_summaries), stop_reason,
    )


if __name__ == "__main__":
    main()
