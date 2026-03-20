"""Tranche-based pilot runner for synthetic QA generation.

Reads planned_tasks.jsonl, executes in tranches, enforces per-tranche
quota targets, and applies stop conditions.
"""

import argparse
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .io_utils import load_json, load_jsonl, save_json, save_jsonl
from .run_synthetic_v2 import generate_synthetic
from .support_context import build_support_index
from .gate3_dedup import InMemoryDedupIndex

logger = logging.getLogger(__name__)


# ── Stop-condition thresholds ──────────────────────────────────────
STOP_BAD_JSON_RATE = 0.05        # > 5% BAD_JSON → stop
STOP_LLM_ERROR_RATE = 0.05      # > 5% LLM_CALL_ERROR → stop
STOP_DUP_RATE = 0.25            # > 25% QUESTION_TEMPLATE_DUP or NEAR_DUP → stop
STOP_MIN_ACCEPTANCE_RATE = 0.75  # < 75% acceptance → stop


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
    """Merge task metadata into a copy of the seed row.

    Task fields (task_key, transform_type, provider_hint, support_record_id)
    overlay the seed's fields. Original seed_q/seed_a are preserved.
    """
    merged = dict(seed_row)
    # overlay task-specific fields
    for key in (
        "task_key",
        "transform_type",
        "provider_hint",
        "support_record_id",
        "template_id",
        "support_relation",
        "support_source",
        "support_score",
        "model_bucket",
        "priority",
        "tier_route_key",
    ):
        if key in task:
            merged[key] = task[key]
    # ensure route_mode from task wins (may differ if task was re-planned)
    if "route_mode" in task:
        merged["route_mode"] = task["route_mode"]
    return merged


def select_tasks_for_tranche(
    tasks: List[Dict[str, Any]],
    tranche_size: int,
    remaining_targets: Optional[Dict[str, int]],
    already_scheduled: Set[str],
) -> List[Dict[str, Any]]:
    """Pick the next tranche of tasks respecting quotas and dedup.

    Tasks are assumed to be pre-sorted by priority (descending).
    Skips any task whose task_key is in already_scheduled.
    If remaining_targets is provided (keyed by route_mode), respects those limits.
    """
    selected: List[Dict[str, Any]] = []
    mode_counts: Counter = Counter()

    for task in tasks:
        if len(selected) >= tranche_size:
            break

        tk = task.get("task_key")
        if tk and tk in already_scheduled:
            continue

        # Respect remaining per-mode targets if provided
        if remaining_targets:
            mode = task.get("route_mode")
            if mode and mode in remaining_targets:
                if mode_counts[mode] >= remaining_targets[mode]:
                    continue

        selected.append(task)
        mode_counts[task.get("route_mode")] += 1

    return selected


def _check_stop_conditions(tranche_summary: Dict[str, Any], tranche_size: int) -> Optional[str]:
    """Check whether a tranche's results trigger a stop condition.

    Returns a stop-reason string, or None if OK.
    """
    by_reason = tranche_summary.get("by_reason", {})
    by_status = tranche_summary.get("by_status", {})
    total = tranche_summary.get("total_events", 0)
    if total == 0:
        return "EMPTY_TRANCHE"

    bad_json = by_reason.get("BAD_JSON", 0)
    llm_error = by_reason.get("LLM_CALL_ERROR", 0)
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


def run_tranche(
    tasks: List[Dict[str, Any]],
    seed_index: Dict[str, Dict[str, Any]],
    profiles: Dict[str, Any],
    runtime: Dict[str, Any],
    support_index: Optional[Dict[str, Any]],
    out_dir: str,
    tranche_num: int,
    dedup_index: Optional[InMemoryDedupIndex] = None,
) -> Dict[str, Any]:
    """Execute one tranche through generate_synthetic and return summary.

    Writes tranche-level accepted.jsonl, rejected.jsonl, and telemetry.jsonl
    under out_dir/tranche_NNN/.
    """
    tranche_dir = os.path.join(out_dir, f"tranche_{tranche_num:03d}")
    os.makedirs(tranche_dir, exist_ok=True)

    accepted_path = os.path.join(tranche_dir, "accepted.jsonl")
    rejected_path = os.path.join(tranche_dir, "rejected.jsonl")
    telemetry_path = os.path.join(tranche_dir, "telemetry.jsonl")

    # Build seed rows from tasks by merging with seed index
    seed_rows = []
    for task in tasks:
        sid = task.get("seed_id")
        base_seed = seed_index.get(sid, {})
        merged = attach_task_to_seed(base_seed, task)
        seed_rows.append(merged)

    llm_client = runtime.get("llm_client")
    embedding_index = runtime.get("embedding_index")
    if dedup_index is None:
        dedup_index = InMemoryDedupIndex(
            similarity_backend=runtime.get("question_similarity_backend")
        )

    summary = generate_synthetic(
        routed_seeds=seed_rows,
        model_profiles=profiles,
        llm_client=llm_client,
        embedding_index=embedding_index,
        dedup_index=dedup_index,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        telemetry_path=telemetry_path,
        support_index=support_index,
    )

    summary["tranche_num"] = tranche_num
    summary["tranche_dir"] = tranche_dir
    summary["task_count"] = len(tasks)

    # Save per-tranche summary
    save_json(summary, os.path.join(tranche_dir, "summary.json"))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tranche-based pilot runner for synthetic QA generation"
    )
    parser.add_argument(
        "--planned-tasks", required=True,
        help="Path to planned_tasks.jsonl (sorted by priority descending)"
    )
    parser.add_argument(
        "--routed-seeds", required=True,
        help="Path to seed_routes.jsonl"
    )
    parser.add_argument(
        "--records", default=None,
        help="Path to all_records.jsonl for support exemplar lookup"
    )
    parser.add_argument(
        "--profiles", required=True,
        help="Path to model_profiles_merged.json"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for tranche results"
    )
    parser.add_argument(
        "--tranche-size", type=int, default=2000,
        help="Number of tasks per tranche (default: 2000)"
    )
    parser.add_argument(
        "--target-total", type=int, default=10000,
        help="Target total accepted count (default: 10000)"
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Path to prior run output directory (for resume)"
    )
    parser.add_argument(
        "--runtime-config", default=None,
        help="Path to production runtime config JSON"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # 1. Load planned tasks (already sorted by priority descending)
    logger.info("Loading planned tasks from %s", args.planned_tasks)
    all_tasks = load_jsonl(args.planned_tasks)
    logger.info("Loaded %d planned tasks", len(all_tasks))

    # 2. Load routed seeds and build index
    logger.info("Loading routed seeds from %s", args.routed_seeds)
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

    # 5. Build runtime (LLM client, embedding, dedup)
    from .production_runtime import build_runtime
    runtime = build_runtime(args.runtime_config)

    # 6. Resume: load prior task keys
    already_scheduled: Set[str] = set()
    total_accepted = 0
    total_rejected = 0
    dedup_index = InMemoryDedupIndex(
        similarity_backend=runtime.get("question_similarity_backend")
    )

    if args.resume_from and Path(args.resume_from).exists():
        logger.info("Resuming from %s", args.resume_from)
        resume_dir = Path(args.resume_from)
        # Collect all accepted/rejected paths from prior tranches
        prior_paths = []
        for tranche_subdir in sorted(resume_dir.glob("tranche_*")):
            for fname in ("accepted.jsonl", "rejected.jsonl"):
                p = tranche_subdir / fname
                if p.exists():
                    prior_paths.append(str(p))

        already_scheduled = load_seen_task_keys(prior_paths)
        logger.info("Loaded %d prior task keys for resume", len(already_scheduled))

        # Seed dedup index from prior accepted
        for tranche_subdir in sorted(resume_dir.glob("tranche_*")):
            acc_path = tranche_subdir / "accepted.jsonl"
            if acc_path.exists():
                for row in load_jsonl(str(acc_path), tolerant=True):
                    dedup_index.add(row)
                    total_accepted += 1
            rej_path = tranche_subdir / "rejected.jsonl"
            if rej_path.exists():
                total_rejected += len(load_jsonl(str(rej_path), tolerant=True))

        logger.info(
            "Resume state: %d accepted, %d rejected, %d scheduled",
            total_accepted, total_rejected, len(already_scheduled)
        )

    # 7. Tranche loop
    os.makedirs(args.out_dir, exist_ok=True)
    tranche_num = 0
    tranche_summaries: List[Dict[str, Any]] = []
    stop_reason: Optional[str] = None

    while total_accepted < args.target_total:
        # Select tasks for this tranche
        remaining = args.target_total - total_accepted
        effective_size = min(args.tranche_size, remaining * 2)  # overshoot for rejections

        tranche_tasks = select_tasks_for_tranche(
            tasks=all_tasks,
            tranche_size=effective_size,
            remaining_targets=None,
            already_scheduled=already_scheduled,
        )

        if not tranche_tasks:
            logger.warning("No more tasks available — exhausted task pool")
            stop_reason = "TASK_POOL_EXHAUSTED"
            break

        logger.info(
            "Tranche %d: %d tasks selected (target remaining: %d)",
            tranche_num, len(tranche_tasks), remaining
        )

        # Mark tasks as scheduled
        for t in tranche_tasks:
            tk = t.get("task_key")
            if tk:
                already_scheduled.add(tk)

        # Execute tranche
        t0 = time.time()
        summary = run_tranche(
            tasks=tranche_tasks,
            seed_index=seed_index,
            profiles=profiles,
            runtime=runtime,
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
            "Tranche %d complete: %d accepted, %d rejected (%.1fs). "
            "Cumulative: %d/%d accepted",
            tranche_num, tranche_accepted, tranche_rejected, elapsed,
            total_accepted, args.target_total
        )

        # Print rejection breakdown
        by_reason = summary.get("by_reason", {})
        if by_reason:
            breakdown = ", ".join(
                f"{k}={v}" for k, v in sorted(by_reason.items(), key=lambda x: -x[1])
            )
            logger.info("  Rejection breakdown: %s", breakdown)

        tranche_summaries.append(summary)

        # Check stop conditions
        stop_reason = _check_stop_conditions(summary, len(tranche_tasks))
        if stop_reason:
            logger.warning("Stop condition triggered: %s", stop_reason)
            break

        tranche_num += 1

    # 8. Write final summary
    final_summary = {
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "target_total": args.target_total,
        "tranches_completed": len(tranche_summaries),
        "stop_reason": stop_reason,
        "tranche_summaries": tranche_summaries,
    }
    final_path = os.path.join(args.out_dir, "final_summary.json")
    save_json(final_summary, final_path)
    logger.info("Final summary written to %s", final_path)
    logger.info(
        "Done: %d accepted / %d target (%d tranches, stop=%s)",
        total_accepted, args.target_total, len(tranche_summaries), stop_reason
    )


if __name__ == "__main__":
    main()
