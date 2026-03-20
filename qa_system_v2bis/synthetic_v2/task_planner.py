import argparse
import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

from .io_utils import load_json, load_jsonl, save_json, save_jsonl
from .text_utils import normalize_model_name


DEFAULT_PLANNER_CONFIG = {
    "support": {
        "top_k_per_seed": 12,
        "allowed_sources": ["corpus", "seed"],
        "allow_cross_model_same_class_support": False,
        "score_same_model": 50,
        "score_same_class": 40,
        "score_same_level": 5,
        "score_corpus_bonus": 8,
        "score_seed_bonus": 3,
        "score_accepted_synth_bonus": 0,
        "max_supports_per_seed_by_mode": {
            "core_paraphrase": 4,
            "core_repair": 3,
            "anchored_extension": 5
        }
    },
    "transforms": {
        "core_paraphrase": [
            "lexical_paraphrase",
            "syntactic_reframe",
            "mechanism_focus_shift",
            "measurement_reframe",
            "experiment_design_reframe",
            "limitation_reframe"
        ],
        "core_repair": [
            "unsupported_identifier_repair",
            "mechanism_grounding_repair",
            "core_operationalization_repair"
        ],
        "anchored_extension": [
            "extension_reframing",
            "extension_operationalization",
            "extension_compare_baseline",
            "extension_experiment_design",
            "extension_limitation_analysis"
        ]
    },
    "caps": {
        "max_tasks_per_seed_by_mode": {
            "core_paraphrase": 8,
            "core_repair": 5,
            "anchored_extension": 10
        },
        "max_tasks_per_support": 60
    },
    "providers": {
        "default_core_provider": "gpt-5.4-nano",
        "default_extension_provider": "gpt-5.4-nano",
        "grok_extension_families": [
            "state_refinement",
            "broadcast_media",
            "long_range_links",
            "network_layer",
            "large_scale_optimization"
        ],
        "grok_fraction_by_family": {
            "state_refinement": 0.60,
            "broadcast_media": 0.35,
            "long_range_links": 0.35,
            "network_layer": 0.20,
            "large_scale_optimization": 0.20
        },
        "grok_provider_name": "grok-4-1-fast-reasoning",
        "nano_provider_name": "gpt-5.4-nano"
    },
    "priority": {
        "tier_weight": {
            "A": 1.00,
            "B": 0.82,
            "C": 0.55
        },
        "route_weight": {
            "core_paraphrase": 0.95,
            "core_repair": 0.82,
            "anchored_extension": 0.90
        },
        "support_score_divisor": 100.0,
        "corpus_support_bonus": 0.05,
        "seed_only_penalty": 0.08
    },
    "model_bucketing": {
        "high_count_models": 8,
        "mid_count_models": 12,
        "labels": {
            "high": "high_volume",
            "mid": "mid_volume",
            "long": "long_tail"
        }
    },
    "selection": {
        "enabled": False,
        "target_total": None,

        "route_mode_targets": {},
        "tier_targets": {},
        "provider_targets": {},
        "family_targets": {},
        "model_bucket_targets": {}
    },
    "template_version": "v1"
}


def norm(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()


def stable_hash_int(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_planner_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = DEFAULT_PLANNER_CONFIG
    if path:
        override = load_json(path)
        cfg = deep_merge(cfg, override)
    return cfg


def safe_text(x: Any) -> str:
    return "" if x is None else str(x)


def build_record_indexes(records: List[Dict[str, Any]], allowed_sources: List[str]) -> Dict[str, Any]:
    filtered = [r for r in records if r.get("source") in allowed_sources]

    by_model = defaultdict(list)
    by_model_class = defaultdict(list)

    for r in filtered:
        model = normalize_model_name(r.get("model_name"))
        class_id = r.get("class_id")
        if model:
            by_model[model].append(r)
            if class_id is not None:
                by_model_class[(model, class_id)].append(r)

    return {
        "records": filtered,
        "by_model": by_model,
        "by_model_class": by_model_class
    }


def support_relation_and_score(seed_row: Dict[str, Any], rec: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Optional[str], float]:
    support_cfg = cfg["support"]

    seed_id = seed_row.get("seed_id")
    seed_model = normalize_model_name(seed_row.get("model_name"))
    seed_class = seed_row.get("class_id")
    seed_level = seed_row.get("level")

    rec_model = normalize_model_name(rec.get("model_name"))
    rec_class = rec.get("class_id")
    rec_level = rec.get("level")

    # Exclude self
    if rec.get("seed_id") == seed_id:
        return None, 0.0
    if rec.get("source") == "seed" and rec.get("source_id") == seed_id:
        return None, 0.0
    if safe_text(rec.get("question")).strip() == safe_text(seed_row.get("seed_q") or seed_row.get("question")).strip():
        # avoid seed-equivalent direct reuse as support
        pass

    same_model = (seed_model and rec_model and seed_model == rec_model)
    same_class = (seed_class is not None and rec_class is not None and seed_class == rec_class)
    same_level = (seed_level is not None and rec_level is not None and seed_level == rec_level)

    if not same_model:
        if not support_cfg["allow_cross_model_same_class_support"]:
            return None, 0.0
        if not same_class:
            return None, 0.0

    score = 0.0
    relation = "other"

    if same_model:
        score += support_cfg["score_same_model"]
        relation = "same_model"

    if same_class:
        score += support_cfg["score_same_class"]
        relation = "same_class_same_model" if same_model else "same_class_cross_model"

    if same_level:
        score += support_cfg["score_same_level"]

    source = rec.get("source")
    if source == "corpus":
        score += support_cfg["score_corpus_bonus"]
    elif source == "seed":
        score += support_cfg["score_seed_bonus"]
    elif source == "accepted_synth":
        score += support_cfg["score_accepted_synth_bonus"]

    return relation, score


def build_support_pool_for_seed(
    seed_row: Dict[str, Any],
    indexes: Dict[str, Any],
    cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    support_cfg = cfg["support"]
    seed_model = normalize_model_name(seed_row.get("model_name"))
    seed_class = seed_row.get("class_id")

    candidates = []

    # Prefer same model records
    same_model_records = indexes["by_model"].get(seed_model, [])

    for rec in same_model_records:
        relation, score = support_relation_and_score(seed_row, rec, cfg)
        if relation is None:
            continue
        candidates.append({
            "record_id": rec.get("record_id"),
            "source": rec.get("source"),
            "model_name": rec.get("model_name"),
            "class_id": rec.get("class_id"),
            "question": rec.get("question"),
            "answer": rec.get("answer"),
            "relation": relation,
            "score": score
        })

    # Optional same-class cross-model fallback if enabled
    if support_cfg["allow_cross_model_same_class_support"] and seed_class is not None:
        same_class_records = indexes["by_model_class"].get((seed_model, seed_class), [])
        # already covered in same_model loop; nothing extra needed here unless using wider index

    # Dedup by record_id, keep highest score
    best_by_id = {}
    for c in candidates:
        rid = c["record_id"]
        if not rid:
            continue
        prev = best_by_id.get(rid)
        if prev is None or c["score"] > prev["score"]:
            best_by_id[rid] = c

    candidates = list(best_by_id.values())
    candidates.sort(key=lambda x: (-x["score"], x["record_id"]))

    return candidates[:support_cfg["top_k_per_seed"]]


def build_support_pools(
    routed_seeds: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    cfg: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    indexes = build_record_indexes(records, allowed_sources=cfg["support"]["allowed_sources"])
    pools = {}

    for seed_row in routed_seeds:
        seed_id = seed_row.get("seed_id")
        if not seed_id:
            continue
        pools[seed_id] = build_support_pool_for_seed(seed_row, indexes, cfg)

    return pools


def compute_model_buckets(routed_seeds: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, str]:
    counts = Counter()
    for row in routed_seeds:
        if row.get("route_mode") == "skip":
            continue
        model_name = normalize_model_name(row.get("model_name"))
        if model_name:
            counts[model_name] += 1

    ordered = [m for m, _ in counts.most_common()]
    high_n = cfg["model_bucketing"]["high_count_models"]
    mid_n = cfg["model_bucketing"]["mid_count_models"]
    labels = cfg["model_bucketing"]["labels"]

    buckets = {}
    for i, model_name in enumerate(ordered):
        if i < high_n:
            buckets[model_name] = labels["high"]
        elif i < high_n + mid_n:
            buckets[model_name] = labels["mid"]
        else:
            buckets[model_name] = labels["long"]

    return buckets


def transforms_for_seed(seed_row: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    mode = seed_row.get("route_mode")
    return cfg["transforms"].get(mode, [])


def choose_provider_hint(
    seed_row: Dict[str, Any],
    transform_type: str,
    support_record_id: Optional[str],
    cfg: Dict[str, Any]
) -> str:
    route_mode = seed_row.get("route_mode")
    family = seed_row.get("extension_family")
    provider_cfg = cfg["providers"]

    if route_mode in {"core_paraphrase", "core_repair"}:
        return provider_cfg["default_core_provider"]

    if route_mode == "anchored_extension":
        if family in provider_cfg["grok_extension_families"]:
            frac = provider_cfg["grok_fraction_by_family"].get(family, 0.0)
            key = "|".join([
                safe_text(seed_row.get("seed_id")),
                safe_text(route_mode),
                safe_text(family),
                safe_text(transform_type),
                safe_text(support_record_id),
            ])
            bucket = stable_hash_int(key) % 100
            if bucket < int(round(frac * 100)):
                return provider_cfg["grok_provider_name"]
        return provider_cfg["default_extension_provider"]

    return provider_cfg["nano_provider_name"]


def compute_priority(
    seed_row: Dict[str, Any],
    support_entry: Dict[str, Any],
    provider_hint: str,
    cfg: Dict[str, Any]
) -> float:
    priority_cfg = cfg["priority"]

    tier = safe_text(seed_row.get("tier")).upper() or "C"
    route_mode = seed_row.get("route_mode")

    tier_w = priority_cfg["tier_weight"].get(tier, 0.5)
    route_w = priority_cfg["route_weight"].get(route_mode, 0.7)

    support_score = float(support_entry.get("score", 0.0))
    support_norm = support_score / float(priority_cfg["support_score_divisor"])

    pr = 0.0
    pr += tier_w
    pr += route_w
    pr += support_norm

    if support_entry.get("source") == "corpus":
        pr += priority_cfg["corpus_support_bonus"]

    if support_entry.get("relation") == "seed_only":
        pr -= priority_cfg["seed_only_penalty"]

    # tiny deterministic jitter to break ties stably
    jitter_key = "|".join([
        safe_text(seed_row.get("seed_id")),
        safe_text(seed_row.get("route_mode")),
        safe_text(seed_row.get("extension_family")),
        safe_text(support_entry.get("record_id")),
        safe_text(provider_hint)
    ])
    pr += (stable_hash_int(jitter_key) % 1000) / 1_000_000.0
    return round(pr, 6)


def build_task_key(
    seed_row: Dict[str, Any],
    transform_type: str,
    support_record_id: Optional[str],
    provider_hint: str,
    template_version: str
) -> str:
    payload = "|".join([
        safe_text(seed_row.get("seed_id")),
        safe_text(seed_row.get("route_mode")),
        safe_text(seed_row.get("extension_family")),
        safe_text(transform_type),
        safe_text(support_record_id),
        safe_text(provider_hint),
        safe_text(template_version),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def enumerate_tasks_for_seed(
    seed_row: Dict[str, Any],
    support_pool: List[Dict[str, Any]],
    model_bucket: str,
    cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    route_mode = seed_row.get("route_mode")
    if route_mode == "skip":
        return []

    transforms = transforms_for_seed(seed_row, cfg)
    if not transforms:
        return []

    cap = cfg["caps"]["max_tasks_per_seed_by_mode"].get(route_mode, 4)
    max_supports = cfg["support"]["max_supports_per_seed_by_mode"].get(route_mode, 3)

    supports = support_pool[:max_supports]
    if not supports:
        supports = [{
            "record_id": None,
            "source": "seed_only",
            "model_name": seed_row.get("model_name"),
            "class_id": seed_row.get("class_id"),
            "question": None,
            "answer": None,
            "relation": "seed_only",
            "score": 0.0
        }]

    tasks = []
    template_version = cfg["template_version"]

    # Round-robin supports × transforms
    combos = []
    for t in transforms:
        for s in supports:
            combos.append((t, s))

    # sort for diversity:
    # support-rich tasks first, but keep transform coverage
    combos.sort(key=lambda pair: (-float(pair[1].get("score", 0.0)), pair[0], safe_text(pair[1].get("record_id"))))

    seen = set()
    for transform_type, support_entry in combos:
        provider_hint = choose_provider_hint(seed_row, transform_type, support_entry.get("record_id"), cfg)
        task_key = build_task_key(
            seed_row=seed_row,
            transform_type=transform_type,
            support_record_id=support_entry.get("record_id"),
            provider_hint=provider_hint,
            template_version=template_version
        )
        if task_key in seen:
            continue
        seen.add(task_key)

        priority = compute_priority(seed_row, support_entry, provider_hint, cfg)

        task = {
            "task_key": task_key,
            "seed_id": seed_row.get("seed_id"),
            "model_name": normalize_model_name(seed_row.get("model_name")),
            "class_id": seed_row.get("class_id"),
            "tier": seed_row.get("tier"),
            "level": seed_row.get("level"),
            "tier_route_key": f"{seed_row.get('tier')}::{route_mode}",
            "route_mode": route_mode,
            "extension_family": seed_row.get("extension_family"),
            "transform_type": transform_type,
            "template_id": f"{route_mode}::{transform_type}::{template_version}",
            "support_record_id": support_entry.get("record_id"),
            "support_relation": support_entry.get("relation"),
            "support_source": support_entry.get("source"),
            "support_score": support_entry.get("score"),
            "provider_hint": provider_hint,
            "model_bucket": model_bucket,
            "priority": priority
        }
        tasks.append(task)

        if len(tasks) >= cap:
            break

    return tasks


def enumerate_all_tasks(
    routed_seeds: List[Dict[str, Any]],
    support_pools: Dict[str, List[Dict[str, Any]]],
    cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    model_buckets = compute_model_buckets(routed_seeds, cfg)
    tasks = []

    for seed_row in routed_seeds:
        seed_id = seed_row.get("seed_id")
        model_name = normalize_model_name(seed_row.get("model_name"))
        model_bucket = model_buckets.get(model_name, cfg["model_bucketing"]["labels"]["long"])
        pool = support_pools.get(seed_id, [])
        tasks.extend(enumerate_tasks_for_seed(seed_row, pool, model_bucket, cfg))

    # Enforce per-support cap
    max_per_support = cfg.get("caps", {}).get("max_tasks_per_support", 60)
    if max_per_support:
        support_counts = Counter()
        capped = []
        for t in tasks:
            sid = t.get("support_record_id")
            if sid:
                support_counts[sid] += 1
                if support_counts[sid] > max_per_support:
                    continue
            capped.append(t)
        tasks = capped

    # stable sort by priority descending, then task_key
    tasks.sort(key=lambda x: (-x["priority"], x["task_key"]))
    return tasks


def assign_target_counts(total: int, share_or_count_map: Dict[str, Any]) -> Dict[str, int]:
    """
    Accepts either:
    - explicit counts (ints)
    - shares (floats summing near 1.0)
    """
    if not share_or_count_map:
        return {}

    # all ints => direct
    if all(isinstance(v, int) for v in share_or_count_map.values()):
        return dict(share_or_count_map)

    # otherwise treat as shares
    raw = {}
    assigned = 0
    keys = list(share_or_count_map.keys())
    for k in keys:
        v = float(share_or_count_map[k])
        c = int(total * v)
        raw[k] = c
        assigned += c

    # distribute remainder to highest share keys
    remainder = total - assigned
    sorted_keys = sorted(keys, key=lambda k: (-float(share_or_count_map[k]), k))
    i = 0
    while remainder > 0 and sorted_keys:
        raw[sorted_keys[i % len(sorted_keys)]] += 1
        remainder -= 1
        i += 1

    return raw


def select_tasks_for_target(tasks: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sel_cfg = cfg.get("selection", {})
    if not sel_cfg.get("enabled"):
        return tasks

    target_total = sel_cfg.get("target_total")
    if not target_total:
        return tasks

    route_targets = assign_target_counts(target_total, sel_cfg.get("route_mode_targets", {}))
    tier_targets = assign_target_counts(target_total, sel_cfg.get("tier_targets", {}))
    provider_targets = assign_target_counts(target_total, sel_cfg.get("provider_targets", {}))
    family_targets = assign_target_counts(target_total, sel_cfg.get("family_targets", {}))
    bucket_targets = assign_target_counts(target_total, sel_cfg.get("model_bucket_targets", {}))

    counts_route = Counter()
    counts_tier = Counter()
    counts_provider = Counter()
    counts_family = Counter()
    counts_bucket = Counter()

    selected = []

    def under(counter: Counter, key: Optional[str], targets: Dict[str, int]) -> bool:
        if not targets:
            return True
        if key is None:
            return True
        if key not in targets:
            return True
        return counter[key] < targets[key]

    # pass 1: enforce all active targets
    for task in tasks:
        if len(selected) >= target_total:
            break

        if not under(counts_route, task["route_mode"], route_targets):
            continue
        if not under(counts_tier, task["tier"], tier_targets):
            continue
        if not under(counts_provider, task["provider_hint"], provider_targets):
            continue
        if not under(counts_bucket, task["model_bucket"], bucket_targets):
            continue
        if task["route_mode"] == "anchored_extension":
            if not under(counts_family, task["extension_family"], family_targets):
                continue

        selected.append(task)
        counts_route[task["route_mode"]] += 1
        counts_tier[task["tier"]] += 1
        counts_provider[task["provider_hint"]] += 1
        counts_bucket[task["model_bucket"]] += 1
        if task["route_mode"] == "anchored_extension":
            counts_family[task["extension_family"]] += 1

    # pass 2: relax family target
    if len(selected) < target_total:
        already = {t["task_key"] for t in selected}
        for task in tasks:
            if len(selected) >= target_total:
                break
            if task["task_key"] in already:
                continue
            if not under(counts_route, task["route_mode"], route_targets):
                continue
            if not under(counts_tier, task["tier"], tier_targets):
                continue
            if not under(counts_provider, task["provider_hint"], provider_targets):
                continue
            if not under(counts_bucket, task["model_bucket"], bucket_targets):
                continue

            selected.append(task)
            counts_route[task["route_mode"]] += 1
            counts_tier[task["tier"]] += 1
            counts_provider[task["provider_hint"]] += 1
            counts_bucket[task["model_bucket"]] += 1
            if task["route_mode"] == "anchored_extension":
                counts_family[task["extension_family"]] += 1
            already.add(task["task_key"])

    # pass 3: relax provider + family
    if len(selected) < target_total:
        already = {t["task_key"] for t in selected}
        for task in tasks:
            if len(selected) >= target_total:
                break
            if task["task_key"] in already:
                continue
            if not under(counts_route, task["route_mode"], route_targets):
                continue
            if not under(counts_tier, task["tier"], tier_targets):
                continue
            if not under(counts_bucket, task["model_bucket"], bucket_targets):
                continue

            selected.append(task)
            counts_route[task["route_mode"]] += 1
            counts_tier[task["tier"]] += 1
            counts_provider[task["provider_hint"]] += 1
            counts_bucket[task["model_bucket"]] += 1
            if task["route_mode"] == "anchored_extension":
                counts_family[task["extension_family"]] += 1
            already.add(task["task_key"])

    # pass 4: fill remaining by priority only
    if len(selected) < target_total:
        already = {t["task_key"] for t in selected}
        for task in tasks:
            if len(selected) >= target_total:
                break
            if task["task_key"] in already:
                continue
            selected.append(task)
            already.add(task["task_key"])

    return selected[:target_total]


def summarize_support_pools(support_pools: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    counts = [len(v) for v in support_pools.values()]
    by_relation = Counter()
    by_source = Counter()

    for _, pool in support_pools.items():
        for item in pool:
            by_relation[item.get("relation")] += 1
            by_source[item.get("source")] += 1

    return {
        "seed_count": len(support_pools),
        "avg_pool_size": (sum(counts) / len(counts)) if counts else 0.0,
        "max_pool_size": max(counts) if counts else 0,
        "min_pool_size": min(counts) if counts else 0,
        "by_relation": dict(by_relation),
        "by_source": dict(by_source)
    }


def summarize_tasks(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode = Counter()
    by_family = Counter()
    by_provider = Counter()
    by_tier = Counter()
    by_model = Counter()
    by_bucket = Counter()
    by_transform = Counter()
    by_support_relation = Counter()
    by_seed = Counter()

    for t in tasks:
        by_mode[t.get("route_mode")] += 1
        if t.get("extension_family"):
            by_family[t["extension_family"]] += 1
        by_provider[t.get("provider_hint")] += 1
        by_tier[t.get("tier")] += 1
        by_model[t.get("model_name")] += 1
        by_bucket[t.get("model_bucket")] += 1
        by_transform[t.get("transform_type")] += 1
        by_support_relation[t.get("support_relation")] += 1
        by_seed[t.get("seed_id")] += 1

    return {
        "task_count": len(tasks),
        "by_mode": dict(by_mode),
        "by_family": dict(by_family),
        "by_provider": dict(by_provider),
        "by_tier": dict(by_tier),
        "by_model_bucket": dict(by_bucket),
        "top_models": by_model.most_common(50),
        "by_transform": dict(by_transform),
        "by_support_relation": dict(by_support_relation),
        "top_seeds_by_task_count": by_seed.most_common(50)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed-seeds", required=True, help="Path to seed_routes.jsonl")
    parser.add_argument("--records", required=True, help="Path to all_records.jsonl")
    parser.add_argument("--out-support-pools", required=True, help="Output path for support_pools.json")
    parser.add_argument("--out-tasks", required=True, help="Output path for planned_tasks.jsonl")
    parser.add_argument("--summary-out", required=True, help="Output path for planning_summary.json")
    parser.add_argument("--config-json", default=None, help="Optional planner config override JSON")
    args = parser.parse_args()

    cfg = load_planner_config(args.config_json)
    routed_seeds = load_jsonl(args.routed_seeds)
    records = load_jsonl(args.records)

    support_pools = build_support_pools(routed_seeds, records, cfg)
    all_tasks = enumerate_all_tasks(routed_seeds, support_pools, cfg)
    selected_tasks = select_tasks_for_target(all_tasks, cfg)

    save_json(support_pools, args.out_support_pools)
    save_jsonl(selected_tasks, args.out_tasks)

    summary = {
        "config": cfg,
        "support_pools": summarize_support_pools(support_pools),
        "all_tasks": summarize_tasks(all_tasks),
        "selected_tasks": summarize_tasks(selected_tasks),
    }
    save_json(summary, args.summary_out)


if __name__ == "__main__":
    main()