import argparse
from collections import Counter, defaultdict
from statistics import median
from typing import Dict, List, Any, Optional

from .io_utils import load_json, load_jsonl, save_json
from .text_utils import normalize_model_name


DEDUP_REASONS = {
    "EXACT_QA_DUP",
    "QUESTION_TEMPLATE_DUP",
    "NEAR_DUP_QUESTION",
}

BAD_MODEL_REASONS = {
    "UNKNOWN_CORE_IDENTIFIER",
    "UNAPPROVED_EXTENSION_IDENTIFIER",
    "CROSS_FAMILY_EXTENSION_IDENTIFIER",
    "BASE_MODEL_MISREPRESENTATION",
    "EXTENSION_REQUIRES_FRAMING",
    "INSUFFICIENT_CORE_ANCHOR",
    "DISALLOWED_THEME",
}


DEFAULT_CAPACITY_CONFIG = {
    "weights": {
        "n_seed_anchors": 0.14,
        "n_corpus_records": 0.12,
        "n_classes_covered": 0.18,
        "n_core_procedures": 0.08,
        "n_core_variables": 0.05,
        "n_core_widgets": 0.05,
        "n_core_breeds": 0.03,
        "n_extension_families": 0.14,
        "n_extension_identifiers": 0.08,
        "support_pool_avg_size": 0.13,
    },
    "dedup_risk_weights": {
        "dedup_rate": 0.55,
        "bad_model_rate": 0.20,
        "reject_rate": 0.15,
        "low_support_penalty": 0.10,
    },
    "buckets": {
        "high_capacity_min": 0.72,
        "mid_capacity_min": 0.45,
    },
    "caps": {
        "min_core_paraphrase": 4,
        "max_core_paraphrase": 12,
        "min_core_repair": 2,
        "max_core_repair": 7,
        "min_anchored_extension": 4,
        "max_anchored_extension": 14,
        "min_total": 10,
        "max_total": 30,
        "min_support_reuse_cap": 25,
        "max_support_reuse_cap": 120,
    }
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_capacity_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = DEFAULT_CAPACITY_CONFIG
    if path:
        override = load_json(path)
        cfg = deep_merge(cfg, override)
    return cfg


# -------------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------------

def count_extension_identifiers(profile: Dict[str, Any]) -> int:
    total = 0
    for fam in profile.get("extensions", {}).get("families", []):
        total += len(fam.get("identifiers", []))
    return total


def model_profile_features(model_profiles: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    models = model_profiles.get("models", model_profiles)
    for model_name, profile in models.items():
        key = normalize_model_name(model_name) or model_name
        core = profile.get("core", {})
        exts = profile.get("extensions", {})

        out[key] = {
            "n_core_procedures": len(core.get("procedures", [])),
            "n_core_variables": len(core.get("variables", [])),
            "n_core_widgets": len(core.get("widgets", [])),
            "n_core_breeds": len(core.get("breeds", [])),
            "n_extension_families": len(exts.get("families", [])),
            "n_extension_identifiers": count_extension_identifiers(profile),
        }
    return out


def routed_seed_features(routed_seeds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_model_seed = defaultdict(set)
    by_model_class = defaultdict(set)
    by_model_route = defaultdict(Counter)
    by_model_family = defaultdict(Counter)
    by_model_tier = defaultdict(Counter)

    for row in routed_seeds:
        model_name = normalize_model_name(row.get("model_name"))
        if not model_name:
            continue

        seed_id = row.get("seed_id")
        class_id = row.get("class_id")
        route_mode = row.get("route_mode")
        family = row.get("extension_family")
        tier = row.get("tier")

        if seed_id:
            by_model_seed[model_name].add(seed_id)
        if class_id is not None:
            by_model_class[model_name].add(class_id)
        if route_mode:
            by_model_route[model_name][route_mode] += 1
        if family:
            by_model_family[model_name][family] += 1
        if tier:
            by_model_tier[model_name][tier] += 1

    out = {}
    all_models = set(by_model_seed) | set(by_model_class) | set(by_model_route) | set(by_model_family) | set(by_model_tier)
    for model_name in all_models:
        out[model_name] = {
            "n_seed_anchors": len(by_model_seed[model_name]),
            "n_classes_covered": len(by_model_class[model_name]),
            "n_routed_core_paraphrase": by_model_route[model_name].get("core_paraphrase", 0),
            "n_routed_core_repair": by_model_route[model_name].get("core_repair", 0),
            "n_routed_extension": by_model_route[model_name].get("anchored_extension", 0),
            "n_extension_families_used": len(by_model_family[model_name]),
            "tier_counts": dict(by_model_tier[model_name]),
        }
    return out


def corpus_record_features(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_model = Counter()
    by_model_class = defaultdict(set)

    for row in records:
        model_name = normalize_model_name(row.get("model_name"))
        if not model_name:
            continue
        by_model[model_name] += 1
        class_id = row.get("class_id")
        if class_id is not None:
            by_model_class[model_name].add(class_id)

    out = {}
    for model_name in by_model:
        out[model_name] = {
            "n_corpus_records": by_model[model_name],
            "n_record_classes": len(by_model_class[model_name]),
        }
    return out


def support_pool_features(support_pools: Optional[Dict[str, List[Dict[str, Any]]]], routed_seeds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not support_pools:
        return {}

    seed_to_model = {}
    for row in routed_seeds:
        seed_id = row.get("seed_id")
        if seed_id:
            seed_to_model[seed_id] = normalize_model_name(row.get("model_name"))

    pool_sizes = defaultdict(list)
    support_reuse = defaultdict(Counter)

    for seed_id, pool in support_pools.items():
        model_name = seed_to_model.get(seed_id)
        if not model_name:
            continue

        pool_sizes[model_name].append(len(pool))
        for item in pool:
            rid = item.get("record_id")
            if rid:
                support_reuse[model_name][rid] += 1

    out = {}
    for model_name, sizes in pool_sizes.items():
        reuse_counts = list(support_reuse[model_name].values())
        out[model_name] = {
            "support_pool_avg_size": round(sum(sizes) / len(sizes), 4) if sizes else 0.0,
            "support_pool_median_size": median(sizes) if sizes else 0.0,
            "support_pool_max_size": max(sizes) if sizes else 0,
            "support_record_reuse_avg": round(sum(reuse_counts) / len(reuse_counts), 4) if reuse_counts else 0.0,
            "support_record_reuse_max": max(reuse_counts) if reuse_counts else 0,
        }
    return out


def telemetry_features(telemetry_rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    if not telemetry_rows:
        return {}

    asked = Counter()
    accepted = Counter()
    rejected = Counter()
    dedup_rejected = Counter()
    bad_model_rejected = Counter()

    for ev in telemetry_rows:
        model_name = normalize_model_name(ev.get("model_name"))
        if not model_name:
            continue

        asked[model_name] += 1
        status = ev.get("status")
        reason = ev.get("reason")

        if status == "accepted":
            accepted[model_name] += 1
        else:
            rejected[model_name] += 1
            if reason in DEDUP_REASONS:
                dedup_rejected[model_name] += 1
            if reason in BAD_MODEL_REASONS:
                bad_model_rejected[model_name] += 1

    out = {}
    all_models = set(asked) | set(accepted) | set(rejected)
    for model_name in all_models:
        a = asked[model_name]
        acc = accepted[model_name]
        rej = rejected[model_name]
        dd = dedup_rejected[model_name]
        bad = bad_model_rejected[model_name]

        out[model_name] = {
            "pilot_asked": a,
            "pilot_accepted": acc,
            "pilot_rejected": rej,
            "pilot_acceptance_rate": (acc / a) if a else None,
            "pilot_reject_rate": (rej / a) if a else None,
            "pilot_dedup_rate": (dd / a) if a else None,
            "pilot_bad_model_rate": (bad / a) if a else None,
        }
    return out


# -------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------

def collect_feature_matrix(model_names: List[str], feature_dicts: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out = {m: {} for m in model_names}
    for fd in feature_dicts:
        for model_name in model_names:
            out[model_name].update(fd.get(model_name, {}))
    return out


def min_max_normalize(feature_matrix: Dict[str, Dict[str, Any]], feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    out = {m: {} for m in feature_matrix}

    for fname in feature_names:
        vals = []
        for model_name, feats in feature_matrix.items():
            v = feats.get(fname)
            if isinstance(v, (int, float)) and v is not None:
                vals.append(float(v))

        if not vals:
            for model_name in feature_matrix:
                out[model_name][fname] = 0.0
            continue

        lo = min(vals)
        hi = max(vals)

        for model_name, feats in feature_matrix.items():
            v = feats.get(fname)
            if v is None:
                out[model_name][fname] = 0.0
            elif hi <= lo:
                out[model_name][fname] = 1.0
            else:
                out[model_name][fname] = (float(v) - lo) / (hi - lo)

    return out


# -------------------------------------------------------------------
# Capacity scoring
# -------------------------------------------------------------------

def compute_capacity_score(
    model_name: str,
    raw_feats: Dict[str, Any],
    norm_feats: Dict[str, float],
    cfg: Dict[str, Any]
) -> float:
    score = 0.0
    for fname, w in cfg["weights"].items():
        score += w * norm_feats.get(fname, 0.0)
    return round(score, 6)


def compute_dedup_risk_score(
    raw_feats: Dict[str, Any],
    cfg: Dict[str, Any]
) -> float:
    w = cfg["dedup_risk_weights"]

    dedup_rate = raw_feats.get("pilot_dedup_rate")
    bad_model_rate = raw_feats.get("pilot_bad_model_rate")
    reject_rate = raw_feats.get("pilot_reject_rate")

    dedup_rate = 0.0 if dedup_rate is None else float(dedup_rate)
    bad_model_rate = 0.0 if bad_model_rate is None else float(bad_model_rate)
    reject_rate = 0.0 if reject_rate is None else float(reject_rate)

    support_avg = float(raw_feats.get("support_pool_avg_size", 0.0))
    low_support_penalty = 0.0
    if support_avg < 2:
        low_support_penalty = 1.0
    elif support_avg < 4:
        low_support_penalty = 0.6
    elif support_avg < 6:
        low_support_penalty = 0.3

    risk = (
        w["dedup_rate"] * dedup_rate +
        w["bad_model_rate"] * bad_model_rate +
        w["reject_rate"] * reject_rate +
        w["low_support_penalty"] * low_support_penalty
    )
    return round(min(max(risk, 0.0), 1.0), 6)


def capacity_bucket(effective_capacity: float, cfg: Dict[str, Any]) -> str:
    if effective_capacity >= cfg["buckets"]["high_capacity_min"]:
        return "high_capacity"
    if effective_capacity >= cfg["buckets"]["mid_capacity_min"]:
        return "mid_capacity"
    return "low_capacity"


def lerp_int(lo: int, hi: int, alpha: float) -> int:
    alpha = min(max(alpha, 0.0), 1.0)
    return int(round(lo + (hi - lo) * alpha))


def suggested_caps(effective_capacity: float, cfg: Dict[str, Any]) -> Dict[str, int]:
    c = cfg["caps"]
    core_paraphrase = lerp_int(c["min_core_paraphrase"], c["max_core_paraphrase"], effective_capacity)
    core_repair = lerp_int(c["min_core_repair"], c["max_core_repair"], effective_capacity)
    anchored_extension = lerp_int(c["min_anchored_extension"], c["max_anchored_extension"], effective_capacity)
    total = lerp_int(c["min_total"], c["max_total"], effective_capacity)
    support_reuse_cap = lerp_int(c["min_support_reuse_cap"], c["max_support_reuse_cap"], effective_capacity)

    return {
        "core_paraphrase": core_paraphrase,
        "core_repair": core_repair,
        "anchored_extension": anchored_extension,
        "total": total,
        "support_reuse_cap": support_reuse_cap,
    }


def build_model_capacity_report(
    model_profiles: Dict[str, Any],
    routed_seeds: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    support_pools: Optional[Dict[str, List[Dict[str, Any]]]],
    telemetry_rows: Optional[List[Dict[str, Any]]],
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    profile_feats = model_profile_features(model_profiles)
    seed_feats = routed_seed_features(routed_seeds)
    record_feats = corpus_record_features(records)
    support_feats = support_pool_features(support_pools, routed_seeds)
    telem_feats = telemetry_features(telemetry_rows)

    models_key = model_profiles.get("models", model_profiles)
    model_names = sorted(
        set(models_key.keys()) |
        set(profile_feats.keys()) |
        set(seed_feats.keys()) |
        set(record_feats.keys()) |
        set(support_feats.keys()) |
        set(telem_feats.keys())
    )

    raw_matrix = collect_feature_matrix(
        model_names,
        [profile_feats, seed_feats, record_feats, support_feats, telem_feats]
    )

    norm_feature_names = list(cfg["weights"].keys())
    norm_matrix = min_max_normalize(raw_matrix, norm_feature_names)

    models_out = {}

    for model_name in model_names:
        raw_feats = raw_matrix[model_name]
        norm_feats = norm_matrix[model_name]

        cap_score = compute_capacity_score(model_name, raw_feats, norm_feats, cfg)
        risk_score = compute_dedup_risk_score(raw_feats, cfg)
        effective_capacity = round(cap_score * (1.0 - risk_score), 6)
        bucket = capacity_bucket(effective_capacity, cfg)
        caps = suggested_caps(effective_capacity, cfg)

        models_out[model_name] = {
            **raw_feats,
            "normalized_features": norm_feats,
            "capacity_score": cap_score,
            "dedup_risk_score": risk_score,
            "effective_capacity": effective_capacity,
            "capacity_bucket": bucket,
            "suggested_seed_caps": {
                "core_paraphrase": caps["core_paraphrase"],
                "core_repair": caps["core_repair"],
                "anchored_extension": caps["anchored_extension"],
                "total": caps["total"],
            },
            "suggested_support_reuse_cap": caps["support_reuse_cap"],
        }

    # summary
    by_bucket = Counter()
    top_effective = []
    for model_name, row in models_out.items():
        by_bucket[row["capacity_bucket"]] += 1
        top_effective.append((model_name, row["effective_capacity"]))

    top_effective.sort(key=lambda x: (-x[1], x[0]))

    return {
        "meta": {
            "config": cfg,
            "model_count": len(models_out),
            "telemetry_used": telemetry_rows is not None,
            "support_pools_used": support_pools is not None,
        },
        "summary": {
            "by_capacity_bucket": dict(by_bucket),
            "top_models_by_effective_capacity": [
                {"model_name": m, "effective_capacity": s}
                for m, s in top_effective[:50]
            ]
        },
        "models": models_out
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True, help="Path to model_profiles_merged.json")
    parser.add_argument("--routed-seeds", required=True, help="Path to seed_routes.jsonl")
    parser.add_argument("--records", required=True, help="Path to all_records_with_homotopy.jsonl")
    parser.add_argument("--support-pools", default=None, help="Optional support_pools.json")
    parser.add_argument("--telemetry-jsonl", default=None, help="Optional telemetry.jsonl")
    parser.add_argument("--config-json", default=None, help="Optional capacity config override JSON")
    parser.add_argument("--out", required=True, help="Output model_capacity_report.json")
    args = parser.parse_args()

    cfg = load_capacity_config(args.config_json)
    profiles = load_json(args.profiles)
    routed_seeds = load_jsonl(args.routed_seeds)
    records = load_jsonl(args.records)

    support_pools = load_json(args.support_pools) if args.support_pools else None
    telemetry_rows = load_jsonl(args.telemetry_jsonl) if args.telemetry_jsonl else None

    report = build_model_capacity_report(
        model_profiles=profiles,
        routed_seeds=routed_seeds,
        records=records,
        support_pools=support_pools,
        telemetry_rows=telemetry_rows,
        cfg=cfg
    )
    save_json(report, args.out)


if __name__ == "__main__":
    main()
