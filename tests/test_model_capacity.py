"""Tests for model_capacity.py — capacity scoring for dedup-aware planning."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qa_system_v2bis.synthetic_v2.model_capacity import (
    model_profile_features,
    routed_seed_features,
    corpus_record_features,
    telemetry_features,
    min_max_normalize,
    compute_capacity_score,
    compute_dedup_risk_score,
    capacity_bucket,
    lerp_int,
    suggested_caps,
    build_model_capacity_report,
    DEFAULT_CAPACITY_CONFIG,
)


# ---------------------------------------------------------------------------
# 1. model_profile_features — extract counts from structured profiles
# ---------------------------------------------------------------------------
def test_model_profile_features_basic():
    profiles = {
        "rebellion": {
            "core": {
                "procedures": ["setup", "go", "move"],
                "variables": ["x"],
                "breeds": ["turtles"],
                "widgets": ["s1"],
            },
            "extensions": {
                "families": [
                    {"name": "f1", "identifiers": ["id1", "id2"]},
                ],
            },
        },
        "fire": {
            "core": {
                "procedures": ["setup", "go"],
                "variables": ["y", "z"],
                "breeds": [],
                "widgets": ["s1", "s2"],
            },
            "extensions": {
                "families": [],
            },
        },
    }
    feats = model_profile_features(profiles)
    assert feats["rebellion"]["n_core_procedures"] == 3
    assert feats["rebellion"]["n_extension_families"] == 1
    assert feats["rebellion"]["n_extension_identifiers"] == 2
    assert feats["fire"]["n_core_procedures"] == 2
    assert feats["fire"]["n_extension_families"] == 0
    assert feats["fire"]["n_extension_identifiers"] == 0


# ---------------------------------------------------------------------------
# 2. routed_seed_features — count seed anchors per model
# ---------------------------------------------------------------------------
def test_routed_seed_features():
    seeds = [
        {"seed_id": "S1", "model_name": "rebellion", "route_mode": "core_paraphrase"},
        {"seed_id": "S2", "model_name": "rebellion", "route_mode": "anchored_extension"},
        {"seed_id": "S3", "model_name": "fire", "route_mode": "core_paraphrase"},
    ]
    feats = routed_seed_features(seeds)
    assert feats["rebellion"]["n_seed_anchors"] == 2
    assert feats["fire"]["n_seed_anchors"] == 1


# ---------------------------------------------------------------------------
# 3. corpus_record_features — count records per model
# ---------------------------------------------------------------------------
def test_corpus_record_features():
    records = [
        {"record_id": "R1", "model_name": "rebellion"},
        {"record_id": "R2", "model_name": "rebellion"},
        {"record_id": "R3", "model_name": "rebellion"},
        {"record_id": "R4", "model_name": "fire"},
        {"record_id": "R5", "model_name": "fire"},
    ]
    feats = corpus_record_features(records)
    assert feats["rebellion"]["n_corpus_records"] == 3
    assert feats["fire"]["n_corpus_records"] == 2


# ---------------------------------------------------------------------------
# 4. telemetry_features — pilot dedup rate from mixed events
# ---------------------------------------------------------------------------
def test_telemetry_features_with_dedup():
    events = [
        {"model_name": "rebellion", "status": "accepted"},
        {"model_name": "rebellion", "status": "accepted"},
        {"model_name": "rebellion", "status": "rejected", "reason": "QUESTION_TEMPLATE_DUP"},
        {"model_name": "rebellion", "status": "rejected", "reason": "UNAPPROVED_EXTENSION_IDENTIFIER"},
        {"model_name": "rebellion", "status": "accepted"},
    ]
    feats = telemetry_features(events)
    # dedup events: QUESTION_TEMPLATE_DUP counts as dedup → 1 out of 5
    assert feats["rebellion"]["pilot_dedup_rate"] == 1 / 5


# ---------------------------------------------------------------------------
# 5. telemetry_features — empty list
# ---------------------------------------------------------------------------
def test_telemetry_features_empty():
    feats = telemetry_features([])
    assert feats == {}


# ---------------------------------------------------------------------------
# 6. min_max_normalize — linear rescaling to [0, 1]
# ---------------------------------------------------------------------------
def test_min_max_normalize():
    per_model = {
        "a": {"val": 10},
        "b": {"val": 20},
        "c": {"val": 30},
    }
    normed = min_max_normalize(per_model, ["val"])
    assert normed["a"]["val"] == 0.0
    assert normed["b"]["val"] == 0.5
    assert normed["c"]["val"] == 1.0


# ---------------------------------------------------------------------------
# 7. compute_capacity_score — all features at 1.0
# ---------------------------------------------------------------------------
def test_compute_capacity_score():
    weights = DEFAULT_CAPACITY_CONFIG["weights"]
    norm_feats = {k: 1.0 for k in weights}
    score = compute_capacity_score("test_model", {}, norm_feats, DEFAULT_CAPACITY_CONFIG)
    expected = sum(weights.values())
    assert abs(score - expected) < 1e-4


# ---------------------------------------------------------------------------
# 8. compute_dedup_risk_score — no telemetry, low support
# ---------------------------------------------------------------------------
def test_compute_dedup_risk_score_no_telemetry():
    model_feats = {"pilot_dedup_rate": None, "support_pool_avg_size": 1}
    score = compute_dedup_risk_score(model_feats, DEFAULT_CAPACITY_CONFIG)
    risk_weights = DEFAULT_CAPACITY_CONFIG["dedup_risk_weights"]
    # Only low_support_penalty should fire (rate at 1.0 since support=1 is min)
    assert score == risk_weights["low_support_penalty"] * 1.0


# ---------------------------------------------------------------------------
# 9. compute_dedup_risk_score — with pilot dedup rate
# ---------------------------------------------------------------------------
def test_compute_dedup_risk_score_with_telemetry():
    model_feats = {"pilot_dedup_rate": 0.2, "pilot_bad_model_rate": 0.0, "pilot_reject_rate": 0.0, "support_pool_avg_size": 100}
    score = compute_dedup_risk_score(model_feats, DEFAULT_CAPACITY_CONFIG)
    risk_weights = DEFAULT_CAPACITY_CONFIG["dedup_risk_weights"]
    # pilot_dedup_rate contributes 0.2 * its weight, support is high so no penalty
    expected = risk_weights["dedup_rate"] * 0.2
    assert abs(score - expected) < 1e-4


# ---------------------------------------------------------------------------
# 10. capacity_bucket — threshold classification
# ---------------------------------------------------------------------------
def test_capacity_bucket_thresholds():
    cfg = DEFAULT_CAPACITY_CONFIG
    assert capacity_bucket(0.72, cfg) == "high_capacity"
    assert capacity_bucket(0.71, cfg) == "mid_capacity"
    assert capacity_bucket(0.45, cfg) == "mid_capacity"
    assert capacity_bucket(0.44, cfg) == "low_capacity"


# ---------------------------------------------------------------------------
# 11. suggested_caps — lerp from min to max
# ---------------------------------------------------------------------------
def test_suggested_caps_lerp():
    cfg = DEFAULT_CAPACITY_CONFIG
    caps_min = suggested_caps(0.0, cfg)
    caps_max = suggested_caps(1.0, cfg)
    # At 0.0, all caps should be at min values
    assert caps_min["core_paraphrase"] == cfg["caps"]["min_core_paraphrase"]
    assert caps_min["core_repair"] == cfg["caps"]["min_core_repair"]
    # At 1.0, all caps should be at max values
    assert caps_max["core_paraphrase"] == cfg["caps"]["max_core_paraphrase"]
    assert caps_max["core_repair"] == cfg["caps"]["max_core_repair"]


# ---------------------------------------------------------------------------
# 12. build_model_capacity_report — end-to-end with minimal data
# ---------------------------------------------------------------------------
def test_build_report_end_to_end():
    profiles = {
        "rebellion": {
            "core": {"procedures": ["setup", "go"], "variables": ["x"], "breeds": [], "widgets": ["s1"]},
            "extensions": {"families": [{"name": "f1", "identifiers": ["id1"]}]},
        },
        "fire": {
            "core": {"procedures": ["setup"], "variables": [], "breeds": [], "widgets": []},
            "extensions": {"families": []},
        },
    }
    seeds = [
        {"seed_id": "S1", "model_name": "rebellion", "route_mode": "core_paraphrase"},
        {"seed_id": "S2", "model_name": "rebellion", "route_mode": "anchored_extension"},
        {"seed_id": "S3", "model_name": "fire", "route_mode": "core_paraphrase"},
    ]
    records = [
        {"record_id": "R1", "model_name": "rebellion"},
        {"record_id": "R2", "model_name": "rebellion"},
        {"record_id": "R3", "model_name": "fire"},
        {"record_id": "R4", "model_name": "fire"},
    ]
    report = build_model_capacity_report(profiles, seeds, records, support_pools=None, telemetry_rows=None, cfg=DEFAULT_CAPACITY_CONFIG)
    assert report["meta"]["model_count"] == 2
    for model_name in ["rebellion", "fire"]:
        entry = report["models"][model_name]
        assert "capacity_score" in entry
        assert "effective_capacity" in entry
        assert "capacity_bucket" in entry
        assert "suggested_seed_caps" in entry


# ---------------------------------------------------------------------------
# 13. backward compat — enumerate_all_tasks still works without capacity arg
# ---------------------------------------------------------------------------
def test_backward_compat_task_planner():
    from qa_system_v2bis.synthetic_v2.task_planner import enumerate_all_tasks, build_support_pools, DEFAULT_PLANNER_CONFIG
    seeds = [
        {"seed_id": "S1", "model_name": "rebellion", "class_id": 5, "tier": "A",
         "level": "L2", "route_mode": "core_paraphrase", "seed_q": "Q1", "seed_a": "A1", "question": "Q1"},
    ]
    records = [
        {"record_id": "R1", "source": "corpus", "model_name": "rebellion",
         "class_id": 5, "question": "Q2", "answer": "A2"},
    ]
    pools = build_support_pools(seeds, records, DEFAULT_PLANNER_CONFIG)
    tasks = enumerate_all_tasks(seeds, pools, DEFAULT_PLANNER_CONFIG)
    assert isinstance(tasks, list)
    assert len(tasks) > 0
