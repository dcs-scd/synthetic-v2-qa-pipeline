import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qa_system_v2bis.synthetic_v2.pilot_runner import (
    attach_task_to_seed,
    index_routed_seeds_by_id,
    load_seen_task_keys,
    select_tasks_for_tranche,
    _check_stop_conditions,
)


def test_index_routed_seeds():
    seeds = [
        {"seed_id": "S1", "model_name": "rebellion", "seed_q": "Q1"},
        {"seed_id": "S2", "model_name": "fire", "seed_q": "Q2"},
        {"seed_id": "S3", "model_name": "wolf", "seed_q": "Q3"},
    ]
    index = index_routed_seeds_by_id(seeds)
    assert len(index) == 3
    assert index["S1"]["model_name"] == "rebellion"
    assert index["S2"]["seed_q"] == "Q2"
    assert "S3" in index


def test_index_routed_seeds_skips_missing_id():
    seeds = [
        {"seed_id": "S1", "model_name": "rebellion"},
        {"model_name": "no_id"},  # no seed_id
    ]
    index = index_routed_seeds_by_id(seeds)
    assert len(index) == 1
    assert "S1" in index


def test_attach_task_to_seed():
    seed = {
        "seed_id": "S1",
        "model_name": "rebellion",
        "seed_q": "What is rebellion?",
        "seed_a": "A model of rebellion.",
        "class_id": 5,
        "route_mode": "core_paraphrase",
    }
    task = {
        "task_key": "abc123",
        "seed_id": "S1",
        "transform_type": "lexical_paraphrase",
        "provider_hint": "gpt-5.4-nano",
        "support_record_id": "R42",
        "route_mode": "core_paraphrase",
        "priority": 1.95,
    }
    merged = attach_task_to_seed(seed, task)

    # task fields overlaid
    assert merged["task_key"] == "abc123"
    assert merged["transform_type"] == "lexical_paraphrase"
    assert merged["provider_hint"] == "gpt-5.4-nano"
    assert merged["support_record_id"] == "R42"

    # seed fields preserved
    assert merged["seed_q"] == "What is rebellion?"
    assert merged["seed_a"] == "A model of rebellion."
    assert merged["model_name"] == "rebellion"
    assert merged["class_id"] == 5

    # original seed not mutated
    assert "task_key" not in seed


def test_select_tasks_for_tranche():
    tasks = [
        {"task_key": f"T{i}", "route_mode": "core_paraphrase", "priority": 10 - i}
        for i in range(10)
    ]
    already = {"T0", "T1"}
    selected = select_tasks_for_tranche(tasks, tranche_size=5, remaining_targets=None, already_scheduled=already)

    assert len(selected) == 5
    # T0 and T1 should be skipped
    selected_keys = {t["task_key"] for t in selected}
    assert "T0" not in selected_keys
    assert "T1" not in selected_keys
    assert "T2" in selected_keys


def test_select_tasks_for_tranche_respects_size():
    tasks = [{"task_key": f"T{i}", "route_mode": "core_paraphrase"} for i in range(100)]
    selected = select_tasks_for_tranche(tasks, tranche_size=3, remaining_targets=None, already_scheduled=set())
    assert len(selected) == 3


def test_select_tasks_for_tranche_with_remaining_targets():
    tasks = [
        {"task_key": "T0", "route_mode": "core_paraphrase"},
        {"task_key": "T1", "route_mode": "core_paraphrase"},
        {"task_key": "T2", "route_mode": "core_paraphrase"},
        {"task_key": "T3", "route_mode": "anchored_extension"},
        {"task_key": "T4", "route_mode": "anchored_extension"},
    ]
    remaining = {"core_paraphrase": 2}
    selected = select_tasks_for_tranche(tasks, tranche_size=10, remaining_targets=remaining, already_scheduled=set())
    # Should pick 2 core_paraphrase + 2 anchored_extension (no limit on extension)
    cp_count = sum(1 for t in selected if t["route_mode"] == "core_paraphrase")
    assert cp_count == 2
    assert len(selected) == 4


def test_stop_condition_bad_json():
    # 6 BAD_JSON out of 100 → 6% > 5% threshold
    summary = {
        "total_events": 100,
        "by_status": {"accepted": 80, "rejected": 20},
        "by_reason": {"BAD_JSON": 6, "ACCEPTED": 80},
    }
    result = _check_stop_conditions(summary, 100)
    assert result is not None
    assert "BAD_JSON" in result


def test_stop_condition_llm_error():
    summary = {
        "total_events": 100,
        "by_status": {"accepted": 80, "rejected": 20},
        "by_reason": {"LLM_CALL_ERROR": 7, "ACCEPTED": 80},
    }
    result = _check_stop_conditions(summary, 100)
    assert result is not None
    assert "LLM_ERROR" in result


def test_stop_condition_dup_rate():
    summary = {
        "total_events": 100,
        "by_status": {"accepted": 60, "rejected": 40},
        "by_reason": {
            "QUESTION_TEMPLATE_DUP": 15,
            "NEAR_DUP_QUESTION": 12,
            "ACCEPTED": 60,
        },
    }
    result = _check_stop_conditions(summary, 100)
    assert result is not None
    assert "DUP_RATE" in result


def test_stop_condition_low_acceptance():
    summary = {
        "total_events": 100,
        "by_status": {"accepted": 70, "rejected": 30},
        "by_reason": {"BAD_JSON": 2, "ACCEPTED": 70},
    }
    result = _check_stop_conditions(summary, 100)
    assert result is not None
    assert "LOW_ACCEPTANCE" in result


def test_stop_condition_ok():
    summary = {
        "total_events": 100,
        "by_status": {"accepted": 90, "rejected": 10},
        "by_reason": {"BAD_JSON": 2, "ACCEPTED": 90},
    }
    result = _check_stop_conditions(summary, 100)
    assert result is None


def test_stop_condition_empty_tranche():
    summary = {"total_events": 0, "by_status": {}, "by_reason": {}}
    result = _check_stop_conditions(summary, 0)
    assert result == "EMPTY_TRANCHE"


def test_resume_loads_prior_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write accepted.jsonl with task_keys
        acc_path = os.path.join(tmpdir, "accepted.jsonl")
        with open(acc_path, "w") as f:
            f.write(json.dumps({"task_key": "TK1", "question": "Q1", "answer": "A1"}) + "\n")
            f.write(json.dumps({"task_key": "TK2", "question": "Q2", "answer": "A2"}) + "\n")

        # Write rejected.jsonl with task_keys
        rej_path = os.path.join(tmpdir, "rejected.jsonl")
        with open(rej_path, "w") as f:
            f.write(json.dumps({"task_key": "TK3", "reason": "BAD_JSON"}) + "\n")

        seen = load_seen_task_keys([acc_path, rej_path])
        assert seen == {"TK1", "TK2", "TK3"}


def test_resume_handles_missing_files():
    seen = load_seen_task_keys(["/nonexistent/path/foo.jsonl"])
    assert seen == set()


def test_resume_skips_rows_without_task_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "accepted.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"task_key": "TK1", "question": "Q"}) + "\n")
            f.write(json.dumps({"question": "no task key"}) + "\n")  # no task_key

        seen = load_seen_task_keys([path])
        assert seen == {"TK1"}
