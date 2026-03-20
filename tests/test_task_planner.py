import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_build_support_pools():
    from qa_system_v2bis.synthetic_v2.task_planner import build_support_pools, DEFAULT_PLANNER_CONFIG
    seeds = [{"seed_id": "S1", "model_name": "rebellion", "class_id": 5, "level": "L2", "route_mode": "core_paraphrase", "seed_q": "Q1", "seed_a": "A1", "question": "Q1"}]
    records = [
        {"record_id": "R1", "source": "corpus", "model_name": "rebellion", "class_id": 5, "question": "Q2", "answer": "A2"},
        {"record_id": "R2", "source": "corpus", "model_name": "rebellion", "class_id": 5, "question": "Q3", "answer": "A3"},
    ]
    pools = build_support_pools(seeds, records, DEFAULT_PLANNER_CONFIG)
    assert "S1" in pools
    assert len(pools["S1"]) >= 2

def test_enumerate_tasks_unique_keys():
    from qa_system_v2bis.synthetic_v2.task_planner import enumerate_tasks_for_seed, DEFAULT_PLANNER_CONFIG
    seed = {"seed_id": "S1", "model_name": "rebellion", "class_id": 5, "tier": "A", "level": "L2", "route_mode": "core_paraphrase"}
    support = [{"record_id": "R1", "source": "corpus", "model_name": "rebellion", "class_id": 5, "relation": "same_class_same_model", "score": 90, "question": "Q", "answer": "A"}]
    tasks = enumerate_tasks_for_seed(seed, support, "high_volume", DEFAULT_PLANNER_CONFIG)
    keys = [t["task_key"] for t in tasks]
    assert len(keys) == len(set(keys))
    assert len(tasks) <= 8  # core_paraphrase cap

def test_skip_seeds():
    from qa_system_v2bis.synthetic_v2.task_planner import enumerate_tasks_for_seed, DEFAULT_PLANNER_CONFIG
    seed = {"seed_id": "S1", "model_name": "x", "class_id": 1, "tier": "A", "level": "L1", "route_mode": "skip"}
    assert enumerate_tasks_for_seed(seed, [], "high_volume", DEFAULT_PLANNER_CONFIG) == []

def test_task_key_deterministic():
    from qa_system_v2bis.synthetic_v2.task_planner import build_task_key
    seed = {"seed_id": "S1", "route_mode": "core_paraphrase", "extension_family": None}
    k1 = build_task_key(seed, "lexical_paraphrase", "R1", "gpt-5.4-nano", "v1")
    k2 = build_task_key(seed, "lexical_paraphrase", "R1", "gpt-5.4-nano", "v1")
    assert k1 == k2
    k3 = build_task_key(seed, "syntactic_reframe", "R1", "gpt-5.4-nano", "v1")
    assert k1 != k3

def test_core_gets_nano():
    from qa_system_v2bis.synthetic_v2.task_planner import choose_provider_hint, DEFAULT_PLANNER_CONFIG
    seed = {"seed_id": "S1", "route_mode": "core_paraphrase", "extension_family": None}
    assert choose_provider_hint(seed, "lexical_paraphrase", None, DEFAULT_PLANNER_CONFIG) == "gpt-5.4-nano"

def test_end_to_end():
    from qa_system_v2bis.synthetic_v2.task_planner import build_support_pools, enumerate_all_tasks, DEFAULT_PLANNER_CONFIG
    seeds = [
        {"seed_id": "S1", "model_name": "rebellion", "class_id": 5, "tier": "A", "level": "L2", "route_mode": "core_paraphrase", "seed_q": "Q1", "seed_a": "A1", "question": "Q1"},
        {"seed_id": "S2", "model_name": "rebellion", "class_id": 5, "tier": "B", "level": "L1", "route_mode": "anchored_extension", "extension_family": "state_refinement", "seed_q": "Q2", "seed_a": "A2", "question": "Q2"},
    ]
    records = [{"record_id": "R1", "source": "corpus", "model_name": "rebellion", "class_id": 5, "question": "Q3", "answer": "A3"}]
    pools = build_support_pools(seeds, records, DEFAULT_PLANNER_CONFIG)
    tasks = enumerate_all_tasks(seeds, pools, DEFAULT_PLANNER_CONFIG)
    assert len(tasks) > 0
    for t in tasks:
        for field in ["task_key", "seed_id", "route_mode", "transform_type", "provider_hint", "priority"]:
            assert field in t, f"Missing field: {field}"