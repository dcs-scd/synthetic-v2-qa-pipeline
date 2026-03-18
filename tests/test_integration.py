"""End-to-end integration test using smoke-test adapters."""
import json
import os
import tempfile
from qa_system_v2bis.synthetic_v2.run_synthetic_v2 import (
    generate_synthetic, EchoSeedLLMClient,
)
from qa_system_v2bis.synthetic_v2.gate1_embedding import AlwaysPassEmbeddingIndex
from qa_system_v2bis.synthetic_v2.gate3_dedup import InMemoryDedupIndex
from qa_system_v2bis.synthetic_v2.io_utils import load_jsonl


PROFILE = {
    "models": {
        "rebellion": {
            "model_name": "rebellion",
            "core": {
                "procedures": ["setup", "go"],
                "variables": ["active?"],
                "breeds": ["agents"],
                "widgets": ["vision"],
                "model_summary": "Rebellion model.",
            },
            "extensions": {
                "families": [],
                "general_rules": [],
                "framing_cues": [],
                "disallowed_unanchored_themes": [],
            }
        }
    }
}

SEEDS = [
    {
        "seed_id": "S1", "model_name": "rebellion", "class_id": 1,
        "level": "L3", "tier": "C", "route_mode": "core_paraphrase",
        "extension_family": None,
        "seed_q": "How does `setup` work?",
        "seed_a": "The `setup` procedure initializes `active?` for all `agents`.",
        "question": "How does `setup` work?",
        "answer": "The `setup` procedure initializes `active?` for all `agents`.",
    },
    {
        "seed_id": "S2", "model_name": "rebellion", "class_id": 1,
        "level": "L3", "tier": "C", "route_mode": "skip",
        "extension_family": None,
        "seed_q": "Irrelevant", "seed_a": "Irrelevant",
        "question": "Irrelevant", "answer": "Irrelevant",
    },
    {
        "seed_id": "S3", "model_name": "rebellion", "class_id": 1,
        "level": "L3", "tier": "C", "route_mode": "core_paraphrase",
        "extension_family": None,
        "seed_q": "What does `go` do?",
        "seed_a": "The `go` procedure runs the main loop with `agents` checking `vision`.",
        "question": "What does `go` do?",
        "answer": "The `go` procedure runs the main loop with `agents` checking `vision`.",
    },
]


class TestEndToEnd:
    def test_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            accepted_path = os.path.join(tmpdir, "accepted.jsonl")
            rejected_path = os.path.join(tmpdir, "rejected.jsonl")
            telemetry_path = os.path.join(tmpdir, "telemetry.jsonl")

            summary = generate_synthetic(
                routed_seeds=SEEDS,
                model_profiles=PROFILE,
                llm_client=EchoSeedLLMClient(),
                embedding_index=AlwaysPassEmbeddingIndex(),
                dedup_index=InMemoryDedupIndex(),
                accepted_path=accepted_path,
                rejected_path=rejected_path,
                telemetry_path=telemetry_path,
                max_workers=2,
            )

            assert summary["total_events"] == 3
            assert summary["by_status"]["accepted"] >= 1
            # S2 is skip, so at least 1 rejected
            assert summary["by_status"].get("rejected", 0) >= 1

            accepted = load_jsonl(accepted_path)
            rejected = load_jsonl(rejected_path)
            assert len(accepted) + len(rejected) == 3

            # Verify accepted records have expected fields
            for rec in accepted:
                assert "question" in rec
                assert "answer" in rec
                assert "gate1" in rec
                assert "gate2" in rec

    def test_dedup_prevents_duplicates(self):
        duped_seeds = [SEEDS[0], SEEDS[0].copy()]  # same seed twice
        duped_seeds[1]["seed_id"] = "S1_dup"

        with tempfile.TemporaryDirectory() as tmpdir:
            accepted_path = os.path.join(tmpdir, "accepted.jsonl")
            rejected_path = os.path.join(tmpdir, "rejected.jsonl")

            summary = generate_synthetic(
                routed_seeds=duped_seeds,
                model_profiles=PROFILE,
                llm_client=EchoSeedLLMClient(),
                embedding_index=AlwaysPassEmbeddingIndex(),
                dedup_index=InMemoryDedupIndex(),
                accepted_path=accepted_path,
                rejected_path=rejected_path,
                max_workers=1,  # single thread to ensure deterministic dedup
            )

            accepted = load_jsonl(accepted_path)
            rejected = load_jsonl(rejected_path)
            # First passes, second should be deduped
            assert len(accepted) == 1
            assert len(rejected) == 1
