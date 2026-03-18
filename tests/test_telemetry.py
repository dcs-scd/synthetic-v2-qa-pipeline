"""Tests for telemetry event creation and summarization."""
from qa_system_v2bis.synthetic_v2.telemetry import make_event, summarize_telemetry


class TestMakeEvent:
    def test_event_shape(self):
        seed = {"seed_id": "S1", "model_name": "rebellion", "class_id": 13,
                "level": "L3", "tier": "C", "route_mode": "core_paraphrase",
                "extension_family": None}
        event = make_event(seed, status="accepted", reason="ACCEPTED", details={})
        assert event["seed_id"] == "S1"
        assert event["model_name"] == "rebellion"
        assert event["status"] == "accepted"
        assert event["reason"] == "ACCEPTED"


class TestSummarizeTelemetry:
    def test_empty(self):
        s = summarize_telemetry([])
        assert s["total_events"] == 0

    def test_counts(self):
        events = [
            {"status": "accepted", "reason": "ACCEPTED", "model_name": "rebellion",
             "route_mode": "core_paraphrase", "extension_family": None},
            {"status": "rejected", "reason": "BAD_JSON", "model_name": "rebellion",
             "route_mode": "core_repair", "extension_family": None},
            {"status": "accepted", "reason": "ACCEPTED", "model_name": "shepherds",
             "route_mode": "anchored_extension", "extension_family": "state_refinement"},
        ]
        s = summarize_telemetry(events)
        assert s["total_events"] == 3
        assert s["by_status"]["accepted"] == 2
        assert s["by_status"]["rejected"] == 1
        assert s["by_model"]["rebellion"] == 2
        assert s["by_family"]["state_refinement"] == 1
