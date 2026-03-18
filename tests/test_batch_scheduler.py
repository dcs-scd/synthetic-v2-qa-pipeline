"""Tests for QuotaTracker and batch scheduling."""
from qa_system_v2bis.synthetic_v2.batch_scheduler import QuotaTracker, DEFAULT_QUOTAS, load_quotas


class TestQuotaTrackerBasic:
    def test_first_seed_passes(self):
        qt = QuotaTracker()
        seed = {"model_name": "rebellion", "route_mode": "core_paraphrase",
                "extension_family": None, "seed_id": "S1"}
        ok, reason = qt.should_process(seed)
        assert ok is True
        assert reason is None

    def test_per_model_cap(self):
        qt = QuotaTracker({"per_model": 2, "per_mode": {}, "per_family": 100,
                           "per_seed_per_mode": 100, "per_seed_per_family": 100})
        seed = {"model_name": "rebellion", "route_mode": "core_paraphrase",
                "extension_family": None, "seed_id": "S1"}
        qt.record_accepted(seed)
        qt.record_accepted(seed)
        ok, reason = qt.should_process(seed)
        assert ok is False
        assert "QUOTA_MODEL" in reason

    def test_per_seed_per_mode_cap(self):
        qt = QuotaTracker({"per_model": 100, "per_mode": {}, "per_family": 100,
                           "per_seed_per_mode": 2, "per_seed_per_family": 100})
        seed = {"model_name": "rebellion", "route_mode": "core_paraphrase",
                "extension_family": None, "seed_id": "S1"}
        qt.record_accepted(seed)
        qt.record_accepted(seed)
        ok, reason = qt.should_process(seed)
        assert ok is False
        assert "QUOTA_SEED_MODE" in reason

    def test_different_seeds_independent(self):
        qt = QuotaTracker({"per_model": 100, "per_mode": {}, "per_family": 100,
                           "per_seed_per_mode": 1, "per_seed_per_family": 100})
        s1 = {"model_name": "rebellion", "route_mode": "core_paraphrase",
              "extension_family": None, "seed_id": "S1"}
        s2 = {"model_name": "rebellion", "route_mode": "core_paraphrase",
              "extension_family": None, "seed_id": "S2"}
        qt.record_accepted(s1)
        ok1, _ = qt.should_process(s1)
        ok2, _ = qt.should_process(s2)
        assert ok1 is False
        assert ok2 is True

    def test_per_family_cap(self):
        qt = QuotaTracker({"per_model": 100, "per_mode": {}, "per_family": 1,
                           "per_seed_per_mode": 100, "per_seed_per_family": 100})
        seed = {"model_name": "rebellion", "route_mode": "anchored_extension",
                "extension_family": "broadcast_media", "seed_id": "S1"}
        qt.record_accepted(seed)
        ok, reason = qt.should_process(seed)
        assert ok is False
        assert "QUOTA_FAMILY" in reason


class TestQuotaTrackerSeeding:
    def test_seed_from_existing(self):
        qt = QuotaTracker({"per_model": 2, "per_mode": {}, "per_family": 100,
                           "per_seed_per_mode": 100, "per_seed_per_family": 100})
        existing = [
            {"model_name": "rebellion", "route_mode": "core_paraphrase",
             "extension_family": None, "seed_id": "S1"},
            {"model_name": "rebellion", "route_mode": "core_repair",
             "extension_family": None, "seed_id": "S2"},
        ]
        qt.seed_from_existing(existing)
        ok, reason = qt.should_process(existing[0])
        assert ok is False


class TestQuotaTrackerSummary:
    def test_summary_shape(self):
        qt = QuotaTracker()
        s = qt.summary()
        assert "model_counts" in s
        assert "family_counts" in s
        assert "skip_reasons" in s
        assert "total_accepted" in s
        assert "total_skipped" in s
