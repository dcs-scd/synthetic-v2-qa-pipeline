"""Batch scheduling with per-model, per-mode, per-family, and per-seed quota enforcement."""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

from .io_utils import load_json
from .text_utils import normalize_model_name

logger = logging.getLogger(__name__)

# Default quota caps (override via config)
DEFAULT_QUOTAS = {
    "per_model": 50000,
    "per_mode": {
        "core_paraphrase": 0.30,   # 30% of per_model cap
        "core_repair": 0.20,       # 20%
        "anchored_extension": 0.50, # 50%
    },
    "per_family": 10000,
    "per_seed_per_mode": 5,
    "per_seed_per_family": 3,
}


class QuotaTracker:
    """Track and enforce generation quotas.

    Quotas are checked BEFORE processing a seed (should_process).
    Counters are updated AFTER a seed is accepted (record_accepted).
    """

    def __init__(self, quotas: Optional[Dict[str, Any]] = None):
        self.quotas = quotas or DEFAULT_QUOTAS
        self._model_counts: Counter = Counter()
        self._model_mode_counts: Dict[str, Counter] = defaultdict(Counter)
        self._family_counts: Counter = Counter()
        self._seed_mode_counts: Dict[str, Counter] = defaultdict(Counter)
        self._seed_family_counts: Dict[str, Counter] = defaultdict(Counter)
        self._skip_reasons: Counter = Counter()

    def should_process(self, seed_row: Dict[str, Any]) -> tuple:
        """Check if this seed should be processed given current quotas.

        Returns (True, None) if OK, (False, reason_str) if quota exceeded.
        """
        model = normalize_model_name(seed_row.get("model_name")) or "unknown"
        mode = seed_row.get("route_mode") or "unknown"
        family = seed_row.get("extension_family")
        seed_id = seed_row.get("seed_id") or "unknown"

        # Per-model cap
        per_model = self.quotas.get("per_model", 50000)
        if self._model_counts[model] >= per_model:
            return False, f"QUOTA_MODEL:{model}"

        # Per-mode cap (percentage of per_model)
        mode_pcts = self.quotas.get("per_mode", {})
        if mode in mode_pcts:
            mode_cap = int(per_model * mode_pcts[mode])
            if self._model_mode_counts[model][mode] >= mode_cap:
                return False, f"QUOTA_MODE:{model}:{mode}"

        # Per-family cap
        if family:
            per_family = self.quotas.get("per_family", 10000)
            if self._family_counts[family] >= per_family:
                return False, f"QUOTA_FAMILY:{family}"

        # Per-seed-per-mode cap
        per_seed_mode = self.quotas.get("per_seed_per_mode", 5)
        if self._seed_mode_counts[seed_id][mode] >= per_seed_mode:
            return False, f"QUOTA_SEED_MODE:{seed_id}:{mode}"

        # Per-seed-per-family cap
        if family:
            per_seed_family = self.quotas.get("per_seed_per_family", 3)
            if self._seed_family_counts[seed_id][family] >= per_seed_family:
                return False, f"QUOTA_SEED_FAMILY:{seed_id}:{family}"

        return True, None

    def record_accepted(self, seed_row: Dict[str, Any]) -> None:
        """Update counters after a seed is accepted."""
        model = normalize_model_name(seed_row.get("model_name")) or "unknown"
        mode = seed_row.get("route_mode") or "unknown"
        family = seed_row.get("extension_family")
        seed_id = seed_row.get("seed_id") or "unknown"

        self._model_counts[model] += 1
        self._model_mode_counts[model][mode] += 1
        self._seed_mode_counts[seed_id][mode] += 1
        if family:
            self._family_counts[family] += 1
            self._seed_family_counts[seed_id][family] += 1

    def record_skip(self, reason: str) -> None:
        """Track skip reasons for reporting."""
        self._skip_reasons[reason] += 1

    def seed_from_existing(self, accepted_rows: List[Dict[str, Any]]) -> None:
        """Pre-seed counters from existing accepted records (for resume)."""
        for row in accepted_rows:
            self.record_accepted(row)
        if accepted_rows:
            logger.info(
                "QuotaTracker seeded from %d existing accepted records",
                len(accepted_rows)
            )

    def summary(self) -> Dict[str, Any]:
        """Return quota state summary for telemetry."""
        return {
            "model_counts": dict(self._model_counts),
            "family_counts": dict(self._family_counts),
            "skip_reasons": dict(self._skip_reasons),
            "total_accepted": sum(self._model_counts.values()),
            "total_skipped": sum(self._skip_reasons.values()),
        }


def load_quotas(path: Optional[str] = None) -> Dict[str, Any]:
    """Load quota config from JSON file, or return defaults."""
    if path is None:
        return DEFAULT_QUOTAS
    return load_json(path)
