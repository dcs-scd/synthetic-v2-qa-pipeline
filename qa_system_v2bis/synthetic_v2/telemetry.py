from collections import Counter, defaultdict
from typing import Dict, Any, List

from .io_utils import append_jsonl, save_json
from .text_utils import normalize_model_name


def make_event(
    seed_row: Dict[str, Any],
    status: str,
    reason: str,
    details: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "seed_id": seed_row.get("seed_id"),
        "model_name": normalize_model_name(seed_row.get("model_name")),
        "class_id": seed_row.get("class_id"),
        "level": seed_row.get("level"),
        "tier": seed_row.get("tier"),
        "route_mode": seed_row.get("route_mode"),
        "extension_family": seed_row.get("extension_family"),
        "status": status,
        "reason": reason,
        "details": details or {}
    }


def record_attempt(telemetry_path: str, event: Dict[str, Any]) -> None:
    append_jsonl(telemetry_path, event)


def summarize_telemetry(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_status = Counter()
    by_reason = Counter()
    by_model = Counter()
    by_mode = Counter()
    by_family = Counter()
    by_model_mode = defaultdict(Counter)
    by_model_reason = defaultdict(Counter)

    for ev in events:
        status = ev.get("status")
        reason = ev.get("reason")
        model = normalize_model_name(ev.get("model_name"))
        mode = ev.get("route_mode")
        family = ev.get("extension_family")

        if status:
            by_status[status] += 1
        if reason:
            by_reason[reason] += 1
        if model:
            by_model[model] += 1
            if mode:
                by_model_mode[model][mode] += 1
            if reason:
                by_model_reason[model][reason] += 1
        if mode:
            by_mode[mode] += 1
        if family:
            by_family[family] += 1

    return {
        "total_events": len(events),
        "by_status": dict(by_status),
        "by_reason": dict(by_reason),
        "by_model": dict(by_model),
        "by_mode": dict(by_mode),
        "by_family": dict(by_family),
        "by_model_mode": {k: dict(v) for k, v in by_model_mode.items()},
        "by_model_reason": {k: dict(v) for k, v in by_model_reason.items()},
    }


class TelemetrySummary:
    """Incremental telemetry accumulator — O(1) memory regardless of event count."""

    def __init__(self):
        self.total_events = 0
        self.by_status: Counter = Counter()
        self.by_reason: Counter = Counter()
        self.by_model: Counter = Counter()
        self.by_mode: Counter = Counter()
        self.by_family: Counter = Counter()
        self.by_model_mode: Dict[str, Counter] = defaultdict(Counter)
        self.by_model_reason: Dict[str, Counter] = defaultdict(Counter)

    def record(self, event: Dict[str, Any]) -> None:
        self.total_events += 1
        status = event.get("status")
        reason = event.get("reason")
        model = normalize_model_name(event.get("model_name"))
        mode = event.get("route_mode")
        family = event.get("extension_family")

        if status:
            self.by_status[status] += 1
        if reason:
            self.by_reason[reason] += 1
        if model:
            self.by_model[model] += 1
            if mode:
                self.by_model_mode[model][mode] += 1
            if reason:
                self.by_model_reason[model][reason] += 1
        if mode:
            self.by_mode[mode] += 1
        if family:
            self.by_family[family] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "by_status": dict(self.by_status),
            "by_reason": dict(self.by_reason),
            "by_model": dict(self.by_model),
            "by_mode": dict(self.by_mode),
            "by_family": dict(self.by_family),
            "by_model_mode": {k: dict(v) for k, v in self.by_model_mode.items()},
            "by_model_reason": {k: dict(v) for k, v in self.by_model_reason.items()},
        }


def save_telemetry_summary(events: List[Dict[str, Any]], out_path: str) -> None:
    summary = summarize_telemetry(events)
    save_json(summary, out_path)
