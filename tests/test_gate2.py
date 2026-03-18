"""Tests for mode-aware Gate 2 validation."""
import pytest
from qa_system_v2bis.synthetic_v2.gate2_model_validation import validate_gate2

PROFILE = {
    "core": {
        "procedures": ["setup", "go", "determine-behavior", "grievance"],
        "variables": ["active?", "jail-term", "risk-aversion"],
        "breeds": ["agents", "cops"],
        "widgets": ["government-legitimacy", "vision"],
        "model_summary": "Rebellion model.",
    },
    "extensions": {
        "families": [
            {
                "name": "broadcast_media",
                "concepts": ["broadcast media"],
                "identifiers": ["media-influence", "media-signal"],
                "rules": [],
            },
            {
                "name": "network_layer",
                "concepts": ["social network"],
                "identifiers": ["homophily-index", "network-density"],
                "rules": [],
            }
        ],
        "general_rules": [],
        "framing_cues": ["extend the model", "you could add"],
        "disallowed_unanchored_themes": ["quantum mechanics"],
    }
}


class TestCoreValidation:
    def test_valid_core_answer_passes(self):
        r = validate_gate2(
            question="How does activation work?",
            answer="The `determine-behavior` procedure checks `grievance` against threshold. When `active?` becomes true, the agent joins the rebellion.",
            mode="core_paraphrase",
            profile=PROFILE,
        )
        assert r["ok"] is True

    def test_unknown_identifier_fails(self):
        r = validate_gate2(
            question="How does activation work?",
            answer="The `fake-procedure` controls activation.",
            mode="core_paraphrase",
            profile=PROFILE,
        )
        assert r["ok"] is False
        assert r["reason"] == "UNKNOWN_CORE_IDENTIFIER"

    def test_extension_id_in_core_mode_fails(self):
        r = validate_gate2(
            question="How does activation work?",
            answer="The `media-influence` variable affects activation.",
            mode="core_paraphrase",
            profile=PROFILE,
        )
        assert r["ok"] is False
        assert r["reason"] == "UNAPPROVED_EXTENSION_IDENTIFIER"


class TestExtensionValidation:
    def test_valid_extension_passes(self):
        r = validate_gate2(
            question="How could broadcast media affect rebellion?",
            answer="To extend the model, you could add a `media-influence` variable that modulates `grievance`.",
            mode="anchored_extension",
            profile=PROFILE,
            family_name="broadcast_media",
        )
        assert r["ok"] is True

    def test_cross_family_fails(self):
        r = validate_gate2(
            question="How could broadcast media affect rebellion?",
            answer="To extend the model, you could add `media-influence` and `homophily-index`.",
            mode="anchored_extension",
            profile=PROFILE,
            family_name="broadcast_media",
        )
        assert r["ok"] is False
        assert r["reason"] == "CROSS_FAMILY_EXTENSION_IDENTIFIER"

    def test_no_core_anchor_fails(self):
        r = validate_gate2(
            question="What about broadcast media?",
            answer="The `media-signal` extends things.",
            mode="anchored_extension",
            profile=PROFILE,
            family_name="broadcast_media",
        )
        # This should fail with EXTENSION_REQUIRES_FRAMING or INSUFFICIENT_CORE_ANCHOR
        assert r["ok"] is False


class TestUnknownMode:
    def test_unknown_mode_fails(self):
        r = validate_gate2(
            question="test", answer="test",
            mode="nonexistent_mode",
            profile=PROFILE,
        )
        assert r["ok"] is False
        assert r["reason"] == "UNKNOWN_MODE"
