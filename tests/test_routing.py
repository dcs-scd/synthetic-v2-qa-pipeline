"""Tests for seed routing logic."""
import pytest
from qa_system_v2bis.synthetic_v2.routing import route_seed, DEFAULT_ROUTING_CONFIG

REBELLION_PROFILE = {
    "core": {
        "procedures": ["setup", "go", "determine-behavior", "grievance", "estimated-arrest-probability"],
        "variables": ["active?", "jail-term", "risk-aversion", "perceived-hardship", "k", "threshold"],
        "breeds": ["agents", "an-agent", "cops", "cop"],
        "widgets": ["government-legitimacy", "vision", "initial-cop-density"],
        "model_summary": "The rebellion model simulates citizen uprising against a central authority.",
    },
    "extensions": {
        "families": [
            {
                "name": "broadcast_media",
                "concepts": ["broadcast media", "global signal", "media influence"],
                "identifiers": ["media-influence", "media-signal", "social-influence-weight"],
                "rules": ["Must be framed as an added global influence."],
            },
            {
                "name": "state_refinement",
                "concepts": ["state diagram", "sympathizer", "protester"],
                "identifiers": ["rebellion-state", "global-unrest-index"],
                "rules": [],
            },
        ],
        "general_rules": [],
        "framing_cues": ["extend the model", "you could add"],
        "disallowed_unanchored_themes": ["quantum mechanics"],
    }
}


class TestCoreParaphrase:
    def test_source_faithful_seed_routes_core(self):
        result = route_seed(
            seed_q="How does the rebellion model decide agent activation?",
            seed_a="The `determine-behavior` procedure checks if `grievance` exceeds `threshold`. When it does, the agent sets `active?` to true.",
            profile=REBELLION_PROFILE,
        )
        assert result["route"] in ("core_paraphrase", "core_repair")
        assert result["family"] is None


class TestAnchoredExtension:
    def test_media_seed_routes_extension(self):
        result = route_seed(
            seed_q="How could I add a broadcast media layer to the rebellion model?",
            seed_a="You could extend the model by adding a `media-influence` global that modulates `grievance`. This broadcast media signal would represent...",
            profile=REBELLION_PROFILE,
        )
        assert result["route"] == "anchored_extension"
        assert result["family"] == "broadcast_media"

    def test_state_seed_routes_extension(self):
        result = route_seed(
            seed_q="How detailed should the state diagram be for rebellion agents?",
            seed_a="You could add a `rebellion-state` variable to replace the binary `active?` with states like sympathizer, protester...",
            profile=REBELLION_PROFILE,
        )
        assert result["route"] == "anchored_extension"
        assert result["family"] == "state_refinement"


class TestSkip:
    def test_unanchored_seed_skips(self):
        result = route_seed(
            seed_q="How do financial derivatives affect stock prices?",
            seed_a="Derivatives are complex instruments...",
            profile=REBELLION_PROFILE,
        )
        assert result["route"] == "skip"
