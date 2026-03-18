"""Tests for prompt building."""
import pytest
from qa_system_v2bis.synthetic_v2.prompt_builder import build_prompt

PROFILE = {
    "core": {
        "procedures": ["setup", "go"],
        "variables": ["active?"],
        "breeds": ["agents"],
        "widgets": ["vision"],
        "model_summary": "Test model.",
    },
    "extensions": {
        "families": [
            {
                "name": "test_family",
                "concepts": ["test concept"],
                "identifiers": ["test-id"],
                "rules": ["test rule"],
            }
        ],
        "general_rules": ["Stay anchored."],
        "framing_cues": ["extend the model"],
        "disallowed_unanchored_themes": [],
    }
}


class TestCoreParaphrasePrompt:
    def test_contains_mode(self):
        seed = {"model_name": "test", "level": "L3", "tier": "C",
                "route_mode": "core_paraphrase", "seed_q": "Q?", "seed_a": "A."}
        prompt = build_prompt(seed, PROFILE)
        assert "core_paraphrase" in prompt
        assert "Do not add extensions" in prompt

    def test_contains_core_block(self):
        seed = {"model_name": "test", "level": "L3", "tier": "C",
                "route_mode": "core_paraphrase", "seed_q": "Q?", "seed_a": "A."}
        prompt = build_prompt(seed, PROFILE)
        assert "setup" in prompt
        assert "active?" in prompt


class TestExtensionPrompt:
    def test_contains_family(self):
        seed = {"model_name": "test", "level": "L3", "tier": "C",
                "route_mode": "anchored_extension", "extension_family": "test_family",
                "seed_q": "Q?", "seed_a": "A."}
        prompt = build_prompt(seed, PROFILE)
        assert "test_family" in prompt
        assert "test-id" in prompt
        assert "test concept" in prompt

    def test_no_cross_family_in_prompt(self):
        seed = {"model_name": "test", "level": "L3", "tier": "C",
                "route_mode": "anchored_extension", "extension_family": "test_family",
                "seed_q": "Q?", "seed_a": "A."}
        prompt = build_prompt(seed, PROFILE)
        # Only the selected family should appear
        assert "test_family" in prompt


class TestCoreRepairPrompt:
    def test_contains_skip_option(self):
        seed = {"model_name": "test", "level": "L3", "tier": "C",
                "route_mode": "core_repair", "seed_q": "Q?", "seed_a": "A."}
        prompt = build_prompt(seed, PROFILE)
        assert "SEED_CONFLICT" in prompt


class TestInvalidMode:
    def test_raises_on_skip(self):
        seed = {"route_mode": "skip"}
        with pytest.raises(ValueError):
            build_prompt(seed, PROFILE)
