"""Tests for LLM JSON response parsing."""
from qa_system_v2bis.synthetic_v2.run_synthetic_v2 import parse_llm_json, MAX_LLM_RESPONSE_SIZE


class TestParseValid:
    def test_valid_qa(self):
        r = parse_llm_json('{"question": "Q?", "answer": "A."}')
        assert r["ok"] is True
        assert r["data"]["question"] == "Q?"

    def test_extra_keys_preserved_in_data(self):
        r = parse_llm_json('{"question": "Q?", "answer": "A.", "extra": 1}')
        assert r["ok"] is True
        assert r["data"]["extra"] == 1


class TestParseInvalid:
    def test_not_json(self):
        r = parse_llm_json("not json at all")
        assert r["ok"] is False
        assert "json_parse_error" in r["error"]

    def test_not_dict(self):
        r = parse_llm_json('[1, 2, 3]')
        assert r["ok"] is False
        assert "not_json_object" in r["error"]

    def test_missing_question(self):
        r = parse_llm_json('{"answer": "A."}')
        assert r["ok"] is False
        assert "missing_question_or_answer" in r["error"]

    def test_skip_response(self):
        r = parse_llm_json('{"skip": "SEED_CONFLICT"}')
        assert r["ok"] is False
        assert "model_returned_skip" in r["error"]

    def test_too_large(self):
        big = '{"question": "' + "x" * (MAX_LLM_RESPONSE_SIZE + 1) + '", "answer": "A."}'
        r = parse_llm_json(big)
        assert r["ok"] is False
        assert "response_too_large" in r["error"]

    def test_non_string_question(self):
        r = parse_llm_json('{"question": 123, "answer": "A."}')
        assert r["ok"] is False
        assert "not_string" in r["error"]
