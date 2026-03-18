"""Tests for LLM retry logic."""
from qa_system_v2bis.synthetic_v2.run_synthetic_v2 import _call_llm_with_retry, _safe_error_str


class MockLLMClient:
    def __init__(self, fail_times=0, error_msg="429 rate limit exceeded"):
        self.fail_times = fail_times
        self.error_msg = error_msg
        self.call_count = 0

    def generate(self, prompt, seed_row=None):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise Exception(self.error_msg)
        return '{"question": "Q?", "answer": "A."}'


class TestRetrySuccess:
    def test_succeeds_first_try(self):
        client = MockLLMClient(fail_times=0)
        result = _call_llm_with_retry(client, "prompt", {})
        assert "question" in result
        assert client.call_count == 1

    def test_succeeds_after_transient_failure(self):
        client = MockLLMClient(fail_times=1, error_msg="429 rate limit exceeded")
        result = _call_llm_with_retry(client, "prompt", {})
        assert "question" in result
        assert client.call_count == 2


class TestRetryExhausted:
    def test_raises_after_max_retries(self):
        client = MockLLMClient(fail_times=10, error_msg="429 rate limit exceeded")
        try:
            _call_llm_with_retry(client, "prompt", {})
            assert False, "Should have raised"
        except Exception as e:
            assert "429" in str(e)
        assert client.call_count == 3  # MAX_LLM_RETRIES


class TestNonTransientError:
    def test_non_transient_raises_immediately(self):
        client = MockLLMClient(fail_times=10, error_msg="Invalid API key")
        try:
            _call_llm_with_retry(client, "prompt", {})
            assert False, "Should have raised"
        except Exception as e:
            assert "Invalid API key" in str(e)
        assert client.call_count == 1  # No retry on non-transient


class TestSafeErrorStr:
    def test_redacts_api_keys(self):
        result = _safe_error_str(Exception("Error: sk-abc123def456 is invalid"))
        assert "sk-" not in result or "REDACTED" in result

    def test_truncates_long_messages(self):
        result = _safe_error_str(Exception("x" * 500), max_len=100)
        assert len(result) <= 110  # some slack for redaction
