"""
Tests for batch_api_utils.py

Tests the non-API functions (chunking, request building, file writing).
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_chunk_requests():
    """Test chunking of requests into batches."""
    from batch_api_utils import chunk_requests

    # Test normal chunking
    reqs = [{"id": i} for i in range(2500)]
    chunks = chunk_requests(reqs, chunk_size=1000)
    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000
    assert len(chunks[2]) == 500

    # Verify all items are present
    flat_items = [item for chunk in chunks for item in chunk]
    assert len(flat_items) == 2500
    assert [item["id"] for item in flat_items] == list(range(2500))


def test_chunk_small():
    """Test chunking with fewer items than chunk size."""
    from batch_api_utils import chunk_requests

    reqs = [{"id": i} for i in range(50)]
    chunks = chunk_requests(reqs, chunk_size=1000)
    assert len(chunks) == 1
    assert len(chunks[0]) == 50
    assert chunks[0] == reqs


def test_chunk_empty():
    """Test chunking with empty input."""
    from batch_api_utils import chunk_requests

    reqs = []
    chunks = chunk_requests(reqs, chunk_size=1000)
    assert len(chunks) == 0


def test_chunk_exact_multiple():
    """Test chunking when input size is exact multiple of chunk size."""
    from batch_api_utils import chunk_requests

    reqs = [{"id": i} for i in range(2000)]
    chunks = chunk_requests(reqs, chunk_size=1000)
    assert len(chunks) == 2
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000


def test_build_request():
    """Test building OpenAI batch request objects."""
    from batch_api_utils import build_openai_batch_request

    req = build_openai_batch_request("test_0", "Hello", model="gpt-5.4-nano")

    # Check structure
    assert req["custom_id"] == "test_0"
    assert req["method"] == "POST"
    assert req["url"] == "/v1/chat/completions"

    # Check body
    body = req["body"]
    assert body["model"] == "gpt-5.4-nano"
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["temperature"] == 0.7  # default
    assert body["max_tokens"] == 2048  # default


def test_build_request_custom_params():
    """Test building request with custom parameters."""
    from batch_api_utils import build_openai_batch_request

    req = build_openai_batch_request(
        "test_1",
        "What is 2+2?",
        model="gpt-4",
        temperature=0.1,
        max_tokens=100
    )

    body = req["body"]
    assert body["model"] == "gpt-4"
    assert body["messages"][0]["content"] == "What is 2+2?"
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 100


def test_write_file(tmp_path):
    """Test writing batch input file."""
    from batch_api_utils import write_batch_input_file

    reqs = [
        {"custom_id": "0", "data": "first"},
        {"custom_id": "1", "data": "second"}
    ]

    output_path = str(tmp_path / "batch.jsonl")
    result_path = write_batch_input_file(reqs, output_path)

    # Check return value
    assert result_path == output_path

    # Check file contents
    assert Path(output_path).exists()
    with open(output_path) as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"custom_id": "0", "data": "first"}
    assert json.loads(lines[1]) == {"custom_id": "1", "data": "second"}


def test_write_file_empty(tmp_path):
    """Test writing empty batch input file."""
    from batch_api_utils import write_batch_input_file

    reqs = []
    output_path = str(tmp_path / "empty.jsonl")
    write_batch_input_file(reqs, output_path)

    assert Path(output_path).exists()
    with open(output_path) as f:
        content = f.read()
    assert content == ""


def test_integration_chunking_and_building():
    """Test integration between chunking and request building."""
    from batch_api_utils import chunk_requests, build_openai_batch_request

    # Build requests
    prompts = [f"Prompt {i}" for i in range(150)]
    requests = []
    for i, prompt in enumerate(prompts):
        req = build_openai_batch_request(f"req-{i}", prompt)
        requests.append(req)

    # Chunk requests
    chunks = chunk_requests(requests, chunk_size=100)

    # Verify chunking
    assert len(chunks) == 2
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 50

    # Verify request structure is preserved
    first_req = chunks[0][0]
    assert first_req["custom_id"] == "req-0"
    assert first_req["body"]["messages"][0]["content"] == "Prompt 0"

    last_req = chunks[1][-1]
    assert last_req["custom_id"] == "req-149"
    assert last_req["body"]["messages"][0]["content"] == "Prompt 149"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])