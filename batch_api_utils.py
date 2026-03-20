"""
Reusable batch API utility module for chunked OpenAI batch submissions.

Handles the 1000-request limit per batch by chunking large request sets
and submitting multiple batches with polling and result collection.
"""

import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Configuration constants
OPENAI_BATCH_CHUNK_SIZE = 1000
DEFAULT_POLL_INTERVAL = 15  # seconds
DEFAULT_TIMEOUT = 7200  # 2 hours
DEFAULT_STALL_TIMEOUT = 300  # 5 minutes

logger = logging.getLogger(__name__)


def build_openai_batch_request(
    custom_id: str,
    prompt: str,
    model: str = "gpt-5.4-nano",
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Dict[str, Any]:
    """
    Build a single OpenAI batch API request object.

    Args:
        custom_id: Unique identifier for this request
        prompt: The prompt text to send to the model
        model: OpenAI model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Dict representing a single batch request
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }


def write_batch_input_file(requests: List[Dict[str, Any]], output_path: str) -> str:
    """
    Write batch requests to JSONL file format required by OpenAI.

    Args:
        requests: List of request dictionaries
        output_path: Path to write the JSONL file

    Returns:
        The output path (for convenience)
    """
    with open(output_path, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')

    logger.info(f"Wrote {len(requests)} batch requests to {output_path}")
    return output_path


def chunk_requests(requests: List[Dict[str, Any]], chunk_size: int = OPENAI_BATCH_CHUNK_SIZE) -> List[List[Dict[str, Any]]]:
    """
    Split a list of requests into chunks of the specified size.

    Args:
        requests: List of request dictionaries
        chunk_size: Maximum size of each chunk (default: 1000)

    Returns:
        List of request chunks
    """
    if not requests:
        return []

    chunks = []
    for i in range(0, len(requests), chunk_size):
        chunk = requests[i:i + chunk_size]
        chunks.append(chunk)

    logger.info(f"Split {len(requests)} requests into {len(chunks)} chunks")
    return chunks


def submit_openai_batch_chunked(
    client,  # OpenAI client
    requests: List[Dict[str, Any]],
    temp_dir: str,
    description: str = "Batch API submission",
    inter_chunk_delay: float = 2.0
) -> List[str]:
    """
    Submit requests as chunked OpenAI batches.

    Args:
        client: OpenAI client instance
        requests: List of request dictionaries
        temp_dir: Directory for temporary JSONL files
        description: Description for the batch jobs
        inter_chunk_delay: Seconds to wait between chunk submissions

    Returns:
        List of batch IDs
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    chunks = chunk_requests(requests)
    batch_ids = []

    for i, chunk in enumerate(chunks):
        # Write chunk to temp file
        chunk_file = temp_path / f"batch_chunk_{i:03d}.jsonl"
        write_batch_input_file(chunk, str(chunk_file))

        # Upload file
        with open(chunk_file, 'rb') as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )

        # Create batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"{description} - Chunk {i+1}/{len(chunks)}",
                "chunk_index": str(i),
                "total_chunks": str(len(chunks))
            }
        )

        batch_ids.append(batch.id)
        logger.info(f"Submitted batch {batch.id} (chunk {i+1}/{len(chunks)}, {len(chunk)} requests)")

        # Delay between submissions
        if i < len(chunks) - 1:
            time.sleep(inter_chunk_delay)

        # Clean up temp file
        chunk_file.unlink()

    logger.info(f"Submitted {len(batch_ids)} batches total")
    return batch_ids


def poll_openai_batch(
    client,  # OpenAI client
    batch_id: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    timeout: int = DEFAULT_TIMEOUT,
    stall_timeout: int = DEFAULT_STALL_TIMEOUT
) -> Dict[str, Any]:
    """
    Poll a single OpenAI batch until completion or timeout.

    Args:
        client: OpenAI client instance
        batch_id: The batch ID to poll
        poll_interval: Seconds between polls
        timeout: Maximum total time to wait
        stall_timeout: Maximum time without progress before giving up

    Returns:
        Dict with status information including final batch state
    """
    start_time = time.time()
    last_progress_time = start_time
    last_completed_count = 0

    while time.time() - start_time < timeout:
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status

            # Check for completion
            if status in ['completed', 'failed', 'expired', 'cancelled']:
                elapsed = time.time() - start_time
                logger.info(f"Batch {batch_id} finished with status: {status} (elapsed: {elapsed:.1f}s)")
                return {
                    "status": status,
                    "batch": batch,
                    "elapsed_time": elapsed,
                    "success": status == 'completed'
                }

            # Check for progress (to detect stalls)
            completed_count = getattr(batch.request_counts, 'completed', 0)
            if completed_count > last_completed_count:
                last_completed_count = completed_count
                last_progress_time = time.time()

            # Check for stall
            if time.time() - last_progress_time > stall_timeout:
                logger.warning(f"Batch {batch_id} appears stalled (no progress for {stall_timeout}s)")
                return {
                    "status": "stalled",
                    "batch": batch,
                    "elapsed_time": time.time() - start_time,
                    "success": False,
                    "last_progress_time": last_progress_time
                }

            # Log progress
            if hasattr(batch, 'request_counts'):
                counts = batch.request_counts
                total = getattr(counts, 'total', 0)
                completed = getattr(counts, 'completed', 0)
                failed = getattr(counts, 'failed', 0)
                logger.info(f"Batch {batch_id}: {completed}/{total} completed, {failed} failed")

        except Exception as e:
            logger.error(f"Error polling batch {batch_id}: {e}")

        time.sleep(poll_interval)

    # Timeout reached
    logger.error(f"Batch {batch_id} timed out after {timeout}s")
    try:
        batch = client.batches.retrieve(batch_id)
        return {
            "status": "timeout",
            "batch": batch,
            "elapsed_time": timeout,
            "success": False
        }
    except Exception as e:
        logger.error(f"Failed to retrieve batch {batch_id} after timeout: {e}")
        return {
            "status": "timeout_error",
            "batch": None,
            "elapsed_time": timeout,
            "success": False,
            "error": str(e)
        }


def collect_openai_batch_results(client, batch_id: str) -> Dict[str, str]:
    """
    Download and parse results from a completed OpenAI batch.

    Args:
        client: OpenAI client instance
        batch_id: The completed batch ID

    Returns:
        Dict mapping custom_id to response text content
    """
    try:
        batch = client.batches.retrieve(batch_id)

        if not batch.output_file_id:
            logger.error(f"Batch {batch_id} has no output file")
            return {}

        # Download output file
        file_response = client.files.content(batch.output_file_id)
        content = file_response.text if hasattr(file_response, 'text') else file_response.read().decode()

        # Parse JSONL results
        results = {}
        for line_num, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                custom_id = row.get('custom_id')
                if not custom_id:
                    logger.warning(f"Line {line_num}: missing custom_id")
                    continue

                response = row.get('response', {})
                if response.get('status_code', 200) >= 400:
                    logger.warning(f"Line {line_num}: HTTP error {response.get('status_code')}")
                    continue

                body = response.get('body', {})
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except json.JSONDecodeError:
                        logger.warning(f"Line {line_num}: invalid JSON body")
                        continue

                choices = body.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    results[custom_id] = content
                else:
                    logger.warning(f"Line {line_num}: no choices in response")

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                continue

        logger.info(f"Collected {len(results)} results from batch {batch_id}")
        return results

    except Exception as e:
        logger.error(f"Failed to collect results from batch {batch_id}: {e}")
        return {}


def poll_and_collect_all_chunks(
    client,  # OpenAI client
    batch_ids: List[str],
    chunk_size: int = OPENAI_BATCH_CHUNK_SIZE,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    timeout: int = DEFAULT_TIMEOUT,
    stall_timeout: int = DEFAULT_STALL_TIMEOUT
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Poll all batch chunks until completion and collect results.

    Args:
        client: OpenAI client instance
        batch_ids: List of batch IDs to poll
        chunk_size: Size of each chunk (for index calculation)
        poll_interval: Seconds between polls
        timeout: Maximum total time to wait per batch
        stall_timeout: Maximum time without progress before giving up

    Returns:
        Tuple of (results_by_index, summary) where:
        - results_by_index: Dict mapping original request index to response text
        - summary: Dict with overall statistics and status
    """
    results_by_index = {}
    batch_statuses = {}
    total_collected = 0
    total_expected = len(batch_ids) * chunk_size

    for i, batch_id in enumerate(batch_ids):
        logger.info(f"Polling batch {i+1}/{len(batch_ids)}: {batch_id}")

        # Poll until completion
        poll_result = poll_openai_batch(
            client, batch_id, poll_interval, timeout, stall_timeout
        )
        batch_statuses[batch_id] = poll_result

        # Collect results if successful
        if poll_result['success']:
            chunk_results = collect_openai_batch_results(client, batch_id)

            # Map custom_ids back to original indices
            for custom_id, content in chunk_results.items():
                # Assume custom_id format is 'req-{index}'
                if custom_id.startswith('req-'):
                    try:
                        index = int(custom_id[4:])
                        results_by_index[index] = content
                        total_collected += 1
                    except ValueError:
                        logger.warning(f"Could not parse index from custom_id: {custom_id}")
        else:
            logger.error(f"Batch {batch_id} failed with status: {poll_result['status']}")

    # Calculate summary statistics
    successful_batches = sum(1 for status in batch_statuses.values() if status['success'])
    failed_batches = len(batch_ids) - successful_batches

    summary = {
        'total_batches': len(batch_ids),
        'successful_batches': successful_batches,
        'failed_batches': failed_batches,
        'total_collected': total_collected,
        'total_expected': total_expected,
        'collection_rate': total_collected / total_expected if total_expected > 0 else 0.0,
        'batch_statuses': batch_statuses
    }

    logger.info(f"Collection complete: {total_collected}/{total_expected} results "
                f"({summary['collection_rate']:.1%}) from {successful_batches}/{len(batch_ids)} batches")

    return results_by_index, summary