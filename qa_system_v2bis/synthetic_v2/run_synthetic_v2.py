import argparse
import json
from typing import Dict, Any, List, Optional

from .io_utils import load_json, load_jsonl, append_jsonl, save_json
from .prompt_builder import build_prompt, get_model_profile
from .gate1_embedding import run_gate1_embedding, AlwaysPassEmbeddingIndex
from .gate2_model_validation import validate_generated_row
from .gate3_dedup import InMemoryDedupIndex, run_gate3_dedup
from .telemetry import make_event, record_attempt, summarize_telemetry
from .text_utils import normalize_model_name


# -------------------------------------------------------------------
# LLM client smoke-test adapters
# -------------------------------------------------------------------

class EchoSeedLLMClient:
    """
    Smoke-test LLM adapter.
    Returns the seed's own Q/A as JSON.
    Useful for pipeline testing without calling a real model.
    """
    def generate(self, prompt: str, seed_row: Dict[str, Any]) -> str:
        q = seed_row.get("seed_q") or seed_row.get("question") or ""
        a = seed_row.get("seed_a") or seed_row.get("answer") or ""
        return json.dumps({"question": q, "answer": a}, ensure_ascii=False)


class StaticJSONLLMClient:
    """
    Another smoke-test adapter.
    Returns the same canned JSON object every time.
    """
    def __init__(self, question: str, answer: str):
        self.obj = {"question": question, "answer": answer}

    def generate(self, prompt: str, seed_row: Dict[str, Any]) -> str:
        return json.dumps(self.obj, ensure_ascii=False)


# -------------------------------------------------------------------
# JSON parsing
# -------------------------------------------------------------------

def parse_llm_json(raw: str) -> Dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {"ok": False, "error": f"json_parse_error: {e}", "raw": raw[:1000]}

    if not isinstance(obj, dict):
        return {"ok": False, "error": "response_not_json_object", "raw": raw[:1000]}

    if "skip" in obj:
        return {"ok": False, "error": f"model_returned_skip:{obj.get('skip')}", "raw": raw[:1000]}

    if "question" not in obj or "answer" not in obj:
        return {"ok": False, "error": "missing_question_or_answer", "raw": raw[:1000]}

    if not isinstance(obj["question"], str) or not isinstance(obj["answer"], str):
        return {"ok": False, "error": "question_or_answer_not_string", "raw": raw[:1000]}

    return {"ok": True, "data": obj}


# -------------------------------------------------------------------
# Result shaping
# -------------------------------------------------------------------

def accept_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "accepted",
        "record": record
    }


def reject_record(seed_row: Dict[str, Any], reason: str, details: Dict[str, Any]) -> Dict[str, Any]:
    rec = {
        "seed_id": seed_row.get("seed_id"),
        "model_name": seed_row.get("model_name"),
        "class_id": seed_row.get("class_id"),
        "level": seed_row.get("level"),
        "tier": seed_row.get("tier"),
        "route_mode": seed_row.get("route_mode"),
        "extension_family": seed_row.get("extension_family"),
        "reason": reason,
        "details": details or {}
    }
    return {
        "status": "rejected",
        "record": rec
    }


# -------------------------------------------------------------------
# Seed processing
# -------------------------------------------------------------------

def process_seed(
    seed_row: Dict[str, Any],
    model_profiles: Dict[str, Any],
    llm_client,
    embedding_index,
    dedup_index: InMemoryDedupIndex,
    near_dup_threshold: float = 0.985
) -> Dict[str, Any]:
    route_mode = seed_row.get("route_mode")
    if route_mode == "skip":
        return reject_record(seed_row, "SEED_ROUTED_SKIP", seed_row.get("route_diagnostics", {}))

    model_name = normalize_model_name(seed_row.get("model_name"))
    profile = get_model_profile(model_profiles, model_name)
    if not profile:
        return reject_record(seed_row, "MISSING_MODEL_PROFILE", {"model_name": model_name})

    try:
        prompt = build_prompt(seed_row, profile)
    except Exception as e:
        return reject_record(seed_row, "PROMPT_BUILD_ERROR", {"error": str(e)})

    # LLM call
    try:
        raw = llm_client.generate(prompt, seed_row=seed_row)
    except Exception as e:
        return reject_record(seed_row, "LLM_CALL_ERROR", {"error": str(e)})

    parsed = parse_llm_json(raw)
    if not parsed["ok"]:
        return reject_record(seed_row, "BAD_JSON", parsed)

    qa = parsed["data"]

    # Gate 1
    gate1 = run_gate1_embedding(
        question=qa["question"],
        class_id=seed_row.get("class_id"),
        embedding_index=embedding_index
    )
    if not gate1["ok"]:
        return reject_record(seed_row, gate1["reason"], gate1["details"])

    # Gate 2
    gate2 = validate_generated_row(
        generated_row=qa,
        routed_seed_row=seed_row,
        model_profiles=model_profiles
    )
    if not gate2["ok"]:
        return reject_record(seed_row, gate2["reason"], gate2["details"])

    # Gate 3
    gate3 = run_gate3_dedup(
        question=qa["question"],
        answer=qa["answer"],
        dedup_index=dedup_index,
        near_dup_threshold=near_dup_threshold
    )
    if not gate3["ok"]:
        return reject_record(seed_row, gate3["reason"], gate3["details"])

    accepted = {
        "seed_id": seed_row.get("seed_id"),
        "model_name": model_name,
        "class_id": seed_row.get("class_id"),
        "level": seed_row.get("level"),
        "tier": seed_row.get("tier"),
        "route_mode": seed_row.get("route_mode"),
        "extension_family": seed_row.get("extension_family"),
        "question": qa["question"],
        "answer": qa["answer"],
        "prompt_meta": {
            "route_mode": seed_row.get("route_mode"),
            "extension_family": seed_row.get("extension_family")
        },
        "gate1": gate1,
        "gate2": gate2,
        "gate3": gate3,
    }

    dedup_index.add(accepted)
    return accept_record(accepted)


# -------------------------------------------------------------------
# Batch generation
# -------------------------------------------------------------------

def generate_synthetic(
    routed_seeds: List[Dict[str, Any]],
    model_profiles: Dict[str, Any],
    llm_client,
    embedding_index,
    dedup_index: InMemoryDedupIndex,
    accepted_path: str,
    rejected_path: str,
    telemetry_path: Optional[str] = None,
    near_dup_threshold: float = 0.985,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    events = []

    rows = routed_seeds[:limit] if limit is not None else routed_seeds

    for seed_row in rows:
        result = process_seed(
            seed_row=seed_row,
            model_profiles=model_profiles,
            llm_client=llm_client,
            embedding_index=embedding_index,
            dedup_index=dedup_index,
            near_dup_threshold=near_dup_threshold
        )

        if result["status"] == "accepted":
            append_jsonl(accepted_path, result["record"])
            event = make_event(
                seed_row,
                status="accepted",
                reason="ACCEPTED",
                details={
                    "gate1": result["record"]["gate1"],
                    "gate2": result["record"]["gate2"],
                    "gate3": result["record"]["gate3"],
                }
            )
        else:
            append_jsonl(rejected_path, result["record"])
            event = make_event(
                seed_row,
                status="rejected",
                reason=result["record"]["reason"],
                details=result["record"].get("details", {})
            )

        events.append(event)
        if telemetry_path:
            record_attempt(telemetry_path, event)

    return summarize_telemetry(events)


# -------------------------------------------------------------------
# CLI helpers
# -------------------------------------------------------------------

def build_llm_client(mode: str):
    """
    Built-in smoke-test modes only.
    Production should inject a real LLM adapter programmatically.
    """
    if mode == "echo_seed":
        return EchoSeedLLMClient()
    elif mode == "static_demo":
        return StaticJSONLLMClient(
            question="How does the model work?",
            answer="This is a static smoke-test answer."
        )
    else:
        raise ValueError(
            f"Unknown --llm-mode={mode!r}. "
            "Built-in options: echo_seed, static_demo. "
            "For production, use the library API and pass a real client object."
        )


def build_embedding_index(mode: str):
    """
    Built-in smoke-test modes only.
    Production should inject a real embedding adapter programmatically.
    """
    if mode == "always_pass":
        return AlwaysPassEmbeddingIndex()
    else:
        raise ValueError(
            f"Unknown --embedding-mode={mode!r}. "
            "Built-in option: always_pass. "
            "For production, use the library API and pass a real embedding adapter."
        )


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed-seeds", required=True, help="Path to seed_routes.jsonl")
    parser.add_argument("--profiles", required=True, help="Path to model_profiles_merged.json")
    parser.add_argument("--accepted", required=True, help="Output accepted.jsonl")
    parser.add_argument("--rejected", required=True, help="Output rejected.jsonl")
    parser.add_argument("--telemetry", default=None, help="Optional telemetry.jsonl path")
    parser.add_argument("--summary-out", required=True, help="Output JSON summary path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of seeds to process")
    parser.add_argument("--near-dup-threshold", type=float, default=0.985)

    # smoke-test adapters
    parser.add_argument("--llm-mode", default="echo_seed", help="echo_seed | static_demo")
    parser.add_argument("--embedding-mode", default="always_pass", help="always_pass")
    args = parser.parse_args()

    routed_seeds = load_jsonl(args.routed_seeds)
    model_profiles = load_json(args.profiles)

    llm_client = build_llm_client(args.llm_mode)
    embedding_index = build_embedding_index(args.embedding_mode)
    dedup_index = InMemoryDedupIndex()

    summary = generate_synthetic(
        routed_seeds=routed_seeds,
        model_profiles=model_profiles,
        llm_client=llm_client,
        embedding_index=embedding_index,
        dedup_index=dedup_index,
        accepted_path=args.accepted,
        rejected_path=args.rejected,
        telemetry_path=args.telemetry,
        near_dup_threshold=args.near_dup_threshold,
        limit=args.limit
    )

    save_json(summary, args.summary_out)


if __name__ == "__main__":
    main()
