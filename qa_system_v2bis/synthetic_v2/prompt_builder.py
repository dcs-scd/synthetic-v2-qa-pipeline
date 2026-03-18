import argparse
from typing import Dict, Any, Optional

from .io_utils import load_json, load_jsonl, save_json
from .text_utils import normalize_model_name
from .prompt_templates import (
    core_paraphrase_template,
    core_repair_template,
    anchored_extension_template,
)


def get_model_profile(model_profiles: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    model_name = normalize_model_name(model_name)
    return model_profiles.get("models", {}).get(model_name, {})


def build_core_prompt(seed: Dict[str, Any], profile: Dict[str, Any]) -> str:
    return core_paraphrase_template(seed, profile)


def build_core_repair_prompt(seed: Dict[str, Any], profile: Dict[str, Any]) -> str:
    return core_repair_template(seed, profile)


def build_extension_prompt(seed: Dict[str, Any], profile: Dict[str, Any], family_name: str) -> str:
    if not family_name:
        raise ValueError("anchored_extension prompt requires family_name")
    return anchored_extension_template(seed, profile, family_name)


def build_prompt(seed: Dict[str, Any], profile: Dict[str, Any], route_result: Optional[Dict[str, Any]] = None) -> str:
    """
    Dispatcher.

    Accepts either:
    - a routed seed row (with route_mode / extension_family), or
    - a plain seed row plus route_result
    """
    if route_result is None:
        route_mode = seed.get("route_mode")
        family_name = seed.get("extension_family")
    else:
        route_mode = route_result.get("route")
        family_name = route_result.get("family")

    if route_mode == "core_paraphrase":
        return build_core_prompt(seed, profile)
    elif route_mode == "core_repair":
        return build_core_repair_prompt(seed, profile)
    elif route_mode == "anchored_extension":
        return build_extension_prompt(seed, profile, family_name)
    else:
        raise ValueError(f"Cannot build prompt for route_mode={route_mode!r}")


def preview_prompt_for_seed(routed_seed: Dict[str, Any], model_profiles: Dict[str, Any]) -> Dict[str, Any]:
    model_name = normalize_model_name(routed_seed.get("model_name"))
    profile = get_model_profile(model_profiles, model_name)

    if not profile:
        raise ValueError(f"No model profile found for model_name={model_name!r}")

    prompt = build_prompt(routed_seed, profile)
    return {
        "seed_id": routed_seed.get("seed_id"),
        "model_name": model_name,
        "route_mode": routed_seed.get("route_mode"),
        "extension_family": routed_seed.get("extension_family"),
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed-seeds", required=True, help="Path to seed_routes.jsonl")
    parser.add_argument("--profiles", required=True, help="Path to model_profiles_merged.json")
    parser.add_argument("--seed-id", required=True, help="Seed ID to preview")
    parser.add_argument("--out", required=True, help="Output JSON file for prompt preview")
    args = parser.parse_args()

    routed_seeds = load_jsonl(args.routed_seeds)
    profiles = load_json(args.profiles)

    target = None
    for row in routed_seeds:
        if row.get("seed_id") == args.seed_id:
            target = row
            break

    if target is None:
        raise ValueError(f"Seed ID not found: {args.seed_id}")

    result = preview_prompt_for_seed(target, profiles)
    save_json(result, args.out)


if __name__ == "__main__":
    main()
