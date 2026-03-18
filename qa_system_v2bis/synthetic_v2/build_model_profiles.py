import argparse
from pathlib import Path
from typing import Dict, List, Any

from .io_utils import load_json, save_json
from .text_utils import normalize_model_name
from .nlogo_parser import (
    split_nlogo_sections,
    extract_core_from_code,
    infer_unresolved_identifiers,
    first_info_paragraph,
    fallback_summary,
)
from .interface_parser import (
    extract_widgets_from_interface,
    infer_candidate_widgets_from_code,
    merge_widget_sets,
)


def parse_nlogo_file(path: str, model_name: str = None) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    sections = split_nlogo_sections(text)

    code = sections["code"]
    interface = sections["interface"]
    info = sections["info"]

    core_extract = extract_core_from_code(code)

    unresolved = infer_unresolved_identifiers(code, core_extract)

    interface_parse = extract_widgets_from_interface(interface)
    inferred_widgets = infer_candidate_widgets_from_code(
        code=code,
        extracted_core=core_extract,
        unresolved_identifiers=unresolved["unresolved_identifiers"],
        unresolved_counts=unresolved["unresolved_identifier_counts"],
    )

    widgets = merge_widget_sets(
        interface_parse["widgets"],
        inferred_widgets["widget_candidates"]
    )

    if model_name is None:
        model_name = normalize_model_name(Path(path).stem)
    else:
        model_name = normalize_model_name(model_name)

    summary = first_info_paragraph(info)
    if not summary:
        summary = fallback_summary(
            model_name=model_name,
            procedures=core_extract["procedures"],
            variables=core_extract["variables"],
            breeds=core_extract["breeds"],
            widgets=widgets,
        )

    profile = {
        "model_name": model_name,
        "core": {
            "procedures": core_extract["procedures"],
            "variables": core_extract["variables"],
            "breeds": core_extract["breeds"],
            "widgets": widgets,
            "model_summary": summary,
        },
        "raw_parse": {
            **core_extract["raw_parse"],
            "interface_widgets": interface_parse["widgets"],
            "widgets_by_type": interface_parse["widgets_by_type"],
            "widget_candidates_from_code": inferred_widgets["widget_candidates"],
            "widget_candidate_reasons": inferred_widgets["candidate_reasons"],
            "unresolved_identifiers": unresolved["unresolved_identifiers"],
            "unresolved_identifier_counts": unresolved["unresolved_identifier_counts"],
            "info_excerpt": info[:1000],
            "source_path": str(path),
        }
    }
    return profile


def scan_model_files(model_dir: str) -> List[Path]:
    return sorted(Path(model_dir).rglob("*.nlogo"))


def build_core_profiles(model_dir: str) -> Dict[str, Any]:
    model_files = scan_model_files(model_dir)

    out = {
        "meta": {
            "source": str(model_dir),
            "model_count": 0,
        },
        "models": {}
    }

    for path in model_files:
        profile = parse_nlogo_file(str(path))
        model_name = profile["model_name"]
        out["models"][model_name] = profile

    out["meta"]["model_count"] = len(out["models"])
    return out


def compare_list_fields(parsed: List[str], existing: List[str]) -> Dict[str, List[str]]:
    p = {x.lower(): x for x in parsed or []}
    e = {x.lower(): x for x in existing or []}

    only_parsed = [p[k] for k in sorted(set(p) - set(e))]
    only_existing = [e[k] for k in sorted(set(e) - set(p))]
    common = [p[k] for k in sorted(set(p) & set(e))]

    return {
        "only_parsed": only_parsed,
        "only_existing": only_existing,
        "common": common
    }


def compare_profiles_to_existing(core_profiles: Dict[str, Any], existing_invariants: Dict[str, Any]) -> Dict[str, Any]:
    report = {
        "models": {},
        "summary": {
            "missing_in_existing": 0,
            "missing_in_parsed": 0,
            "shared_models": 0,
            "parsed_only_models": 0,
            "existing_only_models": 0,
        }
    }

    parsed_models = set(core_profiles["models"].keys())
    existing_models = {normalize_model_name(k) for k in existing_invariants.keys()}

    report["summary"]["shared_models"] = len(parsed_models & existing_models)
    report["summary"]["parsed_only_models"] = len(parsed_models - existing_models)
    report["summary"]["existing_only_models"] = len(existing_models - parsed_models)

    for model_name, parsed_prof in core_profiles["models"].items():
        existing = None

        for k, v in existing_invariants.items():
            if normalize_model_name(k) == model_name:
                existing = v
                break

        if existing is None:
            report["models"][model_name] = {
                "status": "parsed_only",
                "parsed_core": parsed_prof["core"]
            }
            continue

        parsed_core = parsed_prof["core"]

        cmp = {
            "status": "shared",
            "procedures": compare_list_fields(parsed_core.get("procedures", []), existing.get("procedures", [])),
            "variables": compare_list_fields(parsed_core.get("variables", []), existing.get("variables", [])),
            "breeds": compare_list_fields(parsed_core.get("breeds", []), existing.get("breeds", [])),
            "widgets": compare_list_fields(parsed_core.get("widgets", []), existing.get("widgets", [])),
        }

        missing_in_existing = (
            len(cmp["procedures"]["only_parsed"]) +
            len(cmp["variables"]["only_parsed"]) +
            len(cmp["breeds"]["only_parsed"]) +
            len(cmp["widgets"]["only_parsed"])
        )
        missing_in_parsed = (
            len(cmp["procedures"]["only_existing"]) +
            len(cmp["variables"]["only_existing"]) +
            len(cmp["breeds"]["only_existing"]) +
            len(cmp["widgets"]["only_existing"])
        )

        report["summary"]["missing_in_existing"] += missing_in_existing
        report["summary"]["missing_in_parsed"] += missing_in_parsed

        report["models"][model_name] = cmp

    for k, v in existing_invariants.items():
        nk = normalize_model_name(k)
        if nk not in parsed_models:
            report["models"][nk] = {
                "status": "existing_only",
                "existing_invariant": v
            }

    return report


def summarize_widget_candidates(core_profiles: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for model_name, prof in core_profiles["models"].items():
        rp = prof.get("raw_parse", {})
        out[model_name] = {
            "interface_widgets": rp.get("interface_widgets", []),
            "widget_candidates_from_code": rp.get("widget_candidates_from_code", []),
            "unresolved_identifiers_top20": rp.get("unresolved_identifiers", [])[:20],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory containing .nlogo files")
    parser.add_argument("--out", required=True, help="Output core_profiles.json")
    parser.add_argument("--existing-invariants", default=None, help="Optional path to existing model_invariants.json")
    parser.add_argument("--diff-out", default=None, help="Optional output path for diff report against existing invariants")
    parser.add_argument("--widget-review-out", default=None, help="Optional output path for widget candidate review report")
    args = parser.parse_args()

    core_profiles = build_core_profiles(args.model_dir)
    save_json(core_profiles, args.out)

    if args.widget_review_out:
        widget_report = summarize_widget_candidates(core_profiles)
        save_json(widget_report, args.widget_review_out)

    if args.existing_invariants and args.diff_out:
        existing = load_json(args.existing_invariants)
        diff_report = compare_profiles_to_existing(core_profiles, existing)
        save_json(diff_report, args.diff_out)


if __name__ == "__main__":
    main()
