import argparse
from copy import deepcopy
from typing import Dict, List, Any, Tuple

from .io_utils import load_json, save_json
from .text_utils import normalize_model_name, norm_lower


DEFAULT_GENERAL_RULES = [
    "Extension content must remain anchored to the original model.",
    "At least one core identifier or clearly core concept should appear in every extension answer.",
    "Do not present extension identifiers as original source-code elements unless they are in the core invariant.",
    "Prefer minimal coherent extensions over broad redesigns."
]

DEFAULT_FRAMING_CUES = [
    "to extend the original model",
    "extend the model",
    "you could add",
    "you could introduce",
    "if you modify the model",
    "in an extended version",
    "beyond the original code",
    "as an extension",
    "replace the original",
    "refine the original"
]

DEFAULT_DISALLOWED_UNANCHORED_THEMES = [
    "quantum mechanics",
    "genetics",
    "financial derivatives",
    "cosmology"
]


def load_core_profiles(path: str) -> Dict[str, Any]:
    data = load_json(path)
    if "models" not in data:
        raise ValueError("Expected core_profiles.json with top-level 'models'")
    return data


def load_audit_reports(path: str) -> Dict[str, Any]:
    data = load_json(path)
    # expected to be artifacts/audit/per_model/_all_models.json
    return data


def load_manual_extension_specs(path: str) -> Dict[str, Any]:
    data = load_json(path)

    if "models" not in data:
        raise ValueError("manual_extension_specs.json must have top-level 'models'")

    if "_defaults" not in data:
        data["_defaults"] = {}

    return data


def dedup_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs or []:
        s = str(x).strip()
        if not s:
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def list_to_count_map(items: List[Dict[str, Any]], key: str = "term") -> Dict[str, int]:
    out = {}
    for item in items or []:
        k = item.get(key)
        if not k:
            continue
        out[norm_lower(k)] = int(item.get("count", 0))
    return out


def phrase_list_to_count_map(items: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {}
    for item in items or []:
        phrase = item.get("phrase")
        if not phrase:
            continue
        out[norm_lower(phrase)] = int(item.get("count", 0))
    return out


def merge_string_lists(defaults: List[str], model_specific: List[str]) -> List[str]:
    return dedup_preserve_order((defaults or []) + (model_specific or []))


def audit_support_for_family(family_spec: Dict[str, Any], audit_report: Dict[str, Any]) -> Dict[str, Any]:
    unsupported_counts = list_to_count_map(audit_report.get("top_unsupported_terms", []), key="term")
    phrase_counts = phrase_list_to_count_map(audit_report.get("top_phrase_candidates", []))
    parser_miss_counts = list_to_count_map(audit_report.get("parser_miss_candidates", []), key="term")
    extension_candidate_counts = list_to_count_map(audit_report.get("extension_candidate_terms", []), key="term")

    identifiers = family_spec.get("identifiers", []) or []
    concepts = family_spec.get("concepts", []) or []

    identifier_hits = []
    concept_hits = []

    for x in identifiers:
        nx = norm_lower(x)
        identifier_hits.append({
            "identifier": x,
            "unsupported_count": unsupported_counts.get(nx, 0),
            "extension_candidate_count": extension_candidate_counts.get(nx, 0),
            "parser_miss_count": parser_miss_counts.get(nx, 0),
        })

    for x in concepts:
        nx = norm_lower(x)
        concept_hits.append({
            "concept": x,
            "phrase_count": phrase_counts.get(nx, 0),
        })

    family_support_score = sum(
        row["unsupported_count"] + row["extension_candidate_count"]
        for row in identifier_hits
    ) + sum(
        row["phrase_count"] for row in concept_hits
    )

    return {
        "support_score": family_support_score,
        "identifier_hits": identifier_hits,
        "concept_hits": concept_hits,
    }


def normalize_family_spec(fam: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": fam["name"],
        "concepts": dedup_preserve_order(fam.get("concepts", [])),
        "identifiers": dedup_preserve_order(fam.get("identifiers", [])),
        "rules": dedup_preserve_order(fam.get("rules", [])),
    }


def normalize_model_extension_spec(
    model_name: str,
    model_spec: Dict[str, Any],
    defaults: Dict[str, Any]
) -> Dict[str, Any]:
    default_general_rules = defaults.get("general_rules", DEFAULT_GENERAL_RULES)
    default_framing_cues = defaults.get("framing_cues", DEFAULT_FRAMING_CUES)
    default_disallowed = defaults.get("disallowed_unanchored_themes", DEFAULT_DISALLOWED_UNANCHORED_THEMES)

    families = [normalize_family_spec(f) for f in model_spec.get("families", [])]

    out = {
        "model_name": model_name,
        "general_rules": merge_string_lists(default_general_rules, model_spec.get("general_rules", [])),
        "framing_cues": merge_string_lists(default_framing_cues, model_spec.get("framing_cues", [])),
        "disallowed_unanchored_themes": merge_string_lists(
            default_disallowed,
            model_spec.get("disallowed_unanchored_themes", [])
        ),
        "families": families,
    }
    return out


def build_extension_profile_for_model(
    model_name: str,
    model_spec: Dict[str, Any],
    defaults: Dict[str, Any],
    audit_report: Dict[str, Any]
) -> Dict[str, Any]:
    normalized = normalize_model_extension_spec(model_name, model_spec, defaults)

    families_out = []
    for fam in normalized["families"]:
        fam_with_evidence = deepcopy(fam)
        fam_with_evidence["audit_support"] = audit_support_for_family(fam, audit_report or {})
        families_out.append(fam_with_evidence)

    return {
        "model_name": model_name,
        "extensions": {
            "families": families_out,
            "general_rules": normalized["general_rules"],
            "framing_cues": normalized["framing_cues"],
            "disallowed_unanchored_themes": normalized["disallowed_unanchored_themes"],
        }
    }


def build_extension_profiles(
    audit_reports: Dict[str, Any],
    manual_specs: Dict[str, Any]
) -> Dict[str, Any]:
    defaults = manual_specs.get("_defaults", {})
    model_specs = manual_specs.get("models", {})

    out = {
        "meta": {
            "model_count": 0,
            "source": "manual_extension_specs + audit_evidence"
        },
        "models": {}
    }

    for raw_model_name, model_spec in model_specs.items():
        model_name = normalize_model_name(raw_model_name)
        if not model_name:
            continue
        audit_report = audit_reports.get(model_name, {})
        profile = build_extension_profile_for_model(
            model_name=model_name,
            model_spec=model_spec,
            defaults=defaults,
            audit_report=audit_report
        )
        out["models"][model_name] = profile

    out["meta"]["model_count"] = len(out["models"])
    return out


def merge_core_and_extension_profiles(
    core_profiles: Dict[str, Any],
    extension_profiles: Dict[str, Any]
) -> Dict[str, Any]:
    out = {
        "meta": {
            "core_model_count": len(core_profiles.get("models", {})),
            "extension_model_count": len(extension_profiles.get("models", {})),
        },
        "models": {}
    }

    core_models = core_profiles.get("models", {})
    ext_models = extension_profiles.get("models", {})

    all_model_names = sorted(set(core_models.keys()) | set(ext_models.keys()))

    for model_name in all_model_names:
        merged = {
            "model_name": model_name
        }

        if model_name in core_models:
            merged["core"] = core_models[model_name].get("core", {})
            merged["raw_parse"] = core_models[model_name].get("raw_parse", {})

        if model_name in ext_models:
            merged["extensions"] = ext_models[model_name].get("extensions", {})
        else:
            merged["extensions"] = {
                "families": [],
                "general_rules": DEFAULT_GENERAL_RULES,
                "framing_cues": DEFAULT_FRAMING_CUES,
                "disallowed_unanchored_themes": DEFAULT_DISALLOWED_UNANCHORED_THEMES,
            }

        out["models"][model_name] = merged

    return out


def summarize_extension_profiles(extension_profiles: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "model_count": len(extension_profiles.get("models", {})),
        "family_count_total": 0,
        "families_by_model": {},
        "top_families_by_support": [],
    }

    family_support_rows = []

    for model_name, prof in extension_profiles.get("models", {}).items():
        families = prof.get("extensions", {}).get("families", [])
        summary["families_by_model"][model_name] = [f["name"] for f in families]
        summary["family_count_total"] += len(families)

        for fam in families:
            family_support_rows.append({
                "model_name": model_name,
                "family_name": fam["name"],
                "support_score": fam.get("audit_support", {}).get("support_score", 0),
                "identifier_hits": fam.get("audit_support", {}).get("identifier_hits", []),
                "concept_hits": fam.get("audit_support", {}).get("concept_hits", []),
            })

    family_support_rows.sort(key=lambda x: (-x["support_score"], x["model_name"], x["family_name"]))
    summary["top_families_by_support"] = family_support_rows[:100]

    return summary


def validate_manual_specs_against_core(
    core_profiles: Dict[str, Any],
    manual_specs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Sanity-check manual extension specs:
    - extension identifiers that accidentally duplicate core identifiers
    - model names in specs missing from core profiles
    """
    report = {
        "models_missing_from_core": [],
        "families_with_identifier_overlap": []
    }

    core_models = core_profiles.get("models", {})
    model_specs = manual_specs.get("models", {})

    for raw_model_name, model_spec in model_specs.items():
        model_name = normalize_model_name(raw_model_name)

        if model_name not in core_models:
            report["models_missing_from_core"].append(model_name)
            continue

        core_terms = set()
        core = core_models[model_name].get("core", {})
        for key in ["procedures", "variables", "breeds", "widgets"]:
            for x in core.get(key, []):
                core_terms.add(norm_lower(x))

        for fam in model_spec.get("families", []):
            fam_name = fam["name"]
            overlaps = []
            for ident in fam.get("identifiers", []):
                if norm_lower(ident) in core_terms:
                    overlaps.append(ident)

            if overlaps:
                report["families_with_identifier_overlap"].append({
                    "model_name": model_name,
                    "family_name": fam_name,
                    "overlapping_identifiers": overlaps
                })

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True, help="Path to core_profiles.json")
    parser.add_argument("--audit", required=True, help="Path to audit _all_models.json")
    parser.add_argument("--manual", required=True, help="Path to manual_extension_specs.json")
    parser.add_argument("--out-ext", required=True, help="Output path for extension_profiles.json")
    parser.add_argument("--out-merged", required=True, help="Output path for model_profiles_merged.json")
    parser.add_argument("--summary-out", default=None, help="Optional summary output path")
    parser.add_argument("--validation-out", default=None, help="Optional manual spec validation report output path")
    args = parser.parse_args()

    core_profiles = load_core_profiles(args.core)
    audit_reports = load_audit_reports(args.audit)
    manual_specs = load_manual_extension_specs(args.manual)

    extension_profiles = build_extension_profiles(
        audit_reports=audit_reports,
        manual_specs=manual_specs
    )
    merged = merge_core_and_extension_profiles(
        core_profiles=core_profiles,
        extension_profiles=extension_profiles
    )

    save_json(extension_profiles, args.out_ext)
    save_json(merged, args.out_merged)

    if args.summary_out:
        summary = summarize_extension_profiles(extension_profiles)
        save_json(summary, args.summary_out)

    if args.validation_out:
        validation = validate_manual_specs_against_core(core_profiles, manual_specs)
        save_json(validation, args.validation_out)


if __name__ == "__main__":
    main()
