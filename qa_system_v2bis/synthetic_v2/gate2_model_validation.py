import argparse
import re
from typing import Dict, List, Set, Any, Optional

from .io_utils import load_json, save_json
from .text_utils import normalize_model_name, norm_lower, STOPWORDS, GLOBAL_ALLOW as DEFAULT_GLOBAL_ALLOW, BACKTICK_RE
from .profile_utils import (
    build_core_identifier_set,
    all_extension_identifiers,
    all_extension_concepts,
    get_family,
    family_identifier_set,
    family_concept_set,
    get_framing_cues,
    get_disallowed_themes,
    summary_keywords,
    extract_backticked_identifiers,
    find_present_phrases,
)
from .extension_profile_builder import DEFAULT_FRAMING_CUES


WORD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_\-]*\??\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    return [x.strip() for x in SENTENCE_SPLIT_RE.split(text or "") if x.strip()]


def extract_words(text: str) -> List[str]:
    return [norm_lower(x) for x in WORD_RE.findall(text or "")]


def has_extension_framing(answer: str, profile: Dict[str, Any]) -> bool:
    a = norm_lower(answer)
    framing_cues = get_framing_cues(profile)

    if any(cue in a for cue in framing_cues):
        return True

    return any(norm_lower(x) in a for x in DEFAULT_FRAMING_CUES)


def implies_base_model_contains_extensions(answer: str, used_extension_ids: Set[str]) -> List[str]:
    """
    Flags extension identifiers that are described as already existing in the original/base model.
    """
    bad_hits = []
    sentences = split_sentences(answer)

    bad_cues = [
        "the model already",
        "the original model already",
        "the base model already",
        "already uses",
        "already tracks",
        "already contains",
        "already includes",
        "already has",
        "the existing code",
        "the original code",
        "built-in",
        "currently defined",
        "is defined in the model",
        "is tracked in the model",
        "is in the original model",
        "the model uses",
        "the model tracks",
        "the model contains",
        "the model includes",
        "the model has"
    ]

    safe_cues = [
        "you could add",
        "you could introduce",
        "extend the model",
        "extend the original model",
        "in an extended version",
        "as an extension",
        "if you modify the model",
        "not part of the original code",
        "beyond the original code",
        "replace the original",
        "refine the original"
    ]

    for sent in sentences:
        s = norm_lower(sent)

        if any(safe in s for safe in safe_cues):
            continue
        has_bad_cue = any(bad in s for bad in bad_cues)
        if not has_bad_cue:
            continue
        for ext_id in used_extension_ids:
            if ext_id in s:
                bad_hits.append(ext_id)

    return sorted(set(bad_hits))


def has_core_anchor(text: str, core_ids: Set[str], model_summary: str) -> bool:
    t = norm_lower(text)

    # direct core-id anchor
    for cid in core_ids:
        if cid in t:
            return True

    # explicit anchor phrases
    anchor_phrases = [
        "original model",
        "base model",
        "original code",
        "original rebellion model",
        "original shepherds model",
        "extend the original model"
    ]
    if any(p in t for p in anchor_phrases):
        return True

    # summary keyword overlap
    kws = summary_keywords(model_summary or "")
    hits = [kw for kw in kws if kw in t]
    return len(hits) >= 2


def detect_disallowed_theme_hits(text: str, profile: Dict[str, Any]) -> List[str]:
    disallowed = get_disallowed_themes(profile)
    return find_present_phrases(text, disallowed)


def detect_extension_concepts_in_core_mode(text: str, profile: Dict[str, Any]) -> List[str]:
    all_concepts = all_extension_concepts(profile)
    return find_present_phrases(text, all_concepts)


def detect_cross_family_extension_ids(
    used_extension_ids: Set[str],
    profile: Dict[str, Any],
    family_name: Optional[str]
) -> List[str]:
    if not family_name:
        return []
    allowed = family_identifier_set(profile, family_name)
    return sorted([x for x in used_extension_ids if x not in allowed])


def _validate_core_mode(
    text: str,
    mode: str,
    core_ids: Set[str],
    used_core_ids: Set[str],
    used_ext_ids: Set[str],
    unknown_ids: Set[str],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    if unknown_ids:
        return {
            "ok": False,
            "reason": "UNKNOWN_CORE_IDENTIFIER",
            "details": {
                "unknown_identifiers": sorted(unknown_ids),
                "used_core_ids": sorted(used_core_ids),
                "used_extension_ids": sorted(used_ext_ids),
            }
        }
    if used_ext_ids:
        return {
            "ok": False,
            "reason": "UNAPPROVED_EXTENSION_IDENTIFIER",
            "details": {"extension_identifiers_in_core_mode": sorted(used_ext_ids)}
        }
    ext_concept_hits = detect_extension_concepts_in_core_mode(text, profile)
    if ext_concept_hits:
        return {
            "ok": False,
            "reason": "EXTENSION_CONTENT_IN_CORE_MODE",
            "details": {"extension_concept_hits": ext_concept_hits[:30]}
        }
    disallowed_hits = detect_disallowed_theme_hits(text, profile)
    if disallowed_hits:
        return {
            "ok": False,
            "reason": "DISALLOWED_THEME",
            "details": {"disallowed_hits": disallowed_hits}
        }
    return {
        "ok": True,
        "reason": None,
        "details": {
            "mode": mode,
            "used_core_ids": sorted(used_core_ids),
            "used_extension_ids": [],
        }
    }


def _validate_extension_mode(
    text: str,
    text_n: str,
    answer: str,
    mode: str,
    family_name: Optional[str],
    core_ids: Set[str],
    used_core_ids: Set[str],
    used_ext_ids: Set[str],
    unknown_ids: Set[str],
    profile: Dict[str, Any],
    summary: str,
) -> Dict[str, Any]:
    if unknown_ids:
        return {
            "ok": False,
            "reason": "UNAPPROVED_EXTENSION_IDENTIFIER",
            "details": {
                "unknown_identifiers": sorted(unknown_ids),
                "used_core_ids": sorted(used_core_ids),
                "used_extension_ids": sorted(used_ext_ids),
            }
        }
    cross_family_ids = detect_cross_family_extension_ids(used_ext_ids, profile, family_name)
    if cross_family_ids:
        return {
            "ok": False,
            "reason": "CROSS_FAMILY_EXTENSION_IDENTIFIER",
            "details": {"family_name": family_name, "cross_family_extension_ids": cross_family_ids}
        }
    if used_ext_ids and not has_extension_framing(answer, profile):
        return {
            "ok": False,
            "reason": "EXTENSION_REQUIRES_FRAMING",
            "details": {"used_extension_ids": sorted(used_ext_ids)}
        }
    misrep = implies_base_model_contains_extensions(answer, used_ext_ids)
    if misrep:
        return {
            "ok": False,
            "reason": "BASE_MODEL_MISREPRESENTATION",
            "details": {"misrepresented_extension_ids": misrep}
        }
    if not has_core_anchor(text_n, core_ids, summary):
        return {
            "ok": False,
            "reason": "INSUFFICIENT_CORE_ANCHOR",
            "details": {
                "used_core_ids": sorted(used_core_ids),
                "family_name": family_name,
                "model_summary": summary
            }
        }
    disallowed_hits = detect_disallowed_theme_hits(text, profile)
    if disallowed_hits:
        return {
            "ok": False,
            "reason": "DISALLOWED_THEME",
            "details": {"disallowed_hits": disallowed_hits}
        }
    return {
        "ok": True,
        "reason": None,
        "details": {
            "mode": mode,
            "family_name": family_name,
            "used_core_ids": sorted(used_core_ids),
            "used_extension_ids": sorted(used_ext_ids),
        }
    }


def validate_gate2(
    question: str,
    answer: str,
    mode: str,
    profile: Dict[str, Any],
    family_name: Optional[str] = None,
    global_allow: Optional[Set[str]] = None
) -> Dict[str, Any]:
    if global_allow is None:
        global_allow = DEFAULT_GLOBAL_ALLOW

    text = f"{question or ''}\n{answer or ''}"
    text_n = norm_lower(text)

    core = profile.get("core", {})
    core_ids = build_core_identifier_set(profile)
    ext_ids_all = all_extension_identifiers(profile)
    summary = core.get("model_summary", "")

    backticked = set(extract_backticked_identifiers(text))
    used_core_ids = {x for x in backticked if x in core_ids}
    used_ext_ids = {x for x in backticked if x in ext_ids_all}
    unknown_ids = {
        x for x in backticked
        if x not in core_ids and x not in ext_ids_all and x not in global_allow
    }

    if mode in {"core_paraphrase", "core_repair"}:
        return _validate_core_mode(text, mode, core_ids, used_core_ids, used_ext_ids, unknown_ids, profile)
    elif mode == "anchored_extension":
        return _validate_extension_mode(
            text, text_n, answer, mode, family_name,
            core_ids, used_core_ids, used_ext_ids, unknown_ids, profile, summary
        )
    else:
        return {"ok": False, "reason": "UNKNOWN_MODE", "details": {"mode": mode}}


def validate_generated_row(
    generated_row: Dict[str, Any],
    routed_seed_row: Dict[str, Any],
    model_profiles: Dict[str, Any],
    global_allow: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Convenience wrapper when you have:
    - generated row with question/answer
    - routed seed row with route_mode / extension_family / model_name
    - merged model profiles
    """
    model_name = normalize_model_name(routed_seed_row.get("model_name"))
    profile = model_profiles.get("models", {}).get(model_name)

    if not profile:
        return {
            "ok": False,
            "reason": "MISSING_MODEL_PROFILE",
            "details": {"model_name": model_name}
        }

    return validate_gate2(
        question=generated_row.get("question", ""),
        answer=generated_row.get("answer", ""),
        mode=routed_seed_row.get("route_mode"),
        profile=profile,
        family_name=routed_seed_row.get("extension_family"),
        global_allow=global_allow,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True, help="Path to model_profiles_merged.json")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--mode", required=True, help="core_paraphrase | core_repair | anchored_extension")
    parser.add_argument("--family-name", default=None, help="Extension family, if mode=anchored_extension")
    parser.add_argument("--question", required=True, help="Generated question text")
    parser.add_argument("--answer", required=True, help="Generated answer text")
    parser.add_argument("--out", required=True, help="Output JSON report")
    args = parser.parse_args()

    profiles = load_json(args.profiles)
    model_name = normalize_model_name(args.model_name)
    profile = profiles.get("models", {}).get(model_name)

    if not profile:
        raise ValueError(f"No model profile found for model_name={model_name!r}")

    result = validate_gate2(
        question=args.question,
        answer=args.answer,
        mode=args.mode,
        profile=profile,
        family_name=args.family_name
    )
    save_json(result, args.out)


if __name__ == "__main__":
    main()
