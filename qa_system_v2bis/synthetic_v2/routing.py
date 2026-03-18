import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set, Tuple, Optional

from .io_utils import load_json, load_jsonl, save_json, save_jsonl
from .text_utils import normalize_model_name, norm_lower, STOPWORDS, GLOBAL_ALLOW, BACKTICK_RE, IDLIKE_RE
from .profile_utils import (
    build_core_identifier_set as core_identifier_set,
    all_extension_identifiers,
    all_extension_concepts,
    summary_keywords,
)

GENERIC_EXTENSION_INTENT_CUES = {
    "add", "adding", "extend", "extension", "modify", "modifying",
    "introduce", "introducing", "layer", "layers", "network", "networks",
    "broadcast", "media", "friends", "friend", "leader", "leaders",
    "state", "states", "diagram", "long-range", "small-world",
    "optimize", "optimization", "cache", "scaling", "hub", "hubs",
    "centrality", "warning", "indicator", "indicators", "metric", "metrics"
}

DEFAULT_ROUTING_CONFIG = {
    # score thresholds
    "family_strong_threshold": 4,
    "family_weak_threshold": 2,

    # core routing thresholds
    "core_paraphrase_min_core_hits": 2,
    "core_repair_min_core_hits": 1,
    "core_paraphrase_max_unknown_ratio": 0.35,
    "core_repair_max_unknown_ratio": 0.75,

    # weighting
    "weight_family_identifier_hit": 3,
    "weight_family_concept_phrase_hit": 2,
    "weight_family_concept_token_hit": 1,

    # fallback behavior
    "prefer_extension_when_intent_present": True,
}


def safe_text(x):
    return "" if x is None else str(x)


def extract_terms_and_counts(text: str):
    """Extract terms as both a set and a counter in one pass."""
    terms = set()
    counts = Counter()
    t = text or ""
    for x in BACKTICK_RE.findall(t):
        nx = norm_lower(x)
        terms.add(nx)
        counts[nx] += 1
    for x in IDLIKE_RE.findall(t):
        nx = norm_lower(x)
        if nx not in STOPWORDS:
            terms.add(nx)
            counts[nx] += 1
    return terms, counts


def extract_terms(text: str) -> set:
    terms, _ = extract_terms_and_counts(text)
    return terms


def family_lookup(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for fam in profile.get("extensions", {}).get("families", []):
        out[fam["name"]] = {
            "name": fam["name"],
            "concepts": {norm_lower(x) for x in fam.get("concepts", [])},
            "identifiers": {norm_lower(x) for x in fam.get("identifiers", [])},
            "rules": fam.get("rules", []),
            "audit_support": fam.get("audit_support", {}),
        }
    return out


def phrase_present(text: str, phrase: str) -> bool:
    return norm_lower(phrase) in norm_lower(text)


def count_phrase_hits(text: str, phrases: Set[str]) -> Tuple[int, List[str]]:
    hits = []
    t = norm_lower(text)
    for p in phrases:
        if p and p in t:
            hits.append(p)
    return len(hits), sorted(set(hits))


def generic_extension_intent(text: str) -> Tuple[bool, List[str]]:
    terms = extract_terms(text)
    hits = sorted([t for t in terms if t in GENERIC_EXTENSION_INTENT_CUES])
    return (len(hits) > 0), hits


def family_score_for_text(
    text: str,
    family_spec: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    terms, term_counts = extract_terms_and_counts(text)
    text_n = norm_lower(text)

    id_hits = sorted([t for t in terms if t in family_spec["identifiers"]])
    phrase_count, phrase_hits = count_phrase_hits(text_n, family_spec["concepts"])

    # token-level concept support:
    # if a concept phrase contains words present individually, count as weak support
    concept_token_hits = set()
    for concept in family_spec["concepts"]:
        for tok in concept.split():
            if tok in terms and tok not in STOPWORDS:
                concept_token_hits.add(tok)

    score = (
        len(id_hits) * config["weight_family_identifier_hit"]
        + phrase_count * config["weight_family_concept_phrase_hit"]
        + len(concept_token_hits) * config["weight_family_concept_token_hit"]
    )

    return {
        "score": score,
        "identifier_hits": id_hits,
        "identifier_hit_counts": {t: term_counts[t] for t in id_hits},
        "concept_phrase_hits": phrase_hits,
        "concept_token_hits": sorted(concept_token_hits),
    }


def choose_best_family(text: str, profile: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    fams = family_lookup(profile)
    family_scores = {}
    family_matches = {}

    best_family = None
    best_score = -1

    for fam_name, fam_spec in fams.items():
        result = family_score_for_text(text, fam_spec, config)
        family_scores[fam_name] = result["score"]
        family_matches[fam_name] = result

        if result["score"] > best_score:
            best_score = result["score"]
            best_family = fam_name

    return {
        "best_family": best_family,
        "best_family_score": max(best_score, 0),
        "family_scores": family_scores,
        "family_matches": family_matches,
    }


def route_seed(
    seed_q: str,
    seed_a: str,
    profile: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if config is None:
        config = DEFAULT_ROUTING_CONFIG

    text = f"{safe_text(seed_q)}\n{safe_text(seed_a)}"
    text_n = norm_lower(text)

    core_ids = core_identifier_set(profile)
    ext_ids = all_extension_identifiers(profile)
    ext_concepts = all_extension_concepts(profile)
    summary_kw = summary_keywords(profile.get("core", {}).get("model_summary", ""))

    terms = extract_terms(text)

    core_hits = sorted([t for t in terms if t in core_ids])
    ext_identifier_hits = sorted([t for t in terms if t in ext_ids])
    ext_concept_hits = sorted([c for c in ext_concepts if c in text_n])

    unknown_terms = sorted([
        t for t in terms
        if t not in core_ids and t not in ext_ids and t not in GLOBAL_ALLOW
    ])

    known_terms = [t for t in terms if t in core_ids or t in ext_ids or t in GLOBAL_ALLOW]
    unknown_ratio = len(unknown_terms) / max(1, len(known_terms) + len(unknown_terms))

    summary_overlap = sorted([kw for kw in summary_kw if kw in terms])

    extension_intent, extension_intent_hits = generic_extension_intent(text)

    fam = choose_best_family(text, profile, config)
    best_family = fam["best_family"]
    best_family_score = fam["best_family_score"]

    route = "skip"
    route_reason = "no_clear_match"

    # 1. Strong extension match
    if best_family and (
        best_family_score >= config["family_strong_threshold"]
        or (
            config["prefer_extension_when_intent_present"]
            and extension_intent
            and best_family_score >= config["family_weak_threshold"]
        )
    ):
        route = "anchored_extension"
        route_reason = "strong_extension_family_match"

    # 2. Strong core match, no extension pressure
    elif (
        len(core_hits) >= config["core_paraphrase_min_core_hits"]
        and best_family_score == 0
        and unknown_ratio <= config["core_paraphrase_max_unknown_ratio"]
    ):
        route = "core_paraphrase"
        route_reason = "strong_core_match"

    # 3. Core match but noisy / needs cleanup
    elif (
        len(core_hits) >= config["core_repair_min_core_hits"]
        and best_family_score == 0
        and unknown_ratio <= config["core_repair_max_unknown_ratio"]
    ):
        route = "core_repair"
        route_reason = "core_match_with_noise"

    # 4. Mixed core + extension content, lean extension if family is plausible
    elif best_family and best_family_score >= config["family_weak_threshold"] and (extension_intent or ext_identifier_hits or ext_concept_hits):
        route = "anchored_extension"
        route_reason = "mixed_core_extension_content"

    # 5. Weak but nonzero core evidence only
    elif len(core_hits) >= 1 or len(summary_overlap) >= 2:
        route = "core_repair"
        route_reason = "weak_core_anchor"

    else:
        route = "skip"
        route_reason = "insufficient_anchor"

    return {
        "route": route,
        "family": best_family if route == "anchored_extension" else None,
        "route_reason": route_reason,
        "core_hits": core_hits[:50],
        "summary_overlap": summary_overlap[:30],
        "extension_identifier_hits": ext_identifier_hits[:50],
        "extension_concept_hits": ext_concept_hits[:30],
        "best_family_score": best_family_score,
        "family_scores": fam["family_scores"],
        "family_matches": fam["family_matches"],
        "extension_intent": extension_intent,
        "extension_intent_hits": extension_intent_hits[:30],
        "unknown_terms_sample": unknown_terms[:50],
        "unknown_ratio": unknown_ratio,
    }


def route_seed_record(seed: Dict[str, Any], profile: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = route_seed(
        seed_q=seed.get("seed_q") or seed.get("question") or "",
        seed_a=seed.get("seed_a") or seed.get("answer") or "",
        profile=profile,
        config=config,
    )

    return {
        **seed,
        "route_mode": result["route"],
        "extension_family": result.get("family"),
        "route_diagnostics": result,
    }


def route_all_seeds(
    seeds: List[Dict[str, Any]],
    model_profiles: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    out = []

    for seed in seeds:
        model_name = normalize_model_name(seed.get("model_name"))
        profile = model_profiles.get("models", {}).get(model_name)

        if profile is None:
            routed = {
                **seed,
                "route_mode": "skip",
                "extension_family": None,
                "route_diagnostics": {
                    "route": "skip",
                    "family": None,
                    "route_reason": "missing_model_profile",
                    "core_hits": [],
                    "summary_overlap": [],
                    "extension_identifier_hits": [],
                    "extension_concept_hits": [],
                    "best_family_score": 0,
                    "family_scores": {},
                    "family_matches": {},
                    "extension_intent": False,
                    "extension_intent_hits": [],
                    "unknown_terms_sample": [],
                    "unknown_ratio": 1.0,
                }
            }
        else:
            routed = route_seed_record(seed, profile, config=config)

        out.append(routed)

    return out


def summarize_routes(routed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode = Counter()
    by_model = Counter()
    by_model_mode = defaultdict(Counter)
    by_family = Counter()
    by_reason = Counter()

    for row in routed_rows:
        model_name = normalize_model_name(row.get("model_name"))
        mode = row.get("route_mode")
        family = row.get("extension_family")
        reason = row.get("route_diagnostics", {}).get("route_reason")

        if model_name:
            by_model[model_name] += 1
            by_model_mode[model_name][mode] += 1
        if mode:
            by_mode[mode] += 1
        if family:
            by_family[family] += 1
        if reason:
            by_reason[reason] += 1

    per_model = {}
    for model_name, ctr in by_model_mode.items():
        per_model[model_name] = dict(ctr)

    return {
        "total_routed": len(routed_rows),
        "by_mode": dict(by_mode),
        "by_family": dict(by_family),
        "by_reason": dict(by_reason),
        "by_model": dict(by_model),
        "by_model_mode": per_model,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, help="Path to seeds_with_text.jsonl")
    parser.add_argument("--profiles", required=True, help="Path to model_profiles_merged.json")
    parser.add_argument("--out", required=True, help="Output path for seed_routes.jsonl")
    parser.add_argument("--summary-out", default=None, help="Optional output path for routing summary JSON")
    args = parser.parse_args()

    seeds = load_jsonl(args.seeds)
    profiles = load_json(args.profiles)

    routed = route_all_seeds(seeds, profiles)
    save_jsonl(routed, args.out)

    if args.summary_out:
        summary = summarize_routes(routed)
        save_json(summary, args.summary_out)


if __name__ == "__main__":
    main()
