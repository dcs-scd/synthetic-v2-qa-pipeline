import argparse
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

from .io_utils import load_json, load_jsonl, save_json
from .text_utils import normalize_model_name
from .text_utils import norm_lower, STOPWORDS as _BASE_STOPWORDS, GLOBAL_ALLOW, BACKTICK_RE, IDLIKE_RE, safe_filename
from .profile_utils import build_core_identifier_set


BRACKET_RE = re.compile(r"\[([^\]]+)\]")
QUOTE_RE = re.compile(r'"([^"\n]{1,120})"|\'([^\'\n]{1,120})\'')

# N-gram candidate extraction
STOPWORDS = _BASE_STOPWORDS | {"use", "using", "used"}


PHRASE_KEYWORDS = {
    "state", "states", "diagram", "transition", "threshold", "vision",
    "grievance", "legitimacy", "risk", "arrest", "active", "quiet",
    "jail", "jailing", "neighbor", "neighbors", "neighborhood", "police",
    "cop", "cops", "agent", "agents", "rebel", "rebels", "rebellion",
    "media", "broadcast", "network", "friends", "friend", "links", "link",
    "cluster", "clustering", "clustered", "path", "small-world", "long-range",
    "leader", "leaders", "homophily", "density", "critical", "tipping",
    "uprising", "phase", "time", "series", "warning", "indicator", "indicators",
    "autocorrelation", "variance", "metric", "metrics", "optimization",
    "speed", "execution", "efficiency", "cache", "hub", "centrality"
}


def short_example(record: Dict[str, Any], max_len: int = 220) -> Dict[str, Any]:
    q = (record.get("question") or "")[:max_len]
    a = (record.get("answer") or "")[:max_len]
    return {
        "source": record.get("source"),
        "record_id": record.get("record_id"),
        "question": q,
        "answer": a,
    }


def load_profiles(path: str) -> Dict[str, Any]:
    data = load_json(path)
    if "models" not in data:
        raise ValueError("Expected profiles JSON with top-level 'models'")
    return data


def build_core_term_set(model_profile: Dict[str, Any]) -> Set[str]:
    return build_core_identifier_set(model_profile)


def build_aux_term_set(model_profile: Dict[str, Any]) -> Set[str]:
    """
    Extra terms derived from source parse that are not currently part of strict core
    validation but are useful for audit support classification.
    """
    rp = model_profile.get("raw_parse", {})
    out = set()

    # string literals may appear in corpus as legitimate model-specific text
    for x in rp.get("string_literals", []):
        out.add(norm_lower(x))

    # direct interface widget extraction
    for x in rp.get("interface_widgets", []):
        out.add(norm_lower(x))

    return out


def build_unresolved_sets(model_profile: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    rp = model_profile.get("raw_parse", {})
    unresolved = {norm_lower(x) for x in rp.get("unresolved_identifiers", [])}
    widget_candidates = {norm_lower(x) for x in rp.get("widget_candidates_from_code", [])}
    return unresolved, widget_candidates


def extract_exact_terms(text: str, core_terms: Set[str], unresolved_hint_terms: Set[str]) -> Counter:
    """
    Extract likely model-specific terms.

    Sources:
    - backtick identifiers
    - [foo] single-token reporter references
    - all plain tokens that match known core terms
    - all plain tokens that match unresolved hints from source parsing
    - hyphenated tokens
    - ?-suffix tokens
    """
    out = Counter()
    t = text or ""

    for x in BACKTICK_RE.findall(t):
        out[norm_lower(x)] += 1

    for x in BRACKET_RE.findall(t):
        x = x.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*\??", x):
            out[norm_lower(x)] += 1

    for tok in IDLIKE_RE.findall(t):
        nt = norm_lower(tok)
        if (
            nt in core_terms
            or nt in unresolved_hint_terms
            or "-" in nt
            or nt.endswith("?")
        ):
            out[nt] += 1

    return out


def extract_phrase_candidates(text: str, min_n: int = 2, max_n: int = 4) -> Counter:
    """
    Collect:
    - quoted phrases
    - short ngrams containing one of the phrase keywords
    """
    out = Counter()
    t = text or ""

    # quoted phrases
    for g1, g2 in QUOTE_RE.findall(t):
        phrase = norm_lower(g1 or g2)
        if phrase and 1 <= len(phrase.split()) <= 5:
            out[phrase] += 1

    # ngrams
    tokens = [norm_lower(x) for x in IDLIKE_RE.findall(t)]
    tokens = [x for x in tokens if x not in STOPWORDS]

    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i+n]
            if any(tok in PHRASE_KEYWORDS for tok in gram):
                phrase = " ".join(gram)
                out[phrase] += 1

    return out


def term_examples_update(storage: Dict[str, List[Dict[str, Any]]], term: str, record: Dict[str, Any], limit: int = 3) -> None:
    if len(storage[term]) < limit:
        storage[term].append(short_example(record))


def classify_terms_for_model(records: List[Dict[str, Any]], model_profile: Dict[str, Any]) -> Dict[str, Any]:
    core_terms = build_core_term_set(model_profile)
    aux_terms = build_aux_term_set(model_profile)
    unresolved_terms, widget_candidate_terms = build_unresolved_sets(model_profile)

    supported_terms = Counter()
    aux_supported_terms = Counter()
    unsupported_terms = Counter()
    phrase_candidates = Counter()

    examples_supported = defaultdict(list)
    examples_aux_supported = defaultdict(list)
    examples_unsupported = defaultdict(list)
    examples_phrases = defaultdict(list)

    source_counts = Counter()

    for rec in records:
        source_counts[rec.get("source") or "unknown"] += 1
        text = f"{rec.get('question', '')}\n{rec.get('answer', '')}"

        exact_terms = extract_exact_terms(text, core_terms=core_terms, unresolved_hint_terms=unresolved_terms)
        phrases = extract_phrase_candidates(text)

        for term, c in exact_terms.items():
            if term in core_terms:
                supported_terms[term] += c
                term_examples_update(examples_supported, term, rec)
            elif term in aux_terms or term in GLOBAL_ALLOW:
                aux_supported_terms[term] += c
                term_examples_update(examples_aux_supported, term, rec)
            elif term in unresolved_terms:
                unsupported_terms[term] += c
                term_examples_update(examples_unsupported, term, rec)
            else:
                unsupported_terms[term] += c
                term_examples_update(examples_unsupported, term, rec)

        for phrase, c in phrases.items():
            phrase_candidates[phrase] += c
            term_examples_update(examples_phrases, phrase, rec, limit=2)

    parser_miss_candidates = []
    extension_candidate_terms = []
    drift_candidate_terms = []

    for term, count in unsupported_terms.most_common():
        item = {
            "term": term,
            "count": count,
            "examples": examples_unsupported[term],
        }

        if term in widget_candidate_terms:
            item["why"] = "overlaps_widget_candidate_from_code"
            parser_miss_candidates.append(item)
        elif term in unresolved_terms:
            item["why"] = "overlaps_unresolved_identifier_from_source_parse"
            parser_miss_candidates.append(item)
        else:
            # unsupported but not obviously a parser miss
            # leave to later extension/drift curation
            if count >= 2:
                extension_candidate_terms.append(item)
            else:
                drift_candidate_terms.append(item)

    return {
        "record_counts": {
            "total": len(records),
            "by_source": dict(source_counts),
        },
        "top_supported_terms": [
            {
                "term": t,
                "count": c,
                "examples": examples_supported[t]
            }
            for t, c in supported_terms.most_common(50)
        ],
        "top_aux_supported_terms": [
            {
                "term": t,
                "count": c,
                "examples": examples_aux_supported[t]
            }
            for t, c in aux_supported_terms.most_common(30)
        ],
        "top_unsupported_terms": [
            {
                "term": t,
                "count": c,
                "examples": examples_unsupported[t]
            }
            for t, c in unsupported_terms.most_common(75)
        ],
        "top_phrase_candidates": [
            {
                "phrase": p,
                "count": c,
                "examples": examples_phrases[p]
            }
            for p, c in phrase_candidates.most_common(100)
        ],
        "parser_miss_candidates": parser_miss_candidates[:40],
        "extension_candidate_terms": extension_candidate_terms[:80],
        "drift_candidate_terms": drift_candidate_terms[:80],
        "source_parse_overlap": {
            "unresolved_identifiers": sorted(list(unresolved_terms))[:100],
            "widget_candidate_terms": sorted(list(widget_candidate_terms))[:100],
        },
    }


def group_records_by_model(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for rec in records:
        model_name = normalize_model_name(rec.get("model_name"))
        if model_name:
            grouped[model_name].append(rec)
    return grouped


def audit_model_records(model_name: str, records: List[Dict[str, Any]], model_profile: Dict[str, Any]) -> Dict[str, Any]:
    classification = classify_terms_for_model(records, model_profile)

    report = {
        "model_name": model_name,
        "core_summary": {
            "procedure_count": len(model_profile.get("core", {}).get("procedures", [])),
            "variable_count": len(model_profile.get("core", {}).get("variables", [])),
            "breed_count": len(model_profile.get("core", {}).get("breeds", [])),
            "widget_count": len(model_profile.get("core", {}).get("widgets", [])),
        },
        **classification
    }
    return report


def summarize_reports(per_model_reports: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "model_count": len(per_model_reports),
        "records_total": 0,
        "records_by_model": {},
        "top_models_by_unsupported_terms": [],
        "top_models_by_parser_miss_candidates": [],
        "top_models_by_extension_candidates": [],
        "top_global_unsupported_terms": [],
        "top_global_phrase_candidates": [],
    }

    global_unsupported = Counter()
    global_phrases = Counter()
    model_unsupported_pressure = []
    model_parser_miss_pressure = []
    model_extension_pressure = []

    for model_name, rep in per_model_reports.items():
        nrec = rep["record_counts"]["total"]
        summary["records_total"] += nrec
        summary["records_by_model"][model_name] = nrec

        unsupported_total = sum(x["count"] for x in rep.get("top_unsupported_terms", []))
        parser_miss_total = sum(x["count"] for x in rep.get("parser_miss_candidates", []))
        extension_total = sum(x["count"] for x in rep.get("extension_candidate_terms", []))

        model_unsupported_pressure.append((model_name, unsupported_total))
        model_parser_miss_pressure.append((model_name, parser_miss_total))
        model_extension_pressure.append((model_name, extension_total))

        for item in rep.get("top_unsupported_terms", []):
            global_unsupported[item["term"]] += item["count"]

        for item in rep.get("top_phrase_candidates", []):
            global_phrases[item["phrase"]] += item["count"]

    model_unsupported_pressure.sort(key=lambda x: (-x[1], x[0]))
    model_parser_miss_pressure.sort(key=lambda x: (-x[1], x[0]))
    model_extension_pressure.sort(key=lambda x: (-x[1], x[0]))

    summary["top_models_by_unsupported_terms"] = [
        {"model_name": m, "count": c}
        for m, c in model_unsupported_pressure[:20]
    ]
    summary["top_models_by_parser_miss_candidates"] = [
        {"model_name": m, "count": c}
        for m, c in model_parser_miss_pressure[:20]
    ]
    summary["top_models_by_extension_candidates"] = [
        {"model_name": m, "count": c}
        for m, c in model_extension_pressure[:20]
    ]
    summary["top_global_unsupported_terms"] = [
        {"term": t, "count": c}
        for t, c in global_unsupported.most_common(100)
    ]
    summary["top_global_phrase_candidates"] = [
        {"phrase": p, "count": c}
        for p, c in global_phrases.most_common(100)
    ]

    return summary


def audit_all_models(core_profiles: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = group_records_by_model(records)

    def _audit_one(item):
        model_name, model_profile = item
        model_records = grouped.get(model_name, [])
        return model_name, audit_model_records(model_name, model_records, model_profile)

    items = list(core_profiles.get("models", {}).items())
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(_audit_one, items))

    return dict(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True, help="Path to core_profiles.json")
    parser.add_argument("--records", required=True, help="Path to all_records.jsonl")
    parser.add_argument("--out-dir", required=True, help="Directory for per-model audit JSON")
    parser.add_argument("--summary-out", required=True, help="Path to summary.json")
    args = parser.parse_args()

    core_profiles = load_profiles(args.profiles)
    records = load_jsonl(args.records)

    per_model_reports = audit_all_models(core_profiles, records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, rep in per_model_reports.items():
        save_json(rep, str(out_dir / f"{safe_filename(model_name)}.json"))

    save_json(per_model_reports, str(out_dir / "_all_models.json"))

    summary = summarize_reports(per_model_reports)
    save_json(summary, args.summary_out)


if __name__ == "__main__":
    main()
