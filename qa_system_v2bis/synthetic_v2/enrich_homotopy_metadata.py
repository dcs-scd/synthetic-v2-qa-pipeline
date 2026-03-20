import argparse
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

from .io_utils import load_jsonl, save_json, save_jsonl


# -----------------------------
# Parquet loading
# -----------------------------

def load_parquet_rows(path: str) -> List[Dict[str, Any]]:
    """
    Load parquet into a list[dict].
    Tries pandas first, then pyarrow.
    """
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    except Exception:
        pass

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.to_pylist()
    except Exception as e:
        raise RuntimeError(f"Could not load parquet file {path}: {e}")


# -----------------------------
# Column detection
# -----------------------------

EXAMPLE_ID_CANDIDATES = [
    "id",
    "example_id",
    "record_id",
    "source_id",
    "global_id",
    "seed_id",
    "corpus_id",
]

CLASS_ID_CANDIDATES = [
    "class_id",
    "cluster_id",
    "class",
    "cluster",
]

TIER_CANDIDATES = [
    "tier",
]

TIGHTNESS_P10_CANDIDATES = [
    "tightness_p10",
    "p10",
    "cluster_tightness_p10",
]

INTERIOR_CANDIDATES = [
    "is_interior",
    "interior",
    "non_boundary",
    "is_non_boundary",
]

BOUNDARY_CANDIDATES = [
    "is_boundary",
    "boundary",
]

REP_CANDIDATES = [
    "is_representative",
    "representative",
    "is_rep",
]


def detect_first_present_column(rows: List[Dict[str, Any]], candidates: List[str]) -> Optional[str]:
    if not rows:
        return None
    keys = set()
    for row in rows[:100]:
        keys |= set(row.keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def detect_columns(rows: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    return {
        "example_id": detect_first_present_column(rows, EXAMPLE_ID_CANDIDATES),
        "class_id": detect_first_present_column(rows, CLASS_ID_CANDIDATES),
        "tier": detect_first_present_column(rows, TIER_CANDIDATES),
        "tightness_p10": detect_first_present_column(rows, TIGHTNESS_P10_CANDIDATES),
        "is_interior": detect_first_present_column(rows, INTERIOR_CANDIDATES),
        "is_boundary": detect_first_present_column(rows, BOUNDARY_CANDIDATES),
        "is_representative": detect_first_present_column(rows, REP_CANDIDATES),
    }


# -----------------------------
# Normalization helpers
# -----------------------------

def safe_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def norm_id(x: Any) -> Optional[str]:
    s = safe_text(x)
    if not s:
        return None
    return s


def coerce_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def compute_tier_from_p10(p10: Optional[float]) -> Optional[str]:
    if p10 is None:
        return None
    if p10 >= 0.82:
        return "A"
    if p10 >= 0.78:
        return "B"
    return "C"


def record_candidate_ids(row: Dict[str, Any]) -> List[str]:
    """
    Generate multiple possible identifiers for matching normalized rows
    to example_metadata rows.
    """
    out = []

    for key in ["source_id", "seed_id", "global_id", "record_id"]:
        v = norm_id(row.get(key))
        if v:
            out.append(v)

    rid = norm_id(row.get("record_id"))
    if rid and "::" in rid:
        out.append(rid.split("::", 1)[1])

    raw_ref = row.get("raw_ref") or {}
    for key in ["id", "global_id", "joined_from_corpus_id"]:
        v = norm_id(raw_ref.get(key))
        if v:
            out.append(v)

    seen = set()
    final = []
    for x in out:
        if x not in seen:
            seen.add(x)
            final.append(x)
    return final


# -----------------------------
# Lookup building
# -----------------------------

def build_example_lookup(rows: List[Dict[str, Any]], cols: Dict[str, Optional[str]]) -> Dict[str, Dict[str, Any]]:
    id_col = cols["example_id"]
    if not id_col:
        raise RuntimeError("Could not detect example id column in example_metadata parquet")

    lookup = {}
    for row in rows:
        ex_id = norm_id(row.get(id_col))
        if ex_id:
            lookup[ex_id] = row
    return lookup


def build_class_lookup(rows: List[Dict[str, Any]], cols: Dict[str, Optional[str]]) -> Dict[Any, Dict[str, Any]]:
    class_col = cols["class_id"]
    if not class_col:
        raise RuntimeError("Could not detect class_id column in classes parquet")

    lookup = {}
    for row in rows:
        cid = row.get(class_col)
        if cid is not None:
            lookup[cid] = row
    return lookup


def extract_example_enrichment(
    ex_row: Dict[str, Any],
    ex_cols: Dict[str, Optional[str]],
    class_lookup: Dict[Any, Dict[str, Any]],
    class_cols: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    out = {}

    class_id = ex_row.get(ex_cols["class_id"]) if ex_cols["class_id"] else None
    if class_id is not None:
        out["class_id"] = class_id

    if ex_cols["is_interior"]:
        out["is_interior"] = coerce_bool(ex_row.get(ex_cols["is_interior"]))
    if ex_cols["is_boundary"]:
        out["is_boundary"] = coerce_bool(ex_row.get(ex_cols["is_boundary"]))
    if ex_cols["is_representative"]:
        out["is_representative"] = coerce_bool(ex_row.get(ex_cols["is_representative"]))

    class_row = class_lookup.get(class_id) if class_id is not None else None
    if class_row:
        tier = None
        if class_cols["tier"]:
            tier = safe_text(class_row.get(class_cols["tier"]))
            if tier:
                tier = tier.upper()

        if not tier and class_cols["tightness_p10"]:
            p10 = coerce_float(class_row.get(class_cols["tightness_p10"]))
            out["tightness_p10"] = p10
            tier = compute_tier_from_p10(p10)

        if tier:
            out["tier"] = tier

        if class_cols["tightness_p10"] and "tightness_p10" not in out:
            out["tightness_p10"] = coerce_float(class_row.get(class_cols["tightness_p10"]))

    return out


def enrich_row(
    row: Dict[str, Any],
    example_lookup: Dict[str, Dict[str, Any]],
    ex_cols: Dict[str, Optional[str]],
    class_lookup: Dict[Any, Dict[str, Any]],
    class_cols: Dict[str, Optional[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidate_ids = record_candidate_ids(row)
    matched_id = None
    ex_row = None

    for cid in candidate_ids:
        if cid in example_lookup:
            matched_id = cid
            ex_row = example_lookup[cid]
            break

    enriched = dict(row)
    debug = {
        "candidate_ids": candidate_ids,
        "matched_example_id": matched_id,
        "matched": ex_row is not None,
    }

    if ex_row is None:
        return enriched, debug

    enrich = extract_example_enrichment(ex_row, ex_cols, class_lookup, class_cols)

    existing_class = enriched.get("class_id")
    incoming_class = enrich.get("class_id")
    if existing_class is not None and incoming_class is not None and existing_class != incoming_class:
        debug["class_id_conflict"] = {
            "existing": existing_class,
            "incoming": incoming_class
        }
    elif existing_class is None and incoming_class is not None:
        enriched["class_id"] = incoming_class

    if not enriched.get("tier") and enrich.get("tier"):
        enriched["tier"] = enrich["tier"]

    for key in ["is_interior", "is_boundary", "is_representative", "tightness_p10"]:
        if enrich.get(key) is not None and enriched.get(key) is None:
            enriched[key] = enrich[key]

    return enriched, debug


def enrich_rows(
    rows: List[Dict[str, Any]],
    example_lookup: Dict[str, Dict[str, Any]],
    ex_cols: Dict[str, Optional[str]],
    class_lookup: Dict[Any, Dict[str, Any]],
    class_cols: Dict[str, Optional[str]],
    source_name: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out = []
    matched = 0
    unmatched = 0
    conflicts = 0
    debug_examples = []

    for row in rows:
        enriched, dbg = enrich_row(row, example_lookup, ex_cols, class_lookup, class_cols)
        out.append(enriched)

        if dbg["matched"]:
            matched += 1
        else:
            unmatched += 1

        if "class_id_conflict" in dbg:
            conflicts += 1

        if len(debug_examples) < 25:
            debug_examples.append({
                "record_id": row.get("record_id"),
                "source_id": row.get("source_id"),
                "seed_id": row.get("seed_id"),
                "matched_example_id": dbg.get("matched_example_id"),
                "candidate_ids": dbg.get("candidate_ids"),
                "class_id_conflict": dbg.get("class_id_conflict"),
            })

    report = {
        "source_name": source_name,
        "row_count": len(rows),
        "matched_examples": matched,
        "unmatched_examples": unmatched,
        "class_id_conflicts": conflicts,
        "debug_examples": debug_examples,
    }
    return out, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-jsonl", required=True, help="Path to seeds_with_text.jsonl")
    parser.add_argument("--all-records-jsonl", required=True, help="Path to all_records.jsonl")
    parser.add_argument("--classes-parquet", required=True, help="Path to classes.parquet")
    parser.add_argument("--example-metadata-parquet", required=True, help="Path to example_metadata.parquet")
    parser.add_argument("--out-seeds", required=True, help="Output enriched seeds JSONL")
    parser.add_argument("--out-records", required=True, help="Output enriched all_records JSONL")
    parser.add_argument("--out-report", required=True, help="Output enrichment report JSON")
    args = parser.parse_args()

    seeds = load_jsonl(args.seeds_jsonl)
    records = load_jsonl(args.all_records_jsonl)

    class_rows = load_parquet_rows(args.classes_parquet)
    example_rows = load_parquet_rows(args.example_metadata_parquet)

    class_cols = detect_columns(class_rows)
    ex_cols = detect_columns(example_rows)

    class_lookup = build_class_lookup(class_rows, class_cols)
    example_lookup = build_example_lookup(example_rows, ex_cols)

    seeds_enriched, seeds_report = enrich_rows(
        seeds, example_lookup, ex_cols, class_lookup, class_cols, source_name="seeds"
    )
    records_enriched, records_report = enrich_rows(
        records, example_lookup, ex_cols, class_lookup, class_cols, source_name="all_records"
    )

    save_jsonl(seeds_enriched, args.out_seeds)
    save_jsonl(records_enriched, args.out_records)

    report = {
        "detected_columns": {
            "classes_parquet": class_cols,
            "example_metadata_parquet": ex_cols,
        },
        "classes_row_count": len(class_rows),
        "example_metadata_row_count": len(example_rows),
        "seeds_report": seeds_report,
        "records_report": records_report,
    }
    save_json(report, args.out_report)


if __name__ == "__main__":
    main()