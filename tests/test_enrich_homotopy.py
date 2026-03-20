import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_compute_tier_from_p10():
    from qa_system_v2bis.synthetic_v2.enrich_homotopy_metadata import compute_tier_from_p10
    assert compute_tier_from_p10(0.85) == "A"
    assert compute_tier_from_p10(0.82) == "A"
    assert compute_tier_from_p10(0.80) == "B"
    assert compute_tier_from_p10(0.78) == "B"
    assert compute_tier_from_p10(0.70) == "C"
    assert compute_tier_from_p10(None) is None

def test_record_candidate_ids():
    from qa_system_v2bis.synthetic_v2.enrich_homotopy_metadata import record_candidate_ids
    row = {"record_id": "corpus::STE-GENERAL-NL-0175", "seed_id": "S1"}
    ids = record_candidate_ids(row)
    assert "corpus::STE-GENERAL-NL-0175" in ids
    assert "STE-GENERAL-NL-0175" in ids
    assert "S1" in ids

def test_coerce_bool():
    from qa_system_v2bis.synthetic_v2.enrich_homotopy_metadata import coerce_bool
    assert coerce_bool(True) is True
    assert coerce_bool(False) is False
    assert coerce_bool("1") is True
    assert coerce_bool("true") is True
    assert coerce_bool("0") is False
    assert coerce_bool(None) is None

def test_detect_columns():
    from qa_system_v2bis.synthetic_v2.enrich_homotopy_metadata import detect_columns
    rows = [{"id": "x", "class_id": 1, "tightness_p10": 0.85, "is_boundary": False}]
    cols = detect_columns(rows)
    assert cols["example_id"] == "id"
    assert cols["class_id"] == "class_id"
    assert cols["tightness_p10"] == "tightness_p10"
    assert cols["is_boundary"] == "is_boundary"

def test_support_context_render():
    from qa_system_v2bis.synthetic_v2.support_context import render_support_block
    rec = {"record_id": "R1", "source": "corpus", "model_name": "rebellion", "class_id": 5, "question": "How does X work?", "answer": "X works by doing Y."}
    block = render_support_block(rec)
    assert "R1" in block
    assert "How does X work?" in block
    assert "LOCAL SUPPORT EXEMPLAR" in block

def test_support_context_none():
    from qa_system_v2bis.synthetic_v2.support_context import render_support_block
    assert render_support_block(None) == ""