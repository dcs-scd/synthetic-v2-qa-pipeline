"""Tests for Gate 3 deduplication."""
import pytest
from qa_system_v2bis.synthetic_v2.gate3_dedup import (
    InMemoryDedupIndex,
    run_gate3_dedup,
    qa_hash,
    q_hash,
)


class TestExactDedup:
    def test_first_entry_passes(self):
        idx = InMemoryDedupIndex()
        r = run_gate3_dedup("What is X?", "X is Y.", idx)
        assert r["ok"] is True

    def test_exact_qa_dup_fails(self):
        idx = InMemoryDedupIndex()
        idx.add({"question": "What is X?", "answer": "X is Y."})
        r = run_gate3_dedup("What is X?", "X is Y.", idx)
        assert r["ok"] is False
        assert r["reason"] == "EXACT_QA_DUP"

    def test_same_question_different_answer_fails(self):
        idx = InMemoryDedupIndex()
        idx.add({"question": "What is X?", "answer": "X is Y."})
        r = run_gate3_dedup("What is X?", "X is Z.", idx)
        assert r["ok"] is False
        assert r["reason"] == "QUESTION_TEMPLATE_DUP"

    def test_different_question_passes(self):
        idx = InMemoryDedupIndex()
        idx.add({"question": "What is X?", "answer": "X is Y."})
        r = run_gate3_dedup("What is Z?", "Z is W.", idx)
        assert r["ok"] is True

    def test_whitespace_normalization(self):
        idx = InMemoryDedupIndex()
        idx.add({"question": "What  is   X?", "answer": "X is Y."})
        r = run_gate3_dedup("What is X?", "X is Y.", idx)
        assert r["ok"] is False  # normalized versions match


class TestFromExisting:
    def test_seed_from_existing_rows(self):
        rows = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        idx = InMemoryDedupIndex.from_existing_rows(rows)
        r1 = run_gate3_dedup("Q1", "A1", idx)
        assert r1["ok"] is False
        r3 = run_gate3_dedup("Q3", "A3", idx)
        assert r3["ok"] is True
