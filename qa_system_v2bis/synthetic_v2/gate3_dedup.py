import hashlib
import re
from typing import Dict, Any, Optional, List


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def qa_hash(question: str, answer: str) -> str:
    payload = normalize_text(question) + "\n---\n" + normalize_text(answer)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def q_hash(question: str) -> str:
    payload = normalize_text(question)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NoopQuestionSimilarityBackend:
    """
    Smoke-test backend: no near-dup detection.
    """
    def max_similarity(self, question: str) -> float:
        return 0.0

    def add_question(self, question: str) -> None:
        return None


class InMemoryDedupIndex:
    """
    Exact dedup:
      - QA hash
      - question-only hash

    Optional near-dup:
      - similarity_backend.max_similarity(question) -> float
      - similarity_backend.add_question(question)
    """

    def __init__(self, similarity_backend: Optional[Any] = None):
        self.qa_hashes = set()
        self.q_hashes = set()
        self.similarity_backend = similarity_backend or NoopQuestionSimilarityBackend()

    def has_qa_hash(self, h: str) -> bool:
        return h in self.qa_hashes

    def has_q_hash(self, h: str) -> bool:
        return h in self.q_hashes

    def max_question_similarity(self, question: str) -> float:
        return float(self.similarity_backend.max_similarity(question))

    def add(self, accepted_record: Dict[str, Any]) -> None:
        q = accepted_record.get("question", "")
        a = accepted_record.get("answer", "")
        self.qa_hashes.add(qa_hash(q, a))
        self.q_hashes.add(q_hash(q))
        self.similarity_backend.add_question(q)

    @classmethod
    def from_existing_rows(cls, rows: List[Dict[str, Any]], similarity_backend: Optional[Any] = None):
        idx = cls(similarity_backend=similarity_backend)
        for row in rows:
            idx.add(row)
        return idx


def run_gate3_dedup(
    question: str,
    answer: str,
    dedup_index: InMemoryDedupIndex,
    near_dup_threshold: float = 0.985
) -> Dict[str, Any]:
    hqa = qa_hash(question, answer)
    hq = q_hash(question)

    if dedup_index.has_qa_hash(hqa):
        return {
            "ok": False,
            "reason": "EXACT_QA_DUP",
            "details": {
                "qa_hash": hqa
            }
        }

    if dedup_index.has_q_hash(hq):
        return {
            "ok": False,
            "reason": "QUESTION_TEMPLATE_DUP",
            "details": {
                "q_hash": hq
            }
        }

    sim = dedup_index.max_question_similarity(question)
    if sim >= near_dup_threshold:
        return {
            "ok": False,
            "reason": "NEAR_DUP_QUESTION",
            "details": {
                "sim": sim,
                "threshold": near_dup_threshold
            }
        }

    return {
        "ok": True,
        "reason": None,
        "details": {
            "qa_hash": hqa,
            "q_hash": hq,
            "near_dup_sim": sim,
            "threshold": near_dup_threshold
        }
    }
