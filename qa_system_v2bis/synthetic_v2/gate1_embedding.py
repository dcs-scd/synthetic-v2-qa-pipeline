from typing import Dict, Any, Optional


class AlwaysPassEmbeddingIndex:
    """
    Smoke-test adapter.
    Always passes Gate 1.
    """

    def threshold_for_class(self, class_id: int) -> float:
        return 0.0

    def embed(self, text: str):
        return None

    def search_best(self, vec, class_id: Optional[int] = None) -> Dict[str, Any]:
        return {
            "class_id": class_id,
            "sim": 1.0,
            "margin": 1.0
        }


def run_gate1_embedding(
    question: str,
    class_id: int,
    embedding_index
) -> Dict[str, Any]:
    """
    Production expectation for embedding_index:
      - embed(text) -> vector
      - search_best(vec, class_id=None) -> {
            "class_id": int,
            "sim": float,
            "margin": float
        }
      - threshold_for_class(class_id) -> float

    This wrapper keeps Gate 1 logic separate from the underlying embedding stack.
    """
    try:
        vec = embedding_index.embed(question)
        best = embedding_index.search_best(vec, class_id=class_id)
        threshold = embedding_index.threshold_for_class(class_id)
    except Exception as e:
        return {
            "ok": False,
            "reason": "GATE1_BACKEND_ERROR",
            "details": {"error": str(e)}
        }

    best_class = best.get("class_id")
    sim = float(best.get("sim", 0.0))
    margin = float(best.get("margin", 0.0))

    if class_id is not None and best_class is not None and best_class != class_id:
        return {
            "ok": False,
            "reason": "WRONG_CLASS_NEIGHBORHOOD",
            "details": {
                "expected_class_id": class_id,
                "best_class_id": best_class,
                "best_sim": sim,
                "margin": margin,
                "threshold": threshold
            }
        }

    if sim < threshold:
        return {
            "ok": False,
            "reason": "LOW_EMBED_SIM",
            "details": {
                "best_sim": sim,
                "margin": margin,
                "threshold": threshold
            }
        }

    MIN_CLASS_MARGIN = 0.01
    if margin < MIN_CLASS_MARGIN:
        return {
            "ok": False,
            "reason": "LOW_CLASS_MARGIN",
            "details": {
                "best_sim": sim,
                "margin": margin,
                "threshold": threshold
            }
        }

    return {
        "ok": True,
        "reason": None,
        "details": {
            "best_sim": sim,
            "margin": margin,
            "threshold": threshold,
            "best_class_id": best_class
        }
    }
