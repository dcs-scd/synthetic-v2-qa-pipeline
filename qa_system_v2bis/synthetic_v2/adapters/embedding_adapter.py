from typing import Dict, Any, Optional


class FAISSEmbeddingAdapter:
    """
    Adapter around a FAISS-based embedding index.

    Expected interface for run_synthetic_v2:
        - embed(text: str) -> vector
        - search_best(vec, class_id=None) -> {"class_id": int, "sim": float, "margin": float}
        - threshold_for_class(class_id: int) -> float

    Usage:
        adapter = FAISSEmbeddingAdapter(
            faiss_index=my_faiss_index,
            embed_fn=my_embed_function,
            class_thresholds=my_threshold_dict,
        )
        # Then pass adapter as embedding_index to run_synthetic_v2
    """

    def __init__(self, faiss_index, embed_fn, class_thresholds: Dict[int, float], default_threshold: float = 0.65):
        self.faiss_index = faiss_index
        self.embed_fn = embed_fn
        self.class_thresholds = class_thresholds
        self.default_threshold = default_threshold

    def threshold_for_class(self, class_id: int) -> float:
        return self.class_thresholds.get(class_id, self.default_threshold)

    def embed(self, text: str):
        """Return embedding vector for the given text."""
        return self.embed_fn(text)

    def search_best(self, vec, class_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Override with actual FAISS search logic.
        Should return {"class_id": int, "sim": float, "margin": float}.
        """
        # Example pseudo-implementation:
        # D, I = self.faiss_index.search(vec.reshape(1, -1), k=10)
        # best_sim = float(D[0][0])
        # best_class = class_labels[I[0][0]]
        # margin = float(D[0][0] - D[0][1]) if len(D[0]) > 1 else 1.0
        # return {"class_id": best_class, "sim": best_sim, "margin": margin}
        raise NotImplementedError(
            "FAISSEmbeddingAdapter.search_best() must be implemented "
            "with your actual FAISS search logic."
        )
