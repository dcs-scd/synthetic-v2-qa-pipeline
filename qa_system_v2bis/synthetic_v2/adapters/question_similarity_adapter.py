from typing import Optional


class FAISSQuestionSimilarityBackend:
    """
    Adapter for near-duplicate question detection using FAISS.

    Expected interface for gate3_dedup.InMemoryDedupIndex:
        - max_similarity(question: str) -> float
        - add_question(question: str) -> None

    Usage:
        backend = FAISSQuestionSimilarityBackend(
            faiss_index=my_dedup_index,
            embed_fn=my_embed_function,
        )
        dedup_index = InMemoryDedupIndex(similarity_backend=backend)
    """

    def __init__(self, faiss_index, embed_fn):
        self.faiss_index = faiss_index
        self.embed_fn = embed_fn

    def max_similarity(self, question: str) -> float:
        """
        Override with actual FAISS similarity search logic.
        Should return the maximum cosine similarity to any
        previously added question.
        """
        # Example pseudo-implementation:
        # vec = self.embed_fn(question)
        # if self.faiss_index.ntotal == 0:
        #     return 0.0
        # D, I = self.faiss_index.search(vec.reshape(1, -1), k=1)
        # return float(D[0][0])
        raise NotImplementedError(
            "FAISSQuestionSimilarityBackend.max_similarity() must be "
            "implemented with your actual FAISS search logic."
        )

    def add_question(self, question: str) -> None:
        """
        Override with actual FAISS add logic.
        """
        # Example pseudo-implementation:
        # vec = self.embed_fn(question)
        # self.faiss_index.add(vec.reshape(1, -1))
        raise NotImplementedError(
            "FAISSQuestionSimilarityBackend.add_question() must be "
            "implemented with your actual FAISS add logic."
        )
