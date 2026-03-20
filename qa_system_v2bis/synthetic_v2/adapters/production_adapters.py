"""
Production adapters that support explicit callable binding for config-driven runtime.
These extend the base adapters to support dynamic function injection.
"""

from typing import Dict, Any, Optional, Callable


class RotatingClientAdapter:
    """
    Production adapter for rotating LLM clients with explicit callable support.

    Can use either:
    - backend's native methods (if complete_fn/chat_fn are None)
    - explicit callable functions passed in
    """

    def __init__(
        self,
        backend,
        model_name: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 1400,
        extra_generation_kwargs: Optional[Dict[str, Any]] = None,
        complete_fn: Optional[Callable] = None,
        chat_fn: Optional[Callable] = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_generation_kwargs = extra_generation_kwargs or {}
        self.complete_fn = complete_fn
        self.chat_fn = chat_fn

    def generate(self, prompt: str, seed_row: Dict[str, Any]) -> str:
        """Generate response using either explicit complete_fn or backend's default method."""
        metadata = {
            "seed_id": seed_row.get("seed_id"),
            "model_name": self.model_name or seed_row.get("model_name"),
        }

        if self.complete_fn:
            # Use explicit callable
            result = self.complete_fn(
                backend=self.backend,
                prompt=prompt,
                metadata=metadata,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_generation_kwargs=self.extra_generation_kwargs,
                seed_row=seed_row,
            )
        else:
            # Fallback to backend's presumed interface
            if hasattr(self.backend, 'complete'):
                result = self.backend.complete(
                    prompt=prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    metadata=metadata,
                    **self.extra_generation_kwargs,
                )
            else:
                raise NotImplementedError(
                    "RotatingClientAdapter: backend has no 'complete' method and no complete_fn provided"
                )

        # Extract text from result if it's an object
        if hasattr(result, 'text'):
            return result.text
        elif isinstance(result, str):
            return result
        else:
            return str(result)


class ExistingEmbeddingIndexAdapter:
    """
    Production adapter for embedding backends with explicit callable support.

    Can use either:
    - backend's native methods (if callables are None)
    - explicit callable functions passed in
    """

    def __init__(
        self,
        backend,
        embed_fn: Optional[Callable] = None,
        search_fn: Optional[Callable] = None,
        threshold_fn: Optional[Callable] = None,
        default_threshold: float = 0.55,
    ):
        self.backend = backend
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.threshold_fn = threshold_fn
        self.default_threshold = default_threshold

    def embed(self, text: str):
        """Embed text using either explicit embed_fn or backend's default method."""
        if self.embed_fn:
            return self.embed_fn(self.backend, text)
        elif hasattr(self.backend, 'embed'):
            return self.backend.embed(text)
        else:
            raise NotImplementedError(
                "ExistingEmbeddingIndexAdapter: backend has no 'embed' method and no embed_fn provided"
            )

    def search_best(self, vec, class_id: Optional[int] = None) -> Dict[str, Any]:
        """Search using either explicit search_fn or backend's default method."""
        if self.search_fn:
            return self.search_fn(self.backend, vec, class_id)
        elif hasattr(self.backend, 'search_best'):
            return self.backend.search_best(vec, class_id=class_id)
        else:
            raise NotImplementedError(
                "ExistingEmbeddingIndexAdapter: backend has no 'search_best' method and no search_fn provided"
            )

    def threshold_for_class(self, class_id: int) -> float:
        """Get threshold using either explicit threshold_fn or backend's default method."""
        if self.threshold_fn:
            return self.threshold_fn(self.backend, class_id)
        elif hasattr(self.backend, 'threshold_for_class'):
            return self.backend.threshold_for_class(class_id)
        else:
            # Fallback to default threshold
            return self.default_threshold


class ExistingQuestionSimilarityAdapter:
    """
    Production adapter for question similarity backends with explicit callable support.

    Can use either:
    - backend's native methods (if callables are None)
    - explicit callable functions passed in
    """

    def __init__(
        self,
        backend,
        max_similarity_fn: Optional[Callable] = None,
        add_question_fn: Optional[Callable] = None,
    ):
        self.backend = backend
        self.max_similarity_fn = max_similarity_fn
        self.add_question_fn = add_question_fn

    def max_similarity(self, question: str) -> float:
        """Get max similarity using either explicit max_similarity_fn or backend's default method."""
        if self.max_similarity_fn:
            return self.max_similarity_fn(self.backend, question)
        elif hasattr(self.backend, 'max_similarity'):
            return self.backend.max_similarity(question)
        else:
            raise NotImplementedError(
                "ExistingQuestionSimilarityAdapter: backend has no 'max_similarity' method and no max_similarity_fn provided"
            )

    def add_question(self, question: str) -> None:
        """Add question using either explicit add_question_fn or backend's default method."""
        if self.add_question_fn:
            self.add_question_fn(self.backend, question)
        elif hasattr(self.backend, 'add_question'):
            self.backend.add_question(question)
        else:
            raise NotImplementedError(
                "ExistingQuestionSimilarityAdapter: backend has no 'add_question' method and no add_question_fn provided"
            )