"""
Project-specific glue functions for synthetic_v2 production runtime.

Replace the internals of these functions to match your actual existing codebase.
"""

from typing import Dict, Any


# -----------------------------
# LLM bindings
# -----------------------------

def build_rotating_client(provider: str = None, **kwargs):
    # Replace with your actual builder/import
    # Example:
    # from qa_system_v2bis.llm.rotating_clients import RotatingClients
    # return RotatingClients(provider=provider, **kwargs)
    raise NotImplementedError("Implement build_rotating_client()")


def rotating_client_complete(
    backend,
    prompt: str,
    metadata: Dict[str, Any],
    model_name: str,
    temperature: float,
    max_tokens: int,
    extra_generation_kwargs: Dict[str, Any],
    seed_row: Dict[str, Any],
):
    """
    Replace with your actual rotating client completion call.
    Must return a provider response object or plain text.
    """
    # Example:
    # return backend.complete(
    #     prompt=prompt,
    #     model=model_name,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     metadata=metadata,
    #     **extra_generation_kwargs,
    # )
    raise NotImplementedError("Implement rotating_client_complete()")


# -----------------------------
# Embedding / Gate 1 bindings
# -----------------------------

def build_embedding_backend(**kwargs):
    # Replace with your actual FAISS / embedding backend builder
    raise NotImplementedError("Implement build_embedding_backend()")


def embed_question(backend, text: str):
    # Replace with your actual embedding call
    raise NotImplementedError("Implement embed_question()")


def search_best_neighbor(backend, vec, class_id):
    """
    Must return either:
      {"class_id": ..., "sim": ..., "margin": ...}
    or something the adapter can normalize.
    """
    raise NotImplementedError("Implement search_best_neighbor()")


def get_class_threshold(backend, class_id: int) -> float:
    # Replace with your threshold lookup
    raise NotImplementedError("Implement get_class_threshold()")


# -----------------------------
# Question similarity / Gate 3 bindings
# -----------------------------

def build_question_similarity_backend(**kwargs):
    # Replace with your actual question similarity backend builder
    raise NotImplementedError("Implement build_question_similarity_backend()")


def max_question_similarity(backend, question: str) -> float:
    raise NotImplementedError("Implement max_question_similarity()")


def add_question_to_similarity_index(backend, question: str) -> None:
    raise NotImplementedError("Implement add_question_to_similarity_index()")