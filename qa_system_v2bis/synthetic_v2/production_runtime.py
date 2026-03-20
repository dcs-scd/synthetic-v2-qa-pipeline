import importlib
from typing import Dict, Any, Optional, Callable

from .io_utils import load_json
from .adapters import (
    RotatingClientAdapter,
    ExistingEmbeddingIndexAdapter,
    ExistingQuestionSimilarityAdapter,
)


class RuntimeConfigError(RuntimeError):
    pass


def import_from_path(path: str) -> Any:
    """
    Supports:
      module.submodule:attr
      module.submodule.attr
    """
    if ":" in path:
        module_name, attr_path = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise RuntimeConfigError(f"Invalid import path: {path}")
        module_name = ".".join(parts[:-1])
        attr_path = parts[-1]

    mod = importlib.import_module(module_name)
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def maybe_import_callable(path: Optional[str]) -> Optional[Callable]:
    if not path:
        return None
    obj = import_from_path(path)
    if not callable(obj):
        raise RuntimeConfigError(f"Imported object is not callable: {path}")
    return obj


def build_backend_from_section(section: Dict[str, Any], section_name: str) -> Any:
    """
    Section supports either:
      - backend_object_path
      - backend_builder_path + backend_builder_kwargs

    Builder invocation tries:
      builder(**kwargs)
      builder(config=section, **kwargs)
      builder(section)
    """
    if not section:
        raise RuntimeConfigError(f"Missing config section: {section_name}")

    if section.get("backend_object_path"):
        return import_from_path(section["backend_object_path"])

    if section.get("backend_builder_path"):
        builder = import_from_path(section["backend_builder_path"])
        if not callable(builder):
            raise RuntimeConfigError(f"Builder for {section_name} is not callable")

        kwargs = section.get("backend_builder_kwargs", {})

        try:
            return builder(**kwargs)
        except TypeError:
            pass

        try:
            return builder(config=section, **kwargs)
        except TypeError:
            pass

        try:
            return builder(section)
        except TypeError as e:
            raise RuntimeConfigError(
                f"Could not invoke backend builder for {section_name}: {e}"
            )

    raise RuntimeConfigError(
        f"{section_name} config must define backend_object_path or backend_builder_path"
    )


def build_runtime(runtime_config_json: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds production runtime with config-driven explicit adapter bindings.

    Expected config structure:
    {
      "llm": {
        "backend_builder_path": "...:build_rotating_client",
        "backend_builder_kwargs": {...},
        "model_name": "...",
        "temperature": 0.4,
        "max_tokens": 1400,
        "extra_generation_kwargs": {},
        "complete_fn_path": "...:complete_fn",   # optional
        "chat_fn_path": "...:chat_fn"            # optional
      },
      "embedding": {
        "backend_builder_path": "...:build_embedding_backend",
        "backend_builder_kwargs": {...},
        "embed_fn_path": "...:embed_fn",         # optional
        "search_fn_path": "...:search_fn",       # optional
        "threshold_fn_path": "...:threshold_fn", # optional
        "default_threshold": 0.55
      },
      "dedup": {
        "backend_builder_path": "...:build_question_similarity_backend",
        "backend_builder_kwargs": {...},
        "max_similarity_fn_path": "...:max_similarity_fn", # optional
        "add_question_fn_path": "...:add_question_fn"      # optional
      }
    }
    """
    config = load_json(runtime_config_json) if runtime_config_json else {}

    llm_cfg = config.get("llm", {})
    emb_cfg = config.get("embedding", {})
    dedup_cfg = config.get("dedup", {})

    # Build raw backends
    rotating_backend = build_backend_from_section(llm_cfg, "llm")
    embedding_backend = build_backend_from_section(emb_cfg, "embedding")
    question_sim_backend = build_backend_from_section(dedup_cfg, "dedup")

    # Import optional explicit binding callables
    complete_fn = maybe_import_callable(llm_cfg.get("complete_fn_path"))
    chat_fn = maybe_import_callable(llm_cfg.get("chat_fn_path"))

    embed_fn = maybe_import_callable(emb_cfg.get("embed_fn_path"))
    search_fn = maybe_import_callable(emb_cfg.get("search_fn_path"))
    threshold_fn = maybe_import_callable(emb_cfg.get("threshold_fn_path"))

    max_similarity_fn = maybe_import_callable(dedup_cfg.get("max_similarity_fn_path"))
    add_question_fn = maybe_import_callable(dedup_cfg.get("add_question_fn_path"))

    llm_client = RotatingClientAdapter(
        backend=rotating_backend,
        model_name=llm_cfg.get("model_name"),
        temperature=llm_cfg.get("temperature", 0.4),
        max_tokens=llm_cfg.get("max_tokens", 1400),
        extra_generation_kwargs=llm_cfg.get("extra_generation_kwargs", {}),
        complete_fn=complete_fn,
        chat_fn=chat_fn,
    )

    embedding_index = ExistingEmbeddingIndexAdapter(
        backend=embedding_backend,
        embed_fn=embed_fn,
        search_fn=search_fn,
        threshold_fn=threshold_fn,
        default_threshold=emb_cfg.get("default_threshold", 0.55),
    )

    question_similarity_backend = ExistingQuestionSimilarityAdapter(
        backend=question_sim_backend,
        max_similarity_fn=max_similarity_fn,
        add_question_fn=add_question_fn,
    )

    return {
        "llm_client": llm_client,
        "embedding_index": embedding_index,
        "question_similarity_backend": question_similarity_backend,
        "dedup_index": None,  # runner builds/loads this later
        "config": config,
        "raw_backends": {
            "rotating_backend": rotating_backend,
            "embedding_backend": embedding_backend,
            "question_similarity_backend": question_sim_backend,
        }
    }