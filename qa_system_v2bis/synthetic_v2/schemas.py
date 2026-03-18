from typing import TypedDict, Optional, List, Dict, Any, Literal

RouteMode = Literal[
    "core_paraphrase",
    "core_repair",
    "anchored_extension",
    "skip"
]

SourceType = Literal[
    "seed",
    "corpus",
    "accepted_synth",
    "rejected_synth"
]


class SeedRecord(TypedDict, total=False):
    seed_id: str
    class_id: int
    model_name: str
    level: str
    tier: str
    seed_type: str
    seed_q: str
    seed_a: str


class NormalizedRecord(TypedDict, total=False):
    record_id: str
    source: SourceType
    source_id: str
    model_name: str
    question: str
    answer: str

    # optional metadata
    seed_id: str
    class_id: int
    level: str
    tier: str
    seed_type: str
    global_id: str
    tags: List[str]
    raw_ref: Dict[str, Any]


class RouteResult(TypedDict, total=False):
    route: RouteMode
    family: Optional[str]
    core_hits: List[str]
    best_family_score: int
    family_scores: Dict[str, int]
    family_matches: Dict[str, Any]
    extension_intent: bool
    unknown_terms_sample: List[str]


class CoreProfile(TypedDict, total=False):
    procedures: List[str]
    variables: List[str]
    breeds: List[str]
    widgets: List[str]
    model_summary: str


class ExtensionFamily(TypedDict, total=False):
    name: str
    concepts: List[str]
    identifiers: List[str]
    rules: List[str]


class ExtensionProfile(TypedDict, total=False):
    families: List[ExtensionFamily]
    general_rules: List[str]
    framing_cues: List[str]
    disallowed_unanchored_themes: List[str]


class ModelProfile(TypedDict, total=False):
    model_name: str
    core: CoreProfile
    extensions: ExtensionProfile


class AcceptedRecord(TypedDict, total=False):
    seed_id: str
    model_name: str
    class_id: int
    level: str
    tier: str
    route_mode: RouteMode
    extension_family: Optional[str]
    question: str
    answer: str
    gate1: Dict[str, Any]
    gate2: Dict[str, Any]
    gate3: Dict[str, Any]


class RejectedRecord(TypedDict, total=False):
    seed_id: str
    model_name: str
    class_id: int
    level: str
    tier: str
    route_mode: Optional[RouteMode]
    extension_family: Optional[str]
    reason: str
    details: Dict[str, Any]


REJECTION_CODES = {
    # normalization / input
    "MISSING_MODEL_NAME",
    "MISSING_QUESTION",
    "MISSING_ANSWER",
    "SEED_JOIN_MISS",
    "DUPLICATE_RECORD_ID",

    # generation / validation
    "SEED_ROUTED_SKIP",
    "BAD_JSON",
    "LOW_EMBED_SIM",
    "LOW_CLASS_MARGIN",
    "WRONG_CLASS_NEIGHBORHOOD",
    "UNKNOWN_CORE_IDENTIFIER",
    "UNAPPROVED_EXTENSION_IDENTIFIER",
    "CROSS_FAMILY_EXTENSION_IDENTIFIER",
    "EXTENSION_REQUIRES_FRAMING",
    "BASE_MODEL_MISREPRESENTATION",
    "INSUFFICIENT_CORE_ANCHOR",
    "DISALLOWED_THEME",
    "EXACT_QA_DUP",
    "QUESTION_TEMPLATE_DUP",
    "NEAR_DUP_QUESTION",
}
