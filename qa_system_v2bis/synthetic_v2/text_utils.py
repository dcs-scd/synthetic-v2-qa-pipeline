import re
from typing import List, Optional, Set


MODEL_TAG_RE = re.compile(r"^model:(.+)$", re.IGNORECASE)
LEVEL_TAG_RE = re.compile(r"^level:(.+)$", re.IGNORECASE)

# Shared compiled regexes for identifier extraction
_WHITESPACE_RE = re.compile(r"\s+")
BACKTICK_RE = re.compile(r"`([^`]+)`")
IDLIKE_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_\-]*\??\b")

# Canonical stopwords set — extend locally if needed
STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "with",
    "and", "or", "is", "are", "was", "were", "be", "as", "from", "that",
    "this", "these", "those", "it", "its", "their", "them", "what", "how",
    "why", "when", "where", "which", "do", "does", "did", "can", "could",
    "would", "should", "into", "about", "than", "then", "if", "we", "you",
    "your", "our", "they", "he", "she", "i", "my", "me", "also", "not",
    "only", "more", "less", "same", "different", "new", "old", "model",
    "netlogo",
}

# Canonical global allow-list for NetLogo/general terms
GLOBAL_ALLOW: Set[str] = {
    "ticks", "tick", "turtles", "patches", "links", "behaviorspace",
    "monitor", "plot", "reporter", "button", "chooser", "slider",
    "switch", "world", "agent", "agents", "ask", "count", "mean",
    "sum", "min", "max", "distance", "with", "of", "one-of", "n-of",
    "if", "ifelse", "let", "set", "run", "repeat", "parameter",
    "parameters", "metric", "metrics", "experiment", "experiments",
    "variance", "standard", "deviation", "autocorrelation",
    "spatial", "global", "local", "threshold", "ratio", "time", "series",
    "sweep", "sweeps", "validation", "density",
}


def norm_space(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s.strip())


def norm_lower(s: str) -> str:
    """Normalize whitespace, strip, and lowercase. Canonical normalization for identifiers."""
    return _WHITESPACE_RE.sub(" ", str(s).strip().lower())


def normalize_model_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = norm_space(str(value))
    if not value:
        return None
    return value.lower()


def extract_model_from_tags(tags: List[str]) -> Optional[str]:
    for t in tags or []:
        m = MODEL_TAG_RE.match(str(t).strip())
        if m:
            return normalize_model_name(m.group(1))
    return None


def extract_level_from_tags(tags: List[str]) -> Optional[str]:
    for t in tags or []:
        m = LEVEL_TAG_RE.match(str(t).strip())
        if m:
            return norm_space(m.group(1))
    return None


def safe_text(value) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value if value else None


def safe_text_or_empty(value) -> str:
    """Like safe_text but returns '' instead of None. For use in f-strings/prompts."""
    result = safe_text(value)
    return result if result is not None else ""


def safe_filename(name: str) -> str:
    """Strip path separators and special characters for safe use as a filename."""
    name = re.sub(r'[/\\]', '_', name)
    name = re.sub(r'[^\w\-.]', '_', name)
    if not name or name.startswith('.'):
        name = '_' + name
    return name


def normalize_question(q: Optional[str]) -> Optional[str]:
    q = safe_text(q)
    if q is None:
        return None
    return q


def normalize_answer(a: Optional[str]) -> Optional[str]:
    a = safe_text(a)
    if a is None:
        return None
    return a
