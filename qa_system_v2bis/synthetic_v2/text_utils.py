import re
from typing import List, Optional


MODEL_TAG_RE = re.compile(r"^model:(.+)$", re.IGNORECASE)
LEVEL_TAG_RE = re.compile(r"^level:(.+)$", re.IGNORECASE)


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


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
