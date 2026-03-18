"""Shared profile-querying utilities used by routing, validation, audit, and prompts."""

from typing import Any, Dict, List, Optional, Set

from .text_utils import norm_lower, BACKTICK_RE, IDLIKE_RE, STOPWORDS


def build_core_identifier_set(profile: Dict[str, Any]) -> Set[str]:
    """Extract all core identifiers from a model profile as a normalized set."""
    core = profile.get("core", {})
    out = set()
    for key in ["procedures", "variables", "breeds", "widgets"]:
        for x in core.get(key, []):
            out.add(norm_lower(x))
    return out


def all_extension_identifiers(profile: Dict[str, Any]) -> Set[str]:
    out = set()
    for fam in profile.get("extensions", {}).get("families", []):
        for x in fam.get("identifiers", []):
            out.add(norm_lower(x))
    return out


def all_extension_concepts(profile: Dict[str, Any]) -> Set[str]:
    out = set()
    for fam in profile.get("extensions", {}).get("families", []):
        for x in fam.get("concepts", []):
            out.add(norm_lower(x))
    return out


def get_family(profile: Dict[str, Any], family_name: Optional[str]) -> Dict[str, Any]:
    if not family_name:
        return {}
    for fam in profile.get("extensions", {}).get("families", []):
        if fam.get("name") == family_name:
            return fam
    return {}


def family_identifier_set(profile: Dict[str, Any], family_name: Optional[str]) -> Set[str]:
    fam = get_family(profile, family_name)
    return {norm_lower(x) for x in fam.get("identifiers", [])}


def family_concept_set(profile: Dict[str, Any], family_name: Optional[str]) -> Set[str]:
    fam = get_family(profile, family_name)
    return {norm_lower(x) for x in fam.get("concepts", [])}


def get_framing_cues(profile: Dict[str, Any]) -> Set[str]:
    return {norm_lower(x) for x in profile.get("extensions", {}).get("framing_cues", [])}


def get_disallowed_themes(profile: Dict[str, Any]) -> Set[str]:
    return {norm_lower(x) for x in profile.get("extensions", {}).get("disallowed_unanchored_themes", [])}


def summary_keywords(summary: str) -> Set[str]:
    out = set()
    for tok in IDLIKE_RE.findall(summary or ""):
        nt = norm_lower(tok)
        if nt not in STOPWORDS and len(nt) >= 5:
            out.add(nt)
    return out


def extract_backticked_identifiers(text: str) -> List[str]:
    return [norm_lower(x) for x in BACKTICK_RE.findall(text or "")]


def find_present_phrases(text: str, phrases: Set[str]) -> List[str]:
    t = norm_lower(text)
    return sorted({p for p in phrases if p and p in t})
