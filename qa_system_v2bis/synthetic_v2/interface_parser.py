import re
from typing import Dict, List, Any

from .nlogo_parser import strip_netlogo_comments

WIDGET_HEADERS = {
    "GRAPHICS-WINDOW",
    "BUTTON",
    "PLOT",
    "TEXTBOX",
    "CHOOSER",
    "SLIDER",
    "SWITCH",
    "MONITOR",
    "INPUTBOX",
    "OUTPUT",
    "CC-WINDOW",
}

CONTROL_WIDGETS = {"SLIDER", "SWITCH", "CHOOSER", "INPUTBOX"}

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*\??$")
NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

IGNORED_WIDGET_TOKENS = {
    "nil", "horizontal", "vertical", "observer", "turtle", "patch", "link",
    "true", "false"
}


def split_interface_blocks(interface_text: str) -> List[List[str]]:
    lines = [x.rstrip("\n") for x in interface_text.splitlines()]
    blocks = []
    current = []

    for line in lines:
        s = line.strip()
        if s in WIDGET_HEADERS:
            if current:
                blocks.append(current)
            current = [s]
        else:
            if current:
                current.append(line)
    if current:
        blocks.append(current)

    return blocks


def is_identifier_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if NUMBER_RE.match(s):
        return False
    if s.lower() in IGNORED_WIDGET_TOKENS:
        return False
    return bool(IDENT_RE.match(s))


def candidate_identifiers_from_block(block: List[str]) -> List[str]:
    cands = []
    for line in block[1:]:
        s = line.strip()
        if is_identifier_line(s):
            cands.append(s)
    return cands


def choose_control_variable(kind: str, block: List[str]) -> str:
    cands = candidate_identifiers_from_block(block)
    if not cands:
        return ""

    if kind in CONTROL_WIDGETS:
        if len(cands) >= 2:
            return cands[1]
        return cands[0]

    return ""


def extract_widgets_from_interface(interface_text: str) -> Dict[str, Any]:
    blocks = split_interface_blocks(interface_text)

    widgets = []
    by_type = {}

    for block in blocks:
        kind = block[0]
        var_name = choose_control_variable(kind, block)
        if var_name:
            widgets.append(var_name)
            by_type.setdefault(kind, []).append(var_name)

    seen = set()
    out = []
    for w in widgets:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(w)

    by_type_dedup = {}
    for k, vals in by_type.items():
        seen = set()
        tmp = []
        for v in vals:
            lv = v.lower()
            if lv not in seen:
                seen.add(lv)
                tmp.append(v)
        by_type_dedup[k] = tmp

    return {
        "widgets": out,
        "widgets_by_type": by_type_dedup,
        "block_count": len(blocks),
    }


def merge_widget_sets(interface_widgets: List[str], inferred_widgets: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in list(interface_widgets) + list(inferred_widgets):
        lx = x.lower()
        if lx not in seen:
            seen.add(lx)
            out.append(x)
    return sorted(out, key=lambda x: x.lower())


def infer_candidate_widgets_from_code(
    code: str,
    extracted_core: Dict[str, Any],
    unresolved_identifiers: List[str],
    unresolved_counts: Dict[str, int],
) -> Dict[str, Any]:
    candidates = []
    reasons = {}

    def score(tok: str, count: int) -> int:
        s = 0
        if "-" in tok:
            s += 3
        if tok.endswith("?"):
            s += 2
        if count >= 2:
            s += 2
        if any(tok.startswith(prefix) for prefix in [
            "initial-", "num-", "max-", "min-", "show-", "plot-", "vision", "movement"
        ]):
            s += 2
        if tok in {"vision", "visualization"}:
            s += 2
        return s

    for tok in unresolved_identifiers:
        n = unresolved_counts.get(tok, 1)
        sc = score(tok, n)
        if sc >= 3:
            candidates.append(tok)
            reasons[tok] = {
                "count": n,
                "score": sc
            }

    candidates = sorted(set(candidates), key=lambda x: (-reasons[x]["score"], -reasons[x]["count"], x))

    return {
        "widget_candidates": candidates,
        "candidate_reasons": reasons
    }
