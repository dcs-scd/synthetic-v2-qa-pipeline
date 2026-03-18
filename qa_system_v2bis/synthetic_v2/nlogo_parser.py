import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Any

SECTION_SEP = "@#$#@#$#@"

# NetLogo-ish identifier
ID = r"[A-Za-z_][A-Za-z0-9_\-]*\??"

# Declarations
BREED_RE = re.compile(rf"\bbreed\s*\[\s*({ID})\s+({ID})\s*\]", re.IGNORECASE)
DIR_LINK_BREED_RE = re.compile(rf"\bdirected-link-breed\s*\[\s*({ID})\s+({ID})\s*\]", re.IGNORECASE)
UNDIR_LINK_BREED_RE = re.compile(rf"\bundirected-link-breed\s*\[\s*({ID})\s+({ID})\s*\]", re.IGNORECASE)

GLOBALS_RE = re.compile(r"\bglobals\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
TURTLES_OWN_RE = re.compile(r"\bturtles-own\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
PATCHES_OWN_RE = re.compile(r"\bpatches-own\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
LINKS_OWN_RE = re.compile(r"\blinks-own\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
BREED_OWN_RE = re.compile(rf"\b({ID})-own\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)

# Procedure headers
TO_HEADER_RE = re.compile(
    rf"(?mi)^\s*to\s+({ID})(?:\s*\[(.*?)\])?"
)
TO_REPORT_HEADER_RE = re.compile(
    rf"(?mi)^\s*to-report\s+({ID})(?:\s*\[(.*?)\])?"
)

LET_RE = re.compile(rf"\blet\s+({ID})\b", re.IGNORECASE)
FOREACH_ARROW_RE = re.compile(rf"->\s*([A-Za-z_][A-Za-z0-9_\-]*\??(?:\s+[A-Za-z_][A-Za-z0-9_\-]*\??)*)")
STRING_RE = re.compile(r'"([^"\n]*)"')
TOKEN_RE = re.compile(ID)

# First-pass NetLogo keywords and common primitives.
NETLOGO_RESERVED = {
    "to", "to-report", "end", "breed", "directed-link-breed", "undirected-link-breed",
    "globals", "turtles-own", "patches-own", "links-own",
    "extensions", "includes", "let", "set", "report", "if", "ifelse", "while",
    "repeat", "foreach", "map", "reduce", "filter", "carefully", "stop", "tick",
    "reset-ticks", "clear-all", "display", "wait", "show", "print", "type", "output-print",
    "ask", "of", "with", "in-radius", "at-points", "one-of", "n-of", "up-to-n-of",
    "min-one-of", "max-one-of", "sort-on", "sort-by", "count", "sum", "mean", "median",
    "max", "min", "random", "random-float", "random-normal", "exp", "ln", "log", "sqrt",
    "sin", "cos", "tan", "asin", "acos", "atan", "abs", "floor", "ceiling", "precision",
    "round", "mod", "word", "sentence", "fput", "lput", "item", "length", "first", "last",
    "butfirst", "butlast", "member?", "remove", "remove-duplicates", "position", "substring",
    "patch", "patches", "turtle", "turtles", "link", "links", "nobody", "self", "myself",
    "other", "patch-here", "patch-ahead", "patch-at", "patch-left-and-ahead", "patch-right-and-ahead",
    "neighbors", "neighbors4", "turtles-here", "other-end", "in-link-neighbors", "out-link-neighbors",
    "link-neighbors", "in-links", "out-links", "my-links", "create-link-with", "create-links-with",
    "create-link-to", "create-links-to", "create-link-from", "create-links-from",
    "create-temporary-plot-pen", "set-current-plot", "set-current-plot-pen", "plot", "histogram",
    "create-turtles", "create-ordered-turtles", "sprout", "hatch", "die",
    "move-to", "fd", "bk", "lt", "rt", "jump", "can-move?", "distance", "distancexy",
    "towards", "towardsxy", "face", "facexy", "heading", "xcor", "ycor", "setxy",
    "stamp", "pen-down", "pen-up", "home", "hide-turtle", "show-turtle",
    "create-custom-turtles", "clear-drawing", "clear-output",
    "subject", "observer", "is-agent?", "is-agentset?", "is-boolean?", "is-list?",
    "is-number?", "is-string?", "is-turtle?", "is-patch?", "is-link?",
    "new-seed", "random-seed", "behaviorspace-run-number", "ticks",
    "user-message", "user-input", "mouse-xcor", "mouse-ycor", "mouse-down?",
    "true", "false", "not", "and", "or", "xor",
    "who", "breed", "shape", "size", "label", "label-color", "color", "pcolor",
    "plabel", "plabel-color", "hidden?", "pen-size", "pen-mode",
    "countdown", "timer", "reset-timer",
    "black", "gray", "grey", "white", "red", "orange", "brown", "yellow", "green",
    "lime", "turquoise", "cyan", "sky", "blue", "violet", "magenta", "pink",
    "subject", "world-width", "world-height", "min-pxcor", "max-pxcor", "min-pycor", "max-pycor",
}

IGNORE_IN_STRING_LITERALS = {
    "horizontal", "vertical", "n/a", "nil"
}


def split_nlogo_sections(text: str) -> Dict[str, Any]:
    parts = text.split(SECTION_SEP)
    code = parts[0] if len(parts) > 0 else ""
    interface = parts[1] if len(parts) > 1 else ""
    info = parts[2] if len(parts) > 2 else ""
    extra = parts[3:] if len(parts) > 3 else []
    return {
        "code": code,
        "interface": interface,
        "info": info,
        "extra": extra,
    }


def strip_netlogo_comments(code: str) -> str:
    out = []
    in_string = False
    i = 0
    while i < len(code):
        ch = code[i]

        if ch == '"':
            in_string = not in_string
            out.append(ch)
            i += 1
            continue

        if ch == ";" and not in_string:
            while i < len(code) and code[i] != "\n":
                i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def parse_block_vars(block_text: str) -> List[str]:
    return TOKEN_RE.findall(block_text)


def normalize_identifier_list(xs: List[str]) -> List[str]:
    return sorted(set(xs), key=lambda x: x.lower())


def extract_breeds(code_nc: str) -> List[Dict[str, str]]:
    breeds = []
    for rx in [BREED_RE, DIR_LINK_BREED_RE, UNDIR_LINK_BREED_RE]:
        for plural, singular in rx.findall(code_nc):
            breeds.append({"plural": plural, "singular": singular})
    seen = set()
    out = []
    for b in breeds:
        key = (b["plural"].lower(), b["singular"].lower())
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


def extract_declared_vars(code_nc: str) -> Dict[str, Any]:
    globals_ = []
    turtles_own = []
    patches_own = []
    links_own = []
    breed_own = {}

    for block in GLOBALS_RE.findall(code_nc):
        globals_.extend(parse_block_vars(block))

    for block in TURTLES_OWN_RE.findall(code_nc):
        turtles_own.extend(parse_block_vars(block))

    for block in PATCHES_OWN_RE.findall(code_nc):
        patches_own.extend(parse_block_vars(block))

    for block in LINKS_OWN_RE.findall(code_nc):
        links_own.extend(parse_block_vars(block))

    for breed_name, block in BREED_OWN_RE.findall(code_nc):
        lname = breed_name.lower()
        if lname in {"turtles", "patches", "links"}:
            continue
        breed_own.setdefault(breed_name, [])
        breed_own[breed_name].extend(parse_block_vars(block))

    return {
        "globals": normalize_identifier_list(globals_),
        "turtles_own": normalize_identifier_list(turtles_own),
        "patches_own": normalize_identifier_list(patches_own),
        "links_own": normalize_identifier_list(links_own),
        "breed_own": {k: normalize_identifier_list(v) for k, v in breed_own.items()}
    }


def extract_procedure_headers(code_nc: str) -> Dict[str, Any]:
    procedures = []
    procedure_inputs = {}
    procedure_kinds = {}

    for name, inputs in TO_HEADER_RE.findall(code_nc):
        procedures.append(name)
        procedure_kinds[name] = "command"
        procedure_inputs[name] = parse_block_vars(inputs or "")

    for name, inputs in TO_REPORT_HEADER_RE.findall(code_nc):
        procedures.append(name)
        procedure_kinds[name] = "reporter"
        procedure_inputs[name] = parse_block_vars(inputs or "")

    procedures = normalize_identifier_list(procedures)
    return {
        "procedures": procedures,
        "procedure_inputs": {k: normalize_identifier_list(v) for k, v in procedure_inputs.items()},
        "procedure_kinds": procedure_kinds,
    }


def extract_local_variables(code_nc: str) -> Dict[str, Any]:
    let_vars = LET_RE.findall(code_nc)

    lambda_vars = []
    for chunk in FOREACH_ARROW_RE.findall(code_nc):
        lambda_vars.extend(parse_block_vars(chunk))

    return {
        "let_vars": normalize_identifier_list(let_vars),
        "lambda_vars": normalize_identifier_list(lambda_vars),
    }


def extract_string_literals(code_nc: str) -> List[str]:
    vals = []
    for s in STRING_RE.findall(code_nc):
        s2 = s.strip()
        if not s2:
            continue
        if s2.lower() in IGNORE_IN_STRING_LITERALS:
            continue
        vals.append(s2)
    return sorted(set(vals), key=lambda x: x.lower())


def all_identifiers_in_code(code: str) -> List[str]:
    code_nc = strip_netlogo_comments(code)
    return normalize_identifier_list(TOKEN_RE.findall(code_nc))


def breed_generated_primitives(breeds: List[Dict[str, str]]) -> Set[str]:
    out = set()
    for b in breeds:
        plural = b["plural"]
        singular = b["singular"]

        out.add(f"create-{plural}")
        out.add(f"{plural}-on")
        out.add(f"{plural}-here")
        out.add(f"is-{singular}?")
        out.add(f"{singular}-set")
    return {x.lower() for x in out}


def infer_unresolved_identifiers(
    code: str,
    extracted_core: Dict[str, Any]
) -> Dict[str, Any]:
    code_nc = strip_netlogo_comments(code)
    all_ids = TOKEN_RE.findall(code_nc)
    freq = Counter([x.lower() for x in all_ids])

    declared = set()
    declared |= {x.lower() for x in extracted_core.get("procedures", [])}
    declared |= {x.lower() for x in extracted_core.get("variables", [])}
    declared |= {x.lower() for x in extracted_core.get("breeds", [])}

    raw_parse = extracted_core.get("raw_parse", {})
    for _, vals in raw_parse.get("procedure_inputs", {}).items():
        declared |= {x.lower() for x in vals}
    declared |= {x.lower() for x in raw_parse.get("let_vars", [])}
    declared |= {x.lower() for x in raw_parse.get("lambda_vars", [])}

    breeds_struct = raw_parse.get("breeds_struct", [])
    reserved = {x.lower() for x in NETLOGO_RESERVED}
    reserved |= breed_generated_primitives(breeds_struct)

    unresolved = []
    for tok, n in freq.items():
        if tok in declared:
            continue
        if tok in reserved:
            continue
        if tok.isdigit():
            continue
        unresolved.append((tok, n))

    unresolved.sort(key=lambda x: (-x[1], x[0]))
    return {
        "unresolved_identifiers": [t for t, _ in unresolved],
        "unresolved_identifier_counts": {t: n for t, n in unresolved},
    }


def first_info_paragraph(info: str) -> str:
    if not info:
        return ""

    paragraphs = []
    buf = []
    for line in info.splitlines():
        if line.strip():
            buf.append(line.strip())
        else:
            if buf:
                paragraphs.append(" ".join(buf).strip())
                buf = []
    if buf:
        paragraphs.append(" ".join(buf).strip())

    def looks_like_heading(p: str) -> bool:
        s = p.strip()
        if not s:
            return True
        if s.startswith("#"):
            return True
        if len(s.split()) <= 4 and s.upper() == s:
            return True
        return False

    for p in paragraphs:
        if looks_like_heading(p):
            continue
        if len(p.split()) >= 8:
            return p
    return ""


def fallback_summary(model_name: str, procedures: List[str], variables: List[str], breeds: List[str], widgets: List[str]) -> str:
    parts = []
    if breeds:
        parts.append(f"breeds {', '.join(breeds[:4])}")
    if procedures:
        parts.append(f"procedures such as {', '.join(procedures[:5])}")
    if variables:
        parts.append(f"tracked variables like {', '.join(variables[:5])}")
    if widgets:
        parts.append(f"interface controls such as {', '.join(widgets[:5])}")

    if parts:
        return f"The original {model_name} model includes " + "; ".join(parts) + "."
    return f"The original {model_name} model is a NetLogo model with source-derived identifiers."


def extract_core_from_code(code: str) -> Dict[str, Any]:
    code_nc = strip_netlogo_comments(code)

    breeds_struct = extract_breeds(code_nc)
    decl = extract_declared_vars(code_nc)
    proc = extract_procedure_headers(code_nc)
    locals_ = extract_local_variables(code_nc)
    strings = extract_string_literals(code_nc)

    variables = []
    variables.extend(decl["globals"])
    variables.extend(decl["turtles_own"])
    variables.extend(decl["patches_own"])
    variables.extend(decl["links_own"])
    for _, vals in decl["breed_own"].items():
        variables.extend(vals)

    breed_names = []
    for b in breeds_struct:
        breed_names.append(b["plural"])
        breed_names.append(b["singular"])

    out = {
        "procedures": proc["procedures"],
        "breeds": normalize_identifier_list(breed_names),
        "variables": normalize_identifier_list(variables),
        "raw_parse": {
            "globals": decl["globals"],
            "turtles_own": decl["turtles_own"],
            "patches_own": decl["patches_own"],
            "links_own": decl["links_own"],
            "breed_own": decl["breed_own"],
            "procedure_inputs": proc["procedure_inputs"],
            "procedure_kinds": proc["procedure_kinds"],
            "let_vars": locals_["let_vars"],
            "lambda_vars": locals_["lambda_vars"],
            "string_literals": strings,
            "breeds_struct": breeds_struct,
        }
    }
    return out


def parse_nlogo_code_only(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    sections = split_nlogo_sections(text)
    core = extract_core_from_code(sections["code"])
    return {
        "sections": sections,
        "core_extract": core,
    }
