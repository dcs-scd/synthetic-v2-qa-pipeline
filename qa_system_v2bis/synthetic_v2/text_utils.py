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
    # Core language / agentsets
    "ticks", "tick", "turtles", "patches", "links", "behaviorspace",
    "agent", "agents", "observer", "breed",
    # Interface widget types
    "monitor", "plot", "reporter", "button", "chooser", "slider",
    "switch", "world",
    # Common primitives / reporters
    "ask", "count", "mean", "sum", "min", "max", "distance", "with",
    "of", "one-of", "n-of", "if", "ifelse", "let", "set", "run",
    "repeat", "foreach", "map", "filter", "sort-on", "sort-by",
    "random", "random-float",
    # Built-in turtle variables
    "color", "heading", "xcor", "ycor", "size", "shape", "who",
    "hidden?", "label", "label-color", "pen-size", "pen-mode",
    # Built-in patch variables
    "pcolor", "plabel", "plabel-color", "pxcor", "pycor",
    # Movement / spatial primitives
    "forward", "fd", "back", "bk", "right", "rt", "left", "lt",
    "face", "facexy", "towards", "towardsxy", "move-to", "setxy",
    "in-radius", "neighbors", "neighbors4", "patch-here", "patch-at",
    "distance", "distancexy", "diffuse",
    # Agentset constructors
    "patch-set", "turtle-set", "link-set",
    "turtles-here", "turtles-on", "other",
    # Creation / lifecycle
    "create-turtles", "crt", "create-ordered-turtles", "cro",
    "hatch", "sprout", "die",
    # Declaration keywords (appear in backticked code snippets)
    "globals", "turtles-own", "patches-own", "links-own",
    "to", "to-report", "end",
    # Math / logic
    "abs", "sqrt", "ln", "exp", "log", "mod", "floor", "ceiling",
    "round", "precision", "remainder",
    "not", "and", "or", "xor", "true", "false",
    # List operations
    "list", "item", "length", "first", "last", "fput", "lput",
    "but-first", "but-last", "member?", "position", "remove",
    "remove-duplicates", "sentence", "word", "substring",
    # Plotting
    "set-current-plot", "set-current-plot-pen", "plot",
    "histogram", "plotxy",
    # Colors
    "red", "green", "blue", "black", "white", "yellow", "brown",
    "orange", "pink", "violet", "cyan", "gray", "grey",
    "scale-color",
    # Statistical / experiment terms
    "parameter", "parameters", "metric", "metrics",
    "experiment", "experiments",
    "variance", "standard", "deviation", "autocorrelation",
    "spatial", "global", "local", "threshold", "ratio",
    "time", "series", "sweep", "sweeps", "validation", "density",
    # Conceptual terms commonly used in model descriptions
    "alignment", "separation", "cohesion",
    "cluster", "chip", "energy", "speed",
    "inheritance", "reproduction", "mutation", "selection",
    "trajectory", "equilibrium", "bifurcation", "tipping",
    # Common widget-label aliases (differ from code variable names)
    "number-of-termites", "number-of-turtles", "number-of-agents",
    "number-of-sheep", "number-of-wolves",
    # Model-specific reporters/variables often referenced in answers
    "gini-index", "gini-coefficient", "lorenz-curve",
    "average-degree", "giant-component-fraction", "giant-component-size",
    "car-ahead-here",
    # NetLogo built-in shapes (used in turtle-shape discussions)
    "circle", "person", "square", "triangle", "arrow", "bug",
    "butterfly", "car", "default", "dot", "fish", "house",
    # Common state-machine / extension conceptual terms
    "chasing", "herding", "patrolling", "cooldown",
    "carrying?", "carrying-chip?",
    # Turtle visibility / drawing primitives
    "hide-turtle", "ht", "show-turtle", "st",
    "stamp", "stamp-erase",
    "pen-up", "pu", "pen-down", "pd", "pen-erase", "pe",
    # Drawing / display primitives
    "clear-drawing", "cd", "clear-all", "ca", "clear-turtles", "ct",
    "clear-patches", "cp", "clear-links",
    "clear-output", "clear-all-plots",
    "display", "no-display", "tick-advance",
    # Inspection / output primitives
    "show", "print", "type", "write", "output-show", "output-print",
    "output-type", "output-write",
    "inspect", "stop", "wait",
    # Link primitives
    "create-link-with", "create-link-to", "create-link-from",
    "create-links-with", "create-links-to", "create-links-from",
    "link-neighbors", "link-with", "my-links", "my-in-links", "my-out-links",
    "in-link-neighbor?", "out-link-neighbor?", "link-neighbor?",
    "both-ends", "end1", "end2", "tie", "untie",
    # Breed-related primitives
    "is-turtle?", "is-patch?", "is-link?", "is-agent?", "is-agentset?",
    "is-breed?",
    # Additional common primitives
    "nobody", "self", "myself", "patch-ahead", "can-move?",
    "max-one-of", "min-one-of", "max-n-of", "min-n-of",
    "any?", "all?", "is-number?", "is-string?", "is-list?",
    "user-input", "user-message", "user-yes-or-no?",
    "file-open", "file-close", "file-read", "file-write", "file-print",
    "reset-ticks", "reset-timer", "timer",
    "import-world", "export-world", "import-pcolors",
    "rgb", "hsb", "extract-hsb", "extract-rgb",
    "approximate-hsb", "approximate-rgb",
    "wrap-color", "shade-of?",
    "random-xcor", "random-ycor", "random-pxcor", "random-pycor",
    "dx", "dy", "uphill", "downhill", "downhill4", "uphill4",
    "diffuse4",
    "layout-spring", "layout-circle", "layout-radial", "layout-tutte",
    "mouse-xcor", "mouse-ycor", "mouse-down?", "mouse-inside?",
    "max-pxcor", "max-pycor", "min-pxcor", "min-pycor",
    "world-width", "world-height",
    "new-seed", "random-seed",
    "reduce", "n-values", "range", "reverse", "sort", "modes",
    "median", "empty?", "remove-item", "replace-item", "sublist",
    "read-from-string", "is-boolean?",
    "carefully", "error", "error-message",

    # === AUTO-ADDED BY ITERATIVE REFINEMENT ===
    "chance-to-tell", "forever", "step",
    "cv", "remaining-immunity--", "standard-deviation",
    "phenotype",
    "prestige", "prestige-weight",
    "off", "on", "random-one-of", "run-at-300", "run-at-500", "spread-chance", "update-plots",
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
    return value.lower().replace(" ", "_")


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
