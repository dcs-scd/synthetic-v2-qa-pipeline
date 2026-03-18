from typing import Dict, Any, List, Optional


def safe_text(x: Any) -> str:
    return "" if x is None else str(x)


def comma_join(xs: List[str]) -> str:
    return ", ".join(xs or [])


def bullet_lines(xs: List[str], default: str = "None") -> str:
    xs = xs or []
    if not xs:
        return f"- {default}"
    return "\n".join(f"- {x}" for x in xs)


def render_core_block(profile: Dict[str, Any]) -> str:
    core = profile.get("core", {})
    summary = safe_text(core.get("model_summary", ""))

    procedures = comma_join(core.get("procedures", []))
    variables = comma_join(core.get("variables", []))
    breeds = comma_join(core.get("breeds", []))
    widgets = comma_join(core.get("widgets", []))

    return (
        f"BASE MODEL SUMMARY:\n"
        f"{summary}\n\n"
        f"CORE MODEL INVARIANTS:\n"
        f"- procedures: {procedures}\n"
        f"- variables: {variables}\n"
        f"- breeds: {breeds}\n"
        f"- widgets: {widgets}\n"
    )


def get_family(profile: Dict[str, Any], family_name: Optional[str]) -> Dict[str, Any]:
    if not family_name:
        return {}
    for fam in profile.get("extensions", {}).get("families", []):
        if fam.get("name") == family_name:
            return fam
    return {}


def render_extension_block(profile: Dict[str, Any], family_name: str) -> str:
    fam = get_family(profile, family_name)
    extensions = profile.get("extensions", {})

    fam_concepts = fam.get("concepts", [])
    fam_identifiers = fam.get("identifiers", [])
    fam_rules = fam.get("rules", [])

    general_rules = extensions.get("general_rules", [])
    framing_cues = extensions.get("framing_cues", [])

    return (
        f"ALLOWED EXTENSION FAMILY:\n"
        f"- {family_name}\n\n"
        f"ALLOWED EXTENSION CONCEPTS:\n"
        f"{bullet_lines(fam_concepts)}\n\n"
        f"ALLOWED EXTENSION IDENTIFIERS:\n"
        f"{bullet_lines(fam_identifiers)}\n\n"
        f"FAMILY RULES:\n"
        f"{bullet_lines(fam_rules)}\n\n"
        f"GENERAL EXTENSION RULES:\n"
        f"{bullet_lines(general_rules)}\n\n"
        f"GOOD FRAMING CUES:\n"
        f"{bullet_lines(framing_cues)}\n"
    )


def common_header() -> str:
    return (
        "You generate high-quality synthetic training data for NetLogo Q&A.\n"
    )


def common_style_block(level: str = "L3") -> str:
    # Keep this intentionally simple and stable.
    if level.upper() == "L3":
        style = (
            "Style: L3 = scientific coaching.\n"
            "Include:\n"
            "- exactly one hypothesis\n"
            "- exactly one experiment\n"
            "- exactly one limitation\n"
            "Encourage scientific reasoning."
        )
    elif level.upper() == "L2":
        style = (
            "Style: L2 = clear technical explanation.\n"
            "Include:\n"
            "- one main causal explanation\n"
            "- one concrete test or example\n"
            "- one limitation or caveat."
        )
    else:
        style = (
            f"Style: {level}.\n"
            "Keep the answer pedagogically useful and technically correct."
        )
    return style


def core_paraphrase_template(
    seed: Dict[str, Any],
    profile: Dict[str, Any]
) -> str:
    return f"""{common_header()}
Hard rules:
- Output JSON only: {{"question": "...", "answer": "..."}}
- No markdown fences, no preamble, no commentary.
- Preserve technical correctness.
- Use only source-faithful content from the original model.
- Do not invent model-specific identifiers beyond the core invariant.
- If you use model-specific identifiers, keep them backticked and unchanged.

Priority order:
1. Core model correctness
2. Preserve the seed's educational intent
3. Stay close to the seed
4. Style

CONTEXT:
- model_name: {safe_text(seed.get("model_name"))}
- level: {safe_text(seed.get("level"))}
- tier: {safe_text(seed.get("tier"))}
- mode: core_paraphrase

{common_style_block(safe_text(seed.get("level", "L3")))}

{render_core_block(profile)}

TASK:
Paraphrase the seed into a new natural variant while remaining faithful to the original model.
Do not add extensions, new layers, or invented model-specific mechanisms.

SEED QUESTION:
{safe_text(seed.get("seed_q") or seed.get("question"))}

SEED ANSWER:
{safe_text(seed.get("seed_a") or seed.get("answer"))}

OUTPUT:
Return ONE JSON object only: {{"question": "...", "answer": "..."}}
""".strip()


def core_repair_template(
    seed: Dict[str, Any],
    profile: Dict[str, Any]
) -> str:
    return f"""{common_header()}
Hard rules:
- Output JSON only: {{"question": "...", "answer": "..."}} or {{"skip":"SEED_CONFLICT"}}
- No markdown fences, no preamble, no commentary.
- Preserve technical correctness.
- Use only source-faithful content from the original model.
- If the seed contains unsupported identifiers or mechanisms, replace them with the nearest correct core-model formulation.
- Do not invent model-specific identifiers beyond the core invariant.
- If you use model-specific identifiers, keep them backticked and unchanged.

Priority order:
1. Core model correctness
2. Preserve as much of the seed's educational intent as possible
3. Repair unsupported specificity
4. Style

CONTEXT:
- model_name: {safe_text(seed.get("model_name"))}
- level: {safe_text(seed.get("level"))}
- tier: {safe_text(seed.get("tier"))}
- mode: core_repair

{common_style_block(safe_text(seed.get("level", "L3")))}

{render_core_block(profile)}

TASK:
Repair and rewrite the seed into the nearest technically correct source-faithful Q&A for the original model.
Do not preserve unsupported details literally.
If you cannot produce a technically correct core-model answer, return {{"skip":"SEED_CONFLICT"}}.

SEED QUESTION:
{safe_text(seed.get("seed_q") or seed.get("question"))}

SEED ANSWER:
{safe_text(seed.get("seed_a") or seed.get("answer"))}

OUTPUT:
Return ONE JSON object only:
{{"question": "...", "answer": "..."}}
or
{{"skip":"SEED_CONFLICT"}}
""".strip()


def anchored_extension_template(
    seed: Dict[str, Any],
    profile: Dict[str, Any],
    family_name: str
) -> str:
    return f"""{common_header()}
Hard rules:
- Output JSON only: {{"question": "...", "answer": "..."}}
- No markdown fences, no preamble, no commentary.
- Preserve technical correctness.
- Distinguish clearly between the original model and proposed extensions.
- Core identifiers from the source model may be backticked.
- Extension identifiers may be backticked only if they appear in the approved family below.
- Do not present extension identifiers as if they already exist in the original source code.
- Keep the answer anchored to the original model.

Priority order:
1. Technical correctness
2. Preserve the seed's educational intent
3. Stay anchored to the original model
4. Style

CONTEXT:
- model_name: {safe_text(seed.get("model_name"))}
- level: {safe_text(seed.get("level"))}
- tier: {safe_text(seed.get("tier"))}
- mode: anchored_extension
- extension_family: {family_name}

{common_style_block(safe_text(seed.get("level", "L3")))}

{render_core_block(profile)}

{render_extension_block(profile, family_name)}

TASK:
Create a natural variant of the seed that stays anchored to the original model while allowing a coherent extension in the approved family above.
If you introduce new states, variables, links, or metrics, explicitly frame them as additions or modifications to the original model.

SEED QUESTION:
{safe_text(seed.get("seed_q") or seed.get("question"))}

SEED ANSWER:
{safe_text(seed.get("seed_a") or seed.get("answer"))}

OUTPUT:
Return ONE JSON object only: {{"question": "...", "answer": "..."}}
""".strip()
