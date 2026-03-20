from typing import Dict, Any, List, Optional

from .support_context import render_support_block


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
    return "You generate high-quality synthetic training data for NetLogo Q&A.\n"


def common_style_block(level: str = "L3") -> str:
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


def render_transform_block(transform_type: Optional[str], mode: str) -> str:
    """
    Phase 9: use transform_type to steer prompt behavior.
    """
    if not transform_type:
        return ""

    transform_map = {
        "lexical_paraphrase": (
            "TRANSFORM OBJECTIVE:\n"
            "- Perform a close semantic paraphrase.\n"
            "- Change wording and sentence structure while preserving the same local meaning."
        ),
        "syntactic_reframe": (
            "TRANSFORM OBJECTIVE:\n"
            "- Reframe the same topic with a different sentence structure or question angle.\n"
            "- Keep the same semantic neighborhood and technical content."
        ),
        "mechanism_focus_shift": (
            "TRANSFORM OBJECTIVE:\n"
            "- Keep the same topic, but foreground the mechanism or causal logic.\n"
            "- Do not change the model or the local semantic target."
        ),
        "measurement_reframe": (
            "TRANSFORM OBJECTIVE:\n"
            "- Keep the same topic, but emphasize what should be measured or tracked.\n"
            "- Preserve the underlying mechanism and model grounding."
        ),
        "experiment_design_reframe": (
            "TRANSFORM OBJECTIVE:\n"
            "- Keep the same topic, but make the answer more explicitly experimental.\n"
            "- Include a concrete model-grounded experiment and prediction."
        ),
        "limitation_reframe": (
            "TRANSFORM OBJECTIVE:\n"
            "- Keep the same topic, but make the limitation or caveat more explicit.\n"
            "- Do not let the answer drift into a different topic."
        ),
        "unsupported_identifier_repair": (
            "TRANSFORM OBJECTIVE:\n"
            "- Repair unsupported identifiers or mechanisms into the nearest source-faithful formulation.\n"
            "- Preserve the educational intent while correcting technical details."
        ),
        "mechanism_grounding_repair": (
            "TRANSFORM OBJECTIVE:\n"
            "- Rewrite unsupported mechanism descriptions into a grounded explanation using the base model.\n"
            "- Keep the same local question intent."
        ),
        "core_operationalization_repair": (
            "TRANSFORM OBJECTIVE:\n"
            "- Preserve the topic, but rewrite the answer into a clearer source-faithful operationalization.\n"
            "- Remove unsupported specifics."
        ),
        "extension_reframing": (
            "TRANSFORM OBJECTIVE:\n"
            "- Keep the same extension idea, but restate it naturally as a coherent extension to the original model.\n"
            "- Be explicit that new identifiers are additions, not existing source code."
        ),
        "extension_operationalization": (
            "TRANSFORM OBJECTIVE:\n"
            "- Keep the same extension idea, but focus on how it would be operationalized in the model.\n"
            "- Explain what would be added and how it would affect the original model."
        ),
        "extension_compare_baseline": (
            "TRANSFORM OBJECTIVE:\n"
            "- Compare the original model against the proposed extension.\n"
            "- Make clear what is base model behavior and what is added by the extension."
        ),
        "extension_experiment_design": (
            "TRANSFORM OBJECTIVE:\n"
            "- Keep the same extension idea, but focus on how to test it experimentally.\n"
            "- Include one concrete experiment comparing the extension against the original model."
        ),
        "extension_limitation_analysis": (
            "TRANSFORM OBJECTIVE:\n"
            "- Keep the same extension idea, but foreground limitations, assumptions, or calibration issues.\n"
            "- Make clear that the extension is proposed, not original source code."
        ),
    }

    if transform_type in transform_map:
        return transform_map[transform_type]

    return (
        "TRANSFORM OBJECTIVE:\n"
        f"- Use transform_type `{transform_type}` as a local stylistic guide.\n"
        "- Preserve semantic locality and technical correctness."
    )


def core_paraphrase_template(
    seed: Dict[str, Any],
    profile: Dict[str, Any],
    transform_type: Optional[str] = None,
    support_record: Optional[Dict[str, Any]] = None,
) -> str:
    transform_block = render_transform_block(transform_type, "core_paraphrase")
    support_block = render_support_block(support_record)

    extra = ""
    if transform_block:
        extra += transform_block + "\n\n"
    if support_block:
        extra += support_block + "\n"

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
- transform_type: {safe_text(transform_type)}

{common_style_block(safe_text(seed.get("level", "L3")))}

{render_core_block(profile)}

{extra}TASK:
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
    profile: Dict[str, Any],
    transform_type: Optional[str] = None,
    support_record: Optional[Dict[str, Any]] = None,
) -> str:
    transform_block = render_transform_block(transform_type, "core_repair")
    support_block = render_support_block(support_record)

    extra = ""
    if transform_block:
        extra += transform_block + "\n\n"
    if support_block:
        extra += support_block + "\n"

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
- transform_type: {safe_text(transform_type)}

{common_style_block(safe_text(seed.get("level", "L3")))}

{render_core_block(profile)}

{extra}TASK:
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
    family_name: str,
    transform_type: Optional[str] = None,
    support_record: Optional[Dict[str, Any]] = None,
) -> str:
    transform_block = render_transform_block(transform_type, "anchored_extension")
    support_block = render_support_block(support_record)

    extra = ""
    if transform_block:
        extra += transform_block + "\n\n"
    if support_block:
        extra += support_block + "\n"

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
- transform_type: {safe_text(transform_type)}

{common_style_block(safe_text(seed.get("level", "L3")))}

{render_core_block(profile)}

{render_extension_block(profile, family_name)}

{extra}TASK:
Create a natural variant of the seed that stays anchored to the original model while allowing a coherent extension in the approved family above.
If you introduce new states, variables, links, or metrics, explicitly frame them as additions or modifications to the original model.

SEED QUESTION:
{safe_text(seed.get("seed_q") or seed.get("question"))}

SEED ANSWER:
{safe_text(seed.get("seed_a") or seed.get("answer"))}

OUTPUT:
Return ONE JSON object only: {{"question": "...", "answer": "..."}}
""".strip()