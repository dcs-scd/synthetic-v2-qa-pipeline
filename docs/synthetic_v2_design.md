# Synthetic V2 Design

## Purpose

Synthetic V2 generates NetLogo Q&A examples in two valid regimes:

1. **Core/model-faithful**
2. **Model-anchored extension**

The old source-only invariant system incorrectly rejected many coherent extension examples.
This design treats those as first-class outputs when properly framed.

---

## Routing outcomes

Each seed is routed to one of:

- `core_paraphrase`
- `core_repair`
- `anchored_extension`
- `skip`

### `core_paraphrase`
Use when the seed is already faithful to the original model and only needs paraphrasing.

### `core_repair`
Use when the seed is mostly faithful to the original model but contains unsupported specifics
that should be rewritten into source-grounded form.

### `anchored_extension`
Use when the seed proposes a coherent extension to the original model, such as:
- extra states
- network layers
- media layers
- long-range links
- scaling optimizations
- added analytical metrics

Extension content is valid only if:
- it remains anchored to the original model
- it is framed as an addition/modification
- it does not falsely claim to exist in the original source code

### `skip`
Use when the seed is incoherent, too far from the model, or not worth repairing.

---

## Validation regimes

### Core validation
Allowed:
- source-derived model identifiers
- standard NetLogo/global terms

Forbidden:
- extension identifiers
- unsupported model-specific mechanisms

### Extension validation
Allowed:
- source-derived model identifiers
- approved extension-family identifiers
- standard NetLogo/global terms

Required:
- explicit framing as an extension/addition
- explicit anchoring to the original model

Forbidden:
- extension identifiers falsely presented as original source code
- unapproved cross-family extension identifiers
- off-theme unrelated drift

---

## Data artifacts

### `core_profiles.json`
Source-derived core identifiers per model.

### `extension_profiles.json`
Approved extension families per model.

### `model_profiles_merged.json`
Merged file used by routing, prompting, and validation.

### `seeds_with_text.jsonl`
Seed records with `seed_q` and `seed_a` populated.

### `all_records.jsonl`
Unified normalized records from:
- corpus
- seeds
- optionally accepted synthetic
- optionally rejected synthetic

### `input_consistency_report.json`
Counts and issues found during normalization.

---

## Rejection codes

### Input / normalization
- `MISSING_MODEL_NAME`
- `MISSING_QUESTION`
- `MISSING_ANSWER`
- `SEED_JOIN_MISS`
- `DUPLICATE_RECORD_ID`

### Routing / generation / validation
- `SEED_ROUTED_SKIP`
- `BAD_JSON`
- `LOW_EMBED_SIM`
- `LOW_CLASS_MARGIN`
- `WRONG_CLASS_NEIGHBORHOOD`
- `UNKNOWN_CORE_IDENTIFIER`
- `UNAPPROVED_EXTENSION_IDENTIFIER`
- `CROSS_FAMILY_EXTENSION_IDENTIFIER`
- `EXTENSION_REQUIRES_FRAMING`
- `BASE_MODEL_MISREPRESENTATION`
- `INSUFFICIENT_CORE_ANCHOR`
- `DISALLOWED_THEME`
- `EXACT_QA_DUP`
- `QUESTION_TEMPLATE_DUP`
- `NEAR_DUP_QUESTION`

---

## Normalized record format

All normalized records should contain:

- `record_id`
- `source`
- `source_id`
- `model_name`
- `question`
- `answer`
- optional metadata such as:
  - `class_id`
  - `level`
  - `tier`
  - `seed_type`
  - `seed_id`
  - `global_id`
  - `tags`

---

## Notes

- Model names are normalized to lowercase.
- Seeds should carry `seed_q` and `seed_a` directly after normalization.
- If a seed lacks text, join it to the corpus using `seed_id -> corpus.id`.
- The normalization layer should not silently drop malformed rows; it should report them.
