"""Extract unmet information needs and AI product ideas from a situation report.

Makes exactly one LLM call per report (via pipeline.llm), then parses and
validates the JSON response against pipeline.schemas.ExtractionResult.
Responses that are not valid JSON, or that fail schema validation, are
logged and skipped rather than raising - a single bad report should not
stop the run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from .llm import call_llm_json
from .schemas import ExtractionResult

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are helping humanitarian analysts turn field situation reports into \
concrete AI product ideas.

Read the situation report below and do two things:

1. Identify UNMET INFORMATION NEEDS: places where the report shows that responders \
lack information they need (explicitly stated gaps, or gaps strongly implied by what \
is missing from an otherwise detailed report).
2. For EACH need, propose 1 to 3 AI PRODUCT IDEAS that could help close that gap. \
Ideas can use any AI approach - text analysis, forecasting, chatbots/conversational \
tools, computer vision, or geospatial analysis (satellite, mapping, or location data). \
Cover a broad range of approaches across the report as a whole, not just one kind.

IMPORTANT - geospatial coverage: for every need where a geospatial approach is \
plausible (anything involving physical locations, infrastructure, population, \
weather/climate hazards, or movement), at least one of that need's ideas MUST use \
geospatial data (satellite imagery, mapping data, or location-tagged data).

IMPORTANT - datasets: Idea.input_datasets must only name datasets from this exact \
catalogue list. Never invent a dataset name or assume a dataset exists that is not \
in this list. If an idea genuinely needs data that is not in the catalogue, name \
that data source in other_data_needed instead and leave it out of input_datasets.

Catalogue datasets (use these exact names in input_datasets):
{catalogue_names}

For novelty, compare each idea to this list of tools that already exist in the \
humanitarian sector. Two examples of the product pattern to compare against: DISHA's \
AI building damage assessment from satellite imagery, and DISHA's population \
displacement insights from mobile network data - both listed below.

Known existing tools:
{known_tools}

For each idea, set novelty_label to "exists" if it essentially duplicates a known \
tool, "extension" if it meaningfully extends one, or "new" if nothing in the known \
tools list is close. Set closest_known_tool to the nearest match (or "none" if \
novelty_label is "new"), and explain your reasoning briefly in novelty_reason.

Return ONLY valid JSON (no markdown fences, no commentary) matching exactly this shape:

{{
  "needs": [
    {{
      "location": "string - place name mentioned in the report",
      "sector": "string - e.g. health, shelter, WASH, food security, protection",
      "hazard": "string - e.g. flood, conflict, drought, earthquake, disease outbreak",
      "information_gap": "string - one sentence describing what information is missing",
      "supporting_quote": "string - a short quote or close paraphrase from the report",
      "explicit": true,
      "ideas": [
        {{
          "title": "string - short product name",
          "problem": "string - the specific problem this idea addresses",
          "ai_approach": "geospatial | text/NLP | forecasting | conversational | computer vision (non-satellite) | other",
          "uses_geospatial_data": true,
          "input_datasets": ["exact catalogue dataset name", "..."],
          "other_data_needed": ["any non-catalogue data source needed, else empty list"],
          "ai_method": "string - e.g. fine-tuned classifier, LLM extraction, time-series forecasting model",
          "output": "string - what the product produces, e.g. a weekly risk map",
          "end_user": "string - who uses this, e.g. cluster coordinators, field teams",
          "novelty_label": "new | extension | exists",
          "closest_known_tool": "string - name from the known tools list, or 'none'",
          "novelty_reason": "string - brief justification"
        }}
      ]
    }}
  ]
}}

If the report has no identifiable needs, return {{"needs": []}}.

SITUATION REPORT:
{report_text}
"""


def build_prompt(report_text: str, catalogue_names: list[str], known_tools: list[dict]) -> str:
    """Assemble the extraction prompt for a single report."""
    tools_block = "\n".join(f"- {t['name']}: {t['description']}" for t in known_tools)
    catalogue_block = "\n".join(f"- {name}" for name in catalogue_names)
    return PROMPT_TEMPLATE.format(
        catalogue_names=catalogue_block,
        known_tools=tools_block,
        report_text=report_text,
    )


def extract_from_report(
    report_text: str,
    catalogue_names: list[str],
    known_tools: list[dict],
    model: str,
    cache_dir: Path,
) -> ExtractionResult | None:
    """Run the one LLM call for a report and validate its JSON output.

    Returns None (after logging why) if the response is not parseable JSON
    or does not match ExtractionResult.
    """
    prompt = build_prompt(report_text, catalogue_names, known_tools)
    raw = call_llm_json(prompt, model=model, cache_dir=cache_dir)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse JSON from LLM response: %s", exc)
        return None

    try:
        return ExtractionResult.model_validate(data)
    except ValidationError as exc:
        logger.warning("LLM response failed schema validation: %s", exc)
        return None
