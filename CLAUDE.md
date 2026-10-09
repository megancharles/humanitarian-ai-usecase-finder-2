# CLAUDE.md

## Purpose

A prototype that reads humanitarian situation reports, uses an LLM to find
unmet information needs, suggests AI product ideas that could address them
(across approaches: geospatial, text/NLP, forecasting, conversational,
computer vision, other), scores feasibility against a data catalogue, and
shows results in a Streamlit dashboard. Every run deliberately looks for
geospatial ideas (satellite, mapping, or location data) in addition to
everything else. Output ideas are hypotheses for discussion, not
recommendations.

## Folder structure

```
humanitarian-ai-usecase-finder/
  config.yaml           # model name, known_tools list, file paths
  data/
    reports/            # input .txt situation reports (ISO3 / URL / text format)
    catalogue.csv        # dataset catalogue for feasibility scoring
    output/               # pipeline output (needs.csv, ideas.csv, run_info.json) - committed
    cache/                 # LLM response cache, gitignored
  pipeline/
    llm.py                # all provider-specific (Anthropic) code lives here
    schemas.py             # Pydantic models for the LLM's JSON output
    extract.py              # one LLM call per report, parses + validates
    feasibility.py           # catalogue scoring, plain Python, no LLM
    run.py                    # entry point
  app.py                 # Streamlit dashboard, reads only data/output/
```

## Running it

```bash
python -m pipeline.run       # runs the pipeline, writes data/output/
streamlit run app.py          # runs the dashboard
```

## Conventions

- **Provider isolation**: all Anthropic SDK calls live in `pipeline/llm.py`.
  Nothing else imports `anthropic` directly, so the provider can be swapped
  later without touching `extract.py`, `feasibility.py`, or `run.py`.
- **JSON-only LLM outputs, validated with Pydantic**: the extraction prompt
  asks for JSON only (no markdown fences). The response is `json.loads`'d
  and then validated against `pipeline/schemas.py:ExtractionResult`. Any
  response that fails to parse or fails validation is logged and skipped -
  it never raises and never silently produces bad rows.
- **Never invent datasets**: the extraction prompt is given the exact list
  of dataset names from `data/catalogue.csv` and instructed to only use
  those names in `input_datasets`. Anything else goes in
  `other_data_needed`. `pipeline/feasibility.py` treats an idea naming an
  unknown dataset as *unscored* (not a 0) so the mostly-geospatial catalogue
  doesn't unfairly penalize non-geospatial ideas.
- **LLM responses are cached on disk** (`data/cache/`) keyed by a SHA256
  hash of the model + prompt. Re-running the pipeline on unchanged reports
  makes no API calls. Delete `data/cache/` to force a fresh run.
- **The dashboard never calls the LLM or any API.** `app.py` reads only
  `data/output/*.csv` / `*.json`. This is what makes it deployable to
  Streamlit Community Cloud with no secrets.
- **Never commit `.env`** (holds `ANTHROPIC_API_KEY`) or anything under
  `data/cache/` - both are gitignored. `data/output/` *is* committed, since
  the dashboard depends on it being present in the repo.
- One LLM call per report, always. If you need more structured extraction,
  extend the JSON schema rather than adding a second call.
