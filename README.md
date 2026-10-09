# Humanitarian AI Use-Case Finder

A prototype that reads humanitarian situation reports, uses an LLM to find
unmet information needs, suggests AI product ideas (text analysis,
forecasting, chatbots, geospatial analysis, computer vision, or other
approaches) that could address them, scores each idea's feasibility against a
data catalogue, and shows the results in a Streamlit dashboard.

Every run deliberately looks for geospatial ideas (satellite, mapping or
location data) alongside everything else, and the dashboard makes it easy to
filter for them.

Ideas produced by this tool are **hypotheses for discussion, not
recommendations** - they still need human review before anyone acts on them.

## How it works

1. `pipeline/run.py` loads every report in `data/reports/`.
2. For each report, one LLM call (`pipeline/extract.py` + `pipeline/llm.py`)
   extracts unmet information needs and proposes 1-3 AI product ideas per
   need, grounded in the dataset catalogue and a list of existing tools (for
   novelty comparison).
3. `pipeline/feasibility.py` scores each idea's feasibility against
   `data/catalogue.csv`, in plain Python - no LLM involved.
4. Results are written to `data/output/needs.csv`, `data/output/ideas.csv`
   and `data/output/run_info.json`.
5. `app.py` is a Streamlit dashboard that reads only from `data/output/` -
   it never calls the LLM.

## Setup

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Adding reports

Drop `.txt` files into `data/reports/` - see `data/reports/README.md` for
the exact format (ISO3 code on line 1, source URL on line 2, report text
after that). A placeholder `sample_report.txt` is included so you can test
the pipeline before adding real reports.

## Running the pipeline

```bash
python -m pipeline.run
```

This writes `data/output/needs.csv`, `data/output/ideas.csv` and
`data/output/run_info.json`. LLM responses are cached on disk in
`data/cache/` (gitignored) keyed by a hash of the prompt, so re-running on
unchanged reports costs nothing. Delete `data/cache/` to force a fresh run.

## Running the dashboard

```bash
streamlit run app.py
```

The dashboard reads only `data/output/` files and needs no API key.

## Deploying the dashboard to Streamlit Community Cloud

Because `app.py` never calls an LLM, the deployed app needs no secrets:

1. Push this repo to GitHub (see below) with `data/output/` committed -
   that's the data the deployed dashboard will show.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the
   repo, and set the main file to `app.py`.
3. Deploy. No `ANTHROPIC_API_KEY` or other secret is required.
4. To update the live dashboard, re-run `python -m pipeline.run` locally,
   commit the updated `data/output/` files, and push.

## Project layout

```
humanitarian-ai-usecase-finder/
  config.yaml            # model name, known_tools, file paths
  data/
    reports/             # input: .txt situation reports
    catalogue.csv         # dataset catalogue used for feasibility scoring
    output/               # pipeline output, committed so the dashboard can read it
    cache/                 # LLM response cache, gitignored
  pipeline/
    llm.py                # Anthropic SDK wrapper + disk cache
    schemas.py             # Pydantic models for the LLM's JSON output
    extract.py              # one LLM call per report
    feasibility.py           # catalogue-based scoring, plain Python
    run.py                    # entry point: python -m pipeline.run
  app.py                 # Streamlit dashboard
```

See `CLAUDE.md` for the conventions this codebase follows.
