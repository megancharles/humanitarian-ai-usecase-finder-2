"""Pipeline entry point: python -m pipeline.run

Loads every report in data/reports/, makes one LLM call per report to
extract needs and ideas, scores each idea's feasibility against the
catalogue, and writes data/output/needs.csv, ideas.csv and run_info.json.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .extract import extract_from_report
from .feasibility import load_catalogue, score_idea

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_report(path: Path) -> tuple[str, str, str]:
    """Parse a report file: line 1 is the ISO3 code, line 2 is the report URL, the rest is text."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("expected ISO3 code on line 1, URL on line 2, then report text")
    iso3 = lines[0].strip()
    url = lines[1].strip()
    text = "\n".join(lines[2:]).strip()
    if not iso3 or not url or not text:
        raise ValueError("ISO3 code, URL and report text must all be non-empty")
    return iso3, url, text


def main() -> None:
    config = load_config()
    paths = config["paths"]
    reports_dir = ROOT / paths["reports_dir"]
    output_dir = ROOT / paths["output_dir"]
    cache_dir = ROOT / paths["cache_dir"]
    catalogue_path = ROOT / paths["catalogue"]
    output_dir.mkdir(parents=True, exist_ok=True)

    catalogue = load_catalogue(catalogue_path)
    catalogue_names = list(catalogue.index)
    known_tools = config["known_tools"]
    model = config["model"]

    report_files = sorted(reports_dir.glob("*.txt"))
    need_rows: list[dict] = []
    idea_rows: list[dict] = []
    processed_reports = 0

    for report_path in report_files:
        try:
            iso3, url, text = load_report(report_path)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", report_path.name, exc)
            continue

        result = extract_from_report(
            report_text=text,
            catalogue_names=catalogue_names,
            known_tools=known_tools,
            model=model,
            cache_dir=cache_dir,
        )
        if result is None:
            logger.warning("Skipping %s: LLM output failed validation", report_path.name)
            continue
        processed_reports += 1

        for need_index, need in enumerate(result.needs):
            need_rows.append(
                {
                    "report_file": report_path.name,
                    "country_iso3": iso3,
                    "report_url": url,
                    "need_index": need_index,
                    "location": need.location,
                    "sector": need.sector,
                    "hazard": need.hazard,
                    "information_gap": need.information_gap,
                    "supporting_quote": need.supporting_quote,
                    "explicit": need.explicit,
                }
            )

            for idea_index, idea in enumerate(need.ideas):
                feas = score_idea(idea.input_datasets, idea.other_data_needed, catalogue)
                idea_rows.append(
                    {
                        "report_file": report_path.name,
                        "country_iso3": iso3,
                        "report_url": url,
                        "need_index": need_index,
                        "idea_index": idea_index,
                        "title": idea.title,
                        "problem": idea.problem,
                        "ai_approach": idea.ai_approach,
                        "uses_geospatial_data": idea.uses_geospatial_data,
                        "input_datasets": ";".join(idea.input_datasets),
                        "other_data_needed": ";".join(idea.other_data_needed),
                        "ai_method": idea.ai_method,
                        "output": idea.output,
                        "end_user": idea.end_user,
                        "novelty_label": idea.novelty_label,
                        "closest_known_tool": idea.closest_known_tool,
                        "novelty_reason": idea.novelty_reason,
                        "feasibility": feas.score,
                        "feasibility_unscored": feas.unscored,
                        "feasibility_warning": feas.warning or "",
                        "dataset_scores": json.dumps(feas.dataset_scores),
                        "supporting_quote": need.supporting_quote,
                    }
                )

    needs_df = pd.DataFrame(need_rows)
    ideas_df = pd.DataFrame(idea_rows)
    needs_df.to_csv(output_dir / "needs.csv", index=False)
    ideas_df.to_csv(output_dir / "ideas.csv", index=False)

    run_info = {
        "run_date": datetime.now(timezone.utc).date().isoformat(),
        "report_count": processed_reports,
        "report_files_found": len(report_files),
        "needs_count": len(need_rows),
        "ideas_count": len(idea_rows),
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    logger.info(
        "Done: %d/%d reports processed, %d needs, %d ideas written to %s",
        processed_reports,
        len(report_files),
        len(need_rows),
        len(idea_rows),
        output_dir,
    )


if __name__ == "__main__":
    main()
