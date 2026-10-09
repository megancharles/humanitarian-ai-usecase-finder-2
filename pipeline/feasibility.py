"""Score idea feasibility against the dataset catalogue.

Plain Python, no LLM calls. For each idea, feasibility is the minimum of its
input datasets' scores (each dataset's score is the mean of its coverage,
resolution, timeliness and access columns, 0-3), scaled to 0-1.

Ideas that name a dataset not in the catalogue, or that depend on
other_data_needed, are marked unscored (feasibility=None) with a warning
rather than scored as 0 - the catalogue is mostly geospatial, so a
non-geospatial idea shouldn't be penalised just because its data source
isn't in it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCORE_COLUMNS = ["coverage", "resolution", "timeliness", "access"]
MAX_SCORE = 3.0


@dataclass
class FeasibilityResult:
    score: float | None
    unscored: bool
    dataset_scores: dict[str, float] = field(default_factory=dict)
    warning: str | None = None


def load_catalogue(path: Path) -> pd.DataFrame:
    """Load catalogue.csv indexed by dataset name."""
    df = pd.read_csv(path)
    return df.set_index("name")


def _dataset_score(catalogue: pd.DataFrame, dataset_name: str) -> float | None:
    """Mean of a dataset's coverage/resolution/timeliness/access scores, or None if unknown."""
    if dataset_name not in catalogue.index:
        return None
    row = catalogue.loc[dataset_name]
    return round(float(row[SCORE_COLUMNS].mean()), 3)


def score_idea(
    input_datasets: list[str],
    other_data_needed: list[str],
    catalogue: pd.DataFrame,
) -> FeasibilityResult:
    """Feasibility = min(per-dataset mean score) / 3, or unscored with a warning."""
    if other_data_needed:
        return FeasibilityResult(
            score=None,
            unscored=True,
            warning="relies on data outside the catalogue (other_data_needed)",
        )
    if not input_datasets:
        return FeasibilityResult(score=None, unscored=True, warning="no input datasets specified")

    dataset_scores: dict[str, float] = {}
    for name in input_datasets:
        score = _dataset_score(catalogue, name)
        if score is None:
            return FeasibilityResult(
                score=None,
                unscored=True,
                warning=f"unknown dataset not in catalogue: {name}",
            )
        dataset_scores[name] = score

    feasibility = min(dataset_scores.values()) / MAX_SCORE
    return FeasibilityResult(score=round(feasibility, 3), unscored=False, dataset_scores=dataset_scores)
