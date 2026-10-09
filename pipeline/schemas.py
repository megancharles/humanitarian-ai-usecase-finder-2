"""Pydantic models for the LLM extraction output.

One LLM call per report returns JSON matching ExtractionResult. Anything that
fails validation is logged and skipped by pipeline/extract.py - these models
are the enforcement point for that contract.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AIApproach = Literal[
    "geospatial",
    "text/NLP",
    "forecasting",
    "conversational",
    "computer vision (non-satellite)",
    "other",
]

NoveltyLabel = Literal["new", "extension", "exists"]


class Idea(BaseModel):
    """One AI product idea addressing a specific information need."""

    title: str
    problem: str
    ai_approach: AIApproach
    uses_geospatial_data: bool
    input_datasets: list[str] = Field(default_factory=list)
    other_data_needed: list[str] = Field(default_factory=list)
    ai_method: str
    output: str
    end_user: str
    novelty_label: NoveltyLabel
    closest_known_tool: str
    novelty_reason: str


class Need(BaseModel):
    """One unmet information need found in a report, with its candidate ideas."""

    location: str
    sector: str
    hazard: str
    information_gap: str
    supporting_quote: str
    explicit: bool
    ideas: list[Idea] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Full output of the single LLM call made for one report."""

    needs: list[Need] = Field(default_factory=list)
