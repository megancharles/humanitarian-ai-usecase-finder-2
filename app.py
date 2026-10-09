"""Streamlit dashboard for exploring AI use-case ideas from humanitarian situation reports.

Reads only from data/output/ - no API or LLM calls happen here, so this can
be deployed without an API key (e.g. on Streamlit Community Cloud).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "output"

NOVELTY_SCORE = {"new": 1.0, "extension": 0.5, "exists": 0.0}

# Fixed hue order per AI approach, so a given approach is always the same
# color across the bar chart and the scatter plot.
APPROACH_COLORS = {
    "geospatial": "#2a78d6",
    "text/NLP": "#eb6834",
    "forecasting": "#1baf7a",
    "conversational": "#eda100",
    "computer vision (non-satellite)": "#e87ba4",
    "other": "#4a3aa7",
}

# Single-hue sequential ramp (light -> dark) for the choropleth.
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

st.set_page_config(page_title="Humanitarian AI Use-Case Finder", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    needs_path = OUTPUT_DIR / "needs.csv"
    ideas_path = OUTPUT_DIR / "ideas.csv"
    run_info_path = OUTPUT_DIR / "run_info.json"
    if not (needs_path.exists() and ideas_path.exists() and run_info_path.exists()):
        return None
    needs_df = pd.read_csv(needs_path)
    ideas_df = pd.read_csv(ideas_path)
    with open(run_info_path, encoding="utf-8") as f:
        run_info = json.load(f)
    return needs_df, ideas_df, run_info


def main() -> None:
    st.title("Humanitarian AI Use-Case Finder")

    data = load_data()
    if data is None:
        st.info(
            "No pipeline output found yet. Add reports to `data/reports/` and run "
            "`python -m pipeline.run`, then reload this page."
        )
        return

    _needs_df, ideas_df, run_info = data

    st.sidebar.header("Filters")
    countries = sorted(ideas_df["country_iso3"].dropna().unique())
    selected_countries = st.sidebar.multiselect("Country", countries, default=countries)

    approaches = sorted(ideas_df["ai_approach"].dropna().unique())
    selected_approaches = st.sidebar.multiselect("AI approach", approaches, default=approaches)

    geospatial_only = st.sidebar.checkbox("Geospatial only", value=False)

    novelty_labels = sorted(ideas_df["novelty_label"].dropna().unique())
    selected_novelty = st.sidebar.multiselect("Novelty", novelty_labels, default=novelty_labels)

    include_unscored = st.sidebar.checkbox("Include unscored ideas", value=True)
    min_feasibility = st.sidebar.slider("Minimum feasibility", 0.0, 1.0, 0.0, 0.05)

    filtered = ideas_df[
        ideas_df["country_iso3"].isin(selected_countries)
        & ideas_df["ai_approach"].isin(selected_approaches)
        & ideas_df["novelty_label"].isin(selected_novelty)
    ]
    if geospatial_only:
        filtered = filtered[filtered["uses_geospatial_data"] == True]  # noqa: E712

    scored_mask = filtered["feasibility_unscored"] == False  # noqa: E712
    feasible_enough = filtered["feasibility"].fillna(-1) >= min_feasibility
    if include_unscored:
        filtered = filtered[(scored_mask & feasible_enough) | ~scored_mask]
    else:
        filtered = filtered[scored_mask & feasible_enough]

    st.caption(f"{len(filtered)} ideas match the current filters (of {len(ideas_df)} total).")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ideas by country")
        counts_by_country = filtered.groupby("country_iso3").size().reset_index(name="idea_count")
        if counts_by_country.empty:
            st.write("No ideas to show on the map.")
        else:
            fig_map = px.choropleth(
                counts_by_country,
                locations="country_iso3",
                color="idea_count",
                color_continuous_scale=SEQUENTIAL_BLUE,
                labels={"idea_count": "Ideas"},
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_map, width="stretch")

    with col2:
        st.subheader("Ideas by AI approach")
        counts_by_approach = filtered.groupby("ai_approach").size().reset_index(name="idea_count")
        if counts_by_approach.empty:
            st.write("No ideas to show.")
        else:
            fig_bar = px.bar(
                counts_by_approach,
                x="ai_approach",
                y="idea_count",
                color="ai_approach",
                color_discrete_map=APPROACH_COLORS,
                labels={"idea_count": "Ideas", "ai_approach": "AI approach"},
            )
            fig_bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_bar, width="stretch")

    st.subheader("Ideas")
    table = filtered[["title", "country_iso3", "ai_approach", "novelty_label", "feasibility"]].sort_values(
        "feasibility", ascending=False, na_position="last"
    )
    st.dataframe(table, width="stretch")

    st.subheader("Idea details")
    for _, row in filtered.iterrows():
        feasibility_label = "unscored" if row["feasibility_unscored"] else f"{row['feasibility']:.2f}"
        with st.expander(f"{row['title']} — {row['country_iso3']} ({feasibility_label})"):
            st.markdown(f"**Problem:** {row['problem']}")
            st.markdown(f"**Supporting quote:** _{row['supporting_quote']}_")
            st.markdown(f"**Source report:** [{row['report_url']}]({row['report_url']})")
            st.markdown(f"**Closest known tool:** {row['closest_known_tool']}")
            st.markdown(f"**Novelty reason:** {row['novelty_reason']}")

            dataset_scores = json.loads(row["dataset_scores"]) if row["dataset_scores"] else {}
            if dataset_scores:
                st.markdown("**Catalogue datasets (score out of 3):**")
                st.table(pd.DataFrame(dataset_scores.items(), columns=["dataset", "score"]))

            if row["feasibility_warning"]:
                st.warning(row["feasibility_warning"])

            if isinstance(row["other_data_needed"], str) and row["other_data_needed"]:
                st.markdown(f"**Other data needed:** {row['other_data_needed']}")

    st.subheader("Feasibility vs. novelty")
    scatter_df = filtered[filtered["feasibility_unscored"] == False].copy()  # noqa: E712
    if scatter_df.empty:
        st.write("No scored ideas to plot (unscored ideas are excluded here).")
    else:
        scatter_df["novelty_score"] = scatter_df["novelty_label"].map(NOVELTY_SCORE)
        fig_scatter = px.scatter(
            scatter_df,
            x="feasibility",
            y="novelty_score",
            color="ai_approach",
            color_discrete_map=APPROACH_COLORS,
            hover_data=["title", "country_iso3"],
            labels={
                "feasibility": "Feasibility",
                "novelty_score": "Novelty (0=exists, 0.5=extension, 1=new)",
                "ai_approach": "AI approach",
            },
        )
        fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(0,0,0,0.2)")))
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_scatter, width="stretch")

    st.divider()
    st.caption(
        f"Pipeline last run on {run_info['run_date']} across {run_info['report_count']} report(s). "
        "These are hypotheses for discussion, not recommendations."
    )


if __name__ == "__main__":
    main()
