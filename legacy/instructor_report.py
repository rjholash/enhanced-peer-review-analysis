#!/usr/bin/env python3

#python instructor_report.py "Peer evaluation Winter 2025(1-264).xlsx" \
#                            "Student-File.xlsx" \
#                            -o winter25_peer_report.html



"""
Instructor peer‑evaluation report generator – refactored April 2025
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from jinja2 import Template
import numpy as np
from pandas.api.types import is_numeric_dtype
from jinja2 import Template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ‑ %(levelname)s ‑ %(message)s",
)
LOG = logging.getLogger(__name__)


# ──────────────────────────────── Helpers ──────────────────────────────── #

def clean(series: pd.Series, pad: int | None = None) -> pd.Series:
    """Strip, lowercase; optionally zero‑pad numeric IDs."""
    s = series.astype(str).str.strip().str.lower()
    if pad:
        s = s.str.zfill(pad)
    return s


def read_review(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["email_clean"] = clean(df["Email"])
    df["student_id_clean"] = clean(df["StudentID"])
    return df


def read_roster(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["email_clean"] = clean(df["Email"])
    df["student_id_clean"] = clean(df["OrgDefinedId"])
    # pull last two chars as group #
    df["Group"] = df["Product Review Groups"].astype(str).str[-2:]
    return df


def build_report(review: pd.DataFrame, roster: pd.DataFrame) -> str:
    """Return full HTML report (marks %, SD, and time‑on‑task)."""
    # ── merge on email ───────────────────────────────────────────────
    merged = review.merge(
        roster[["email_clean", "student_id_clean", "Group"]],
        on="email_clean",
        how="left",
        indicator=True,
    )

    # ── submission pattern flags ────────────────────────────────────
    counts      = merged.groupby("email_clean")["Group#_reviewing"].count()
    one_only    = counts[counts == 1].index.tolist()
    three_plus  = counts[counts > 2].index.tolist()
    submitted   = set(merged["email_clean"])
    no_submit   = sorted(set(roster["email_clean"]) - submitted)

    # map email → ID for nicer lists
    email_to_id = roster.set_index("email_clean")["student_id_clean"].to_dict()
    tag = lambda lst: [f"{email_to_id.get(e,'?')} ({e})" for e in lst]

    # ── duration (minutes) ──────────────────────────────────────────
    merged["DurationMin"] = (
        pd.to_datetime(merged["Completion time"]) -
        pd.to_datetime(merged["Start time"])
    ).dt.total_seconds() / 60

    # ── rating columns: numeric only, exclude helper cols ───────────
    rating_cols = [
    "Video_Quality", "Presenters", "Explanation",
    "Mechanism", "Side_Effects", "Bias",
    "Critical_review", "Study_Quality", "Study_participants",
    ]
    merged["OverallPct"] = merged[rating_cols].mean(axis=1) * 10  # 0‑100 %

    # ── group stats: mean % and SD (%), plus count ──────────────────
    group_stats = (
        merged.groupby("Group#_reviewing")["OverallPct"]
        .agg(Mean="mean", SD="std", Reviews="count")
        .round({"Mean": 1, "SD": 1})
        .reset_index(names="Group")
        .sort_values("Group")
    )

    html_group = group_stats.to_html(
        index=False, classes="table", border=0,
        justify="left", float_format="%.1f"
    )

    html_time = (
        merged[["email_clean", "Group#_reviewing", "DurationMin"]]
        .sort_values("email_clean")
        .rename(columns={
            "email_clean": "Email",
            "Group#_reviewing": "Group",
            "DurationMin": "Minutes"
        })
        .to_html(index=False, classes="table", border=0,
                 justify="left", float_format="%.1f")
    )

    # ── Jinja2 template ─────────────────────────────────────────────
    tmpl = Template(
        """
        <html><head><title>Peer‑Evaluation Report</title>
        <style>
          body{font-family:sans-serif;max-width:820px;margin:0 auto}
          .table{border-collapse:collapse;width:100%}
          .table th,.table td{border:1px solid #ddd;padding:6px}
          .table th{background:#f2f2f2}
        </style></head><body>
        <h1>Instructor Report</h1>

        <h2>Group Results (percentage out of 100)</h2>
        {{ group_html | safe }}

        {% if one_only %}
        <h2>Exactly one submission ({{ one_only|length }})</h2>
        <p>{{ one_only|join(", ") }}</p>{% endif %}

        {% if three_plus %}
        <h2>Three + submissions ({{ three_plus|length }})</h2>
        <p>{{ three_plus|join(", ") }}</p>{% endif %}

        {% if no_submit %}
        <h2>No submission received ({{ no_submit|length }})</h2>
        <p>{{ no_submit|join(", ") }}</p>{% endif %}

        <h2>Time on Task (each review, minutes)</h2>
        {{ time_html | safe }}

        </body></html>
        """
    )

    return tmpl.render(
        group_html=html_group,
        time_html=html_time,
        one_only=tag(one_only),
        three_plus=tag(three_plus),
        no_submit=tag(no_submit),
    )
# ──────────────────────────────── Main ──────────────────────────────── #

def cli() -> None:
    ap = argparse.ArgumentParser(
        description="Generate peer‑evaluation instructor report."
    )
    ap.add_argument("reviews",  type=Path, help="Excel file with peer reviews")
    ap.add_argument("roster",   type=Path, help="Excel class‑list file")
    ap.add_argument("-o", "--out", type=Path, default="peer_report.html",
                    help="HTML output path")
    args = ap.parse_args()

    LOG.info("Loading data …")
    review_df  = read_review(args.reviews)
    roster_df  = read_roster(args.roster)

    LOG.info("Building report …")
    html = build_report(review_df, roster_df)

    args.out.write_text(html, encoding="utf‑8")
    LOG.info("Report saved to %s", args.out)


if __name__ == "__main__":
    cli()
