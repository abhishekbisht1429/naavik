"""Summaries of finished sessions, to compare hyperparameters.

`discover_sessions` finds the sessions under one or more directories (local
`sessions_dir`s, HPC batches, `hpc fetch` results) and reads their
TensorBoard scalars and configs; `summarize_sessions` compares them (the
hyperparameters that differ, each metric per session, sessions that differ only in
their seed grouped into mean ± std); `render_markdown` / `render_csv` /
`render_json` write the result. From the shell:
`python -m naavik.reporting summarize ROOT [ROOT ...]`.
"""

from naavik.reporting.render import render_csv, render_json, render_markdown
from naavik.reporting.sessions import Attempt, Discovery, ReportError, SessionRecord, discover_sessions
from naavik.reporting.summary import (
    DEFAULT_IGNORED_KEYS,
    DEFAULT_SEED_KEYS,
    Column,
    GroupSummary,
    Metric,
    MetricSummary,
    SessionSummary,
    Stats,
    Summary,
    summarize_sessions,
)

__all__ = [
    "DEFAULT_IGNORED_KEYS",
    "DEFAULT_SEED_KEYS",
    "Attempt",
    "Column",
    "Discovery",
    "GroupSummary",
    "Metric",
    "MetricSummary",
    "ReportError",
    "RunRecord",
    "RunSummary",
    "Stats",
    "Summary",
    "discover_sessions",
    "render_csv",
    "render_json",
    "render_markdown",
    "summarize_sessions",
]
