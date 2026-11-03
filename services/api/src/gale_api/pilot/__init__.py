"""Pilot & continuous-improvement tooling (docs/06 §6, docs/02 §6, ADR-020).

- ``report``: weekly metrics report (precision from reviews, alert acceptance, latency, drift,
  fairness slices, business KPIs, exit criteria) → JSON/Markdown/HTML + email / webhook.
- ``sampler``: weekly active-learning sample → Label Studio tasks.
- ``export``: nightly export of reviewed events (training feedback for the retrain pipelines).
- ``register``: registers a retrained model that passed the promotion gate as a shadow candidate.

Everything here works on aggregates and redacted text only (CLAUDE.md rule 1); nothing logs or
sends transcript text, and reports name no agent.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo


def iso_week(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def parse_week(week: str) -> date:
    """Monday of an ISO week ``YYYY-Www``."""
    try:
        y, w = week.split("-W")
        return date.fromisocalendar(int(y), int(w), 1)
    except ValueError as e:
        raise ValueError(f"not an ISO week (YYYY-Www): {week!r}") from e


def week_bounds(week: str, tz: str) -> tuple[datetime, datetime]:
    """[Monday 00:00, next Monday 00:00) of ``week`` in the call centre's time zone, as UTC."""
    monday = parse_week(week)
    start = datetime(monday.year, monday.month, monday.day, tzinfo=ZoneInfo(tz))
    return start.astimezone(UTC), (start + timedelta(days=7)).astimezone(UTC)


def previous_week(week: str) -> str:
    return iso_week(parse_week(week) - timedelta(days=7))


def last_full_week(now: datetime, tz: str) -> str:
    """The week before the one ``now`` falls in (the weekly job runs on Monday morning)."""
    return previous_week(iso_week(now.astimezone(ZoneInfo(tz)).date()))


def current_week(now: datetime, tz: str) -> str:
    return iso_week(now.astimezone(ZoneInfo(tz)).date())
