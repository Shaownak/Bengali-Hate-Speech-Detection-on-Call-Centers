"""Post-call outputs: ``report.json`` and a single-file ``report.html``.

The HTML is self-contained (inline CSS + SVG, no scripts, no external assets): waveform image
per speaker, transcript with event highlights, loudness / shout / escalation timelines, and the
events table. Text is redacted (D-01) and HTML-escaped. Never includes raw audio.
"""

from __future__ import annotations

import html
import json
from typing import Any

import numpy as np

from gale_postcall.pipeline import PostCallResult
from galecore.schemas import Category, Channel, EventStatus

REPORT_VERSION = "1"
COLORS = {
    Category.SLANG_PROFANITY: "#c0392b",
    Category.INSULT: "#d35400",
    Category.THREAT: "#8e44ad",
    Category.SEXUAL_HARASSMENT: "#a5006e",
    Category.HATE_IDENTITY: "#6d214f",
    Category.DISRESPECT_REGISTER: "#b9770e",
    Category.HARSH_TONE: "#7f8c8d",
    Category.SHOUTING: "#e74c3c",
    Category.SARCASM_MOCKERY: "#16a085",
    Category.DISMISSIVE_RUDE: "#2c3e50",
    Category.INTERRUPTION_OVERTALK: "#2980b9",
}
SPEAKER = {"agent": "এজেন্ট / Agent", "customer": "গ্রাহক / Customer", "mixed": "মিশ্র / Mixed"}
W, PAD = 1000, 40


def to_json(r: PostCallResult) -> dict[str, Any]:
    return {
        "report_version": REPORT_VERSION,
        "call_id": r.call_id,
        "source_file": r.source_file,
        "duration_s": r.duration_s,
        "sample_rate_in": r.sample_rate_in,
        "layout": r.layout,
        "models": r.models,
        "llm_status": r.llm_status,
        "notes": r.notes,
        "stages_s": r.stages_s,
        "events": [e.model_dump(mode="json") for e in r.events],
        "scorecard": r.scorecard.model_dump(mode="json") if r.scorecard else None,
        "segments": [s.model_dump(mode="json") for s in r.segments],
        "signals": [s.model_dump(mode="json") for s in r.signals],
        "timelines": r.timelines,
    }


def to_json_text(r: PostCallResult) -> str:
    return json.dumps(to_json(r), ensure_ascii=False, indent=2) + "\n"


def _x(t: float, dur: float) -> float:
    return PAD + (W - 2 * PAD) * t / max(dur, 1e-6)


def _event_bands(r: PostCallResult, speaker: str, y: float, h: float) -> str:
    out = []
    for e in r.events:
        if e.speaker.value != speaker:
            continue
        x0, x1 = _x(e.ts_start, r.duration_s), _x(e.ts_end, r.duration_s)
        op = "0.15" if e.status == EventStatus.DISMISSED_BY_LLM else "0.35"
        out.append(
            f'<rect x="{x0:.1f}" y="{y}" width="{max(2.0, x1 - x0):.1f}" height="{h}" '
            f'fill="{COLORS[e.category]}" opacity="{op}"><title>{html.escape(e.category.value)} '
            f"sev {e.severity} {e.ts_start:.1f}-{e.ts_end:.1f}s</title></rect>"
        )
    return "".join(out)


def _axis(dur: float, y: float) -> str:
    step = 10 if dur <= 120 else 30 if dur <= 600 else 60
    ticks = []
    for t in np.arange(0, dur + 1e-6, step):
        x = _x(float(t), dur)
        ticks.append(
            f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + 4}" stroke="#888"/>'
            f'<text x="{x:.1f}" y="{y + 15}" font-size="10" text-anchor="middle" fill="#666">'
            f"{int(t // 60)}:{int(t % 60):02d}</text>"
        )
    return "".join(ticks)


def waveform_svg(r: PostCallResult) -> str:
    rows = [ch for ch in (Channel.AGENT, Channel.CUSTOMER, Channel.MIXED) if ch in r.audio_16k]
    row_h, h = 70, 70 * len(rows) + 30
    parts = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="waveform">']
    for i, ch in enumerate(rows):
        y0 = i * row_h
        x = r.audio_16k[ch]
        bins = W - 2 * PAD
        n = max(1, len(x) // bins)
        env = np.abs(x[: n * bins]).reshape(-1, n).max(axis=1) if len(x) >= bins else np.abs(x)
        mid = y0 + row_h / 2
        pts = " ".join(f"{PAD + j:.0f},{mid - v * row_h * 0.45:.1f}" for j, v in enumerate(env))
        pts_low = " ".join(
            f"{PAD + j:.0f},{mid + v * row_h * 0.45:.1f}" for j, v in reversed(list(enumerate(env)))
        )
        parts.append(_event_bands(r, ch.value, y0 + 4, row_h - 8))
        parts.append(f'<polygon points="{pts} {pts_low}" fill="#34495e" opacity="0.8"/>')
        parts.append(
            f'<text x="4" y="{mid + 4:.0f}" font-size="11" fill="#333">{"A" if ch == Channel.AGENT else "C" if ch == Channel.CUSTOMER else "M"}</text>'
        )
    parts.append(_axis(r.duration_s, len(rows) * row_h + 2))
    parts.append("</svg>")
    return "".join(parts)


def timeline_svg(r: PostCallResult) -> str:
    h = 260
    parts = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="timelines">']

    def line(
        points: list[list[float]], lo: float, hi: float, y0: float, hh: float, color: str
    ) -> str:
        if not points:
            return ""
        pts = " ".join(
            f"{_x(t, r.duration_s):.1f},{y0 + hh - hh * (min(hi, max(lo, v)) - lo) / (hi - lo):.1f}"
            for t, v in points
        )
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2"/>'

    colors = {"agent": "#1f77b4", "customer": "#d62728"}
    panels = [
        ("loudness", "Loudness dBFS", -70.0, 0.0),
        ("shout_p", "Shout p", 0.0, 1.0),
        ("escalation", "Escalation (30 s)", 0.0, 1.0),
    ]
    for k, (key, label, lo, hi) in enumerate(panels):
        y0 = 8 + k * 78
        parts.append(f'<text x="{PAD}" y="{y0 + 8}" font-size="11" fill="#333">{label}</text>')
        parts.append(
            f'<line x1="{PAD}" y1="{y0 + 66}" x2="{W - PAD}" y2="{y0 + 66}" stroke="#ddd"/>'
        )
        for spk, color in colors.items():
            parts.append(line(r.timelines.get(key, {}).get(spk, []), lo, hi, y0 + 12, 54, color))
    for m in r.timelines.get("overtalk", []):
        x = _x(m["t"], r.duration_s)
        parts.append(
            f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="236" stroke="{colors.get(m["speaker"], "#999")}" '
            f'stroke-dasharray="3,3" opacity="0.7"><title>overtalk ({m["speaker"]})</title></line>'
        )
    parts.append(_axis(r.duration_s, 238))
    parts.append("</svg>")
    return "".join(parts)


def transcript_html(r: PostCallResult) -> str:
    rows = []
    for s in r.segments:
        evs = [
            e
            for e in r.events
            if e.speaker.value == s.channel.value
            and e.ts_start <= s.ts_end
            and s.ts_start <= e.ts_end
        ]
        words = []
        for w in s.words:
            hits = [
                e
                for e in evs
                if e.ts_start - 0.01 <= (w.start + w.end) / 2 <= e.ts_end + 0.01
                and e.category != Category.INTERRUPTION_OVERTALK
            ]
            # the most specific (shortest) event wins: a gali word inside a shouted turn shows as gali
            hit = min(hits, key=lambda e: e.ts_end - e.ts_start, default=None)
            text = html.escape(w.word)
            if hit is not None:
                cls = "dismissed" if hit.status == EventStatus.DISMISSED_BY_LLM else ""
                text = (
                    f'<mark class="{cls}" style="--c:{COLORS[hit.category]}" '
                    f'title="{html.escape(hit.category.value)} sev {hit.severity}">{text}</mark>'
                )
            words.append(text)
        rows.append(
            f'<tr class="{s.channel.value}"><td class="t">{s.ts_start:6.1f}</td>'
            f'<td class="spk">{html.escape(SPEAKER[s.channel.value])}</td>'
            f'<td lang="bn">{" ".join(words)}</td></tr>'
        )
    return "<table class='transcript'>" + "".join(rows) + "</table>"


def events_html(r: PostCallResult) -> str:
    if not r.events:
        return "<p>কোনো ঘটনা নেই / No events.</p>"
    head = (
        "<tr><th>Time</th><th>Speaker</th><th>Category</th><th>Severity</th><th>Probability</th>"
        "<th>Status</th><th>Alert</th><th>Explanation</th></tr>"
    )
    rows = [
        f"<tr><td>{e.ts_start:.1f}–{e.ts_end:.1f} s</td><td>{html.escape(SPEAKER[e.speaker.value])}</td>"
        f'<td><span class="dot" style="background:{COLORS[e.category]}"></span>{html.escape(e.category.value)}</td>'
        f"<td>{e.severity}</td><td>{e.probability:.2f}</td><td>{html.escape(e.status.value)}</td>"
        f"<td>{'⚠' if e.alert else ''}</td><td lang='bn'>{html.escape(e.explanation_bn or '')}</td></tr>"
        for e in r.events
    ]
    return f"<table class='events'>{head}{''.join(rows)}</table>"


def scorecard_html(r: PostCallResult) -> str:
    c = r.scorecard
    if c is None:
        return "<p>Scorecard unavailable (LLM judge offline).</p>"
    a, cu, es = c.agent, c.customer, c.escalation
    tips = "".join(f"<li lang='bn'>{html.escape(t)}</li>" for t in c.coaching_points_bn)
    return (
        "<div class='cards'>"
        f"<div class='card'><h3>এজেন্ট / Agent</h3><p>Professionalism {a.professionalism}/5 · Empathy "
        f"{a.empathy}/5 · De-escalation {a.de_escalation}/5</p><p>Register kept: "
        f"{'yes' if a.respectful_register_maintained else 'no'}</p>"
        f"<p>Violations: {html.escape(', '.join(a.violations) or '—')}</p></div>"
        f"<div class='card'><h3>গ্রাহক / Customer</h3><p>Hostility {cu.hostility}/5 · Abuse toward agent: "
        f"{'yes' if cu.abuse_toward_agent else 'no'}</p><p lang='bn'>{html.escape(cu.main_issue_bn)}</p></div>"
        f"<div class='card'><h3>Escalation</h3><p>{'escalated' if es.escalated else 'calm'} · first: "
        f"{html.escape(es.who_escalated_first)}</p></div></div>"
        f"<p lang='bn'>{html.escape(c.summary_bn)}</p><p>{html.escape(c.summary_en)}</p>"
        f"<ul>{tips}</ul>"
    )


CSS = """
:root{--fg:#1d1d1f;--bg:#fff;--muted:#666;--line:#e5e5e5}
@media (prefers-color-scheme:dark){:root{--fg:#eee;--bg:#161618;--muted:#aaa;--line:#333}}
body{font-family:"Noto Sans Bengali","Hind Siliguri",system-ui,sans-serif;color:var(--fg);background:var(--bg);
     max-width:1080px;margin:0 auto;padding:16px;line-height:1.5}
h1{font-size:1.3rem;margin:.2rem 0} h2{font-size:1.05rem;margin-top:1.6rem;border-bottom:1px solid var(--line)}
.meta{color:var(--muted);font-size:.85rem} .notes{background:#fff4e5;color:#7a4b00;padding:6px 10px;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:.9rem} td,th{padding:4px 6px;border-bottom:1px solid var(--line);
     text-align:left;vertical-align:top} .transcript td.t{color:var(--muted);width:4.5em;font-variant-numeric:tabular-nums}
.transcript td.spk{width:9em;color:var(--muted)} tr.agent td.spk{color:#1f77b4} tr.customer td.spk{color:#d62728}
mark{background:color-mix(in srgb,var(--c) 30%,transparent);color:inherit;border-bottom:2px solid var(--c);border-radius:3px}
mark.dismissed{background:none;border-bottom-style:dotted;opacity:.7}
.dot{display:inline-block;width:.7em;height:.7em;border-radius:50%;margin-right:.4em}
.cards{display:flex;flex-wrap:wrap;gap:12px}.card{flex:1 1 240px;border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.card h3{margin:.2rem 0;font-size:.95rem} svg text{font-family:inherit}
.legend span{margin-right:12px;font-size:.8rem;white-space:nowrap}
"""


def to_html(r: PostCallResult) -> str:
    legend = "".join(
        f'<span><span class="dot" style="background:{c}"></span>{html.escape(k.value)}</span>'
        for k, c in COLORS.items()
    )
    notes = "".join(f"<p class='notes'>{html.escape(n)}</p>" for n in r.notes)
    alerts = sum(e.alert for e in r.events)
    return f"""<!doctype html>
<html lang="bn"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GAL-E call report</title><style>{CSS}</style></head><body>
<h1>GAL-E কল রিপোর্ট / Call report</h1>
<p class="meta">call {html.escape(r.call_id)} · {html.escape(r.source_file)} · {r.duration_s:.1f} s · {r.layout} ·
models: {html.escape(r.models.get("mode", "?"))} · LLM: {r.llm_status} · {len(r.events)} events, {alerts} alerts ·
processed in {sum(r.stages_s.values()):.1f} s. Advisory only: a supervisor reviews every alert.</p>
{notes}
<h2>Waveform & events</h2>{waveform_svg(r)}<div class="legend">{legend}</div>
<h2>Events</h2>{events_html(r)}
<h2>Scorecard</h2>{scorecard_html(r)}
<h2>Timelines</h2>{timeline_svg(r)}
<p class="meta"><span style="color:#1f77b4">■</span> agent <span style="color:#d62728">■</span> customer · dashed: interruptions</p>
<h2>Transcript (redacted)</h2>{transcript_html(r)}
</body></html>
"""
