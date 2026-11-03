"""Report notifications: SMTP email and a chat webhook (Slack-compatible or plain JSON).

Only the short summary (aggregates, pass/fail) and a dashboard link are sent; never call ids,
agent names or text. Credentials and the webhook URL (it embeds a token) come from the
environment: ``GALE_SMTP_USER``, ``GALE_SMTP_PASSWORD``, ``GALE_REPORT_WEBHOOK_URL``.
"""

from __future__ import annotations

import os
import smtplib
from collections.abc import Callable
from email.message import EmailMessage
from typing import Any

import httpx

from galecore.logging import get_logger
from galecore.pilot import EmailNotify, Notify, WebhookNotify

log = get_logger(__name__)


def send_email(
    cfg: EmailNotify,
    subject: str,
    text: str,
    html_body: str,
    smtp: type[smtplib.SMTP] = smtplib.SMTP,
) -> bool:
    if not cfg.enabled or not cfg.to:
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, cfg.sender, ", ".join(cfg.to)
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    with smtp(cfg.smtp_host, cfg.smtp_port, timeout=30) as s:
        if cfg.starttls:
            s.starttls()
        user, password = os.environ.get("GALE_SMTP_USER"), os.environ.get("GALE_SMTP_PASSWORD")
        if user and password:
            s.login(user, password)
        s.send_message(msg)
    return True


def send_webhook(
    cfg: WebhookNotify,
    text: str,
    payload: dict[str, Any],
    transport: httpx.BaseTransport | None = None,
) -> bool:
    url = os.environ.get("GALE_REPORT_WEBHOOK_URL", "")
    if not cfg.enabled or not url:
        return False
    body: dict[str, Any] = (
        {"text": text}
        if cfg.format == "slack"
        else {
            "week": payload["week"],
            "stage": payload["stage"],
            "headline": payload["headline"],
            "exit": payload["exit"],
        }
    )
    with httpx.Client(transport=transport, timeout=30) as c:
        c.post(url, json=body).raise_for_status()
    return True


def notify(cfg: Notify, payload: dict[str, Any], text: str, html_body: str) -> dict[str, bool]:
    subject = f"GAL-E pilot report {payload['week']} ({payload['stage']})"
    out: dict[str, bool] = {}
    senders: list[tuple[str, Callable[[], bool]]] = [
        ("email", lambda: send_email(cfg.email, subject, text, html_body)),
        ("webhook", lambda: send_webhook(cfg.webhook, text, payload)),
    ]
    for name, fn in senders:
        try:
            out[name] = fn()
        except Exception as e:  # one channel failing must not lose the report
            log.warning("pilot_notify_failed", channel=name, error=type(e).__name__)
            out[name] = False
    return out
