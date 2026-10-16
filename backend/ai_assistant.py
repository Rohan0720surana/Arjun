"""
Claude: ask your cameras, and write the incident report.

Two jobs, both of which take **structured events and nothing else**. No frames,
no crops, no snapshots, no track histories are ever sent to an external
service. What leaves this machine, when a key is configured, is a list of rows
that look like this::

    {"id": 41, "ts": "18:42:07", "camera": "Main Corridor", "type": "fall",
     "severity": "critical", "zone": "West stairwell",
     "message": "Person #14 dropped 31% of their height ...",
     "meta": {"drop_body_heights": 0.31, "down_seconds": 2.4}}

That is the entire payload. It is worth being blunt about why: a privacy-first
system that quietly posted camera footage to an API would be a lie, and a
reviewer should be able to confirm the claim by reading one function,
:func:`_event_rows`, which is the only place an outbound payload is built.

**Both features work without a key.** Without ``CLAUDE_API_KEY`` the question
box falls back to a local keyword-and-filter search over SQLite, and the report
button renders the same facts from a template. The answers are plainer. Nothing
is disabled, and the dashboard says which engine answered.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import config
from backend import events_db

_CLIENT: Any = None
_CLIENT_ERROR = ""


@dataclass
class Answer:
    """What came back, and who produced it."""

    text: str
    engine: str                       # "claude" | "local" | "error"
    model: str = ""
    events_used: int = 0
    elapsed_ms: int = 0
    note: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.text, "engine": self.engine, "model": self.model,
            "events_used": self.events_used, "elapsed_ms": self.elapsed_ms,
            "note": self.note, "sources": self.sources,
        }


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #

def available() -> bool:
    return bool(config.CLAUDE_API_KEY)


def _client():
    """Build the Anthropic client once, lazily."""
    global _CLIENT, _CLIENT_ERROR
    if _CLIENT is not None or _CLIENT_ERROR:
        return _CLIENT
    try:
        import anthropic
        _CLIENT = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
    except Exception as exc:                                # noqa: BLE001
        _CLIENT_ERROR = f"{type(exc).__name__}: {exc}"
        _CLIENT = None
    return _CLIENT


def status() -> dict[str, Any]:
    return {
        "configured": available(),
        "model": config.CLAUDE_MODEL,
        "error": _CLIENT_ERROR,
        "fallback": "local keyword search over the event database",
        "sends": "structured event rows only — never images or video",
    }


# --------------------------------------------------------------------------- #
# The only place an outbound payload is built
# --------------------------------------------------------------------------- #

def _event_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce database rows to the minimum an answer needs.

    Deliberately explicit rather than a dict comprehension over everything:
    anything not named here cannot leave the building, and adding a field is a
    decision somebody has to write down.
    """
    rows: list[dict[str, Any]] = []
    for e in events:
        meta = e.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        rows.append({
            "id": e.get("id"),
            "time": e.get("iso"),
            "camera": e.get("camera"),
            "type": e.get("type"),
            "severity": e.get("severity"),
            "zone": e.get("zone_name"),
            "message": e.get("message"),
            "acknowledged": bool(e.get("acknowledged")),
            "has_clip": bool(e.get("clip_path")),
            "simulated": bool(e.get("simulated")),
            # The rule's own numbers, which is what makes an answer specific
            # rather than a paraphrase of the message.
            "evidence": {k: v for k, v in meta.items()
                         if isinstance(v, (int, float, str, bool))},
        })
    return rows


def _system_prompt() -> str:
    return (
        f"You are the duty analyst for {config.APP_NAME}, a privacy-first "
        "campus safety system at an engineering college.\n\n"
        "The system never stores continuous footage and never identifies "
        "anyone. People exist only as temporary track numbers that are "
        "forgotten within seconds, so you cannot say who anybody is, and must "
        "not speculate about identity, gender, appearance or intent.\n\n"
        "You will be given recent incident records as JSON. Answer only from "
        "those records. If the records do not contain the answer, say exactly "
        "what is missing rather than guessing.\n\n"
        "Write for a security guard reading a phone at two in the morning: "
        "plain sentences, the specific camera, zone and time, the numbers that "
        "triggered the rule when they matter, and the action to take. Be "
        "brief. No preamble, no bullet-point padding, no restating the "
        "question. Where an event is marked simulated, say so."
    )


# --------------------------------------------------------------------------- #
# Ask your cameras
# --------------------------------------------------------------------------- #

def ask(question: str, limit: int | None = None,
        camera: str | None = None) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer("Ask me something about the last few hours.", "local")

    started = time.perf_counter()
    window = _window_from_question(question)
    events = events_db.list_events(
        limit=limit or config.CLAUDE_EVENT_LIMIT,
        camera=camera, since=window,
    )

    if not available():
        answer = _local_search(question, events)
        answer.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return answer

    client = _client()
    if client is None:
        answer = _local_search(question, events)
        answer.note = f"Claude unavailable ({_CLIENT_ERROR}); used local search."
        answer.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return answer

    rows = _event_rows(events)
    import json
    payload = json.dumps(rows, ensure_ascii=False, default=str)

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.CLAUDE_MAX_TOKENS,
            system=_system_prompt(),
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": (
                    f"Incident records (most recent first), as JSON:\n\n"
                    f"{payload}\n\n"
                    f"Local time now is "
                    f"{time.strftime('%A %d %B %Y, %H:%M')}.\n\n"
                    f"Question: {question}"
                ),
            }],
        )
    except Exception as exc:                                # noqa: BLE001
        answer = _local_search(question, events)
        answer.note = (f"Claude call failed ({type(exc).__name__}); "
                       f"answered from the local index instead.")
        answer.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return answer

    if getattr(response, "stop_reason", "") == "refusal":
        answer = _local_search(question, events)
        answer.note = "Claude declined that question; used local search."
        answer.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return answer

    text = "".join(b.text for b in response.content
                   if getattr(b, "type", "") == "text").strip()
    return Answer(
        text=text or "No answer came back. Try rephrasing the question.",
        engine="claude", model=config.CLAUDE_MODEL, events_used=len(rows),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        sources=[{"id": r["id"], "type": r["type"], "camera": r["camera"],
                  "time": r["time"]} for r in rows[:8]],
    )


# --------------------------------------------------------------------------- #
# Incident report
# --------------------------------------------------------------------------- #

def incident_report(event_id: int) -> Answer:
    started = time.perf_counter()
    event = events_db.get_event(event_id)
    if event is None:
        return Answer(f"No incident {event_id} in the database.", "error")

    related = [e for e in events_db.related_events(event_id, window_s=240.0)
               if e["id"] != event_id]

    if not available() or _client() is None:
        answer = _template_report(event, related)
        answer.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return answer

    import json
    body = json.dumps({
        "incident": _event_rows([event])[0],
        "surrounding_events": _event_rows(related),
    }, ensure_ascii=False, default=str)

    spec = config.alert_spec(event["type"])
    try:
        response = _client().messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.CLAUDE_MAX_TOKENS,
            system=_system_prompt(),
            output_config={"effort": "medium"},
            messages=[{
                "role": "user",
                "content": (
                    "Write a formal incident report in Markdown from the "
                    "records below. Use exactly these sections:\n\n"
                    "# Incident report\n"
                    "**Reference**, **Type**, **Camera**, **Zone**, **Time**, "
                    "**Severity** as a short list\n"
                    "## What the system observed\n"
                    "## Why the rule fired\n"
                    "(quote the actual measurements and the thresholds)\n"
                    "## Timeline\n"
                    "(the surrounding events, in order, with times)\n"
                    "## Recommended action\n"
                    "## Privacy note\n"
                    "(state what evidence exists: an anonymous skeleton "
                    "snapshot, and an incident clip only if one was saved; "
                    "no continuous footage and no identification)\n\n"
                    f"The rule that fired is '{spec.title}': {spec.detail}\n\n"
                    f"Records:\n{body}"
                ),
            }],
        )
    except Exception as exc:                                # noqa: BLE001
        answer = _template_report(event, related)
        answer.note = (f"Claude call failed ({type(exc).__name__}); "
                       f"generated the report locally.")
        answer.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return answer

    text = "".join(b.text for b in response.content
                   if getattr(b, "type", "") == "text").strip()
    if not text:
        answer = _template_report(event, related)
        answer.note = "Empty response; generated the report locally."
        return answer
    return Answer(text=text, engine="claude", model=config.CLAUDE_MODEL,
                  events_used=1 + len(related),
                  elapsed_ms=int((time.perf_counter() - started) * 1000))


# --------------------------------------------------------------------------- #
# The offline half
# --------------------------------------------------------------------------- #

#: Words people actually type, mapped to the rule they mean.
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fall": ("fall", "fell", "fallen", "collapse", "collapsed", "injured",
             "injury", "down"),
    "sos": ("sos", "signal", "help", "distress", "gesture"),
    "abandoned": ("bag", "backpack", "luggage", "suitcase", "unattended",
                  "abandoned", "handbag"),
    "loiter": ("loiter", "loitering", "waiting", "hanging", "lingering"),
    "restricted": ("restricted", "trespass", "forbidden", "unauthorised",
                   "unauthorized", "off-limits"),
    "crowd_warning": ("crowd", "crowded", "busy", "queue", "congestion"),
    "crowd_critical": ("crowd", "crush", "dangerous", "overcrowded"),
    "panic": ("panic", "running", "ran", "stampede", "rush", "fleeing"),
    "exit_blocked": ("exit", "fire", "blocked", "blocking", "doorway",
                     "stairwell"),
    "after_hours": ("night", "after hours", "late", "closed", "curfew"),
    "lights_on": ("lights", "energy", "power", "empty", "waste"),
    "exam_phone": ("phone", "exam", "cheating", "mobile", "invigilator"),
    "exam_gaze": ("looking", "head", "turned", "gaze", "exam"),
    "line_cross": ("entry", "exit", "count", "inside", "occupancy", "how many"),
}

_SEVERITY_WORDS = {"critical": ("critical", "serious", "emergency", "urgent"),
                   "warning": ("warning", "warn"),
                   "info": ("info", "minor", "informational")}


def _window_from_question(question: str) -> float:
    """Pull a time window out of the words, defaulting to today."""
    q = question.lower()
    now = time.time()
    m = re.search(r"last\s+(\d+)\s*(minute|min|hour|hr|day)", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        per = 60 if unit.startswith("min") else (3600 if unit.startswith(("hour", "hr")) else 86400)
        return now - n * per
    if "week" in q:
        return now - 7 * 86400
    if "yesterday" in q:
        return now - 2 * 86400
    if "month" in q:
        return now - 30 * 86400
    hour = re.search(r"after\s+(\d{1,2})\s*(am|pm)", q)
    if hour:
        h = int(hour.group(1)) % 12 + (12 if hour.group(2) == "pm" else 0)
        lt = time.localtime()
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, 0, 0,
                            lt.tm_wday, lt.tm_yday, lt.tm_isdst))
    return events_db.today_start()


def _local_search(question: str, events: list[dict[str, Any]]) -> Answer:
    """A keyword search that writes its findings as a few plain sentences.

    Not a language model and not pretending to be one. It matches the words
    people actually use against the rule names, filters, counts, and reports.
    """
    q = question.lower()
    wanted_types = {etype for etype, words in _KEYWORDS.items()
                    if any(w in q for w in words)}
    wanted_sev = {sev for sev, words in _SEVERITY_WORDS.items()
                  if any(w in q for w in words)}
    cameras = {e["camera"] for e in events}
    wanted_cams = {c for c in cameras if c.lower() in q}

    hits = events
    if wanted_types:
        hits = [e for e in hits if e["type"] in wanted_types]
    if wanted_sev:
        hits = [e for e in hits if e["severity"] in wanted_sev]
    if wanted_cams:
        hits = [e for e in hits if e["camera"] in wanted_cams]
    if not wanted_types and not wanted_sev and not wanted_cams:
        words = [w for w in re.findall(r"[a-z]{4,}", q)
                 if w not in {"what", "when", "where", "there", "have", "been",
                              "show", "tell", "anything", "about", "today",
                              "that", "with", "from", "this", "were", "which"}]
        if words:
            hits = [e for e in events
                    if any(w in (e["message"] + " " + e["type"]).lower()
                           for w in words)]

    if not hits:
        scope = "today" if not wanted_types else \
            "matching " + ", ".join(sorted(wanted_types)).replace("_", " ")
        return Answer(
            f"Nothing {scope}. {len(events)} events are in the window I "
            f"searched. (Offline answer: no Claude key is configured, so this "
            f"is a keyword search over the event database.)",
            engine="local", events_used=len(events),
            note="Set CLAUDE_API_KEY in .env for written answers.")

    by_type: dict[str, int] = {}
    for e in hits:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    worst = min(hits, key=lambda e: -config.severity_rank(e["severity"]))

    parts = [f"{len(hits)} matching "
             f"{'event' if len(hits) == 1 else 'events'}: "
             + ", ".join(f"{n} x {config.alert_spec(t).title.lower()}"
                         for t, n in sorted(by_type.items(),
                                            key=lambda kv: -kv[1])) + "."]
    parts.append(f"Most serious: {config.alert_spec(worst['type']).title} on "
                 f"{worst['camera']} at {worst['clock']}"
                 + (f" ({worst['zone_name']})" if worst["zone_name"] else "")
                 + f" — {worst['message']}")
    recent = hits[:5]
    if len(recent) > 1:
        parts.append("Most recent: " + "; ".join(
            f"{e['clock']} {config.alert_spec(e['type']).title.lower()} on "
            f"{e['camera']}" for e in recent))

    return Answer(
        text="\n\n".join(parts), engine="local", events_used=len(events),
        note="Offline keyword search. Set CLAUDE_API_KEY for written answers.",
        sources=[{"id": e["id"], "type": e["type"], "camera": e["camera"],
                  "time": e["iso"]} for e in hits[:8]],
    )


def _template_report(event: dict[str, Any],
                     related: list[dict[str, Any]]) -> Answer:
    """The report, assembled locally. Same facts, plainer prose."""
    spec = config.alert_spec(event["type"])
    meta = event.get("meta") or {}
    lines = [
        "# Incident report",
        "",
        f"- **Reference:** ARJUN-{event['id']:05d}",
        f"- **Type:** {spec.title}",
        f"- **Camera:** {event['camera']}"
        + ("  _(simulated feed)_" if event.get("simulated") else ""),
        f"- **Zone:** {event.get('zone_name') or 'not inside a defined zone'}",
        f"- **Time:** {event['iso'].replace('T', ' ')}",
        f"- **Severity:** {event['severity'].upper()}",
        "",
        "## What the system observed",
        "",
        event["message"],
        "",
        "## Why the rule fired",
        "",
        spec.detail,
        "",
    ]
    if isinstance(meta, dict) and meta:
        lines.append("Measurements recorded at the moment the rule fired:")
        lines.append("")
        for key, value in meta.items():
            if isinstance(value, (int, float, str, bool)):
                lines.append(f"- `{key}`: {value}")
        lines.append("")

    lines += ["## Timeline", ""]
    if related:
        for e in sorted(related, key=lambda r: r["ts"]):
            lines.append(f"- `{e['clock']}` {config.alert_spec(e['type']).title}"
                         f" — {e['message']}")
    else:
        lines.append("- No other events were recorded on this camera within "
                     "four minutes either side.")
    lines += [
        "",
        "## Recommended action",
        "",
        _recommendation(event["type"]),
        "",
        "## Privacy note",
        "",
        "The evidence attached to this incident is an anonymous skeleton "
        "render" + (" and a 15 second incident clip"
                    if event.get("clip_path") else "")
        + ". No continuous footage of this location exists, at any time, and "
          "no person in this record is identified or identifiable: the track "
          "number was assigned seconds before the incident and discarded "
          "seconds after it.",
        "",
        f"_Generated locally by {config.APP_NAME} {config.APP_VERSION} on "
        f"{time.strftime('%d %B %Y at %H:%M')}. Set CLAUDE_API_KEY in .env "
        f"for a written narrative report._",
    ]
    return Answer(text="\n".join(lines), engine="local",
                  events_used=1 + len(related),
                  note="Template report; no Claude key configured.")


_ACTIONS = {
    "fall": "Send the nearest responder to this location now and check whether "
            "the person needs medical help. Review the incident clip before "
            "closing the report.",
    "sos": "Respond discreetly and in person. Do not broadcast this alert, "
           "call out, or approach in a way that identifies the person who "
           "signalled. Assume somebody nearby is the reason for it.",
    "abandoned": "Have someone look at the bag. If the owner cannot be found "
                 "nearby, follow the campus unattended-item procedure.",
    "exit_blocked": "Clear the fire exit immediately and record who was "
                    "responsible for the obstruction.",
    "crowd_critical": "Send staff to manage flow and stop further entry until "
                      "the count falls.",
    "crowd_warning": "Keep the zone under observation; be ready to manage "
                     "entry if the count rises further.",
    "panic": "Look at the live view for this camera now and establish the "
             "cause before dispatching anyone.",
    "restricted": "Check who is in the area and whether they have authority to "
                  "be there.",
    "loiter": "Observe before approaching. Prolonged presence is not by itself "
              "a problem; context decides.",
    "after_hours": "Confirm whether the person has authorisation to be in the "
                   "building at this hour.",
    "lights_on": "No safety action. Pass to facilities as an energy note.",
    "exam_phone": "An invigilator should look at the desk. This is an "
                  "observation for a human to judge, not a finding.",
    "exam_gaze": "No action unless the invigilator sees something themselves. "
                 "Logged for completeness only.",
}


def _recommendation(etype: str) -> str:
    return _ACTIONS.get(etype, "Review the incident and acknowledge it once "
                               "it has been dealt with.")
