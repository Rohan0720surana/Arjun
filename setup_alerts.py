#!/usr/bin/env python3
"""
ARJUN — connect Telegram and Claude.

    python setup_alerts.py

Walks through the two keys that turn on push alerts and written answers, and
checks each one against the live service before writing it down, so you find
out here rather than on stage.

Nothing you type is echoed to the screen, and nothing is printed back
afterwards: the tokens go straight into ``.env`` and the only thing this
script ever shows you is the last four characters, so you can tell which key
is in there without the whole thing ending up in a screenshot or a terminal
scrollback.

Re-run it whenever you like. It edits the keys it manages and leaves every
other line of ``.env`` exactly where it was.
"""

from __future__ import annotations

import getpass
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"

GOLD = "\033[33m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
OFF = "\033[0m"


def supports_colour() -> bool:
    return sys.stdout.isatty()


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{OFF}" if supports_colour() else text


def rule(title: str) -> None:
    print()
    print(c(f"── {title} " + "─" * max(4, 58 - len(title)), GOLD))


def tail(secret: str) -> str:
    """Show only enough to recognise a key, never enough to use it."""
    return f"…{secret[-4:]}" if len(secret) > 4 else "…"


# --------------------------------------------------------------------------- #
# .env editing
# --------------------------------------------------------------------------- #

def read_env() -> str:
    if ENV.exists():
        return ENV.read_text(encoding="utf-8")
    if EXAMPLE.exists():
        print(c("  No .env yet — starting from .env.example.", DIM))
        return EXAMPLE.read_text(encoding="utf-8")
    return ""


def current(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def set_key(text: str, key: str, value: str) -> str:
    """Replace the key in place, or append it if it is not there at all."""
    line = f"{key}={value}"
    pattern = rf"^{re.escape(key)}=.*$"
    if re.search(pattern, text, re.MULTILINE):
        return re.sub(pattern, line, text, count=1, flags=re.MULTILINE)
    sep = "" if text.endswith("\n") or not text else "\n"
    return f"{text}{sep}{line}\n"


def write_env(text: str) -> None:
    ENV.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def get_json(url: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, data: dict, headers: dict | None = None,
              timeout: float = 30.0) -> tuple[int, dict]:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"error": raw[:300]}


def ask_secret(prompt: str) -> str:
    """Read a secret without echoing it."""
    try:
        return getpass.getpass(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #

def setup_telegram(text: str) -> str:
    rule("Telegram — push alerts to your phone")
    print("""
  This is the channel that matters for the demo: when a rule fires, the alert
  lands on a phone with a skeleton image attached. It is free and takes about
  two minutes.

  On your phone:

    1. Open Telegram and search for  @BotFather
    2. Send it:  /newbot
    3. Give the bot any name, then a username ending in "bot"
    4. BotFather replies with a token that looks like
         8123456789:AAH1a2B3c4D5e6F7g8H9i0J1k2L3m4N5o6P
""")

    existing = current(text, "TELEGRAM_BOT_TOKEN")
    if existing:
        print(c(f"  A token is already saved ({tail(existing)}).", DIM))
        if input("  Keep it and skip to the chat ID? [Y/n] ").strip().lower() \
                not in ("n", "no"):
            token = existing
        else:
            token = ""
    else:
        token = ""

    while not token:
        token = ask_secret("  Paste the bot token (hidden, nothing is echoed): ")
        if not token:
            print(c("  Skipped Telegram. Alerts will stay on the dashboard.",
                    DIM))
            return text
        if not re.match(r"^\d{6,}:[A-Za-z0-9_-]{30,}$", token):
            print(c("  That does not look like a bot token — it should be "
                    "digits, a colon, then a long string.", RED))
            token = ""
            continue
        try:
            info = get_json(f"https://api.telegram.org/bot{token}/getMe")
        except Exception as exc:                            # noqa: BLE001
            print(c(f"  Could not reach Telegram: {exc}", RED))
            token = ""
            continue
        if not info.get("ok"):
            print(c("  Telegram rejected that token. Check you copied all of "
                    "it.", RED))
            token = ""
            continue
        bot = info["result"]
        print(c(f"  Connected to @{bot['username']} ({bot.get('first_name')})",
                GREEN))

    text = set_key(text, "TELEGRAM_BOT_TOKEN", token)

    # ---- chat id -------------------------------------------------------- #
    chat = current(text, "TELEGRAM_CHAT_ID")
    if chat:
        print(c(f"  A chat ID is already saved ({chat}).", DIM))
        if input("  Keep it? [Y/n] ").strip().lower() not in ("n", "no"):
            return text

    try:
        username = get_json(
            f"https://api.telegram.org/bot{token}/getMe")["result"]["username"]
    except Exception:                                       # noqa: BLE001
        username = "your bot"

    print(f"""
  Now tell the bot who you are. On your phone:

    5. Open  https://t.me/{username}
    6. Press Start, or just send it the word  hello
""")
    input(c("  Press Enter here once you have messaged the bot… ", GOLD))

    chat_id, who = discover_chat_id(token)
    if not chat_id:
        print(c("  No message found yet.", RED))
        print("  Make sure you pressed Start in the chat, then enter the ID by")
        print("  hand if you know it (or press Enter to skip).")
        manual = input("  Chat ID: ").strip()
        if not manual:
            print(c("  Token saved, but without a chat ID Telegram stays off.",
                    DIM))
            return text
        chat_id, who = manual, "entered by hand"

    print(c(f"  Found chat {chat_id} ({who})", GREEN))
    return set_key(text, "TELEGRAM_CHAT_ID", chat_id)


def discover_chat_id(token: str, attempts: int = 6) -> tuple[str, str]:
    """Poll getUpdates until the user's own message turns up."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    for n in range(attempts):
        try:
            data = get_json(url)
        except Exception:                                   # noqa: BLE001
            time.sleep(1.5)
            continue
        for update in reversed(data.get("result", [])):
            message = (update.get("message") or update.get("edited_message")
                       or update.get("channel_post") or {})
            chat = message.get("chat") or {}
            if chat.get("id") is not None:
                name = (chat.get("username") or chat.get("first_name")
                        or chat.get("title") or "you")
                return str(chat["id"]), name
        if n < attempts - 1:
            print(c("  …waiting for your message", DIM))
            time.sleep(2.0)
    return "", ""


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #

def setup_claude(text: str) -> str:
    rule("Claude — written answers and incident reports")
    print("""
  Optional. Without it, "Ask the cameras" answers from a local keyword search
  and reports come from a template — nothing breaks, the answers are just
  plainer.

    1. Open  https://console.anthropic.com
    2. Settings → API keys → Create key
    3. Copy it. It starts with  sk-ant-
""")

    existing = current(text, "CLAUDE_API_KEY")
    if existing:
        print(c(f"  A key is already saved ({tail(existing)}).", DIM))
        if input("  Keep it? [Y/n] ").strip().lower() not in ("n", "no"):
            return text

    key = ask_secret("  Paste the API key (hidden, nothing is echoed): ")
    if not key:
        print(c("  Skipped. The local fallbacks stay in use.", DIM))
        return text
    if not key.startswith("sk-ant-"):
        print(c("  That does not look like an Anthropic key (they start with "
                "sk-ant-). Saving it anyway — check it if answers fail.", RED))

    model = current(text, "CLAUDE_MODEL") or "claude-opus-5-5"
    print(c(f"  Checking the key against {model}…", DIM))
    status, body = post_json(
        "https://api.anthropic.com/v1/messages",
        {"model": model, "max_tokens": 16,
         "messages": [{"role": "user", "content": "Reply with the word ready."}]},
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    if status == 200:
        print(c("  Claude answered. Written reports are on.", GREEN))
    elif status in (401, 403):
        print(c("  Anthropic rejected that key. Saving it anyway, but "
                "'Ask the cameras' will fall back to the local search.", RED))
    elif status == 404:
        print(c(f"  The key works, but this account cannot reach {model}.",
                RED))
        print("  Set CLAUDE_MODEL in .env to a model you do have access to.")
    else:
        detail = (body.get("error") or {}).get("message") or body.get("error")
        print(c(f"  Unexpected reply ({status}): {str(detail)[:120]}", RED))

    return set_key(text, "CLAUDE_API_KEY", key)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    print(c("""
  ARJUN — connecting your alert channels

  Nothing you type here is shown on screen or written anywhere except .env.
""", GOLD))

    text = read_env()
    text = setup_telegram(text)
    text = setup_claude(text)
    write_env(text)

    rule("Saved")
    telegram_on = bool(current(text, "TELEGRAM_BOT_TOKEN")
                       and current(text, "TELEGRAM_CHAT_ID"))
    claude_on = bool(current(text, "CLAUDE_API_KEY"))
    print(f"""
  .env updated.

    Telegram   {c('on', GREEN) if telegram_on else c('off', DIM)}
    Claude     {c('on', GREEN) if claude_on else c('off', DIM)}

  Now send yourself a real alert — the same code path a genuine incident
  takes, with a skeleton snapshot attached:

      python run.py --test-alert

  Then start the system as usual:

      python run.py
""")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n  Cancelled. Nothing was written.")
        raise SystemExit(1)
