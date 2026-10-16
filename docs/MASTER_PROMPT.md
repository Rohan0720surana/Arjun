# Master Brief — ARJUN (Renaissance Buildathon 2026)

> This file is the original build brief for the project, archived verbatim so the
> team (and the judges) can see exactly what was asked for and compare it against
> what was delivered. See `../README.md` for the delivery notes and
> `DELIVERY_VS_BRIEF.md` for a clause-by-clause map.

---

## CONTEXT: HACKATHON BUILD

We are a student team competing in "Renaissance", CSI KJSSE's Buildathon at
Abhiyantriki 2026 (K J Somaiya School of Engineering). The tagline is
"Build things artistically." You are our lead engineer. Build the entire project
below end to end, fully working, in this session. Do not stop at scaffolding or
placeholders. Every feature must actually run.

## HACKATHON RULES YOU MUST FOLLOW

* All code is written fresh now, from scratch. Do not copy any existing project.
* Initialize a git repo immediately and commit after every working step with
  clear, descriptive messages. Never make one giant commit at the end.
* Open-source libraries, pretrained models and APIs are allowed only if credited.
  Maintain a CREDITS section in the README as you add each dependency.

## JUDGING CRITERIA (optimize for all five)

1. Boldness & Originality
2. Theme & Vision
3. Execution & Proof of Concept
4. Technical Depth & Understanding
5. Presentation & Storytelling

## PILLARS WE TARGET

* A New Worldview: surveillance is flawed because it records everyone; we redesign it.
* Challenge the Perspective: safety does not require watching people.
* Human Potential: getting help to someone who has fallen or is in danger, instantly.

## THE PRODUCT: "CAMPUS GUARDIAN"

A privacy-first AI campus safety system. Cameras see people only as anonymous
skeletons with temporary IDs. Raw footage is never stored, except a short clip
saved automatically when an incident occurs, viewable only by authorized staff.

One-line pitch: "Normal CCTV records thousands of innocent students to catch one
incident. Ours watches no one, until someone gives it a reason to."

Second line: "CCTV today is a witness. Campus Guardian is a first responder."

## TECH STACK

* Python 3.10+, OpenCV, NumPy
* Ultralytics YOLOv8n pretrained on COCO, with built-in ByteTrack tracking
* Google MediaPipe: Pose, Hands, Face Mesh
* FastAPI + WebSockets (uvicorn) for the backend
* SQLite for events, zones and settings (no external database server)
* Frontend: a single-page dashboard in plain HTML/CSS/JS served by FastAPI,
  no build step. Use Chart.js from a CDN for charts.
* Alerts: Telegram Bot API (primary), Twilio WhatsApp sandbox (optional), both
  configured in .env and safely disabled if keys are missing
* Anthropic Claude API for natural-language search and incident reports
* python-dotenv for configuration

## ARCHITECTURE

```
campus-guardian/
  run.py              one command starts everything
  config.py           every threshold and setting in one place
  .env.example        CLAUDE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                      TWILIO_*, ADMIN_PIN, CAMERA_SOURCES
  requirements.txt
  README.md
  backend/
    main.py           FastAPI app, routes, WebSocket hub
    camera_manager.py one thread per camera, auto-reconnect
    pipeline.py       per-frame detection and tracking
    privacy.py        skeleton rendering, rolling raw buffer, clip saving
    detectors/
      fall.py
      zones.py        restricted zones, loitering, crowd density
      abandoned.py
      exam.py         phone at desk, head turn
      sos.py          Signal for Help gesture
      after_hours.py
      running.py      sudden running / panic movement
      line_counter.py entry/exit counting across a line
      exit_blocked.py object blocking a fire-exit zone
    alerts.py         cooldowns, Telegram, WhatsApp, dashboard push
    events_db.py      SQLite access
    ai_assistant.py   Claude API: search and reports
  frontend/
    index.html, style.css, app.js
  clips/              incident clips only (gitignored)
  tests/              unit tests for detection rules using synthetic keypoints
```

## CAMERAS

* CAMERA_SOURCES in .env as a list of name=source pairs, e.g.
  "Main Corridor=0, Exam Hall=http://192.168.x.x:4747/video"
* Support laptop webcams (integer index) and phone streams (DroidCam / IP Webcam URLs)
* Each camera runs in its own thread. If a stream drops, show "Reconnecting" on
  its tile and retry. One camera failing must never crash the app.
* Resize frames and skip frames as needed to stay real time on a laptop CPU.

## PRIVACY CORE (build this first, everything depends on it)

* For every frame: YOLOv8n + ByteTrack gives persons and objects with stable IDs.
  Run MediaPipe Pose on each person crop and map keypoints back to frame coordinates.
* Privacy view: dark canvas showing only glowing skeletons, "Person #ID" labels,
  object outlines and zones. No faces, no textures. Make it look beautiful.
* Rolling raw buffer: keep the last 15 seconds of raw frames in memory only,
  continuously overwritten. Never write raw frames to disk unless an alert fires.
* On alert: save 10 seconds before and 5 seconds after as an MP4 in clips/,
  linked to that event. This is the only raw footage that ever exists.
* Privacy toggle per camera: switch a tile to raw view only after entering ADMIN_PIN.
  Log every raw-view access as an audit event.

## FEATURES (build all of them)

1. **FALL DETECTION** — Trigger when: hip centre drops sharply within ~0.5s, AND
   torso angle becomes near horizontal (or person box becomes wider than tall),
   AND the person stays down for more than 2 seconds. Severity: critical.
2. **RESTRICTED ZONES AND LOITERING** — Zones are polygons drawn on the dashboard,
   stored per camera. Types: restricted, loiter, desk, crowd, exit, counting line.
   Use the person's foot point for point-in-polygon. Restricted: alert on entry.
   Loiter: alert if a person stays more than N seconds.
3. **ABANDONED OBJECT** — backpack / handbag / suitcase stationary for more than N
   seconds with no person within a radius -> alert. Track the object; reset if a
   person returns to it.
4. **INCIDENT CLIP UNLOCK** — As described in the privacy core. The alert card
   shows the clip player.
5. **EXAM MODE (per camera toggle)** — For desk zones: "cell phone" detected inside
   a desk for more than 2s -> flag. Head yaw from Face Mesh beyond a threshold for
   more than 3s -> flag. Flag text like "Desk 3: phone visible (0:04)". Present as
   "for invigilator review". Never accuse, never identify by name.
6. **SOS "SIGNAL FOR HELP" GESTURE** — MediaPipe Hands: open palm facing camera with
   thumb tucked into palm, then four fingers fold down over the thumb, within ~3
   seconds -> priority silent alert.
7. **CROWD DENSITY** — Count persons per crowd zone. Warning above a set limit,
   critical above a higher limit. Show a live count on each zone.
8. **RUNNING / PANIC MOVEMENT** — If several tracked people in one camera suddenly
   move much faster than normal in the same direction -> "possible panic" alert.
9. **ENTRY / EXIT COUNTING** — Counting line drawn on screen. Count crossings in each
   direction by track ID. Show "people inside" for that area.
10. **FIRE EXIT BLOCKED** — Any object or stationary person inside an exit zone for
    more than N seconds -> alert.
11. **AFTER-HOURS OCCUPANCY** — Configurable closing time per camera. Any person
    present after hours -> alert, also shown as an energy-waste hint for empty-room
    lights.
12. **ANONYMOUS HEATMAP** — Accumulate foot points over time into a heatmap overlay
    on the privacy view, showing where people move without showing who.
13. **ASK YOUR CAMERAS (Claude API)** — Text box on the dashboard. Send the question
    plus recent events (JSON, no images) to Claude. Show a plain-English answer,
    e.g. "Any falls near the library today?" or "Summarize everything after 6 PM."
14. **AUTO INCIDENT REPORT (Claude API)** — Button on every alert card. Generates a
    formal report: incident type, camera, zone, time, duration, severity, timeline
    of related events, recommended action. Downloadable as a text or Markdown file.
15. **DEMO MODE** — One switch that shortens every timer (abandoned object 10s,
    loitering 8s, etc.) so features can be shown live in minutes. Display a small
    "Demo mode" badge.

## ALERTS

* Every alert: saved to SQLite, pushed live to the dashboard over WebSocket, and
  sent to Telegram (and WhatsApp if configured) with type, camera, zone, time,
  severity and a skeleton snapshot image (never a raw image in the message).
* Cooldown per alert type per track ID so one incident does not spam.
* Severity levels: info, warning, critical. SOS and fall are critical.
* Dashboard plays a subtle sound and flashes the tile on critical alerts.

## DASHBOARD DESIGN ("build things artistically")

* Dark, elegant theme inspired by Renaissance art: deep black/charcoal background,
  warm gold accents, a refined serif display font for headings (Google Fonts),
  clean sans-serif for data. It must look crafted, not like a default admin panel.
* Header: project name, live clock, cameras online, people on campus, alerts today.
* Camera grid: live tiles via WebSocket, each with privacy/raw toggle, exam mode
  toggle, zone editor button, heatmap toggle, status indicator.
* Zone editor: click to draw polygons and lines on a frozen frame, choose zone type
  and name, save.
* Event timeline: real-time feed with icons and severity colours, filter by camera
  and type.
* Alert detail panel: clip player, skeleton snapshot, "Generate report" button.
* Analytics: alerts by type (chart), crowd count over time per zone (chart).
* "Ask your cameras" panel.
* Privacy panel: shows what is stored ("0 hours of raw footage stored, 3 incident
  clips") and the audit log of raw-view access. Make this prominent: it is our pitch.
* Responsive enough to open on a phone for security staff.

## BUILD ORDER

1. git init, project structure, requirements, config, .env.example, README skeleton. Commit.
2. Camera manager with multi-source support and reconnect. Commit.
3. YOLOv8 + ByteTrack + MediaPipe Pose pipeline, privacy view rendering. Commit.
4. FastAPI + WebSocket streaming to a basic dashboard. Commit.
5. SQLite events + alert system with cooldowns and Telegram. Commit.
6. Rolling buffer and incident clip saving. Commit.
7. Each detector, one at a time, each with a unit test using synthetic keypoints
   or boxes. Commit after each.
8. Zone editor. Commit.
9. Exam mode, SOS, heatmap, counting, exit, after-hours, running. Commit after each.
10. Claude API search and reports. Commit.
11. Full dashboard design pass, analytics, privacy panel. Commit.
12. Demo mode. Commit.
13. README completion, final testing, cleanup. Commit.

## VERIFICATION (do this before saying you are done)

* Run all unit tests and fix failures.
* Start the app with run.py and confirm the dashboard loads and streams.
* Confirm the app still runs with no API keys set (features degrade gracefully
  with a clear message instead of crashing).
* Confirm raw frames are never written to disk outside clips/.
* Confirm the system runs at a usable frame rate on CPU; report the FPS you measured.

## README CONTENTS

* Project name, pitch lines, the problem with traditional CCTV
* Which hackathon pillars it addresses and how
* Feature list with a short explanation of each detection rule (judges will ask)
* Privacy design: what is stored, what is never stored, who can unlock what
* Architecture diagram in Mermaid
* Setup: install, .env, connecting a phone as a camera, running
* Demo guide: how to trigger each feature live
* Ethics section: exam mode flags for human review only, no face recognition,
  no names, audit logging
* Limitations and future work
* CREDITS: Ultralytics YOLOv8, COCO dataset, ByteTrack, Google MediaPipe, OpenCV,
  FastAPI, Uvicorn, Chart.js, Google Fonts, Anthropic Claude API, Telegram Bot API,
  Twilio, and every other library used
* Team section with placeholders for our names

## ALSO GENERATE

* DEMO_SCRIPT.md: a 3-minute live demo script with what to say and what to do for
  each feature, opening with the pitch line and ending with the privacy panel.
* PITCH_NOTES.md: likely judge questions with strong answers (privacy, accuracy,
  false positives, scalability, cost, ethics of exam mode, why skeletons still
  allow catching wrongdoers).

## WORKING STYLE

* Work autonomously and keep going until everything is built and verified.
* If something fails, debug and fix it rather than leaving a TODO.
* Prefer simple, reliable solutions over clever fragile ones. A feature that works
  every time on stage beats a sophisticated one that fails.
* At the end, give me a summary: what was built, how to run it, how to trigger each
  feature for the demo, measured FPS, and anything that needs my action (API keys,
  phone camera URL).

---

## Project name

The project is named **ARJUN**.
