# Delivered against the brief

A clause-by-clause map from [`MASTER_PROMPT.md`](MASTER_PROMPT.md) to what
exists in the repository, including the four places we deliberately did
something different and why.

Legend: **done** · **done, differently** (with the reason) · **added**

---

## Hackathon rules

| Asked for | Delivered |
|---|---|
| All code written fresh | **done** — every file in this repository was written in this session |
| Git repo, committed after each working step | **done** — see `git log`; eleven commits, each a working state |
| Credit every library, model and API | **done** — README § CREDITS, with licences |

## Tech stack

| Asked for | Delivered |
|---|---|
| Python 3.10+, OpenCV, NumPy | **done** (developed on 3.14) |
| YOLOv8n pretrained on COCO, ByteTrack | **done** — `backend/vision/engine.py` |
| MediaPipe Pose, Hands, Face Mesh | **done, differently** — see note 1 |
| FastAPI + WebSockets (uvicorn) | **done** — `backend/main.py` |
| SQLite for events, zones, settings | **done** — `backend/events_db.py`, plus audit and counts tables |
| Single-page dashboard, plain HTML/CSS/JS, no build step | **done** — `frontend/`, three files |
| Chart.js from a CDN | **done** |
| Telegram primary, Twilio WhatsApp optional, both safe if keys missing | **done** — `backend/alerts.py` |
| Claude API for search and reports | **done** — `backend/ai_assistant.py`, with offline fallbacks |
| python-dotenv | **done** |

## Architecture

Every file named in the brief exists with the responsibility it was given.
Six modules were added: `state.py` (track histories), `world.py` (the
vocabulary rules see), `zone_store.py` (zone geometry), and under `vision/`:
`skeleton.py`, `synth.py`, `simscene.py`, `sources.py`, `engine.py`,
`mp_tasks.py`, `model_store.py`.

## Cameras

| Asked for | Delivered |
|---|---|
| `CAMERA_SOURCES` as `name=source` pairs | **done** |
| Webcam indices and phone streams | **done** — plus RTSP and looping video files |
| One thread per camera, "Reconnecting" on drop, retry | **done** — with widening backoff |
| One camera failing never crashes the app | **done** — and a rule that throws is muted while the other ten run |
| Resize and skip frames to stay real time | **done** — queued frames are dropped, not processed |
| | **added** — `sim:` scripted scenes (note 2) |

## Privacy core

| Asked for | Delivered |
|---|---|
| YOLOv8n + ByteTrack for persons and objects with stable ids | **done** |
| Pose keypoints mapped back to frame coordinates | **done** |
| Privacy view: dark canvas, glowing skeletons, "Person #ID", object outlines, zones | **done** — `backend/privacy.py` |
| 15-second rolling raw buffer, memory only, never written to disk | **done** — and trimmed by age as well as count, so the advertised number is the true one |
| On alert: 10s before + 5s after as MP4 in `clips/` | **done** — verified as real H.264, browser-playable |
| Raw view per camera behind `ADMIN_PIN`, every access audited | **done** — raw pixels are not even encoded until a valid token asks |

## The fifteen features

| # | Feature | Status |
|---|---|---|
| 1 | Fall detection | **done** — `detectors/fall.py`, 11 tests |
| 2 | Restricted zones and loitering | **done** — `detectors/zones.py` |
| 3 | Abandoned object | **done** — `detectors/abandoned.py`, see note 3 |
| 4 | Incident clip unlock | **done** — clip player in the alert panel, PIN-gated |
| 5 | Exam mode | **done** — `detectors/exam.py`, opt-in per camera |
| 6 | SOS Signal for Help | **done** — `detectors/sos.py`, silent alert |
| 7 | Crowd density | **done** — live count drawn on each zone |
| 8 | Running / panic | **done** — `detectors/running.py`, circular statistics |
| 9 | Entry / exit counting | **done** — survives a restart |
| 10 | Fire exit blocked | **done** — objects and standing people |
| 11 | After-hours occupancy + lights hint | **done** — midnight-spanning hours |
| 12 | Anonymous heatmap | **done** — per-camera toggle |
| 13 | Ask your cameras | **done** — events only, never images |
| 14 | Auto incident report | **done** — Markdown, downloadable |
| 15 | Demo mode | **done** — shortens timers only, never geometry |

## Alerts

| Asked for | Delivered |
|---|---|
| Saved to SQLite, pushed over WebSocket, sent to Telegram/WhatsApp | **done** |
| Skeleton snapshot attached, never a raw image | **done** |
| Cooldown per type per track | **done** — plus a per-rule cooldown that deliberately never suppresses a *critical* alert for a different person |
| Severities info / warning / critical; SOS and fall critical | **done** |
| Dashboard sound and tile flash on critical | **done** — and the SOS alert is exempt from the sound, by design |

## Dashboard

Every element asked for is present: header with clock and counts, camera grid
with per-tile toggles, zone editor, filterable event timeline, alert detail
with clip player and report button, analytics charts, the "Ask your cameras"
panel, a prominent privacy panel with the audit log, and a layout that works
at phone width.

## Verification

| Asked for | Result |
|---|---|
| Run all unit tests and fix failures | **91 tests, all passing**, in about one second |
| Start with `run.py`, dashboard loads and streams | **done** — verified in a browser; three tiles painting 640×360 |
| Runs with no API keys, degrading clearly | **done** — verified with no `.env` at all |
| Raw frames never written outside `clips/` | **done** — and enforced by `tests/test_privacy_invariants.py` rather than asserted |
| Report measured FPS | **done** — README § Measured performance, with the throttling caveat |

---

## The four deliberate departures

**1. MediaPipe is used for hands and face, not for pose.**
The brief says "run MediaPipe Pose on each person crop". MediaPipe Pose
handles one subject per call, so a corridor with six people costs seven
inferences per frame and the frame rate collapses on a laptop CPU. We track
`yolov8n-pose` on every frame instead — one pass, every skeleton, stable ids —
and keep MediaPipe for the two jobs it is genuinely unbeaten at: 21 hand
landmarks for the Signal for Help, and a 3D face transform for head yaw. Both
run on small crops, only when the feature that needs them is on. Setting
`POSE_BACKEND=mediapipe` restores the per-crop path; the rest of the system
cannot tell the difference, because both write into the same 17-slot layout.

**2. Simulated camera scenes were added.**
Not in the brief. Three scripted scenes stage a fall, an unattended bag,
loitering, a crowd, a panic run, a blocked fire exit, a phone at a desk and
people crossing a counting line, on a loop, with no camera. They exist because
a judge's laptop may have no webcam, a venue's wifi may not carry a phone
stream, and because asking a teammate to throw themselves onto an auditorium
floor repeatedly on cue is a bad plan. They feed the real detectors through
the real cooldowns, clip recorder and alert path — and every frame is stamped
SIMULATED in the pixels, labelled on the tile and flagged in the event record.

**3. Proximity for the unattended-bag rule means "near and stopped".**
The brief says "no person within a radius". Implemented literally, a busy
corridor keeps somebody within that radius almost continuously, the reclaim
timer resets forever and the rule never fires at all — which is worse than not
having it, because you would believe you had it. Custody now requires being
inside the radius *and* below walking pace. There is a test that sweeps a
pedestrian past the bag every few seconds and insists the alert still arrives.

**4. Storage locations are configurable, and warn about cloud sync.**
Not in the brief, added after OneDrive's sync service became the largest CPU
consumer on the development machine — clips are megabytes each and the
database changes on every event. ARJUN detects a synced folder at startup and
prints the two lines that fix it.

---

## Things the brief asked for that are worth pointing at specifically

* **Every threshold is in one file.** `config.py`, with a demo table that
  swaps only the timers — never the geometry — so what a judge sees triggering
  is the real rule, just impatient.
* **Each rule has tests that drive it with synthetic keypoints**, as asked, on
  a virtual clock against the real thresholds. A thirty-second loitering timer
  is tested in about two milliseconds.
* **The privacy claim is enforced by the test suite.**
  `tests/test_privacy_invariants.py` fails if a rule gains an `imwrite`, if
  the renderer is ever given the frame, if the ring buffer touches the disk,
  or if the outbound Claude payload grows a field.
