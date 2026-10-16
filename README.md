<div align="center">

# ARJUN

### Campus Guardian — a safety system that refuses to watch you

*“Arjuna saw only the eye of the bird.”*

**Ordinary CCTV records thousands of innocent students to catch one incident.
ARJUN watches no one, until someone gives it a reason to.**

*CCTV today is a witness. ARJUN is a first responder.*

</div>

---

## The problem with the cameras we already have

A campus with two hundred cameras is a campus recording roughly eight thousand
hours of innocent people every week, so that on the rare day something happens,
a human can go back and look.

Three things are wrong with that bargain, and they are not fixable by adding
features:

1. **It is backwards in time.** A recording is only useful after the fact. A
   student who falls down a stairwell at 7pm is found when someone walks past,
   not when the camera sees it. The camera was a witness; nobody was watching.
2. **It is indiscriminate.** To catch one incident it keeps everybody. Every
   person who walked past that lens is now a file on a disk that somebody can
   search, leak, subpoena or sell.
3. **It makes people behave differently.** A camera that might be watching
   changes how a campus feels, and that cost is paid continuously by people who
   will never be involved in an incident.

ARJUN takes the opposite position: **safety does not require watching people.**
It requires noticing events. Those are not the same problem, and conflating
them is why surveillance looks the way it does.

So ARJUN throws the pixels away at the first opportunity, reasons about
geometry, and keeps fifteen seconds of video in volatile memory in case
somebody needs help. It is not a camera that respects privacy as a setting. It
is a camera that is *structurally incapable* of being used for surveillance,
and the rest of this document is the argument for why that claim holds.

---

## What it actually does

```
camera  →  skeletons + boxes  →  rules  →  incident  →  alert, clip, report
             ↑                                              ↑
       pixels stop here                        the only moment pixels are kept
```

Eleven rules watch for things that matter. When one fires, the system saves a
fifteen second clip, pushes an alert to the dashboard and to Telegram with an
*anonymous skeleton image*, and can write a formal incident report.

When nothing is happening — which is almost always — nothing is recorded,
because nothing in the codebase is capable of recording it.

---

## The hackathon pillars

**A new worldview — surveillance is flawed because it records everyone.**
We did not add privacy to a surveillance system. We inverted the pipeline. The
privacy view is not a filter applied to a camera image; it is a render of
geometry onto a blank canvas, and the camera frame is never a layer in it.
There is no setting, no bug and no race condition that can put a face on the
dashboard, because the dashboard is never sent one.

**Challenge the perspective — safety does not require watching people.**
Every rule in the system is written against joint positions and timestamps. A
fall is a hip dropping 16% of a body's height in 0.6 seconds and staying down
for two more. None of that needs to know what anyone looks like. Read
`backend/detectors/` and you will not find a single line that touches a pixel.

**Human potential — getting help to someone instantly.**
The fall rule fires 2.4 seconds after someone hits the floor, and a guard's
phone buzzes about a second later. The same system recognises the Signal for
Help — the silent hand gesture for "I am in danger and cannot say so" — which
is the one channel a person can deliberately open to a system that has
promised not to identify them.

---

## The eleven rules, and exactly how each decides

Judges will ask. Every threshold below lives in `config.py`, and every rule has
unit tests in `tests/` that drive it with synthetic skeletons.

Speeds are in **body heights per second** and distances in **multiples of the
subject's own height**, never pixels — which is why one set of numbers works
for somebody beside the lens and somebody at the far end of a corridor.

| # | Rule | Fires when | Severity |
|---|---|---|---|
| 1 | **Fall** | Hip centre drops ≥16% of body height within 0.6 s, measured against the *highest* hip in that window; body ends near-horizontal (torso >55° from vertical, or box wider than tall when the pose is poor); still down after 2 s. | critical |
| 2 | **Signal for Help** | Open palm with the thumb tucked across it, *then* the four fingers folding over the thumb, within 3 s, each stage held ≥0.25 s. | critical, silent |
| 3 | **Restricted zone** | A person's **feet** enter a restricted polygon and stay ≥0.6 s. Re-arms when they leave. | warning |
| 4 | **Loitering** | Feet inside a loiter zone for longer than the zone's patience (25 s default). Exits under 1.5 s are forgiven. | warning |
| 5 | **Crowd density** | More people inside a crowd zone than its limit, held ≥2.5 s. Two limits per zone: warning and critical. | warning / critical |
| 6 | **Unattended bag** | A backpack/handbag/suitcase drifts <0.35 of its own size for 30 s with nobody within 1.2 body heights **who has stopped**. | warning |
| 7 | **Fire exit blocked** | An object left in an exit zone, or a person standing (not walking) in it, for 15 s. | critical |
| 8 | **Panic movement** | ≥2 people above 2.2 body heights/second whose headings agree within 70°, held ≥0.5 s. Headings averaged as unit vectors. | critical |
| 9 | **Entry / exit counting** | A foot point crosses a drawn line, with a dead-band so standing on the threshold is not counted repeatedly. Gives a live "people inside". | info |
| 10 | **After-hours** | Anyone present outside the camera's opening hours for 10 s. Also the inverse: a room empty 2 min after closing is an energy hint. | warning / info |
| 11 | **Exam mode** | *Per camera, opt-in.* A phone-shaped object at a desk for 2 s, or a head turned >32° away for 3 s. | warning / info |

### Why the rules say "no" as often as "yes"

A rule nobody trusts is a rule nobody acts on, so each one is built to reject
the thing it most resembles:

* Sitting down drops the hips fast — but the torso stays vertical, so it is not
  a fall. (`test_sitting_down_is_not_a_fall`)
* A fist is not the Signal for Help; the thumb has to be *underneath* the
  fingers, and the palm has to have come first. (`test_a_plain_fist_is_not_the_signal`)
* A bag being carried is moving, so its clock never starts.
* Walking through a fire exit is what a fire exit is for.
* One person sprinting is late for a lecture. Five sprinting *the same way* is
  an emergency.
* A group crossing a hall is not a crowd gathering in it.

---

## Privacy design: what is kept, what is not, who can unlock what

### Never stored, anywhere, at any time

| | Why it cannot happen |
|---|---|
| Continuous footage | Nothing in the codebase can write it. The only writer of pixels is `IncidentRecorder`, and it only runs when a rule fires. |
| Faces | The privacy view is drawn from geometry onto a blank canvas. The renderer is never handed the frame. Facial keypoints inform head position and are never drawn. |
| Names or identities | There is no face recognition, no appearance descriptor, no embedding. A track number is invented when somebody appears and destroyed seconds after they leave. |
| Biometric templates | None are computed. MediaPipe's face mesh is used for one number — a yaw angle — and the angle is all that is kept. |
| Anything sent to an external service | The Claude integration sends structured event rows. One function, `_event_rows` in `backend/ai_assistant.py`, builds the entire outbound payload. |

### Held, deliberately

| | Where | For how long |
|---|---|---|
| 15 seconds of raw video | **Memory only**, a fixed-length ring buffer, continuously overwritten | until the next frame pushes it out |
| Incident clips | `clips/` — 10 s before an alert, 5 s after | 7 days, then deleted automatically |
| Skeleton snapshots | `data/snapshots/` — line drawings, safe to send to a phone | with the event |
| Event records | `data/arjun.db` — a time, a camera, a rule, a zone, a track number | until purged |
| Audit log | `data/arjun.db` — every raw unlock and clip open | permanently |

### Who can unlock what

* **Anyone at the dashboard** sees skeletons, zones, counts, alerts and
  anonymous snapshots. No PIN.
* **The staff PIN** (`ADMIN_PIN`) unlocks the raw view of a camera and the
  incident clips, for ten minutes.
* **Every unlock is itself an incident**, written to the audit table, shown on
  the privacy ledger, and pushed live to every open dashboard. The watchers are
  watched.
* Raw pixels are not even JPEG-encoded until a valid token asks for them.

---

## Architecture

```mermaid
flowchart TB
    subgraph CAM["Capture · one thread per camera"]
        W["Webcam / DroidCam / RTSP / video file"]
        SIM["sim: scripted scene"]
    end

    subgraph VIS["Vision · pixels stop here"]
        Y1["YOLOv8n-pose + ByteTrack<br/>every frame · people + skeletons + ids"]
        Y2["YOLOv8n COCO<br/>every 3rd frame · bags, phones"]
        MP["MediaPipe<br/>hands · face geometry<br/>small crops, only when needed"]
    end

    subgraph MEM["Memory"]
        RING["Ring buffer<br/>15 s of JPEG, RAM only"]
        TRK["Track histories<br/>8 s, then forgotten"]
    end

    subgraph RULES["Eleven rules · geometry only, never pixels"]
        R["fall · sos · restricted · loiter · crowd<br/>abandoned · exit · panic · counting<br/>after-hours · exam"]
    end

    subgraph OUT["Out"]
        REND["Privacy renderer<br/>skeletons on a blank canvas"]
        REC["Incident recorder<br/>the only writer of pixels"]
        AL["Alerts · cooldowns"]
    end

    DB[("SQLite<br/>events · zones · audit")]
    UI["Dashboard<br/>WebSocket"]
    TG["Telegram / WhatsApp<br/>skeleton image only"]
    CL["Claude<br/>event rows only"]

    W --> VIS
    SIM -.->|"ready-made world"| RULES
    VIS --> TRK --> RULES
    W --> RING
    RULES --> AL
    AL --> REC
    RING -->|"on an incident, 10 s before + 5 s after"| REC
    REC --> CLIPS[("clips/")]
    RULES --> REND --> UI
    AL --> DB --> UI
    AL --> TG
    DB --> CL --> UI
```

### Why the models are arranged this way

The obvious reading of the brief is "detector, then MediaPipe Pose on each
person crop". That is one inference per frame **plus one per person**, and
MediaPipe Pose handles a single subject per call — a corridor with six people
costs seven inferences and collapses to single digits on a laptop CPU.

So the work is inverted. `yolov8n-pose` is tracked on *every* frame: one pass
returns every skeleton with a stable id, and people are the safety-critical
subject, so they get full temporal resolution — a fall lasts half a second and
cannot be sampled at 4 Hz. The full COCO detector runs every third frame and
only for the handful of object classes any rule cares about, because a bag that
has been motionless for thirty seconds does not need finding sixty times a
second. MediaPipe is kept for the two jobs it is genuinely unbeaten at — hand
landmarks and a 3D face transform — and runs on small crops, only when the
feature that needs it is switched on, behind a cheap pose-level gate.

`POSE_BACKEND=mediapipe` restores the per-crop path if you want to compare.

### Layout

```
arjun/
  run.py                  one command; --check, --sim, --demo
  config.py               every threshold, with a demo-mode table
  backend/
    main.py               FastAPI, routes, WebSocket hub
    camera_manager.py     one thread per camera, reconnect, frame dropping
    pipeline.py           see · remember · reason · decide · draw
    privacy.py            renderer, ring buffer, incident recorder
    state.py              track histories, zone timers, scale-free speeds
    world.py              Detection / FrameWorld — the vocabulary rules see
    zone_store.py         zones in normalised coordinates
    alerts.py             cooldowns, Telegram, WhatsApp, broadcast
    events_db.py          SQLite: events, zones, audit, counts
    ai_assistant.py       Claude: search + reports (and the offline fallbacks)
    detectors/            the eleven rules, one file each
    vision/
      engine.py           YOLO detect + track + pose
      mp_tasks.py         MediaPipe hands / face / pose
      skeleton.py         COCO-17 layout and the geometry rules are phrased in
      synth.py            a parametric human, for tests and scenes
      simscene.py         scripted camera scenes
      sources.py          webcam / stream / file / sim
      model_store.py      weight download and caching
  frontend/               index.html · style.css · app.js  (no build step)
  tests/                  91 tests, run in about a second
  clips/   data/   models/    (all gitignored)
```

---

## Setup

### Install

```bash
python -m pip install -r requirements.txt
```

Python 3.10 or newer. The first run downloads about 30 MB of pretrained
weights into `models/`; after that it works offline.

### Configure

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

**Every key is optional.** With an empty `.env` the system still boots, detects,
records incidents and serves the dashboard. A missing key disables exactly one
feature and says so in the interface.

| Key | Without it |
|---|---|
| `CAMERA_SOURCES` | falls back to webcam 0 plus a simulated scene |
| `ADMIN_PIN` | defaults to `2026` — change it before any real pilot |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | alerts reach the dashboard but not a phone |
| `TWILIO_*` | no WhatsApp; Telegram is the primary channel anyway |
| `CLAUDE_API_KEY` | questions are answered by a local keyword search, reports by a local template |

### Run

```bash
python run.py
```

Then open <http://localhost:8000>.

```bash
python run.py --check        # verify models, rules, cameras, codecs, keys
python run.py --sim          # three scripted scenes, no camera needed
python run.py --demo         # short timers for a live demonstration
python run.py --cameras "Corridor=0, Gate=http://192.168.1.7:4747/video"
```

### Connecting a phone as a camera

A phone is a better camera than most laptops have, and it means you can point
it at a real corridor.

1. Install **DroidCam** (Android/iOS) or **IP Webcam** (Android).
2. Put the phone and the laptop on the same wifi. Hotspot from the phone works
   and is more reliable than campus wifi, which often blocks device-to-device
   traffic.
3. Start the app; it shows a URL such as `http://192.168.1.7:4747`.
4. Put it in `.env`:

```ini
CAMERA_SOURCES=Main Corridor=0, Exam Hall=http://192.168.1.7:4747/video
```

DroidCam serves `/video`; IP Webcam serves `/video` on port 8080. For a forced
resolution, DroidCam also offers
`http://192.168.1.7:4747/mjpegfeed?640x480`. RTSP cameras work too:
`rtsp://user:pass@192.168.1.9/stream1` — credentials are redacted everywhere
they are displayed.

If a stream drops, the tile says **Reconnecting** and the worker retries with
a widening backoff. One camera failing never affects another.

---

## Demo guide — triggering each feature

Turn **Demo timings** on in the header first. It shortens every waiting
period — loitering 25 s → 8 s, unattended bag 30 s → 10 s — and changes nothing
else. The rules that fire are the real rules.

Run `python run.py --sim --demo` and the three scripted scenes stage most of
this for you on a loop, with no camera and no volunteers. The **Atrium** scene
runs a 78-second cycle: a bag is left at t+16, a loiterer settles at t+22, a
person falls at t+46, three people run at t+58, a crowd gathers at t+66.

On a real camera:

| Feature | How to trigger it |
|---|---|
| **Fall** | Lie down on the floor and stay there three seconds. Sitting down will not do it — that is the point. |
| **Signal for Help** | Face the camera, palm up, fold the thumb into the palm, hold ~1 s, then close the fingers over it. Alert is silent by design. |
| **Restricted zone** | Draw one with **Zones**, then walk into it. |
| **Loitering** | Draw a loiter zone and stand in it for 8 s. Walking in and out briefly will not reset the clock. |
| **Crowd** | Draw a crowd zone, set the warning limit to 2, and get three people into it. |
| **Unattended bag** | Put a backpack down, then walk away. ~10 s. Walking *past* it does not reset the timer; stopping beside it does. |
| **Fire exit blocked** | Draw an exit zone, stand still in it for 6 s — or leave a bag there. |
| **Panic** | Two or more people run across the frame in the same direction. |
| **Entry / exit counting** | Draw a counting line across a doorway and walk through it both ways. Watch "inside" on the Patterns panel. |
| **After-hours** | Turn on **Night mode** on the tile — it pretends the building is closed so you can show this at 2pm. |
| **Exam mode** | Turn on **Exam mode**, draw a desk zone, sit in it and put a phone on the desk. Then turn your head away for four seconds. |
| **Heatmap** | Turn on **Heatmap** and walk around for a minute. |
| **Incident clip** | Click any alert in Observations. The clip asks for the PIN (`2026`) — then look at the privacy ledger: the unlock is already logged. |
| **Ask the cameras** | "Any falls today?" or "Summarise everything after 6 PM". |
| **Incident report** | Open an alert, click **Write the report**, then download it as Markdown. |

See **DEMO_SCRIPT.md** for a three-minute version with the words to say.

---

## Ethics

**Exam mode flags for human review and nothing else.** It raises "Desk 3: phone
visible (0:04) — for invigilator review". It cannot say who is at desk 3,
because the system does not know and has no way to find out. The flag is a
prompt for an invigilator to walk over and look, which is what invigilators do
anyway. The tests assert the wording never accuses. Head-turn flags are logged
at *info* severity, because students look at clocks.

**No face recognition. No names. No re-identification.** Two visits by the same
person are two unrelated tracks. We consider that a feature, not a limitation:
the alternative is a database of where every student has been.

**Exam mode is opt-in per camera, and off by default.** It should be switched
on in an exam hall during an exam, and switched off afterwards. Nothing
enforces that but policy — which is exactly why the audit log exists.

**Every raw-footage access is recorded and visible.** Staff can see faces when
they genuinely need to. They cannot do it quietly.

**Simulated feeds are labelled everywhere** — on the tile, in the status
payload, in the event record, and burned into the pixels of the clip itself.
There is no code path by which a simulated world can be presented as a real
one.

**What this system should not be used for.** Attendance. Productivity
monitoring. Discipline based on an alert alone. Any of those would require
identity, and identity is the thing we deliberately did not build.

---

## Limitations, honestly

* **A single camera has no depth.** Two people standing in line look closer
  together than they are. Crowd limits need tuning per camera, which is why
  every zone can override every threshold.
* **Occlusion breaks tracking.** A person hidden behind a pillar for four
  seconds comes back as a new track number. Rules that depend on duration
  restart. The track-memory window softens this; it does not solve it.
* **Pose estimation degrades at distance.** Below roughly 60 px of body height
  the skeleton gets unreliable, and the fall rule falls back to box shape.
* **The Signal for Help needs a hand the camera can resolve** — roughly 48 px
  across. It works at conversational distance, not across a quadrangle.
* **Exam-mode phone detection is a COCO "cell phone" box.** It will miss a
  phone under a desk and can flag a dark wallet. It is a prompt for a human,
  which is the only claim we make for it.
* **Simulated scenes are scripted, not learned.** They prove the rules fire
  correctly; they do not prove detection accuracy on real footage.
* **No night-vision handling, no multi-camera handover, no re-identification** —
  the last of those on purpose.
* **Thresholds are tuned by hand**, on a handful of people, in one building.
  A real deployment needs a week of calibration per site.

### Future work

* Depth from a second camera, for genuinely metric crowd limits.
* On-device inference (Jetson / Coral) so frames never leave the camera at all.
* A formal privacy-impact assessment and a student-facing consent page.
* Per-zone schedules, so an exam hall becomes a corridor at 5pm.
* An evaluation set of real falls, to replace hand-tuned thresholds with
  measured ones.

---

## Tests

```bash
python -m pytest
```

91 tests, about a second, no camera and no model weights required. Each rule is
driven by synthetic skeletons on a virtual clock, against its **real**
thresholds — a thirty-second loitering timer is tested in about two
milliseconds. They cover each rule's yes case, its no case, and its re-arming;
scale invariance for the fall rule; rotation and scale invariance for the hand
gesture; and midnight-spanning opening hours.

**The privacy guarantees are tested, not asserted.**
`tests/test_privacy_invariants.py` reads the source and fails if anything
other than the incident recorder gains a `VideoWriter` or an `imwrite`, if a
detection rule reaches for the frame or writes to disk, if the renderer's
signature ever accepts an image, if the ring buffer references a file path, or
if the payload bound for Claude grows a field. A privacy property that lives
only in a README is a property that will be gone in three months.

---

## Measured performance

Measured on the development laptop — Windows 11, 12 logical cores, **CPU-only
inference, no GPU** — at 640×360 processing resolution with a YOLO input size
of 416. Frame rates are steady-state, counted as frames delivered over a
30-second window rather than read off an instantaneous meter.

| Configuration | Result |
|---|---|
| One real camera (webcam), full pipeline | **13.8 fps** |
| One simulated scene | **15.0 fps** (capped by `TARGET_FPS`) |
| Three simulated scenes at once | **43.2 fps aggregate** — 14.2 / 14.2 / 14.8 |
| Real camera + one simulated scene | 6.9 + 9.8 fps |

Per-frame cost on a cool machine:

| Stage | Cost |
|---|---|
| Inference — pose every frame, objects every third | 46 ms |
| All eleven rules | 0.2 – 4 ms, depending on how many zones and people |
| Privacy render — zones, skeletons, bloom | 4 – 6 ms |
| Scripted scene generation (simulated cameras only) | 1.5 – 7 ms |

### Two things worth knowing before you quote a number

**A laptop throttles.** After an hour of sustained load during development,
the identical benchmark on the identical code went from 33 ms per model pass
to 93 ms — a 2.7× slowdown with nothing changed but the temperature of the
machine. The end-to-end figure settles around 8 fps once that happens. If you
are demonstrating from a laptop, plug it in, set the power profile to
performance, and don't leave it running flat out for an hour beforehand.

**A cloud sync client will take the machine from you.** This project was
developed in a OneDrive folder, and OneDrive's sync service became the single
largest consumer of CPU on the system — incident clips are megabytes each and
the database changes on every event. ARJUN now detects this at startup and
prints the two lines to add to `.env` that fix it. The difference was roughly
half the frame rate.

### Tuning

Faster: `INFER_SIZE=320` (~1.6× quicker), `PROC_WIDTH=480`, `RENDER_BLOOM=0`,
or raise `OBJ_EVERY_N`. More accurate: `INFER_SIZE=512`, `OBJ_EVERY_N=1`.

Thread allocation is worked out at startup from the core count and how many
cameras actually run models — simulated scenes don't — because both extremes
measurably hurt: giving torch nearly every core made three camera threads
fight over twenty threads, and splitting the budget evenly took a lone camera
from 13 fps to 8. Override with `TORCH_THREADS` if you know better than the
heuristic.

## CREDITS

Every pretrained model, library and API used, with its licence. Nothing here
was written by us except the code in this repository.

### Models and datasets

| | Used for | Licence |
|---|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — `yolov8n.pt`, `yolov8n-pose.pt` | person/object detection and pose | AGPL-3.0 |
| [COCO dataset](https://cocodataset.org) | the 80 classes those weights know | CC BY 4.0 |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | multi-object tracking, via Ultralytics | MIT |
| [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | 21 hand landmarks for the Signal for Help | Apache-2.0 |
| [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | head yaw for exam mode | Apache-2.0 |
| [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) | alternative pose backend | Apache-2.0 |

### Libraries

| | Used for | Licence |
|---|---|---|
| [OpenCV](https://opencv.org) | capture, drawing, H.264 clip encoding | Apache-2.0 |
| [NumPy](https://numpy.org) | geometry and the heatmap accumulator | BSD-3-Clause |
| [PyTorch](https://pytorch.org) | the runtime under Ultralytics | BSD-3-Clause |
| [FastAPI](https://fastapi.tiangolo.com) | HTTP and WebSocket backend | MIT |
| [Uvicorn](https://www.uvicorn.org) | ASGI server | BSD-3-Clause |
| [Starlette](https://www.starlette.io) | WebSockets and static files, via FastAPI | BSD-3-Clause |
| [Pydantic](https://pydantic.dev) | request validation, via FastAPI | MIT |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | configuration | BSD-3-Clause |
| [Requests](https://requests.readthedocs.io) | Telegram and Twilio REST calls | Apache-2.0 |
| [python-multipart](https://github.com/Kludex/python-multipart) | form posts | Apache-2.0 |
| [pytest](https://pytest.org) | the test suite | MIT |
| [SQLite](https://sqlite.org) | the database, via Python's stdlib | public domain |

### Services and APIs

| | Used for |
|---|---|
| [Anthropic Claude API](https://www.anthropic.com) | natural-language search over events, and incident reports |
| [Telegram Bot API](https://core.telegram.org/bots/api) | push alerts |
| [Twilio WhatsApp API](https://www.twilio.com/whatsapp) | optional push alerts |

### Frontend

| | Used for | Licence |
|---|---|---|
| [Chart.js](https://www.chartjs.org) | the two analytics charts | MIT |
| [EB Garamond](https://fonts.google.com/specimen/EB+Garamond) | display type — a revival of Claude Garamont's 16th-century roman | OFL-1.1 |
| [IBM Plex Sans / Mono](https://fonts.google.com/specimen/IBM+Plex+Sans) | interface and data | OFL-1.1 |

### Other credits

* The **Signal for Help** gesture was created by the
  [Canadian Women's Foundation](https://canadianwomen.org/signal-for-help/) in
  2020 and released for public use. We implement it; we did not invent it.
* The **Arjuna** reference is from the Mahābhārata: asked what he could see,
  Arjuna replied that he saw only the eye of the bird, and nothing else. It is
  the oldest statement we know of the principle this system is built on.

---

## The team

| Name | Role |
|---|---|
| _Your name_ | _role_ |
| _Your name_ | _role_ |
| _Your name_ | _role_ |
| _Your name_ | _role_ |

Built at **Renaissance**, CSI KJSSE's Buildathon at Abhiyantriki 2026,
K J Somaiya School of Engineering. The brief is archived verbatim at
[`docs/MASTER_PROMPT.md`](docs/MASTER_PROMPT.md).

---

<div align="center">

**It watches no one, until someone gives it a reason to.**

</div>
