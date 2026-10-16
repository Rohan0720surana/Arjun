"""
ARJUN — central configuration.

Everything tunable in the whole system lives in this file. Nothing else should
hard-code a threshold, a timeout or a colour: if you find yourself wanting a
magic number somewhere in backend/, add it here instead.

Two ideas worth knowing before you read on:

1.  **Demo mode.** Every duration that a judge would have to stand around
    waiting for exists twice — a realistic value and a stage value. Flip
    ``set_demo_mode(True)`` and the loitering timer drops from 25s to 8s, the
    abandoned-bag timer from 30s to 10s, and so on. The detectors read the
    active value on every frame, so the switch takes effect live without a
    restart.

2.  **Scale-free thresholds.** A person two metres from the lens is four times
    taller in pixels than one at the far end of a corridor. So we never express
    a distance or a speed in pixels. Speeds are in *body heights per second* and
    distances in *multiples of the subject's own bounding-box height*. The same
    numbers then work on a 480p webcam and a 1080p phone stream, near or far.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent

ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

MODELS_DIR = ROOT / "models"          # auto-downloaded weights (gitignored)
DATA_DIR = ROOT / "data"              # sqlite db + skeleton snapshots
CLIPS_DIR = ROOT / "clips"            # THE ONLY place raw pixels ever land
SNAPSHOT_DIR = DATA_DIR / "snapshots"  # anonymised skeleton renders (safe)
FRONTEND_DIR = ROOT / "frontend"
DB_PATH = DATA_DIR / "arjun.db"

for _d in (MODELS_DIR, DATA_DIR, CLIPS_DIR, SNAPSHOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Small env helpers
# --------------------------------------------------------------------------- #

def _env(key: str, default: str = "") -> str:
    """Read an env var, treating placeholder text in .env.example as unset."""
    raw = (os.getenv(key) or "").strip()
    if raw.lower() in {"", "none", "changeme", "your_key_here", "paste_here"}:
        return default
    return raw


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(_env(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Identity — used in the UI, the Telegram messages and the generated reports
# --------------------------------------------------------------------------- #

APP_NAME = "ARJUN"
APP_SUBTITLE = "Campus Guardian"
APP_TAGLINE = "It watches no one, until someone gives it a reason to."
APP_MOTTO = "Arjuna saw only the eye of the bird."
APP_VERSION = "1.0.0"
APP_VENUE = "Renaissance · Abhiyantriki 2026 · CSI KJSSE"


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

HOST = _env("HOST", "0.0.0.0")
PORT = _env_int("PORT", 8000)
OPEN_BROWSER = _env_bool("OPEN_BROWSER", True)


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
# CAMERA_SOURCES is a comma separated list of  name=source  pairs.
#
#   Main Corridor=0                               laptop webcam
#   Gate Cam=http://192.168.1.7:4747/video        DroidCam / IP Webcam
#   Atrium=sim:atrium                             built-in simulated scene
#   Archive=clips/something.mp4                   a video file, loops forever
#
# "sim:" sources are scripted scenes that let the whole system be demonstrated
# on a laptop with no camera attached. They are labelled as simulated
# everywhere in the UI — see backend/vision/simscene.py.

DEFAULT_CAMERA_SOURCES = "Main Corridor=0, Atrium (Simulated)=sim:atrium"

PROC_WIDTH = _env_int("PROC_WIDTH", 640)        # frames resized to this width
TARGET_FPS = _env_float("TARGET_FPS", 15.0)     # per-camera capture ceiling
STREAM_FPS = _env_float("STREAM_FPS", 12.0)     # what the browser is fed
JPEG_QUALITY = _env_int("JPEG_QUALITY", 72)     # dashboard stream quality
BUFFER_JPEG_QUALITY = _env_int("BUFFER_JPEG_QUALITY", 80)  # incident clips
RECONNECT_DELAY_S = _env_float("RECONNECT_DELAY_S", 3.0)
READ_FAIL_LIMIT = _env_int("READ_FAIL_LIMIT", 12)  # consecutive bad reads -> reconnect


def parse_camera_sources(raw: str | None = None) -> list[tuple[str, str]]:
    """``"A=0, B=http://x"`` -> ``[("A", "0"), ("B", "http://x")]``.

    Entries without an ``=`` are given a generated name, so ``"0"`` alone works.
    """
    raw = raw if raw is not None else _env("CAMERA_SOURCES", DEFAULT_CAMERA_SOURCES)
    out: list[tuple[str, str]] = []
    for i, chunk in enumerate(raw.split(",")):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            name, src = chunk.split("=", 1)
            name, src = name.strip(), src.strip()
        else:
            name, src = f"Camera {i + 1}", chunk
        if not src:
            continue
        # Keep names unique — they are used as dict keys and in URLs.
        base, n = name, 2
        while any(name == existing for existing, _ in out):
            name = f"{base} ({n})"
            n += 1
        out.append((name, src))
    return out


CAMERA_SOURCES = parse_camera_sources()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# Why two YOLO models instead of the single detector + per-crop MediaPipe Pose
# that the brief sketched: MediaPipe Pose handles exactly one person per call,
# so a corridor with six people costs six inferences per frame and collapses to
# single-digit FPS on a laptop CPU. YOLOv8n-pose returns every skeleton in the
# frame from one pass. We keep MediaPipe for the two jobs it is unbeaten at —
# hand landmarks for the Signal-for-Help gesture, and a face mesh with a 3D
# transform matrix for head yaw in exam mode — and run those only on the small
# crops that need them. POSE_BACKEND=mediapipe switches pose back if you want
# to compare the two.

YOLO_DET_WEIGHTS = _env("YOLO_DET_WEIGHTS", "yolov8n.pt")
YOLO_POSE_WEIGHTS = _env("YOLO_POSE_WEIGHTS", "yolov8n-pose.pt")
POSE_BACKEND = _env("POSE_BACKEND", "yolo").lower()      # yolo | mediapipe | off
TRACKER_CFG = _env("TRACKER_CFG", "bytetrack.yaml")

INFER_SIZE = _env_int("INFER_SIZE", 416)        # YOLO imgsz
DET_CONF = _env_float("DET_CONF", 0.35)
POSE_CONF = _env_float("POSE_CONF", 0.35)
KEYPOINT_CONF = _env_float("KEYPOINT_CONF", 0.30)  # per-joint visibility floor
POSE_EVERY_N = _env_int("POSE_EVERY_N", 2)      # mediapipe backend: pose every Nth
OBJ_EVERY_N = _env_int("OBJ_EVERY_N", 3)        # objects are slow; look less often
TORCH_THREADS = _env_int("TORCH_THREADS", 0)    # 0 = let torch decide

MP_POSE_TASK = "pose_landmarker_lite.task"
MP_HAND_TASK = "hand_landmarker.task"
MP_FACE_TASK = "face_landmarker.task"

# COCO classes we care about. Everything else YOLO sees is discarded on the
# frame it is seen — we never even keep a label for it.
CLASS_PERSON = "person"
BAG_CLASSES = ("backpack", "handbag", "suitcase")
PHONE_CLASSES = ("cell phone",)
BLOCKER_CLASSES = BAG_CLASSES + ("chair", "bench", "bicycle", "motorcycle", "box")
TRACKED_CLASSES = (CLASS_PERSON,) + BAG_CLASSES + PHONE_CLASSES + BLOCKER_CLASSES


# --------------------------------------------------------------------------- #
# Privacy
# --------------------------------------------------------------------------- #

ADMIN_PIN = _env("ADMIN_PIN", "2026")
ADMIN_SESSION_TTL_S = _env_float("ADMIN_SESSION_TTL_S", 600.0)  # raw-view unlock
RAW_BUFFER_SECONDS = _env_float("RAW_BUFFER_SECONDS", 15.0)
CLIP_PRE_SECONDS = _env_float("CLIP_PRE_SECONDS", 10.0)
CLIP_POST_SECONDS = _env_float("CLIP_POST_SECONDS", 5.0)
CLIP_FPS = _env_float("CLIP_FPS", 12.0)
CLIP_CODECS = ("avc1", "mp4v")   # tried in order; avc1 plays in every browser
CLIP_RETENTION_DAYS = _env_int("CLIP_RETENTION_DAYS", 7)
MAX_CLIPS_ON_DISK = _env_int("MAX_CLIPS_ON_DISK", 60)

# Rendering of the privacy view.
RENDER_BLOOM = _env_bool("RENDER_BLOOM", True)   # the glow on the skeletons
RENDER_GRID = _env_bool("RENDER_GRID", True)
RENDER_VIGNETTE = _env_bool("RENDER_VIGNETTE", True)

HEATMAP_DECAY = _env_float("HEATMAP_DECAY", 0.997)   # per frame
HEATMAP_GAIN = _env_float("HEATMAP_GAIN", 0.55)
HEATMAP_RADIUS = _env_int("HEATMAP_RADIUS", 13)


# --------------------------------------------------------------------------- #
# Alert plumbing
# --------------------------------------------------------------------------- #

TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID")
TWILIO_ACCOUNT_SID = _env("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = _env("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = _env("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TWILIO_WHATSAPP_TO = _env("TWILIO_WHATSAPP_TO")
PUBLIC_BASE_URL = _env("PUBLIC_BASE_URL")   # needed for WhatsApp media links

CLAUDE_API_KEY = _env("CLAUDE_API_KEY") or _env("ANTHROPIC_API_KEY")
CLAUDE_MODEL = _env("CLAUDE_MODEL", "claude-sonnet-4-5")
CLAUDE_MAX_TOKENS = _env_int("CLAUDE_MAX_TOKENS", 1600)
CLAUDE_EVENT_LIMIT = _env_int("CLAUDE_EVENT_LIMIT", 120)

SEVERITY_ORDER = ("info", "warning", "critical")

# Which notification channels each severity reaches.
NOTIFY_MIN_SEVERITY = _env("NOTIFY_MIN_SEVERITY", "warning")


# --------------------------------------------------------------------------- #
# Detection thresholds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Thresholds:
    """Every number a detector is allowed to believe in."""

    # -- fall ------------------------------------------------------------- #
    fall_drop_window_s: float = 0.6     # look back this far for a sudden drop
    fall_drop_fraction: float = 0.16    # hip must fall this much of body height
    fall_horizontal_ratio: float = 1.15  # box w/h above this = lying down
    fall_torso_angle_deg: float = 55.0  # torso tilt from vertical that counts
    fall_down_seconds: float = 2.0      # must stay down this long to confirm
    fall_recover_seconds: float = 1.5   # upright this long = incident over

    # -- loitering / restricted ------------------------------------------- #
    loiter_seconds: float = 25.0
    restricted_grace_s: float = 0.6     # ignore a foot clipping the edge
    zone_exit_grace_s: float = 1.5      # brief exits do not reset the timer

    # -- abandoned object -------------------------------------------------- #
    abandoned_seconds: float = 30.0
    abandoned_owner_radius: float = 1.2  # body heights (~2 m) to count as with it
    abandoned_claim_speed_bh_s: float = 0.5  # faster than this = walking past
    abandoned_move_tolerance: float = 0.35  # drift allowed, in the bag's own size
    abandoned_reclaim_s: float = 1.0     # how long you must linger to reclaim it

    # -- crowd ------------------------------------------------------------- #
    crowd_warn: int = 5
    crowd_critical: int = 9
    crowd_sustain_s: float = 2.5        # must hold, so a passing group is fine
    crowd_sample_interval_s: float = 10.0  # how often we log a count for charts

    # -- running / panic --------------------------------------------------- #
    run_speed_bh_s: float = 2.2          # body heights per second = running
    run_min_people: int = 2
    run_direction_spread_deg: float = 70.0
    run_window_s: float = 1.0
    run_sustain_s: float = 0.5

    # -- exam mode --------------------------------------------------------- #
    exam_phone_seconds: float = 2.0
    exam_yaw_deg: float = 32.0
    exam_yaw_seconds: float = 3.0
    exam_face_every_n: int = 3            # face mesh is the expensive one
    exam_min_face_crop_px: int = 64

    # -- SOS signal for help ------------------------------------------------ #
    sos_window_s: float = 3.0            # palm -> tuck -> fold must fit in here
    sos_hold_s: float = 0.25             # each stage must be held this long
    sos_hands_every_n: int = 2
    sos_min_hand_crop_px: int = 48

    # -- fire exit --------------------------------------------------------- #
    exit_blocked_seconds: float = 15.0
    exit_person_still_bh_s: float = 0.35  # slower than this counts as standing

    # -- line counting ----------------------------------------------------- #
    line_cross_margin: float = 0.012     # normalised dead-band around the line
    line_min_track_age_s: float = 0.4

    # -- after hours -------------------------------------------------------- #
    after_hours_open: str = "07:00"
    after_hours_close: str = "21:00"
    after_hours_grace_s: float = 10.0
    after_hours_empty_room_s: float = 120.0  # no one seen -> lights-on hint

    # -- alert hygiene ------------------------------------------------------ #
    cooldown_s: float = 45.0             # per camera + type + track
    cooldown_global_s: float = 8.0       # per camera + type, any track
    track_memory_s: float = 6.0          # keep a vanished track this long
    history_seconds: float = 8.0         # per-track motion history kept

    label: str = "live"


#: Values swapped in when demo mode is on. Only timers — never the geometry,
#: so what the judges see triggering is the real rule, just impatient.
DEMO_OVERRIDES: dict[str, object] = {
    "loiter_seconds": 8.0,
    "abandoned_seconds": 10.0,
    "fall_down_seconds": 1.2,
    "exit_blocked_seconds": 6.0,
    "crowd_sustain_s": 1.2,
    "crowd_warn": 3,
    "crowd_critical": 5,
    "exam_phone_seconds": 1.5,
    "exam_yaw_seconds": 2.0,
    "after_hours_grace_s": 3.0,
    "after_hours_empty_room_s": 30.0,
    "run_min_people": 1,
    "run_sustain_s": 0.3,
    "cooldown_s": 15.0,
    "cooldown_global_s": 4.0,
    "crowd_sample_interval_s": 4.0,
    "label": "demo",
}

LIVE_THRESHOLDS = Thresholds()
DEMO_THRESHOLDS = replace(Thresholds(), **DEMO_OVERRIDES)  # type: ignore[arg-type]

_demo_mode = _env_bool("DEMO_MODE", False)
ACTIVE: Thresholds = DEMO_THRESHOLDS if _demo_mode else LIVE_THRESHOLDS


def set_demo_mode(on: bool) -> Thresholds:
    """Flip the whole system between stage timings and realistic timings."""
    global _demo_mode, ACTIVE
    _demo_mode = bool(on)
    ACTIVE = DEMO_THRESHOLDS if _demo_mode else LIVE_THRESHOLDS
    return ACTIVE


def demo_mode() -> bool:
    return _demo_mode


def thresholds() -> Thresholds:
    """Always call this rather than caching ``ACTIVE`` — demo mode is live."""
    return ACTIVE


# --------------------------------------------------------------------------- #
# Alert catalogue — titles, icons and default severity in one table
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AlertSpec:
    key: str
    title: str
    icon: str
    severity: str
    detail: str
    clip: bool = True      # save an incident clip when this fires
    silent: bool = False   # no dashboard chime (used by the SOS signal)


ALERT_SPECS: dict[str, AlertSpec] = {
    a.key: a for a in [
        AlertSpec("fall", "Possible fall", "fall", "critical",
                  "A person dropped suddenly and stayed down."),
        AlertSpec("sos", "Signal for Help", "sos", "critical",
                  "The silent hand signal for help was recognised.", silent=True),
        AlertSpec("restricted", "Restricted zone entry", "restricted", "warning",
                  "Someone stepped into a zone marked restricted."),
        AlertSpec("loiter", "Loitering", "loiter", "warning",
                  "A person stayed in one place far longer than normal."),
        AlertSpec("abandoned", "Unattended bag", "bag", "warning",
                  "A bag has been still, with nobody near it."),
        AlertSpec("crowd_warning", "Crowd building", "crowd", "warning",
                  "A zone is holding more people than it should."),
        AlertSpec("crowd_critical", "Crowd critical", "crowd", "critical",
                  "A zone is dangerously crowded."),
        AlertSpec("panic", "Possible panic", "run", "critical",
                  "Several people started moving fast the same way at once."),
        AlertSpec("exit_blocked", "Fire exit blocked", "exit", "critical",
                  "Something is standing in a fire exit."),
        AlertSpec("after_hours", "After-hours occupancy", "moon", "warning",
                  "Someone is present outside opening hours."),
        AlertSpec("lights_on", "Lights left on", "bulb", "info",
                  "An empty room is still lit.", clip=False),
        AlertSpec("exam_phone", "Exam flag: phone visible", "phone", "warning",
                  "A phone-shaped object appeared at a desk. For invigilator review."),
        AlertSpec("exam_gaze", "Exam flag: sustained head turn", "gaze", "info",
                  "A head stayed turned away from the desk. For invigilator review."),
        AlertSpec("line_cross", "Occupancy change", "line", "info",
                  "Someone crossed a counting line.", clip=False),
        AlertSpec("raw_view", "Raw view unlocked", "unlock", "info",
                  "A staff member unlocked the raw feed.", clip=False),
    ]
}


def alert_spec(key: str) -> AlertSpec:
    return ALERT_SPECS.get(
        key, AlertSpec(key, key.replace("_", " ").title(), "info", "info", "")
    )


def severity_rank(sev: str) -> int:
    try:
        return SEVERITY_ORDER.index(sev)
    except ValueError:
        return 0


# --------------------------------------------------------------------------- #
# Zone types
# --------------------------------------------------------------------------- #

ZONE_TYPES: dict[str, dict[str, object]] = {
    "restricted": {"label": "Restricted", "colour": "#e0574f", "shape": "polygon"},
    "loiter":     {"label": "Loiter watch", "colour": "#d9a13b", "shape": "polygon"},
    "desk":       {"label": "Exam desk", "colour": "#6f9bd1", "shape": "polygon"},
    "crowd":      {"label": "Crowd zone", "colour": "#8f6fc4", "shape": "polygon"},
    "exit":       {"label": "Fire exit", "colour": "#4fb286", "shape": "polygon"},
    "line":       {"label": "Counting line", "colour": "#c9c2b4", "shape": "line"},
}


# --------------------------------------------------------------------------- #
# Palette — the renderer and the dashboard share one set of colours
# --------------------------------------------------------------------------- #

# Colours are BGR, because OpenCV is. The look is a da Vinci anatomical study:
# an ivory figure drawn on near-black, lit by a warm amber bloom, with gold
# reserved for lettering so that the bones and the labels never compete.
PALETTE = {
    "canvas":      (14, 13, 12),      # near-black, very slightly warm
    "grid":        (30, 29, 27),
    "skeleton":    (214, 224, 236),   # ivory bone
    "glow":        (58, 142, 222),    # amber bloom behind the bone
    "joint":       (238, 244, 250),
    "gold":        (74, 160, 214),    # #D6A04A — labels and accents
    "gold_soft":   (92, 140, 180),
    "object":      (150, 170, 175),
    "critical":    (62, 62, 224),
    "warning":     (58, 160, 230),
    "info":        (170, 160, 150),
    "text":        (226, 226, 230),
}


def as_dict() -> dict:
    """Everything the dashboard is allowed to know about the configuration."""
    t = thresholds()
    return {
        "app": {
            "name": APP_NAME,
            "subtitle": APP_SUBTITLE,
            "tagline": APP_TAGLINE,
            "motto": APP_MOTTO,
            "version": APP_VERSION,
            "venue": APP_VENUE,
        },
        "demo_mode": demo_mode(),
        "thresholds": {
            "loiter_seconds": t.loiter_seconds,
            "abandoned_seconds": t.abandoned_seconds,
            "fall_down_seconds": t.fall_down_seconds,
            "exit_blocked_seconds": t.exit_blocked_seconds,
            "crowd_warn": t.crowd_warn,
            "crowd_critical": t.crowd_critical,
            "exam_phone_seconds": t.exam_phone_seconds,
            "exam_yaw_deg": t.exam_yaw_deg,
            "after_hours_open": t.after_hours_open,
            "after_hours_close": t.after_hours_close,
        },
        "zone_types": ZONE_TYPES,
        "alert_specs": {
            k: {"title": v.title, "icon": v.icon, "severity": v.severity,
                "detail": v.detail}
            for k, v in ALERT_SPECS.items()
        },
        "privacy": {
            "raw_buffer_seconds": RAW_BUFFER_SECONDS,
            "clip_pre_seconds": CLIP_PRE_SECONDS,
            "clip_post_seconds": CLIP_POST_SECONDS,
            "retention_days": CLIP_RETENTION_DAYS,
        },
        "integrations": {
            "claude": bool(CLAUDE_API_KEY),
            "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "whatsapp": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
                             and TWILIO_WHATSAPP_TO),
        },
    }
