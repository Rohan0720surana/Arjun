"""
MediaPipe, used where MediaPipe is the best tool in the room.

Two capabilities in this project genuinely need landmarks that a 17-point body
skeleton cannot provide:

* **Hands** — the Signal for Help gesture is defined by where the *thumb* is
  relative to the palm and the fingers. A wrist keypoint tells you nothing
  about it. :class:`HandLandmarks` gives 21 points per hand.
* **Face** — exam mode's head-turn flag needs a yaw angle, and the honest way
  to get one is the 3D transformation matrix the face landmarker can emit,
  rather than guessing from how far apart two ears look.
  :class:`FaceGeometry` returns yaw/pitch/roll in degrees.

MediaPipe 1.x dropped the old ``mp.solutions`` API, so both classes use the
Tasks API with ``.task`` bundles fetched by :mod:`backend.vision.model_store`.

Everything is lazy and forgiving. The models are only built the first time a
feature that needs them is switched on, and if the bundle cannot be loaded the
object reports ``available = False`` with a readable reason, the feature
switches itself off in the UI, and the rest of the system is untouched.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any

import numpy as np

import config
from backend.vision import model_store

# MediaPipe is chatty on stderr about XNNPACK and feedback tensors; none of it
# is actionable and all of it scrolls a demo console.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_MP_LOCK = threading.Lock()
_MP: dict[str, Any] = {}


def _mp_vision():
    """Import MediaPipe once, on first use."""
    with _MP_LOCK:
        if "vision" not in _MP:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            _MP["mp"] = mp
            _MP["mpp"] = mpp
            _MP["vision"] = vision
    return _MP["mp"], _MP["mpp"], _MP["vision"]


def _as_mp_image(bgr: np.ndarray):
    mp = _MP["mp"]
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


class _Task:
    """Shared lazy-load / fail-soft behaviour."""

    task_file = ""

    def __init__(self) -> None:
        self.error = ""
        self._impl: Any = None
        self._tried = False

    @property
    def available(self) -> bool:
        if not self._tried:
            self._build()
        return self._impl is not None

    def _build(self) -> None:
        self._tried = True
        try:
            path = model_store.ensure_mp_task(self.task_file)
            self._impl = self._create(str(path))
        except Exception as exc:                            # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self._impl = None

    def _create(self, path: str) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        impl = self._impl
        self._impl = None
        if impl is not None:
            try:
                impl.close()
            except Exception:                               # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Hands
# --------------------------------------------------------------------------- #

#: MediaPipe hand landmark indices, named so the gesture rule reads as prose.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
FINGER_MCPS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


class HandLandmarks(_Task):
    """21 landmarks per hand, in the coordinate frame of whatever crop is given."""

    task_file = config.MP_HAND_TASK

    def __init__(self, max_hands: int = 2) -> None:
        super().__init__()
        self.max_hands = max_hands

    def _create(self, path: str):
        mp, mpp, vision = _mp_vision()
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=path),
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=vision.RunningMode.IMAGE,
        )
        return vision.HandLandmarker.create_from_options(opts)

    def detect(self, crop: np.ndarray) -> list[np.ndarray]:
        """Return a list of (21, 3) arrays in crop pixels, z left normalised."""
        if not self.available or crop.size == 0:
            return []
        h, w = crop.shape[:2]
        if w < 16 or h < 16:
            return []
        try:
            res = self._impl.detect(_as_mp_image(crop))
        except Exception as exc:                            # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return []
        out: list[np.ndarray] = []
        for hand in getattr(res, "hand_landmarks", []) or []:
            arr = np.array([[lm.x * w, lm.y * h, lm.z] for lm in hand],
                           dtype=np.float32)
            out.append(arr)
        return out


# --------------------------------------------------------------------------- #
# Face
# --------------------------------------------------------------------------- #

class FaceGeometry(_Task):
    """Head orientation from the face landmarker's 3D transform matrix."""

    task_file = config.MP_FACE_TASK

    def __init__(self, max_faces: int = 1) -> None:
        super().__init__()
        self.max_faces = max_faces

    def _create(self, path: str):
        mp, mpp, vision = _mp_vision()
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=path),
            num_faces=self.max_faces,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE,
        )
        return vision.FaceLandmarker.create_from_options(opts)

    def head_angles(self, crop: np.ndarray) -> tuple[float, float, float] | None:
        """``(yaw, pitch, roll)`` in degrees, or ``None`` if no face is found.

        Yaw is what exam mode uses: 0 is facing the camera, positive is a turn
        to one side. Only the angles leave this method — no landmarks, no
        descriptor, nothing that could identify whose face it was.
        """
        if not self.available or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if min(w, h) < 24:
            return None
        try:
            res = self._impl.detect(_as_mp_image(crop))
        except Exception as exc:                            # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return None

        mats = getattr(res, "facial_transformation_matrixes", None) or []
        if mats:
            return _euler_from_matrix(np.asarray(mats[0], dtype=np.float64))

        # The matrix is normally there, but if the build was compiled without
        # it we can still recover a usable yaw from the landmark geometry.
        faces = getattr(res, "face_landmarks", None) or []
        if not faces:
            return None
        return _euler_from_landmarks(faces[0])


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 4x4 head pose matrix into yaw, pitch and roll in degrees."""
    r = m[:3, :3]
    sy = math.sqrt(float(r[0, 0]) ** 2 + float(r[1, 0]) ** 2)
    if sy > 1e-6:
        pitch = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(-float(r[2, 0]), sy)
        roll = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:                                                   # gimbal lock
        pitch = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        yaw = math.atan2(-float(r[2, 0]), sy)
        roll = 0.0
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


#: Landmark indices in MediaPipe's 478-point face mesh.
_L_EYE_OUTER, _R_EYE_OUTER, _NOSE_TIP, _CHIN = 263, 33, 1, 152


def _euler_from_landmarks(face) -> tuple[float, float, float] | None:
    """Fallback yaw: how far the nose sits from the midpoint of the eyes."""
    try:
        le, re = face[_L_EYE_OUTER], face[_R_EYE_OUTER]
        nose, chin = face[_NOSE_TIP], face[_CHIN]
    except (IndexError, TypeError):
        return None
    eye_mid_x = (le.x + re.x) / 2.0
    eye_span = abs(le.x - re.x) or 1e-6
    # Offset of the nose from centre, as a fraction of eye separation. The
    # factor converts that ratio into something close to degrees; it is a
    # rough instrument, which is why the matrix path is preferred.
    yaw = float(np.clip((nose.x - eye_mid_x) / eye_span, -1.2, 1.2)) * 75.0
    eye_mid_y = (le.y + re.y) / 2.0
    face_h = abs(chin.y - eye_mid_y) or 1e-6
    pitch = float(np.clip((nose.y - eye_mid_y) / face_h - 0.38, -1.0, 1.0)) * 60.0
    roll = math.degrees(math.atan2(float(re.y - le.y), float(re.x - le.x)))
    return (yaw, pitch, roll)


# --------------------------------------------------------------------------- #
# Pose (the alternative backend)
# --------------------------------------------------------------------------- #

class PoseLandmarks(_Task):
    """MediaPipe Pose, used only when ``POSE_BACKEND=mediapipe``."""

    task_file = config.MP_POSE_TASK

    def _create(self, path: str):
        mp, mpp, vision = _mp_vision()
        opts = vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=path),
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE,
        )
        return vision.PoseLandmarker.create_from_options(opts)

    def landmarks(self, crop: np.ndarray):
        """Normalised landmarks for the single largest person in the crop."""
        if not self.available or crop.size == 0:
            return None
        try:
            res = self._impl.detect(_as_mp_image(crop))
        except Exception as exc:                            # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return None
        poses = getattr(res, "pose_landmarks", None) or []
        return poses[0] if poses else None
