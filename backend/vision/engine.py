"""
The neural end of the system: boxes, track ids and skeletons.

**Why the models are arranged this way.** The obvious reading of the brief is
"run the detector, then run MediaPipe Pose on each person crop". That is one
inference for the frame plus one *per person*, and MediaPipe Pose handles a
single subject per call — so a corridor with six people costs seven inferences
and the frame rate falls through the floor on a laptop CPU.

So the work is inverted:

* ``yolov8n-pose`` is tracked on **every frame**. One pass returns every
  person in the frame, with a skeleton and a ByteTrack id. People are the
  safety-critical subject, so they get full temporal resolution — a fall lasts
  half a second and cannot be sampled at 4 Hz.
* ``yolov8n`` (full COCO) is tracked on **every third frame**, and only for the
  handful of object classes any rule cares about. A bag that has been
  motionless for thirty seconds does not need to be found sixty times a second.
* MediaPipe is kept for the two jobs it is genuinely unbeaten at — hand
  landmarks for the Signal for Help gesture, and a face mesh with a 3D
  transform matrix for head yaw — and is run only on the small crops that need
  it, only when the feature that needs it is switched on.

Setting ``POSE_BACKEND=mediapipe`` restores the per-crop MediaPipe Pose path if
you want to compare them; the rest of the system cannot tell the difference,
because both write into the same 17-slot layout from
:mod:`backend.vision.skeleton`.

Every model here is instantiated **per camera**. That is not wasteful caution:
ByteTrack keeps its track state inside the model object, so two cameras sharing
one model would shuffle each other's identities.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

import config
from backend.vision import model_store, skeleton as sk
from backend.world import Detection, FrameWorld

os.environ.setdefault("YOLO_VERBOSE", "False")

_IMPORT_LOCK = threading.Lock()
_YOLO = None            # the ultralytics class, imported once
_TORCH_TUNED = False


def tune_torch(streams: int = 1, model_streams: int | None = None
               ) -> dict[str, Any]:
    """Divide the CPU between the camera threads, once, at startup.

    Torch defaults its intra-op pool to nearly every core — sensible for a
    process doing one inference at a time, and actively harmful here. Each
    camera runs inference on its own thread, so three cameras ask for three
    pools of ten threads on a twelve-core machine, and the result is more time
    spent switching between threads than running the model. On the development
    laptop this starved a real webcam down to 2 fps while two scripted scenes
    ran fine.

    Set ``TORCH_THREADS`` in .env to override; 0 means "work it out".
    """
    global _TORCH_TUNED
    cores = os.cpu_count() or 4
    # Only cameras that actually run a model compete for the torch pool.
    # Splitting it by total camera count gave a real webcam three threads
    # while two scripted scenes - which run no model at all - held the rest,
    # and the webcam fell to 3 fps.
    model_streams = streams if model_streams is None else max(1, model_streams)

    # Budget against the cores we actually have. Each camera thread needs some
    # OpenCV parallelism for rendering and JPEG encoding; what the cameras
    # that run *no* model take for that is subtracted, and the rest is split
    # between the cameras that do.
    #
    # Both extremes were measured and both are wrong: handing torch nearly
    # every core made three camera threads fight over twenty threads, and
    # splitting the budget evenly took a lone camera from 13 fps to 8, because
    # when YOLO is the only thing running it should have the machine.
    cv_threads = max(1, min(4, cores // max(2, streams * 2)))
    reserved = cv_threads * max(0, streams - model_streams)
    if config.TORCH_THREADS > 0:
        threads = config.TORCH_THREADS
    else:
        threads = max(2, min(cores - 1,
                             (cores - reserved) // max(1, model_streams)))
    # OpenCV has its own pool, and it is the worse offender of the two: every
    # resize, blur and blend in the renderer fans out across every core, from
    # inside each camera thread. Measured on three cameras, capping it was a
    # 40% throughput gain.
    report = {"cores": cores, "streams": streams,
              "model_streams": model_streams, "threads": threads,
              "cv_threads": cv_threads}
    try:
        import cv2
        cv2.setNumThreads(cv_threads)
    except Exception:                             # noqa: BLE001
        pass
    try:
        import torch
        torch.set_num_threads(threads)
        report["applied"] = True
    except Exception as exc:                      # noqa: BLE001
        report["applied"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    _TORCH_TUNED = True
    return report


def _load_yolo_class():
    global _YOLO
    with _IMPORT_LOCK:
        if _YOLO is None:
            from ultralytics import YOLO          # heavy; imported on demand
            _YOLO = YOLO
        if not _TORCH_TUNED:
            tune_torch(1)
    return _YOLO


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class VisionEngine:
    """Per-camera inference. Not thread-safe; one instance per worker thread."""

    def __init__(self, camera: str) -> None:
        self.camera = camera
        self.ready = False
        self.status: dict[str, Any] = {
            "pose": "pending", "objects": "pending", "backend": config.POSE_BACKEND,
        }
        self._pose_model = None
        self._det_model = None
        self._mp_pose = None
        self._person_class_ids: list[int] = []
        self._object_class_ids: list[int] = []
        self._object_cache: list[Detection] = []
        self._objects_valid = False
        self._frames = 0
        self._next_synthetic_id = 1

    # -- loading ------------------------------------------------------------- #

    def load(self) -> dict[str, Any]:
        """Bring up whatever is available; never raise."""
        YOLO = None
        try:
            YOLO = _load_yolo_class()
        except Exception as exc:                            # noqa: BLE001
            self.status["pose"] = f"unavailable ({type(exc).__name__})"
            self.status["objects"] = self.status["pose"]
            return self.status

        # -- persons + skeletons ------------------------------------------- #
        if config.POSE_BACKEND == "yolo":
            try:
                path = model_store.ensure_yolo(config.YOLO_POSE_WEIGHTS)
                self._pose_model = YOLO(str(path))
                self.status["pose"] = "yolo-pose"
            except Exception as exc:                        # noqa: BLE001
                self.status["pose"] = f"unavailable: {exc}"
        elif config.POSE_BACKEND == "mediapipe":
            from backend.vision import mp_tasks
            self._mp_pose = mp_tasks.PoseLandmarks()
            self.status["pose"] = ("mediapipe" if self._mp_pose.available
                                   else f"unavailable: {self._mp_pose.error}")
        else:
            self.status["pose"] = "off"

        # -- objects (and persons, if we have no pose model) ---------------- #
        try:
            path = model_store.ensure_yolo(config.YOLO_DET_WEIGHTS)
            self._det_model = YOLO(str(path))
            names = self._det_model.names
            wanted = set(config.TRACKED_CLASSES)
            self._object_class_ids = [i for i, n in names.items()
                                      if n in wanted and n != config.CLASS_PERSON]
            self._person_class_ids = [i for i, n in names.items()
                                      if n == config.CLASS_PERSON]
            self.status["objects"] = "yolo-coco"
        except Exception as exc:                            # noqa: BLE001
            self.status["objects"] = f"unavailable: {exc}"

        self.ready = self._pose_model is not None or self._det_model is not None
        return self.status

    @property
    def detects_people(self) -> bool:
        return (self._pose_model is not None or self._mp_pose is not None
                or self._det_model is not None)

    # -- inference ------------------------------------------------------------ #

    def infer(self, frame: np.ndarray, frame_idx: int, ts: float) -> FrameWorld:
        h, w = frame.shape[:2]
        world = FrameWorld(camera=self.camera, ts=ts, frame_idx=frame_idx,
                           width=w, height=h)
        self._frames += 1

        persons: list[Detection] = []
        if self._pose_model is not None:
            persons = self._infer_pose_yolo(frame)
        elif self._det_model is not None:
            persons = self._infer_persons_det(frame)
            if self._mp_pose is not None:
                self._attach_mediapipe_pose(frame, persons)

        world.persons = persons
        world.objects = self._infer_objects(frame)
        return world

    # -- persons ---------------------------------------------------------------- #

    def _infer_pose_yolo(self, frame: np.ndarray) -> list[Detection]:
        res = self._pose_model.track(
            frame, persist=True, tracker=config.TRACKER_CFG,
            imgsz=config.INFER_SIZE, conf=config.POSE_CONF, verbose=False,
        )[0]
        return self._persons_from_result(res, with_keypoints=True)

    def _infer_persons_det(self, frame: np.ndarray) -> list[Detection]:
        if not self._person_class_ids:
            return []
        res = self._det_model.track(
            frame, persist=True, tracker=config.TRACKER_CFG,
            classes=self._person_class_ids, imgsz=config.INFER_SIZE,
            conf=config.DET_CONF, verbose=False,
        )[0]
        return self._persons_from_result(res, with_keypoints=False)

    def _persons_from_result(self, res, with_keypoints: bool) -> list[Detection]:
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = (boxes.conf.cpu().numpy() if boxes.conf is not None
                 else np.ones(len(xyxy), np.float32))
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            # Tracking can decline to assign ids on the very first frames or
            # when a detection is too weak. A detection with no identity is
            # still worth reporting, so give it a negative placeholder that can
            # never collide with a real ByteTrack id.
            ids = np.array([-(i + 1) for i in range(len(xyxy))], dtype=int)

        kps_all = None
        if with_keypoints:
            kp = getattr(res, "keypoints", None)
            if kp is not None and kp.data is not None and len(kp.data):
                kps_all = kp.data.cpu().numpy()

        out: list[Detection] = []
        for i, box in enumerate(xyxy):
            kps = None
            if kps_all is not None and i < len(kps_all):
                arr = np.asarray(kps_all[i], dtype=np.float32)
                if arr.shape == (sk.N_KP, 3):
                    kps = arr
                elif arr.shape == (sk.N_KP, 2):      # model without confidences
                    kps = np.concatenate(
                        [arr, np.ones((sk.N_KP, 1), np.float32)], axis=1)
            out.append(Detection(
                track_id=int(ids[i]), label=config.CLASS_PERSON,
                conf=float(confs[i]),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                keypoints=kps,
            ))
        return out

    def _attach_mediapipe_pose(self, frame: np.ndarray,
                               persons: list[Detection]) -> None:
        """The per-crop path: one MediaPipe call per person, mapped back."""
        if self._mp_pose is None or not self._mp_pose.available:
            return
        h, w = frame.shape[:2]
        for det in persons:
            x1, y1, x2, y2 = (int(v) for v in det.box)
            pad = int(0.06 * max(x2 - x1, y2 - y1))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            if x2 - x1 < 24 or y2 - y1 < 48:
                continue
            crop = frame[y1:y2, x1:x2]
            lms = self._mp_pose.landmarks(crop)
            if lms is None:
                continue
            kps = sk.from_mediapipe(lms, x2 - x1, y2 - y1)
            kps[kps[:, 2] > 0, 0] += x1          # back to frame coordinates
            kps[kps[:, 2] > 0, 1] += y1
            det.keypoints = kps

    # -- objects ----------------------------------------------------------------- #

    def _infer_objects(self, frame: np.ndarray) -> list[Detection]:
        if self._det_model is None or not self._object_class_ids:
            return []
        every = max(1, config.OBJ_EVERY_N)
        # Note the cache validity flag rather than a truthiness test on the
        # list: "we looked and found nothing" is a perfectly good answer to
        # reuse, and testing the list meant a camera with no bags in view ran
        # the object model on every single frame.
        if self._objects_valid and (self._frames - 1) % every != 0:
            # Between runs, reuse the last answer. Bags and phones do not move
            # fast enough for this to matter, and it buys back a third of the
            # frame budget for the people.
            return self._object_cache

        res = self._det_model.track(
            frame, persist=True, tracker=config.TRACKER_CFG,
            classes=self._object_class_ids, imgsz=config.INFER_SIZE,
            conf=config.DET_CONF, verbose=False,
        )[0]
        names = self._det_model.names
        boxes = getattr(res, "boxes", None)
        out: list[Detection] = []
        if boxes is not None and boxes.xyxy is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = np.array([-(i + 1) for i in range(len(xyxy))], dtype=int)
            for i, box in enumerate(xyxy):
                out.append(Detection(
                    track_id=int(ids[i]), label=str(names.get(int(clss[i]), "?")),
                    conf=float(confs[i]),
                    box=(float(box[0]), float(box[1]),
                         float(box[2]), float(box[3])),
                ))
        self._object_cache = out
        self._objects_valid = True
        return out


# --------------------------------------------------------------------------- #
# Simulated worlds
# --------------------------------------------------------------------------- #

def world_from_sim(camera: str, payload: dict[str, Any], frame_idx: int,
                   ts: float, width: int, height: int) -> FrameWorld:
    """Turn a scripted scene's dictionary into the same FrameWorld a camera gives.

    Track ids from simulated scenes are offset into a high range so that they
    can never be confused with ByteTrack ids in logs or in the event database.
    """
    world = FrameWorld(camera=camera, ts=ts, frame_idx=frame_idx,
                       width=width, height=height, simulated=True,
                       note=str(payload.get("note", "")))
    for src, bucket in (("persons", world.persons), ("objects", world.objects)):
        for item in payload.get(src, []):
            box = tuple(float(v) for v in item["box"])
            kps = item.get("keypoints")
            bucket.append(Detection(
                track_id=900_000 + int(item.get("track_id", 0)),
                label=str(item.get("label", "person")),
                conf=float(item.get("conf", 0.9)),
                box=box,                                    # type: ignore[arg-type]
                keypoints=None if kps is None else np.asarray(kps, np.float32),
            ))
    return world
