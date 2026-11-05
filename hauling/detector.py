"""Adapter for a local YOLO vehicle model and persistent ByteTrack."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Observation:
    """One vehicle detection in frame coordinates; unconfirmed tracks have no ID."""

    track_id: int | None
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int | None = None
    class_name: str | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def reference(self, anchor: str) -> tuple[float, float]:
        """Point tested against the area bands.

        ``bottom`` is the ground contact: the bands are lines on the road plane
        and the speed basis is a road-plane distance, so only this point follows
        that geometry. ``center`` is the bbox centroid, kept for sites whose
        calibration was derived from it.
        """
        x = (self.bbox[0] + self.bbox[2]) / 2
        return (x, self.bbox[3] if anchor == "bottom" else (self.bbox[1] + self.bbox[3]) / 2)


def observations_from_boxes(boxes: Any, tracks: np.ndarray, names: dict | None = None) -> list[Observation]:
    """Keep every detection, associating confirmed tracker output by detection index.

    Requires the Ultralytics BYTETracker.update contract of the pinned version
    (pyproject: ultralytics==8.4.145): tracks rows are [x1, y1, x2, y2, track_id,
    conf, det_index]; det_index is the last column. Unconfirmed tracks are absent.
    The detector class of each box is carried through so downstream gates can
    restrict plate OCR to trucks (the model has 5 classes).
    """
    by_index = {int(track[-1]): track for track in tracks}
    classes = boxes.cls.astype(int).tolist() if hasattr(boxes, "cls") and len(boxes.xyxy) else []
    observations = []
    for index, (bbox, confidence) in enumerate(zip(boxes.xyxy, boxes.conf, strict=True)):
        track = by_index.get(index)
        coordinates = bbox if track is None else track[:4]
        class_id = classes[index] if index < len(classes) else None
        observations.append(
            Observation(
                None if track is None else int(track[4]),
                tuple(float(x) for x in coordinates),
                float(confidence),
                class_id,
                None if class_id is None or names is None else str(names.get(class_id, class_id)),
            )
        )
    return observations


def resolve_device(requested: str, cuda_available: bool, warnings: list[str]) -> str:
    """Resolve the requested compute device, recording an explicit CPU fallback."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        requested = "0"
    if not requested.isdigit():
        raise ValueError("--device must be cpu, cuda, or a CUDA device index (for example 0)")
    if not cuda_available:
        warnings.append("CUDA unavailable; using CPU")
        return "cpu"
    return requested


def validate_model_path(path: Path | None, device: str) -> None:
    """Accept only local YOLO weights; TensorRT cannot run on CPU."""
    if path is None or not path.is_file() or path.suffix.lower() not in {".pt", ".engine"}:
        raise ValueError(f"Model must be an existing local .pt or .engine file: {path}")
    if path.suffix.lower() == ".engine" and device == "cpu":
        raise ValueError(f"TensorRT model requires a CUDA GPU: {path}")


def validate_tracker(path: Path) -> dict[str, Any]:
    """Read only a local ByteTrack configuration with validated parameters."""
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get("tracker_type") != "bytetrack":
        raise ValueError("Tracker must use tracker_type: bytetrack")
    result = {"tracker_type": "bytetrack"}
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        value = data.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid tracker parameter: {key}")
        result[key] = value
    if result["track_low_thresh"] > result["track_high_thresh"]:
        raise ValueError("track_low_thresh must not exceed track_high_thresh")
    value = data.get("track_buffer")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("track_buffer must be a positive integer")
    result["track_buffer"] = value
    if not isinstance(data.get("fuse_score"), bool):
        raise ValueError("fuse_score must be boolean")
    result["fuse_score"] = data["fuse_score"]
    return result


class VehicleDetector:
    """Load local weights and update ByteTrack on every frame, including empty frames."""

    def __init__(
        self,
        model_path: Path,
        tracker_path: Path,
        config: Any,
        device: str,
        warnings: list[str],
        runtime_dir: Path,
        threads: int = 4,
    ) -> None:
        tracker_args = validate_tracker(tracker_path)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        # Must precede the first Ultralytics import: no probes, telemetry or auto-installs.
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir.resolve())
        os.environ["YOLO_VERBOSE"] = "false"
        import torch
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"sync": False})
        torch.set_num_threads(threads)
        self.device = resolve_device(device, torch.cuda.is_available(), warnings)
        if self.device != "cpu" and int(self.device) >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device index does not exist")
        validate_model_path(model_path, self.device)
        self.model = YOLO(str(model_path.resolve()), task="detect")
        if model_path.suffix.lower() == ".engine":
            # .names initializes an engine predictor; select the requested GPU first.
            self.model.overrides["device"] = self.device
        # CPU predictor setup resets Torch's thread count. Reapply the explicit
        # run setting after setup and before inference, including the first frame.
        self.model.add_callback("on_predict_start", lambda predictor: torch.set_num_threads(threads))
        self.model_names = self.model.names
        # In the pinned Ultralytics version track_buffer is measured in frames.
        self.tracker = BYTETracker(SimpleNamespace(**tracker_args))
        self.tracker.reset()
        self.config = config

    def detect(self, frame: np.ndarray) -> list[Observation]:
        """Return normalized Vehicle observations without saving model side effects."""
        result = self.model.predict(
            source=frame,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            max_det=self.config.max_det,
            agnostic_nms=self.config.agnostic_nms,
            classes=self.config.classes,
            device=self.device,
            verbose=False,
            save=False,
            save_txt=False,
            save_crop=False,
            show=False,
            augment=False,
        )[0]
        boxes = result.boxes.cpu().numpy()
        tracks = self.tracker.update(boxes, frame)
        return observations_from_boxes(boxes, tracks, self.model_names)
