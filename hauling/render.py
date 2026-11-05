"""Video overlays for reviewing offline measurements."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hauling.detector import Observation

AREA_COLORS = {
    "main": (70, 180, 70),
    "start": (255, 190, 0),
    "finish": (0, 200, 255),
    "final": (220, 80, 220),
    "contraflow": (0, 0, 220),
}

COLOR_OK = (40, 220, 40)  # green: past the start band and within the limit
COLOR_ALERT = (30, 30, 240)  # red: overspeed, or still before the start band


def _status_color(measurement: dict, state: Any) -> tuple[int, int, int]:
    """Two-color contract for the vehicle box.

    Red while the vehicle is only inside ``main`` (no start crossing yet),
    whenever the measurement is overspeed, and for a vehicle flagged with an
    unsafe final-area headway or a contraflow violation; green once the start
    band has been crossed and no flag is set.
    """
    if (
        measurement.get("is_overspeed")
        or getattr(state, "is_unsafe_distance", False)
        or getattr(state, "is_contraflow", False)
    ):
        return COLOR_ALERT
    if state is not None and getattr(state, "start_timestamp", None) is not None:
        return COLOR_OK
    return COLOR_ALERT


def draw_frame(
    frame: np.ndarray,
    polygons: dict[str, np.ndarray],
    observations: Sequence[Observation],
    states: dict[int, Any],
    measurements: dict[int, dict],
    timestamp: float,
    plate_enabled: bool = False,
    anchor: str = "bottom",
    video_fps: float | None = None,
    processing_fps: float | None = None,
) -> np.ndarray:
    """Draw all areas, vehicle IDs and speed/status on a frame copy."""
    result = frame.copy()
    for name, polygon in polygons.items():
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(result, [points], True, AREA_COLORS[name], 2)
        x, y = points[0]
        # Keep the band label inside its right edge and above the band.
        (label_w, _), _ = cv2.getTextSize(name.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x = min(x, int(points[:, 0].max()) - label_w - 2)
        cv2.putText(
            result,
            name.upper(),
            (max(0, int(x)), max(18, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            AREA_COLORS[name],
            2,
        )
    for observation in observations:
        x1, y1, x2, y2 = (int(x) for x in observation.bbox)
        measurement = measurements.get(observation.track_id, {})
        state = states.get(observation.track_id)
        color = _status_color(measurement, state)
        track_label = str(observation.track_id) if observation.track_id is not None else "untracked"
        speed = measurement.get("speed_kmh")
        label = f"Vehicle {track_label}"
        if state:
            label += f" {state.direction}"
            fresh_box = getattr(state, "lambung_box", None)
            fresh_ts = getattr(state, "lambung_box_ts", None)
            if fresh_box is not None and fresh_ts is not None and timestamp - fresh_ts <= 0.121:
                # Relative projection drifts as the truck approaches, so show only a fresh box.
                px1, py1, px2, py2 = (int(v) for v in fresh_box)
                cv2.rectangle(result, (px1, py1), (px2, py2), (255, 220, 0), 2)
                confirmed = getattr(state, "lambung_result", lambda: None)()
                plate_label = confirmed[0] if confirmed is not None else "plate"
                cv2.putText(
                    result, plate_label, (px1, max(12, py1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1
                )
        label += f" {speed:.2f} km/h" if speed is not None else f" {measurement.get('speed_status', '')}"
        if getattr(state, "is_unsafe_distance", False):
            label += " UNSAFE DIST"
        if getattr(state, "is_contraflow", False):
            label += " CONTRAFLOW"
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.circle(result, tuple(int(x) for x in observation.reference(anchor)), 3, color, -1)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_x = max(0, min(x1, result.shape[1] - text_size[0] - 8))
        label_y = y1 - 8 if y1 >= text_size[1] + 12 else y1 + text_size[1] + 8
        label_y = max(text_size[1] + 4, min(label_y, result.shape[0] - baseline - 4))
        cv2.rectangle(
            result,
            (label_x, label_y - text_size[1] - 4),
            (label_x + text_size[0] + 8, label_y + baseline + 3),
            (20, 20, 20),
            -1,
        )
        cv2.putText(result, label, (label_x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    height, width = result.shape[:2]
    box_right = min(460, width - 8)
    box_bottom = min(64, height - 8)
    cv2.rectangle(result, (8, 8), (box_right, box_bottom), (20, 20, 20), -1)
    cv2.putText(
        result,
        f"Video time {timestamp:.2f}s | Plate: {'enabled' if plate_enabled else 'disabled'}",
        (16, min(31, box_bottom - 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    footer = (
        f"Source {video_fps:.1f} FPS | Process {processing_fps:.1f} FPS"
        if video_fps is not None and processing_fps is not None
        else "Centroid measurement | Offline hauling"
    )
    cv2.putText(result, footer, (16, min(53, box_bottom - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return result


def open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> Any:
    """Open a local MPEG-4 writer; callers can continue CSV on codec failure."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise OSError("Video writer could not open mp4v codec")
    return writer
