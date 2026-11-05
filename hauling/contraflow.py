"""Contraflow (wrong-way) detection from an area polygon and track direction.

A vehicle is contraflow when it faces the camera (``direction``) while the
majority of its bounding box sits in the lane reserved for oncoming traffic.
The check runs on every frame, but only near frames count: a 16x19 px box at
the horizon can cover 100% of a polygon's narrow tip, which would flag normal
traffic. ``near`` is supplied by the caller (the same closest-approach gate the
plate reader uses) so this module stays free of detector/OCR imports.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Frames that must agree before a violation is reported. One frame is not
# evidence: a tracker jump can put a box inside the polygon for a single frame.
MIN_CONFIRM_FRAMES = 3

# Sample grid inside the bbox. A single point (the anchor) is too easy to
# nudge into the polygon; 5x5 asks for most of the body, matching how real
# wrong-way systems use an overlap ratio rather than point membership.
COVERAGE_GRID = 5

STATUS_INACTIVE = "inactive"
STATUS_OUTSIDE = "outside"
STATUS_DIRECTION_UNKNOWN = "direction_unknown"
STATUS_LEGAL_FLOW = "legal_flow"
STATUS_UNCONFIRMED = "unconfirmed"
STATUS_VIOLATION = "violation"


@dataclass(frozen=True)
class ContraflowOutcome:
    """Verdict for one observation, plus the evidence that produced it.

    ``coverage`` is the largest fraction seen on a near frame, so a CSV row can
    be re-thresholded without another inference run. ``timestamp``/
    ``frame_index`` mark the frame where the violation was confirmed and are
    ``None`` until then.
    """

    coverage: float
    status: str
    is_contraflow: bool
    timestamp: float | None
    frame_index: int | None


def bbox_coverage(
    polygon: np.ndarray, bbox: tuple[float, float, float, float], grid: int = COVERAGE_GRID
) -> float:
    """Fraction of a ``grid x grid`` sample of the bbox that lies in the polygon."""
    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        return 0.0
    step = grid - 1
    inside = 0
    for i in range(grid):
        x = x1 + (x2 - x1) * i / step
        for j in range(grid):
            y = y1 + (y2 - y1) * j / step
            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
                inside += 1
    return inside / (grid * grid)


class ContraflowDetector:
    """Track-level wrong-way verdict with temporal confirmation.

    The verdict is sticky: once confirmed, a track stays a violation and the
    evidence frame is frozen, so one vehicle produces one row and one snapshot.
    """

    def __init__(
        self,
        polygon: np.ndarray,
        direction: str = "South",
        min_coverage: float = 0.6,
        min_frames: int = MIN_CONFIRM_FRAMES,
    ) -> None:
        if direction not in {"South", "North"}:
            raise ValueError("Contraflow direction must be South or North")
        if not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
            raise ValueError("Contraflow min_coverage must be in (0, 1]")
        if min_frames < 1:
            raise ValueError("Contraflow min_frames must be positive")
        self.polygon = np.ascontiguousarray(np.asarray(polygon, dtype=np.float32))
        self.direction = direction
        self.min_coverage = min_coverage
        self.min_frames = min_frames
        self._tracks: dict[int, dict] = {}

    def update(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        direction: str,
        timestamp: float,
        frame_index: int,
        near: bool = True,
    ) -> ContraflowOutcome:
        """Evaluate one observation; only ``near`` frames advance the streak."""
        state = self._tracks.setdefault(
            track_id,
            {"coverage": 0.0, "streak": 0, "status": STATUS_OUTSIDE, "timestamp": None, "frame": None},
        )
        if state["status"] != STATUS_VIOLATION and near:
            coverage = bbox_coverage(self.polygon, bbox)
            state["coverage"] = max(state["coverage"], coverage)
            if coverage < self.min_coverage:
                state["status"] = STATUS_OUTSIDE
                state["streak"] = 0
            elif direction not in {"South", "North"}:
                # A stationary or just-spawned track has no direction yet.
                # Reporting legal_flow here would be a false reassurance.
                state["status"] = STATUS_DIRECTION_UNKNOWN
                state["streak"] = 0
            elif direction != self.direction:
                # The oncoming lane legitimately carries the opposite flow; only
                # a vehicle moving *with* the camera's direction there is wrong.
                state["status"] = STATUS_LEGAL_FLOW
                state["streak"] = 0
            else:
                state["streak"] += 1
                if state["streak"] >= self.min_frames:
                    state["status"] = STATUS_VIOLATION
                    state["timestamp"] = timestamp
                    state["frame"] = frame_index
                else:
                    state["status"] = STATUS_UNCONFIRMED
        return ContraflowOutcome(
            coverage=round(state["coverage"], 3),
            status=state["status"],
            is_contraflow=state["status"] == STATUS_VIOLATION,
            timestamp=state["timestamp"],
            frame_index=state["frame"],
        )
