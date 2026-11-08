"""Reference-resolution polygon transforms and inclusive membership checks."""

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

MIN_CROP_SIZE = 16


def scale_polygons(
    polygons: Mapping[str, Sequence[Sequence[float]]],
    basis_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Scale each polygon independently along x and y; sizes are (width, height)."""
    if any(value <= 0 for value in (*basis_size, *frame_size)):
        raise ValueError("Image dimensions must be positive")
    factors = np.asarray(frame_size, dtype=np.float64) / np.asarray(basis_size)
    return {
        name: np.ascontiguousarray((np.asarray(points, dtype=np.float64) * factors).astype(np.float32))
        for name, points in polygons.items()
    }


def contains(polygon: np.ndarray, point: tuple[float, float]) -> bool:
    """Return whether a point is inside a polygon or on its boundary.

    polygon must already be a float32 array (as produced by scale_polygons);
    conversion is deliberately not repeated here — this runs per point per frame.
    """
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def band_y(polygons: Mapping[str, np.ndarray], name: str) -> tuple[float, float]:
    """Return the vertical extent of one crossing band."""
    points = polygons[name]
    return float(points[:, 1].min()), float(points[:, 1].max())


def clip_vehicle_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Clip a vehicle bbox to a frame and reject degenerate crops."""
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(width, int(x2)), min(height, int(y2))
    if ix2 - ix1 < MIN_CROP_SIZE or iy2 - iy1 < MIN_CROP_SIZE:
        return None
    return ix1, iy1, ix2, iy2
