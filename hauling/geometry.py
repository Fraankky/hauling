"""Reference-resolution polygon transforms and inclusive membership checks."""

from collections.abc import Mapping, Sequence

import cv2
import numpy as np


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
