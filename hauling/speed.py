"""Calibrated crossing-time speed measurements and the overspeed flag."""

import math


def calculate_speed(
    start: float | None,
    end: float | None,
    distance: float,
    limit: float,
    status: str,
) -> dict[str, float | str | bool | None]:
    """Measure speed for valid ordered crossings and flag overspeed.

    A measurement is valid only for a correctly ordered crossing pair with a
    positive interval; otherwise the numeric columns stay empty and
    ``speed_status`` explains why. ``is_overspeed`` is ``speed > limit`` using
    the **unrounded** speed (comparing the two-decimal value would forgive
    breaches smaller than 0.01 km/h); an invalid measurement is never overspeed.
    The only boundary allowance is a floating-point noise guard, not a tolerance.
    ``speed_kmh_raw`` carries the unrounded value and is not a CSV column.
    """
    result = {
        "elapsed_seconds": None,
        "speed_kmh": None,
        "speed_kmh_raw": None,
        "speed_status": status,
        "measurement_valid": False,
        "is_overspeed": False,
    }
    if status != "valid":
        return result
    if start is None or end is None or not math.isfinite(end - start) or end <= start:
        result["speed_status"] = "invalid_elapsed_time"
        return result
    if not math.isfinite(distance) or distance <= 0 or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Distance and speed limit must be finite and positive")
    elapsed = end - start
    raw_speed = distance / elapsed * 3.6
    if not math.isfinite(raw_speed):
        result["speed_status"] = "invalid_elapsed_time"
        return result
    return {
        "elapsed_seconds": elapsed,
        "speed_kmh": round(raw_speed, 2),
        "speed_kmh_raw": raw_speed,
        "speed_status": "valid",
        "measurement_valid": True,
        # Guard only floating-point noise at the boundary: 60.0 computed as
        # 60.000000000001 is "at the limit", not a violation.
        "is_overspeed": raw_speed > limit and not math.isclose(raw_speed, limit, rel_tol=1e-12, abs_tol=1e-12),
    }
