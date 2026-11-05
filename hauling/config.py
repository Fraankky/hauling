"""Read only the offline inference allowlist from the legacy INI configuration."""

import configparser
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hauling.plate import PREPROCESS_MODES

POLYGON_KEYS = {
    "main": "main_area_coordinates_list",
    "start": "speed_area_start_coordinates_list",
    "finish": "speed_area_final_coordinates_list",
    "final": "final_area_coordinates_list",
}

CONTRAFLOW_KEY = "contraflow_area_coordinates_list"

AREA_NAMES = frozenset({"main", "start", "finish", "final", "contraflow"})


@dataclass(frozen=True)
class Config:
    """Validated offline options; unrelated source settings are never retained."""

    resolution: tuple[int, int]
    polygons: dict[str, list[list[float]]]
    distance_meter: float
    speed_limit_kmh: float
    conf: float
    iou: float
    imgsz: int
    max_det: int
    agnostic_nms: bool
    classes: list[int] | None
    device: str
    tracker_config: str
    plate_hue_windows: tuple[tuple[tuple[int, int], ...], ...] = ()
    # Hazy signs use the lower saturation floor only after strict localization fails.
    plate_sat_floor: int = 90
    plate_sat_floor_hazy: int = 40
    plate_val_floor: int = 80
    plate_band_h_min: int = 10
    plate_band_h_max: int = 30
    plate_band_h_max_hazy: int = 90
    plate_letters: int = 3
    plate_digits: int = 4
    plate_min_digits: int = 3
    plate_min_agreement: int = 2
    # Measured sharpness: signs 45-506, road/blank crops 0.2-15.
    plate_min_sharpness: float = 5.0
    # A single close frame can be all the evidence available; None disables this exception.
    plate_single_read_prob: float = 0.95
    # Without agreement, accept one read only when both confidence and sharpness pass.
    plate_trusted_min_confidence: float = 0.90
    plate_trusted_min_sharpness: float = 1000.0
    # Limit search to the cab sign, not other red paint on the vehicle.
    plate_roi: tuple[float, float, float, float] = (0.40, 0.40, 0.99, 0.90)
    plate_pattern: str = ""
    plate_band_pad: int = 4
    plate_upscale: float = 4.0
    plate_char_imgsz: int = 224
    plate_classifier_imgsz: int = 96
    plate_min_crop_size: int = 16
    plate_char_min_conf: float = 0.0
    plate_preprocess: str = "none"
    plate_deskew: bool = False
    plate_max_ocr_frames: int = 0
    plate_dump_max_per_track: int = 8
    # Opt-in finish fallback changes the event timestamp when final was never crossed.
    event_on_finish_fallback: bool = False
    unsafe_distance_limit_s: float = 3.0
    # Road-plane calibration requires the ground contact, not the elevated bbox center.
    vehicle_anchor: str = "bottom"
    plate_ocr_classes: tuple[str, ...] = ()
    contraflow_polygon: list[list[float]] | None = None
    contraflow_direction: str = "South"
    contraflow_min_coverage: float = 0.6
    # An uncalibrated site may hide an area overlay without disabling its detection.
    draw_areas: tuple[str, ...] = ()


def _number(value: str, key: str, *, probability: bool = False) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}") from None
    if not math.isfinite(number) or (not 0 <= number <= 1 if probability else number <= 0):
        raise ValueError(f"Invalid {key}")
    return number


def _positive_integer(value: str, key: str) -> int:
    number = _number(value, key)
    if not number.is_integer():
        raise ValueError(f"Invalid {key}")
    return int(number)


def _polygon(value: str, key: str) -> list[list[float]]:
    try:
        points = json.loads(value)
        if not isinstance(points, list) or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(type(coordinate) not in (int, float) for coordinate in point)
            for point in points
        ):
            raise ValueError(f"Invalid {key}")
        polygon = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"Invalid {key}") from None
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3 or not np.isfinite(polygon).all():
        raise ValueError(f"Invalid {key}")
    x, y = polygon[:, 0], polygon[:, 1]
    area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
    if not math.isfinite(area) or area <= 0:
        raise ValueError(f"Degenerate {key}")
    return polygon.tolist()


def _hue_windows(value: str, key: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Parse optional plate hue windows: JSON list of windows, each a list of
    [h_start, h_end] pairs. Absent/empty means "use the calibrated default"."""
    if not value.strip():
        return ()
    try:
        windows = json.loads(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}") from None
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"Invalid {key}")
    parsed = []
    for window in windows:
        if not isinstance(window, list) or not window:
            raise ValueError(f"Invalid {key}")
        ranges = []
        for item in window:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(type(h) is not int for h in item)
                or not 0 <= item[0] < item[1] <= 180
            ):
                raise ValueError(f"Invalid {key}")
            ranges.append((item[0], item[1]))
        parsed.append(tuple(ranges))
    return tuple(parsed)


def _optional_integer(section, key: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    """Read an optional integer key (0 allowed), falling back to ``default``."""
    if key not in section or not section[key].strip():
        return default
    try:
        value = int(section[key].strip())
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}") from None
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {key}")
    return value


def _optional_float(section, key: str, default: float) -> float:
    """Read an optional non-negative float key, falling back to ``default``."""
    if key not in section or not section[key].strip():
        return default
    try:
        value = float(section[key])
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {key}")
    return value


def _optional_bool(section, key: str, default: bool) -> bool:
    """Read an optional boolean key, falling back to ``default``."""
    if key not in section or not section[key].strip():
        return default
    value = section[key].strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid {key}")


def _optional_float_or_none(section, key: str, default: float | None = None) -> float | None:
    """Read an optional probability in (0, 1]; absent means ``default`` (off if None)."""
    if key not in section or not section[key].strip():
        return default
    try:
        value = float(section[key])
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}") from None
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"Invalid {key}")
    return value


def _optional_preprocess(section, key: str) -> str:
    """Read an optional preprocessing mode and validate it against the known set."""
    if key not in section or not section[key].strip():
        return "none"
    value = section[key].strip().lower()
    if value not in PREPROCESS_MODES:
        raise ValueError(f"Invalid {key}")
    return value


def _optional_classes(section, key: str) -> tuple[str, ...]:
    """Read an optional comma-separated list of vehicle class names."""
    if key not in section or not section[key].strip():
        return ()
    names = tuple(part.strip() for part in section[key].split(",") if part.strip())
    if not names:
        raise ValueError(f"Invalid {key}")
    return names


def _optional_pattern(section, key: str) -> str:
    """Read an optional regex the plate text must match; validate it compiles."""
    if key not in section or not section[key].strip():
        return ""
    pattern = section[key].strip()
    try:
        re.compile(pattern)
    except re.error:
        raise ValueError(f"Invalid {key}") from None
    return pattern


def _optional_roi(
    section, key: str, default: tuple[float, float, float, float] | None = None
) -> tuple[float, float, float, float] | None:
    """Read an optional ``x_min,y_min,x_max,y_max`` window of fractions in (0, 1]."""
    if key not in section or not section[key].strip():
        return default
    try:
        values = tuple(float(part) for part in section[key].split(","))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {key}") from None
    if len(values) != 4 or any(not 0 < v <= 1 for v in values):
        raise ValueError(f"Invalid {key}")
    x_min, y_min, x_max, y_max = values
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Invalid {key}")
    return values


def load_config(path: Path) -> Config:
    """Load validated offline settings without evaluating code or exposing secrets."""
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with Path(path).open(encoding="utf-8") as source:
            parser.read_file(source)
    except configparser.Error:
        raise ValueError("Invalid INI configuration") from None
    if "CONFIG" not in parser:
        raise ValueError("Missing CONFIG section")
    section = parser["CONFIG"]

    def required(key: str) -> str:
        if key not in section:
            raise ValueError(f"Missing {key}")
        return section[key].strip()

    resolution_parts = required("resolution_area").split(",")
    if len(resolution_parts) != 2:
        raise ValueError("Invalid resolution_area")
    resolution = tuple(_positive_integer(value, "resolution_area") for value in resolution_parts)
    polygons = {name: _polygon(required(key), key) for name, key in POLYGON_KEYS.items()}
    contraflow_polygon = (
        _polygon(section[CONTRAFLOW_KEY].strip(), CONTRAFLOW_KEY)
        if section.get(CONTRAFLOW_KEY, "").strip()
        else None
    )
    contraflow_direction = (section.get("contraflow_direction") or "South").strip().capitalize()
    if contraflow_direction not in {"South", "North"}:
        raise ValueError("Invalid contraflow_direction")
    contraflow_min_coverage = _optional_float(section, "contraflow_min_coverage", 0.6)
    draw_areas = _optional_classes(section, "draw_areas")
    unknown_areas = set(draw_areas) - AREA_NAMES
    if unknown_areas:
        raise ValueError(f"Invalid draw_areas: {', '.join(sorted(unknown_areas))}")
    if not 0 < contraflow_min_coverage <= 1:
        raise ValueError("Invalid contraflow_min_coverage")
    try:
        distances = json.loads(required("distance_area_speed"))
    except (ValueError, TypeError):
        raise ValueError("Invalid distance_area_speed") from None
    if not isinstance(distances, list) or len(distances) != 1 or type(distances[0]) not in (int, float):
        raise ValueError("distance_area_speed must contain exactly one distance")
    distance = _number(distances[0], "distance_area_speed")
    try:
        agnostic_nms = section.getboolean("agnostic_nms_model")
    except ValueError:
        raise ValueError("Invalid agnostic_nms_model") from None
    if agnostic_nms is None:
        raise ValueError("Missing agnostic_nms_model")
    classes_text = required("classes_name_model")
    try:
        classes = None if classes_text.lower() == "none" else json.loads(classes_text)
    except ValueError:
        raise ValueError("Invalid classes_name_model") from None
    if classes is not None and (
        not isinstance(classes, list) or any(type(value) is not int or value < 0 for value in classes)
    ):
        raise ValueError("classes_name_model must be a list of nonnegative integer class IDs or None")
    device = required("device").lower()
    if device == "cuda":
        device_id = required("device_id")
        if not device_id.isdecimal():
            raise ValueError("Invalid device_id")
        device = str(int(device_id))
    elif device != "cpu" and not device.isdecimal():
        raise ValueError("Invalid device")
    tracker_config = required("config_tracking")
    if not tracker_config:
        raise ValueError("Missing config_tracking")
    unsafe_distance_limit = _optional_float(section, "unsafe_distance_limit", 3.0)
    if unsafe_distance_limit <= 0:
        raise ValueError("Invalid unsafe_distance_limit")
    vehicle_anchor = (section.get("vehicle_anchor") or "bottom").strip().lower()
    if vehicle_anchor not in {"bottom", "center"}:
        raise ValueError("Invalid vehicle_anchor")
    return Config(
        resolution=resolution,
        polygons=polygons,
        distance_meter=distance,
        speed_limit_kmh=_number(required("speed_limit"), "speed_limit"),
        conf=_number(required("conf_model"), "conf_model", probability=True),
        iou=_number(required("iou_model"), "iou_model", probability=True),
        imgsz=_positive_integer(required("imgsz_model"), "imgsz_model"),
        max_det=_positive_integer(required("max_det_model"), "max_det_model"),
        agnostic_nms=agnostic_nms,
        classes=classes,
        device=device,
        tracker_config=tracker_config,
        plate_hue_windows=_hue_windows(section.get("plate_hue_windows", ""), "plate_hue_windows"),
        plate_sat_floor=_optional_integer(section, "plate_sat_floor", 90, maximum=255),
        plate_sat_floor_hazy=_optional_integer(section, "plate_sat_floor_hazy", 40, maximum=255),
        plate_val_floor=_optional_integer(section, "plate_val_floor", 80, maximum=255),
        plate_band_h_min=_optional_integer(section, "plate_band_h_min", 10),
        plate_band_h_max=_optional_integer(section, "plate_band_h_max", 30),
        plate_band_h_max_hazy=_optional_integer(section, "plate_band_h_max_hazy", 90),
        plate_letters=_optional_integer(section, "plate_letters", 3, minimum=1, maximum=8),
        plate_digits=_optional_integer(section, "plate_digits", 4, minimum=1, maximum=8),
        plate_min_digits=_optional_integer(section, "plate_min_digits", 3, minimum=1, maximum=8),
        plate_min_agreement=_optional_integer(section, "plate_min_agreement", 2, minimum=1, maximum=20),
        plate_min_sharpness=_optional_float(section, "plate_min_sharpness", 5.0),
        plate_single_read_prob=_optional_float_or_none(section, "plate_single_read_prob", 0.95),
        plate_trusted_min_confidence=_optional_float(section, "plate_trusted_min_confidence", 0.90),
        plate_trusted_min_sharpness=_optional_float(section, "plate_trusted_min_sharpness", 1000.0),
        plate_pattern=_optional_pattern(section, "plate_pattern"),
        plate_roi=_optional_roi(section, "plate_roi", (0.40, 0.40, 0.99, 0.90)),
        plate_ocr_classes=_optional_classes(section, "plate_ocr_classes"),
        plate_band_pad=_optional_integer(section, "plate_band_pad", 4, maximum=64),
        plate_upscale=_optional_float(section, "plate_upscale", 4.0),
        plate_char_imgsz=_optional_integer(section, "plate_char_imgsz", 224, minimum=64),
        plate_classifier_imgsz=_optional_integer(section, "plate_classifier_imgsz", 96, minimum=32),
        plate_min_crop_size=_optional_integer(section, "plate_min_crop_size", 16, minimum=4),
        plate_char_min_conf=_optional_float(section, "plate_char_min_conf", 0.0),
        plate_preprocess=_optional_preprocess(section, "plate_preprocess"),
        plate_deskew=_optional_bool(section, "plate_deskew", False),
        plate_max_ocr_frames=_optional_integer(section, "plate_max_ocr_frames", 0, maximum=10000),
        plate_dump_max_per_track=_optional_integer(section, "plate_dump_max_per_track", 8, minimum=1, maximum=200),
        event_on_finish_fallback=_optional_bool(section, "event_on_finish_fallback", False),
        unsafe_distance_limit_s=unsafe_distance_limit,
        vehicle_anchor=vehicle_anchor,
        draw_areas=draw_areas,
        contraflow_polygon=contraflow_polygon,
        contraflow_direction=contraflow_direction,
        contraflow_min_coverage=contraflow_min_coverage,
    )
