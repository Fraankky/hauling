"""Build grouped run_summary.json sections from completed pipeline state."""

import json
import time
from pathlib import Path
from typing import Any

from hauling import __version__
from hauling.config import Config
from hauling.contraflow import (
    COVERAGE_GRID,
    MIN_CONFIRM_FRAMES,
    STATUS_DIRECTION_UNKNOWN,
    STATUS_LEGAL_FLOW,
    STATUS_OUTSIDE,
    STATUS_UNCONFIRMED,
    STATUS_VIOLATION,
)
from hauling.distance import MIN_HEADWAY_S
from hauling.options import RunOptions
from hauling.plate import DISABLED_NOTE
from hauling.state import TrackState


def initialize_summary(options: RunOptions, config: Config) -> dict:
    """Create the flat summary contract before runtime resources open."""
    plate_enabled = options.plate_character_model is not None
    models = {
        name: str(path.resolve())
        for name, path in (
            ("vehicle", options.vehicle_model),
            ("digit_detector", options.plate_character_model),
            ("letter_detector", options.plate_letter_character_model),
            ("digit_classifier", options.plate_digit_model),
            ("letter_classifier", options.plate_letter_model),
        )
        if path is not None
    }
    formats = {Path(path).suffix.lower() for path in models.values()}
    runtime = None
    if formats:
        runtime = (
            {".pt": "pytorch", ".engine": "tensorrt"}.get(next(iter(formats)))
            if len(formats) == 1
            else "mixed"
        )
    summary = {
        "source_video": str(options.video.resolve()),
        "fps": None,
        "resolution": None,
        "total_frames": 0,
        "total_detections": 0,
        "unique_tracks": 0,
        "events": 0,
        "speed_valid": 0,
        "overspeed": 0,
        "unsafe_distance": 0,
        "unsafe_distance_limit_s": config.unsafe_distance_limit_s,
        "headway_normal": 0,
        "headway_artifact": 0,
        "headway_no_lead": 0,
        "headway_no_final": 0,
        "contraflow_violation": 0,
        "contraflow_legal_flow": 0,
        "contraflow_direction_unknown": 0,
        "contraflow_outside": 0,
        "contraflow_unconfirmed": 0,
        "contraflow_inactive": 0,
        "contraflow_evidence": 0,
        "plate_detected": 0,
        "plate_detection_enabled": plate_enabled,
        "unfinished_tracks": 0,
        "lost_before_final": 0,
        "events_from_finish": 0,
        "north_detected": 0,
        "plate_no_read": 0,
        "plate_single_read": 0,
        "plate_no_agreement": 0,
        "ocr_frames": 0,
        "ocr_attempts": 0,
        "ocr_band_found": 0,
        "ocr_chars_found": 0,
        "ocr_decode_ok": 0,
        "plate_crop_suspect": 0,
        "plate_rel_w_max": 0.0,
        "plate_rel_h_max": 0.0,
        "plate_sharpness_max": 0.0,
        "plate_reads_debug": {},
        "_speeds": [],
        "duration_seconds": 0.0,
        "timing_seconds": {
            key: 0.0
            for key in (
                "video_setup",
                "vehicle_model_load",
                "ocr_model_load",
                "video_writer_setup",
                "video_decode",
                "vehicle_detect",
                "ocr_gate_vote",
                "plate_read",
                "event_logic",
                "render_annotation",
                "video_encode_write",
                "image_write",
                "event_finalize_csv",
                "cleanup",
            )
        },
        "timing_calls": {
            key: 0
            for key in (
                "video_setup",
                "vehicle_model_load",
                "ocr_model_load",
                "video_writer_setup",
                "video_decode",
                "vehicle_detect",
                "ocr_gate_vote",
                "plate_read",
                "event_logic",
                "render_annotation",
                "video_encode_write",
                "image_write",
                "event_finalize_csv",
                "cleanup",
            )
        },
        "warnings": [],
        "errors": [],
        "status": "running",
        "stop_reason": None,
        "video_requested": options.save_video,
        "video_written": False,
        "vehicle_model": str(options.vehicle_model.resolve()) if options.vehicle_model else None,
        "models": models,
        "model_runtime": runtime,
        "tracker_config": str(options.tracker_config.resolve()) if options.tracker_config else None,
        "tracker_fallback": options.tracker_fallback,
        "device": None,
        "timestamp_source": "frame_index / video_fps (CFR)",
        "versions": {},
    }
    if not plate_enabled:
        summary["warnings"].append(DISABLED_NOTE)
    if config.unsafe_distance_limit_s <= MIN_HEADWAY_S:
        summary["warnings"].append(
            f"unsafe_distance_limit={config.unsafe_distance_limit_s} is at or below the "
            f"{MIN_HEADWAY_S} s artifact floor; no unsafe-distance violation can be reported"
        )
    if options.tracker_fallback:
        summary["warnings"].append("Reference tracker missing; using local Ultralytics ByteTrack baseline")
    return summary


def _band_y(polygons: dict[str, Any], name: str) -> tuple[float, float]:
    """Vertical extent of a crossing band, in frame pixels."""
    points = polygons[name]
    return float(points[:, 1].min()), float(points[:, 1].max())


def _summary_sections(
    summary: dict,
    config: Config,
    states: dict[int, TrackState],
    polygons: dict[str, Any],
    measurements: dict[int, dict],
) -> dict:
    """Grouped, self-describing view of one run for ``run_summary.json``.

    The flat keys stay for compatibility; these sections restate the same
    numbers together with the formula and the config that produced them, so a
    CSV row can be audited without reading the pipeline source.
    """
    speeds = sorted(summary.pop("_speeds", []))
    headways = sorted(s.headway_s for s in states.values() if s.headway_s is not None)
    # A run that fails before calibration has no polygons; keep the sections
    # present but empty rather than raising in the finally block.
    has_bands = all(name in polygons for name in ("start", "finish", "final"))
    start_band = _band_y(polygons, "start") if has_bands else None
    finish_band = _band_y(polygons, "finish") if has_bands else None
    # Nominal start->finish span; both bands are thin, so comparing their top
    # edges gives the same distance the crossing logic measures between.
    segment_px = finish_band[0] - start_band[0] if has_bands else None
    px_per_meter = (
        round(segment_px / config.distance_meter, 3) if segment_px is not None and config.distance_meter else None
    )

    violations = []
    closest = []
    for state in states.values():
        if state.headway_s is None:
            continue
        lead = measurements.get(state.lead_track_id or -1, {})
        lead_speed = lead.get("speed_kmh")
        gap = lead_speed / 3.6 * state.headway_s if lead_speed is not None else None
        entry = {
            "track_id": state.track_id,
            "lead_track_id": state.lead_track_id,
            "headway_s": round(state.headway_s, 3),
            "lead_speed_kmh": lead_speed,
            "gap_m": round(gap, 2) if gap is not None else None,
            "status": state.headway_status,
        }
        closest.append(entry)
        if state.is_unsafe_distance:
            violations.append(entry)
    closest.sort(key=lambda item: item["headway_s"])
    violations.sort(key=lambda item: item["headway_s"])

    wrong_way = sorted(
        (
            {
                "track_id": state.track_id,
                "contraflow_timestamp_s": state.contraflow_timestamp_s,
                "coverage": round(state.contraflow_coverage, 3),
            }
            for state in states.values()
            if state.is_contraflow
        ),
        key=lambda item: item["contraflow_timestamp_s"] or 0.0,
    )

    return {
        "schema_version": 2,
        "pipeline": {
            "package_version": __version__,
            "description": (
                "Offline detection -> ByteTrack -> area crossing -> speed -> plate OCR. "
                "No RTSP, database or API call; every number below comes from this run."
            ),
            "fps": summary["fps"],
            "resolution": summary["resolution"],
            "total_frames": summary["total_frames"],
            "reported_frame_count": summary.get("reported_frame_count"),
            "timestamp_source": summary["timestamp_source"],
            "device": summary["device"],
            "vehicle_model": summary["vehicle_model"],
            "model_runtime": summary["model_runtime"],
            "models": summary["models"],
            "tracker_config": summary["tracker_config"],
            "tracker_fallback": summary["tracker_fallback"],
            "vehicle_anchor": config.vehicle_anchor,
            "anchor_point": (
                "bottom-center of bbox (ground contact)"
                if config.vehicle_anchor == "bottom"
                else "bbox centroid"
            ),
            "anchor_why": (
                "Area bands and distance_meter are road-plane quantities; only the ground "
                "contact follows that geometry. The centroid sits above the road and reaches "
                "each band at a closer distance, which inflates speed by H/(H - h_c)."
            ),
            "areas_px": {
                name: [[round(float(x), 1), round(float(y), 1)] for x, y in polygon]
                for name, polygon in polygons.items()
            },
            "counts": {
                "total_detections": summary["total_detections"],
                "unique_tracks": summary["unique_tracks"],
                "north_detected": summary["north_detected"],
                "events": summary["events"],
                "unfinished_tracks": summary["unfinished_tracks"],
                "lost_before_final": summary["lost_before_final"],
                "events_from_finish": summary["events_from_finish"],
                "tracks_outside_areas": summary["tracks_outside_areas"],
            },
            "versions": summary["versions"],
        },
        "speed": {
            "description": (
                "Measured between the first entry into the start band and the first entry "
                "into the finish band. Both use the same reference point as the areas."
            ),
            "formula": "speed_kmh = distance_meter / elapsed_seconds * 3.6",
            "distance_meter": config.distance_meter,
            "speed_limit_kmh": config.speed_limit_kmh,
            "start_band_y_px": start_band,
            "finish_band_y_px": finish_band,
            "segment_px": segment_px,
            "px_per_meter": px_per_meter,
            "events": summary["events"],
            "valid": summary["speed_valid"],
            "overspeed": summary["overspeed"],
            "min_kmh": speeds[0] if speeds else None,
            "median_kmh": speeds[len(speeds) // 2] if speeds else None,
            "max_kmh": speeds[-1] if speeds else None,
            "mean_kmh": round(sum(speeds) / len(speeds), 2) if speeds else None,
        },
        "plate": {
            "description": (
                "Cab-sign (nomor lambung) OCR on the vehicle crop, gated to trucks near "
                "closest approach. Empty text is preferred over a wrong plate."
            ),
            "enabled": summary["plate_detection_enabled"],
            "detected": summary["plate_detected"],
            "no_read": summary["plate_no_read"],
            "single_read": summary["plate_single_read"],
            "no_agreement": summary["plate_no_agreement"],
            "ocr_funnel": {
                "frames": summary["ocr_frames"],
                "attempts": summary["ocr_attempts"],
                "band_found": summary["ocr_band_found"],
                "chars_found": summary["ocr_chars_found"],
                "decode_ok": summary["ocr_decode_ok"],
            },
            "crop_audit": {
                "suspect": summary["plate_crop_suspect"],
                "rel_w_max": summary["plate_rel_w_max"],
                "rel_h_max": summary["plate_rel_h_max"],
                "sharpness_max": summary["plate_sharpness_max"],
            },
            "calibration": summary.get("plate_calibration", {}),
            "reads_debug": summary["plate_reads_debug"],
        },
        "unsafe": {
            "description": (
                "Time headway between consecutive final-area crossings. The metric is time, "
                "not metres, so it does not depend on distance_meter."
            ),
            "formula": "headway_s = final_timestamp_now - final_timestamp_lead",
            "limit_s": config.unsafe_distance_limit_s,
            "artifact_floor_s": MIN_HEADWAY_S,
            "decision": "0.9 < headway_s <= limit_s, inclusive at the limit",
            "status": (
                "inactive: limit is at or below the artifact floor, no violation can be reported"
                if config.unsafe_distance_limit_s <= MIN_HEADWAY_S
                else "active"
            ),
            "counters": {
                "unsafe": summary["unsafe_distance"],
                "normal": summary["headway_normal"],
                "artifact": summary["headway_artifact"],
                "no_lead": summary["headway_no_lead"],
                "no_final": summary["headway_no_final"],
            },
            "observed": {
                "pairs": len(headways),
                "min_headway_s": headways[0] if headways else None,
                "median_headway_s": headways[len(headways) // 2] if headways else None,
                "closest_pairs": closest[:5],
            },
            "violations": violations,
        },
        "contraflow": {
            "description": (
                "Wrong-way check: a vehicle facing the camera whose bbox sits mostly in the "
                "lane reserved for oncoming traffic. Evaluated on every frame, but only near "
                "frames (bbox area >= 75% of the track's largest) advance the confirmation "
                "streak, because a horizon-sized box can cover a polygon tip entirely."
            ),
            "formula": (
                "coverage = share of a 5x5 bbox sample inside the area; "
                "violation = coverage >= min_coverage on min_confirm_frames consecutive near frames"
            ),
            "status": (
                "active"
                if config.contraflow_polygon
                else "inactive: no contraflow area configured"
            ),
            "direction": config.contraflow_direction,
            "min_coverage": config.contraflow_min_coverage,
            "coverage_grid": COVERAGE_GRID,
            "min_confirm_frames": MIN_CONFIRM_FRAMES,
            "counters": {
                "violation": summary["contraflow_violation"],
                "legal_flow": summary["contraflow_legal_flow"],
                "direction_unknown": summary["contraflow_direction_unknown"],
                "outside": summary["contraflow_outside"],
                "unconfirmed": summary["contraflow_unconfirmed"],
                "inactive": summary["contraflow_inactive"],
            },
            "violations": wrong_way,
            "evidence_dir": "contraflow/",
        },
    }


def finalize_summary(
    summary: dict,
    config: Config,
    states: dict[int, TrackState],
    polygons: dict[str, Any],
    measurements: dict[int, dict],
    output_dir: Path,
    started: float,
) -> None:
    """Complete runtime counters and always persist ``run_summary.json``."""
    summary["unique_tracks"] = len(states)
    south = [state for state in states.values() if state.first_centroid is not None and state.direction == "South"]
    summary["north_detected"] = sum(
        state.first_centroid is not None and state.direction == "North" for state in states.values()
    )
    unfinished = [state for state in south if not state.event_written]
    summary["unfinished_tracks"] = len(unfinished)
    summary["lost_before_final"] = sum(
        state.last_frame_index < summary["total_frames"] - 1 for state in unfinished
    )
    summary["unfinished_detail"] = [
        {
            "track_id": state.track_id,
            "vehicle_class": state.vehicle_class,
            "direction": state.direction,
            "start_s": state.start_timestamp,
            "finish_s": state.end_timestamp,
            "final_s": state.final_timestamp,
            "last_frame": state.last_frame_index,
            "plate_reads": state._lambung_reads,
        }
        for state in unfinished
    ]
    summary["tracks_outside_areas"] = sum(state.first_centroid is None for state in states.values())
    statuses = [state.contraflow_status for state in states.values()]
    summary["contraflow_violation"] = statuses.count(STATUS_VIOLATION)
    summary["contraflow_legal_flow"] = statuses.count(STATUS_LEGAL_FLOW)
    summary["contraflow_direction_unknown"] = statuses.count(STATUS_DIRECTION_UNKNOWN)
    summary["contraflow_outside"] = statuses.count(STATUS_OUTSIDE)
    summary["contraflow_unconfirmed"] = statuses.count(STATUS_UNCONFIRMED)
    summary["contraflow_inactive"] = statuses.count("inactive")
    summary["duration_seconds"] = round(time.monotonic() - started, 3)
    summary["timing_seconds"] = {
        key: round(value, 3) for key, value in summary["timing_seconds"].items()
    }
    summary.update(_summary_sections(summary, config, states, polygons, measurements))
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
