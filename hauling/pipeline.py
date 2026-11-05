"""Single-file offline inference orchestration."""

import importlib.metadata
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from hauling.config import Config
from hauling.contraflow import (
    STATUS_DIRECTION_UNKNOWN,
    STATUS_LEGAL_FLOW,
    STATUS_OUTSIDE,
    STATUS_UNCONFIRMED,
    STATUS_VIOLATION,
    ContraflowDetector,
)
from hauling.detector import VehicleDetector
from hauling.distance import MIN_HEADWAY_S, FinalCrossingLedger
from hauling.geometry import contains, scale_polygons
from hauling.output import EventWriter, prepare_output
from hauling.plate import (
    DEFAULT_HUE_WINDOWS,
    DISABLED_NOTE,
    ColorSignLocalizer,
    LambungReader,
    disabled_plate_fields,
    ocr_gate,
    plate_fields,
)
from hauling.render import draw_frame, open_video_writer
from hauling.speed import calculate_speed
from hauling.state import TrackState
from hauling.summary import _band_y, _summary_sections

LOGGER = logging.getLogger(__name__)

MIN_CROP_SIZE = 16


def _clip_vehicle_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Clip a vehicle bbox to the frame; reject degenerate crops for plate inference."""
    ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
    if ix2 - ix1 < MIN_CROP_SIZE or iy2 - iy1 < MIN_CROP_SIZE:
        return None
    return ix1, iy1, ix2, iy2


@dataclass(frozen=True)
class RunOptions:
    """Inputs and output controls for one local run."""

    video: Path
    output_dir: Path
    vehicle_model: Path | None = None
    tracker_config: Path | None = None
    device: str | None = None
    max_frames: int | None = None
    save_video: bool = True
    threads: int = 4
    tracker_fallback: bool = False
    plate_character_model: Path | None = None
    plate_letter_character_model: Path | None = None
    plate_digit_model: Path | None = None
    plate_letter_model: Path | None = None
    plate_conf: float = 0.4
    plate_min_prob: float = 0.0
    plate_single_read_prob: float | None = None
    dump_plate_crops: bool = False


def _measurement(state: TrackState, config: Config) -> dict:
    """Speed measurement with its ``is_overspeed`` flag.

    Invalid or incomplete crossings yield empty speed columns and
    ``is_overspeed=false``; the flag is ``speed > limit`` on the unrounded value.
    """
    return calculate_speed(
        state.start_timestamp, state.end_timestamp, config.distance_meter, config.speed_limit_kmh, state.speed_status
    )


def _event(
    state: TrackState,
    config: Config,
    video: Path,
    index: int,
    timestamp: float,
    event_number: int,
    measurement: dict,
    plate_enabled: bool,
    plate_min_prob: float = 0.0,
    plate_single_read_prob: float | None = None,
    plate_min_sharpness: float = 5.0,
    plate_trusted_min_confidence: float = 0.90,
    plate_trusted_min_sharpness: float = 1000.0,
) -> dict:
    return {
        "event_id": f"event-{event_number:06d}",
        "source_video": str(video.resolve()),
        "track_id": state.track_id,
        "event_timestamp_s": timestamp,
        "event_frame_index": index,
        "travel_direction": state.direction,
        "vehicle_confidence": state.vehicle_confidence,
        "speed_start_timestamp_s": state.start_timestamp,
        "speed_end_timestamp_s": state.end_timestamp,
        "distance_meter": config.distance_meter,
        "speed_limit_kmh": config.speed_limit_kmh,
        "lead_track_id": state.lead_track_id,
        "headway_s": round(state.headway_s, 3) if state.headway_s is not None else None,
        "headway_status": state.headway_status,
        "unsafe_distance_limit_s": config.unsafe_distance_limit_s,
        "is_unsafe_distance": state.is_unsafe_distance,
        "is_contraflow": state.is_contraflow,
        "contraflow_status": state.contraflow_status,
        # Blank unless the body was really measured inside the area: coverage
        # 0.0 for outside/inactive vehicles is not a measurement, it noise.
        "contraflow_coverage": (
            round(state.contraflow_coverage, 3)
            if state.contraflow_status in {STATUS_UNCONFIRMED, STATUS_VIOLATION}
            else None
        ),
        "contraflow_timestamp_s": state.contraflow_timestamp_s,
        **measurement,
        **(
            plate_fields(
                state,
                plate_min_prob,
                plate_single_read_prob,
                config.plate_digits,
                config.plate_min_agreement,
                config.plate_pattern or None,
                plate_min_sharpness,
                plate_trusted_min_confidence,
                plate_trusted_min_sharpness,
            )
            if plate_enabled
            else disabled_plate_fields()
        ),
    }




def _dump_plate_attempt(
    reader: Any,
    output_dir: Path,
    track_id: int,
    frame_index: int,
    timestamp: float,
    result: tuple[str, tuple[int, int, int, int]] | None,
    attempt: int,
) -> bool:
    """Persist one OCR attempt: band, detector input, and detected characters.

    Kept behind ``--dump-plate-crops`` because it is diagnostic only. The caller
    limits how many attempts per track are written; this returns whether a dump
    happened so that limit counts dumped files rather than attempts.
    """
    if not reader.last_debug:
        return False
    directory = output_dir / "plate_debug"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"trk{track_id:03d}_f{frame_index:06d}_a{attempt:02d}"
    debug = reader.last_debug
    raw = debug.get("raw")
    band = debug.get("band")
    detector_input = debug.get("input")
    if raw is not None:
        cv2.imwrite(str(directory / f"{stem}_raw.png"), raw)
    if band is not None:
        cv2.imwrite(str(directory / f"{stem}_band.png"), band)
    if detector_input is not None:
        cv2.imwrite(str(directory / f"{stem}_input.png"), detector_input)
    characters = [
        {
            "x": round(char[0], 1),
            "y": round(char[1], 1),
            "char": char[2],
            "channel": char[3],
            "det_conf": round(char[4], 4),
            "class_prob": round(char[5], 4),
        }
        for char in debug.get("characters") or []
    ]
    payload = {
        "track_id": track_id,
        "frame_index": frame_index,
        "timestamp_s": round(timestamp, 3),
        "attempt": attempt,
        "raw_size": [int(raw.shape[1]), int(raw.shape[0])] if raw is not None else None,
        "band_size": [int(band.shape[1]), int(band.shape[0])] if band is not None else None,
        "input_size": [int(detector_input.shape[1]), int(detector_input.shape[0])] if detector_input is not None else None,
        "letters": debug.get("letters"),
        "digits": debug.get("digits"),
        "probability": round(float(debug.get("probability") or 0.0), 4),
        "accepted": result is not None,
        "text": result[0] if result else None,
        "characters": characters,
    }
    (directory / f"{stem}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return True


def _flush_pending_events(
    states: dict[int, TrackState],
    events: EventWriter,
    config: Config,
    video: Path,
    summary: dict,
    plate_enabled: bool,
    current_frame: int | None = None,
    timeout_frames: int = 0,
    plate_min_prob: float = 0.0,
    plate_single_read_prob: float | None = None,
) -> int:
    """Write events for tracks whose final crossing is recorded and that are done.

    A track is only considered gone after ``timeout_frames`` without an
    observation (``current_frame`` is the frame being processed). Waiting lets
    plate OCR keep using the closest frames *after* the crossing: a vehicle
    that misses one detection is no longer finalized on the spot, which is what
    previously cut reading short within ~0.6 s of the final crossing. At
    EOF/max-frames ``current_frame`` is ``None`` and every pending event
    flushes.
    """
    written = 0
    for track_id, state in states.items():
        if current_frame is not None:
            if current_frame - state.last_frame_index <= timeout_frames:
                continue
        if not state.flush_event(config.event_on_finish_fallback):
            continue
        measurement = _measurement(state, config)
        event = _event(
            state,
            config,
            video,
            state.final_frame_index,
            state.final_timestamp if state.final_timestamp is not None else 0.0,
            summary["events"] + 1,
            measurement,
            plate_enabled,
            plate_min_prob,
            plate_single_read_prob,
            config.plate_min_sharpness,
            config.plate_trusted_min_confidence,
            config.plate_trusted_min_sharpness,
        )
        try:
            events.write(event)
        except Exception:
            state.event_written = False
            state.event_pending = True
            raise
        summary["events"] += 1
        summary["speed_valid"] += int(measurement["measurement_valid"])
        summary["overspeed"] += int(bool(measurement["is_overspeed"]))
        if measurement["measurement_valid"] and measurement["speed_kmh"] is not None:
            summary["_speeds"].append(measurement["speed_kmh"])
        summary["unsafe_distance"] += int(state.is_unsafe_distance)
        headway_counter = {
            "normal": "headway_normal",
            "artifact": "headway_artifact",
            "no_lead": "headway_no_lead",
            "no_final": "headway_no_final",
        }.get(state.headway_status)
        if headway_counter is not None:
            summary[headway_counter] += 1
        summary["plate_detected"] += int(bool(event["plate_detected"]))
        if state.final_not_crossed:
            summary["events_from_finish"] += 1
            LOGGER.info(
                "track=%s event_from_finish (track ended before the final band)", track_id
            )
        if plate_enabled:
            summary["plate_reads_debug"][str(track_id)] = {
                "reads": state._lambung_reads,
                "sharpness": round(max((read[6] for read in state._lambung_read_data), default=0.0), 1),
                "top": state.read_digest(),
            }
        if not event["plate_text"]:
            if state._lambung_reads == 0:
                summary["plate_no_read"] += 1
            elif state._lambung_reads == 1:
                summary["plate_single_read"] += 1
            else:
                summary["plate_no_agreement"] += 1
        written += 1
        LOGGER.info(
            "track=%s event_flushed frame=%s speed=%s status=%s",
            track_id,
            state.final_frame_index,
            measurement["speed_kmh"],
            measurement["speed_status"],
        )
    return written


def run_pipeline(options: RunOptions, config: Config, detector: Any = None) -> dict:
    """Process video to EOF or a frame limit; always retain a summary after output setup."""
    if not options.video.is_file():
        raise ValueError("--video must point to an existing local video file")
    if options.max_frames is not None and options.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if options.threads <= 0:
        raise ValueError("--threads must be positive")
    prepare_output(options.output_dir)
    started = time.monotonic()
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
            if len(formats) == 1 else "mixed"
        )
    single_read_prob = (
        options.plate_single_read_prob
        if options.plate_single_read_prob is not None
        else config.plate_single_read_prob
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
        "timing_seconds": {"video_decode": 0.0, "vehicle_detect": 0.0, "plate_read": 0.0},
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
    states: dict[int, TrackState] = {}
    measurements: dict[int, dict] = {}
    polygons: dict[str, Any] = {}
    # Cross-track headway at the final area; in-memory per run, never read back
    # from CSV (rows are written in track-death order, not crossing order).
    headway_ledger = FinalCrossingLedger(config.unsafe_distance_limit_s)
    contraflow: ContraflowDetector | None = None
    evidence_written: set[int] = set()
    pending_evidence: list[int] = []
    character_reader = None  # LambungReader when character models are supplied
    dumped_counts: dict[int, int] = {}  # per-track diagnostic dumps written
    capture, writer = None, None
    failure = None
    try:
        capture = cv2.VideoCapture(str(options.video.resolve()))
        if not capture.isOpened():
            raise ValueError("Video cannot be decoded; check the file and installed codecs")
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not math.isfinite(fps) or fps <= 0 or min(size) <= 0:
            raise ValueError("Video FPS and resolution must be valid and positive")
        expected_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        reported_frames = int(expected_frames) if math.isfinite(expected_frames) and expected_frames > 0 else 0
        summary.update(fps=fps, resolution=list(size), reported_frame_count=reported_frames)
        # A track must be unseen for this many frames before its event is
        # written, so plate OCR can use the closest frames after the crossing
        # instead of stopping at the first missed detection.
        plate_timeout_frames = max(30, round(fps * 1.2))
        summary["plate_timeout_frames"] = plate_timeout_frames
        polygons = scale_polygons(config.polygons, config.resolution, size)
        if config.contraflow_polygon:
            polygons["contraflow"] = scale_polygons(
                {"contraflow": config.contraflow_polygon}, config.resolution, size
            )["contraflow"]
            contraflow = ContraflowDetector(
                polygons["contraflow"],
                config.contraflow_direction,
                config.contraflow_min_coverage,
            )
        # Overlay filter: sites whose conf still carries uncalibrated area
        # copies (KM-19 speed bands are KM-18's) draw only what means
        # something there. Detection logic always sees every polygon.
        draw_polygons = (
            {name: p for name, p in polygons.items() if name in config.draw_areas}
            if config.draw_areas
            else polygons
        )
        # Test segments because fast vehicles can skip a thin band between frames.
        finish_band = _band_y(polygons, "finish")
        final_band = _band_y(polygons, "final")
        if detector is None:
            if options.vehicle_model is None or options.tracker_config is None:
                raise ValueError("Vehicle model and tracker configuration are required")
            detector = VehicleDetector(
                options.vehicle_model,
                options.tracker_config,
                config,
                options.device or config.device,
                summary["warnings"],
                options.output_dir / "runtime",
                options.threads,
            )
        summary["device"] = detector.device
        summary["model_classes"] = detector.model_names
        if options.plate_character_model:
            character_reader = LambungReader(
                options.plate_character_model,  # digit character detector
                options.plate_letter_character_model,  # letter character detector
                options.plate_digit_model,  # digit classifier
                options.plate_letter_model,  # letter classifier
                options.output_dir / "runtime",
                detector.device,
                options.plate_conf,
                options.threads,
                localizer=ColorSignLocalizer(
                    config.plate_hue_windows or DEFAULT_HUE_WINDOWS,
                    sat_floors=(config.plate_sat_floor, config.plate_sat_floor_hazy),
                    val_floor=config.plate_val_floor,
                    band_h_min=config.plate_band_h_min,
                    band_h_max=config.plate_band_h_max,
                    hazy_h_max=config.plate_band_h_max_hazy,
                    roi=config.plate_roi,
                ),
                letters=config.plate_letters,
                digits=config.plate_digits,
                min_digits=config.plate_min_digits,
                band_pad=config.plate_band_pad,
                upscale=config.plate_upscale,
                char_imgsz=config.plate_char_imgsz,
                classifier_imgsz=config.plate_classifier_imgsz,
                min_crop_size=config.plate_min_crop_size,
                char_min_conf=config.plate_char_min_conf,
                preprocess=config.plate_preprocess,
                deskew=config.plate_deskew,
            )
            summary["warnings"].append("Lambung (cab sign) reading enabled")
            summary["plate_calibration"] = {
                "sat_floor": config.plate_sat_floor,
                "sat_floor_hazy": config.plate_sat_floor_hazy,
                "val_floor": config.plate_val_floor,
                "band_h_min": config.plate_band_h_min,
                "band_h_max": config.plate_band_h_max,
                "band_h_max_hazy": config.plate_band_h_max_hazy,
                "letters": config.plate_letters,
                "digits": config.plate_digits,
                "min_digits": config.plate_min_digits,
                "min_agreement": config.plate_min_agreement,
                "band_pad": config.plate_band_pad,
                "upscale": config.plate_upscale,
                "dup_band_px": LambungReader.DUP_BAND_PX,
                "char_imgsz": config.plate_char_imgsz,
                "classifier_imgsz": config.plate_classifier_imgsz,
                "char_min_conf": config.plate_char_min_conf,
                "preprocess": config.plate_preprocess,
                "deskew": config.plate_deskew,
                "max_ocr_frames": config.plate_max_ocr_frames,
                "event_on_finish_fallback": config.event_on_finish_fallback,
                "pattern": config.plate_pattern,
                "roi": list(config.plate_roi) if config.plate_roi else None,
                "ocr_classes": list(config.plate_ocr_classes),
            }
        for package in ("ultralytics", "torch", "torchvision", "opencv-python", "numpy", "lap"):
            try:
                summary["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                summary["versions"][package] = None
        if ".engine" in formats:
            import tensorrt

            summary["versions"]["tensorrt"] = tensorrt.__version__
        if options.save_video:
            try:
                writer = open_video_writer(options.output_dir / "annotated.mp4", fps, size)
            except (OSError, cv2.error) as error:
                summary["warnings"].append(f"Video disabled: {error}")
        index, previous_pts = 0, None
        frames_started = time.monotonic()
        last_progress = time.monotonic()
        with EventWriter(options.output_dir / "events.csv") as events:
            while options.max_frames is None or index < options.max_frames:
                step_started = time.monotonic()
                ok, frame = capture.read()
                summary["timing_seconds"]["video_decode"] += time.monotonic() - step_started
                if not ok:
                    summary["stop_reason"] = "eof"
                    break
                summary["total_frames"] += 1
                pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if previous_pts is not None:
                    delta = pts - previous_pts
                    if not math.isfinite(pts) or not math.isclose(delta, 1 / fps, rel_tol=0.05, abs_tol=0.002):
                        raise ValueError(
                            "Non-uniform or unavailable media timestamps; supply a constant-frame-rate video"
                        )
                previous_pts = pts
                timestamp = index / fps
                step_started = time.monotonic()
                observations = detector.detect(frame)
                summary["timing_seconds"]["vehicle_detect"] += time.monotonic() - step_started
                summary["total_detections"] += len(observations)
                for observation in observations:
                    if observation.track_id is None:
                        continue
                    track_id = observation.track_id
                    state = states.setdefault(track_id, TrackState(track_id))
                    if state.vehicle_class is None and observation.class_name is not None:
                        state.vehicle_class = observation.class_name
                    if state._finish_band is None:
                        state.set_crossing_bands(finish_band, final_band)
                    reference = observation.reference(config.vehicle_anchor)
                    areas = {name for name, polygon in polygons.items() if contains(polygon, reference)}
                    before = (state.start_timestamp, state.end_timestamp, state.event_pending)
                    x1, y1, x2, y2 = observation.bbox
                    state.update(
                        reference,
                        timestamp,
                        index,
                        areas,
                        observation.confidence,
                        bbox_area=(x2 - x1) * (y2 - y1),
                    )
                    if contraflow is not None:
                        # Wrong-way check on every frame, but only near frames
                        # count: at the horizon a 16x19 px box can cover a
                        # polygon tip entirely, which would flag normal traffic.
                        verdict = contraflow.update(
                            track_id,
                            observation.bbox,
                            state.direction,
                            timestamp,
                            index,
                            near=ocr_gate((x2 - x1) * (y2 - y1), state.max_bbox_area),
                        )
                        state.is_contraflow = verdict.is_contraflow
                        state.contraflow_status = verdict.status
                        state.contraflow_coverage = verdict.coverage
                        state.contraflow_timestamp_s = verdict.timestamp
                        state.contraflow_frame_index = verdict.frame_index
                        if verdict.is_contraflow and track_id not in evidence_written:
                            evidence_written.add(track_id)
                            pending_evidence.append(track_id)
                            LOGGER.info(
                                "track=%s contraflow_confirmed frame=%s coverage=%.3f",
                                track_id,
                                index,
                                verdict.coverage,
                            )
                    # Other vehicle classes can carry stickers that OCR mistakes for cab signs.
                    class_ok = not config.plate_ocr_classes or state.vehicle_class in config.plate_ocr_classes
                    if character_reader is not None and class_ok and (areas & TrackState.MEASUREMENT_AREAS):
                        # Gate on confirmed South: North shows its rear and early direction is unresolved.
                        ocr_cap = config.plate_max_ocr_frames
                        within_cap = ocr_cap <= 0 or state.ocr_attempts < ocr_cap
                        if (
                            state.direction == "South"
                            and within_cap
                            and ocr_gate((x2 - x1) * (y2 - y1), state.max_bbox_area)
                        ):
                            # Far frames add OCR noise; read only near closest approach.
                            clipped = _clip_vehicle_box(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
                            if clipped is not None:
                                cx1, cy1, cx2, cy2 = clipped
                                crop = frame[cy1:cy2, cx1:cx2]
                                try:
                                    step_started = time.monotonic()
                                    try:
                                        lambung = character_reader.read(crop)
                                    finally:
                                        summary["timing_seconds"]["plate_read"] += time.monotonic() - step_started
                                    summary["ocr_frames"] += 1
                                    state.ocr_attempts += 1
                                    summary["ocr_attempts"] += 1
                                    if character_reader.last_localized:
                                        summary["ocr_band_found"] += 1
                                        state.plate_localized = True
                                        band_conf = character_reader.last_band_probability
                                        if band_conf and (state.plate_confidence or 0.0) < band_conf:
                                            state.plate_confidence = band_conf
                                    if character_reader.last_debug.get("characters"):
                                        summary["ocr_chars_found"] += 1
                                    if lambung is not None:
                                        summary["ocr_decode_ok"] += 1
                                    if (
                                        options.dump_plate_crops
                                        and character_reader.last_debug
                                        and dumped_counts.get(track_id, 0) < config.plate_dump_max_per_track
                                        and _dump_plate_attempt(
                                            character_reader,
                                            options.output_dir,
                                            track_id,
                                            index,
                                            timestamp,
                                            lambung,
                                            dumped_counts.get(track_id, 0) + 1,
                                        )
                                    ):
                                        dumped_counts[track_id] = dumped_counts.get(track_id, 0) + 1
                                    if lambung is not None:
                                        text, sign_box = lambung
                                        frame_h, frame_w = frame.shape[:2]
                                        # Oversized sign boxes often include body paint, not just text.
                                        veh_w = max(1.0, x2 - x1)
                                        veh_h = max(1.0, y2 - y1)
                                        rel_w = sign_box[2] / veh_w
                                        rel_h = sign_box[3] / veh_h
                                        summary["plate_rel_w_max"] = max(summary["plate_rel_w_max"], rel_w)
                                        summary["plate_rel_h_max"] = max(summary["plate_rel_h_max"], rel_h)
                                        if rel_w > 0.30 or rel_h > 0.06:
                                            summary["plate_crop_suspect"] += 1
                                        sign_crop = crop[
                                            max(0, sign_box[1]) : sign_box[1] + sign_box[3],
                                            max(0, sign_box[0]) : sign_box[0] + sign_box[2],
                                        ]
                                        sharpness = (
                                            float(cv2.Laplacian(
                                                cv2.cvtColor(sign_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                                            ).var())
                                            if sign_crop.size else 0.0
                                        )
                                        summary["plate_sharpness_max"] = max(
                                            summary["plate_sharpness_max"], sharpness
                                        )
                                        state.observe_lambung(
                                            text,
                                            character_reader.last_probability,
                                            timestamp,
                                            weight=sign_box[2] * sign_box[3],
                                            box_frame=(
                                                float(min(frame_w, max(0, cx1 + sign_box[0]))),
                                                float(min(frame_h, max(0, cy1 + sign_box[1]))),
                                                float(min(frame_w, cx1 + sign_box[0] + sign_box[2])),
                                                float(min(frame_h, cy1 + sign_box[1] + sign_box[3])),
                                            ),
                                            sharpness=sharpness,
                                        )
                                except cv2.error as error:
                                    # A bad crop is local to one frame, not a failed model backend.
                                    LOGGER.warning("plate inference failed frame=%s: %s", index, error)
                                except Exception as error:
                                    if ".engine" in formats:
                                        raise  # Do not report success with a broken TensorRT OCR chain.
                                    # PRD: a PyTorch plate failure must never kill the vehicle event.
                                    LOGGER.warning("plate inference failed frame=%s: %s", index, error)
                    measurement = _measurement(state, config)
                    measurements[track_id] = measurement
                    if before[0] is None and state.start_timestamp is not None:
                        LOGGER.info("track=%s entered_start frame=%s time=%.3f", track_id, index, timestamp)
                    if before[1] is None and state.end_timestamp is not None:
                        LOGGER.info("track=%s entered_finish frame=%s time=%.3f", track_id, index, timestamp)
                    if not before[2] and state.event_pending:
                        # Register at crossing: event rows flush later and may be out of crossing order.
                        outcome = headway_ledger.register(track_id, state.final_timestamp)
                        state.lead_track_id = outcome.lead_track_id
                        state.headway_s = outcome.headway_s
                        state.headway_status = outcome.status
                        state.is_unsafe_distance = outcome.is_unsafe
                        LOGGER.info(
                            "track=%s entered_final frame=%s time=%.3f direction=%s class=%s "
                            "headway=%s status=%s",
                            track_id, index, timestamp, state.direction, state.vehicle_class,
                            outcome.headway_s, outcome.status,
                        )
                        # Crop failures must not discard the crossing event.
                        try:
                            crop_dir = options.output_dir / "events_crops"
                            crop_dir.mkdir(parents=True, exist_ok=True)
                            clipped = _clip_vehicle_box(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
                            if clipped is not None:
                                cv2.imwrite(
                                    str(crop_dir / f"trk{track_id:03d}_f{index:06d}.png"),
                                    frame[clipped[1]:clipped[3], clipped[0]:clipped[2]],
                                )
                        except OSError as error:
                            LOGGER.warning("event crop failed frame=%s: %s", index, error)
                processing_fps = (index + 1) / max(time.monotonic() - frames_started, 1e-9)
                if pending_evidence:
                    annotated = draw_frame(
                        frame,
                        draw_polygons,
                        observations,
                        states,
                        measurements,
                        timestamp,
                        plate_enabled,
                        config.vehicle_anchor,
                        video_fps=fps,
                        processing_fps=processing_fps,
                    )
                    evidence_dir = options.output_dir / "contraflow"
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    for evidence_track in pending_evidence:
                        cv2.imwrite(
                            str(evidence_dir / f"trk{evidence_track:03d}_f{index:06d}.png"),
                            annotated,
                        )
                        summary["contraflow_evidence"] += 1
                    pending_evidence = []
                if writer is not None:
                    try:
                        writer.write(
                            draw_frame(
                                frame,
                                draw_polygons,
                                observations,
                                states,
                                measurements,
                                timestamp,
                                plate_enabled,
                                config.vehicle_anchor,
                                video_fps=fps,
                                processing_fps=processing_fps,
                            )
                        )
                        summary["video_written"] = True
                    except (OSError, cv2.error) as error:
                        writer.release()
                        writer = None
                        summary["warnings"].append(f"Video writer stopped: {error}")
                        summary["video_written"] = False
                _flush_pending_events(
                    states, events, config, options.video, summary, plate_enabled,
                    current_frame=index,
                    timeout_frames=plate_timeout_frames,
                    plate_min_prob=options.plate_min_prob,
                    plate_single_read_prob=single_read_prob,
                )
                index += 1
                if time.monotonic() - last_progress >= 30:
                    LOGGER.info("progress frames=%s video_time=%.1fs events=%s", index, timestamp, summary["events"])
                    last_progress = time.monotonic()
            else:
                summary["stop_reason"] = "max_frames"
            _flush_pending_events(
                states, events, config, options.video, summary, plate_enabled,
                plate_min_prob=options.plate_min_prob,
                plate_single_read_prob=single_read_prob,
            )
        if summary["total_frames"] == 0:
            raise ValueError("Video contained no readable frames")
        if summary["stop_reason"] == "eof" and 0 < reported_frames != index:
            summary["warnings"].append(f"Decoded {index} frames but container reported frame count {reported_frames}")
        summary["status"] = "completed"
    except KeyboardInterrupt as error:
        summary["status"] = "interrupted"
        summary["stop_reason"] = "interrupt"
        summary["warnings"].append("Run interrupted; outputs contain only processed frames")
        failure = error
    except Exception as error:
        summary["status"] = "failed"
        summary["stop_reason"] = "error"
        summary["errors"].append(f"{type(error).__name__}: {error}")
        failure = error
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        summary["unique_tracks"] = len(states)
        # North tracks are tracked for counting but never written to events;
        # exclude them from the unfinished counters so a real South miss is not
        # hidden behind the expected North flow.
        south = [s for s in states.values() if s.first_centroid is not None and s.direction == "South"]
        summary["north_detected"] = sum(
            1 for s in states.values() if s.first_centroid is not None and s.direction == "North"
        )
        unfinished = [s for s in south if not s.event_written]
        summary["unfinished_tracks"] = len(unfinished)
        summary["lost_before_final"] = sum(s.last_frame_index < summary["total_frames"] - 1 for s in unfinished)
        summary["unfinished_detail"] = [
            {
                "track_id": s.track_id,
                "vehicle_class": s.vehicle_class,
                "direction": s.direction,
                "start_s": s.start_timestamp,
                "finish_s": s.end_timestamp,
                "final_s": s.final_timestamp,
                "last_frame": s.last_frame_index,
                "plate_reads": s._lambung_reads,
            }
            for s in unfinished
        ]
        summary["tracks_outside_areas"] = sum(1 for s in states.values() if s.first_centroid is None)
        # Counted from the track states, not from flushed events: a Northbound
        # vehicle is never written to the CSV but its legal_flow verdict still
        # proves the area is not flagging normal oncoming traffic.
        contraflow_statuses = [s.contraflow_status for s in states.values()]
        summary["contraflow_violation"] = contraflow_statuses.count(STATUS_VIOLATION)
        summary["contraflow_legal_flow"] = contraflow_statuses.count(STATUS_LEGAL_FLOW)
        summary["contraflow_direction_unknown"] = contraflow_statuses.count(STATUS_DIRECTION_UNKNOWN)
        summary["contraflow_outside"] = contraflow_statuses.count(STATUS_OUTSIDE)
        summary["contraflow_unconfirmed"] = contraflow_statuses.count(STATUS_UNCONFIRMED)
        summary["contraflow_inactive"] = contraflow_statuses.count("inactive")
        summary["duration_seconds"] = round(time.monotonic() - started, 3)
        summary["timing_seconds"] = {
            key: round(value, 3) for key, value in summary["timing_seconds"].items()
        }
        summary.update(_summary_sections(summary, config, states, polygons, measurements))
        (options.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if failure is not None:
        raise failure
    return summary
