"""Single-file offline inference orchestration."""

import importlib.metadata
import logging
import math
import time
from pathlib import Path
from typing import Any

import cv2

from hauling.config import Config
from hauling.contraflow import ContraflowDetector
from hauling.detector import VehicleDetector
from hauling.distance import FinalCrossingLedger
from hauling.events import flush_pending_events, measurement
from hauling.geometry import band_y, clip_vehicle_box, contains, scale_polygons
from hauling.options import RunOptions
from hauling.output import EventWriter, prepare_output
from hauling.plate import ocr_gate
from hauling.plate_runtime import (
    create_plate_reader,
    dump_plate_attempt,
    plate_calibration,
    process_plate_observation,
)
from hauling.render import draw_frame, open_video_writer
from hauling.state import TrackState
from hauling.summary import finalize_summary, initialize_summary

LOGGER = logging.getLogger(__name__)

# Diagnostic scripts historically imported this helper from pipeline.
_clip_vehicle_box = clip_vehicle_box
_dump_plate_attempt = dump_plate_attempt


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
    summary = initialize_summary(options, config)
    formats = {Path(path).suffix.lower() for path in summary["models"].values()}
    single_read_prob = (
        options.plate_single_read_prob
        if options.plate_single_read_prob is not None
        else config.plate_single_read_prob
    )
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

    def record_timing(name: str, step_started: float, calls: int = 1) -> None:
        summary["timing_seconds"][name] += time.monotonic() - step_started
        summary["timing_calls"][name] += calls

    try:
        step_started = time.monotonic()
        capture = cv2.VideoCapture(str(options.video.resolve()))
        record_timing("video_setup", step_started)
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
        finish_band = band_y(polygons, "finish")
        final_band = band_y(polygons, "final")
        if detector is None:
            if options.vehicle_model is None or options.tracker_config is None:
                raise ValueError("Vehicle model and tracker configuration are required")
            step_started = time.monotonic()
            detector = VehicleDetector(
                options.vehicle_model,
                options.tracker_config,
                config,
                options.device or config.device,
                summary["warnings"],
                options.output_dir / "runtime",
                options.threads,
            )
            record_timing("vehicle_model_load", step_started)
        summary["device"] = detector.device
        summary["model_classes"] = detector.model_names
        step_started = time.monotonic()
        character_reader = create_plate_reader(
            options, config, detector.device, options.output_dir / "runtime"
        )
        record_timing("ocr_model_load", step_started)
        if character_reader is not None:
            summary["warnings"].append("Lambung (cab sign) reading enabled")
            summary["plate_calibration"] = plate_calibration(config)
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
                step_started = time.monotonic()
                writer = open_video_writer(options.output_dir / "annotated.mp4", fps, size)
                record_timing("video_writer_setup", step_started)
            except (OSError, cv2.error) as error:
                summary["warnings"].append(f"Video disabled: {error}")
        index, previous_pts = 0, None
        frames_started = time.monotonic()
        last_progress = time.monotonic()
        with EventWriter(options.output_dir / "events.csv") as events:
            while options.max_frames is None or index < options.max_frames:
                step_started = time.monotonic()
                ok, frame = capture.read()
                record_timing("video_decode", step_started)
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
                record_timing("vehicle_detect", step_started)
                summary["total_detections"] += len(observations)
                logic_started = time.monotonic()
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
                    if character_reader is not None:
                        record_timing("event_logic", logic_started, calls=0)
                        step_started = time.monotonic()
                        process_plate_observation(
                            character_reader,
                            frame,
                            observation,
                            state,
                            areas,
                            index,
                            timestamp,
                            options,
                            config,
                            summary,
                            dumped_counts,
                            formats,
                        )
                        record_timing("ocr_gate_vote", step_started)
                        logic_started = time.monotonic()
                    speed = measurement(state, config)
                    measurements[track_id] = speed
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
                            clipped = clip_vehicle_box(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
                            if clipped is not None:
                                record_timing("event_logic", logic_started, calls=0)
                                step_started = time.monotonic()
                                try:
                                    cv2.imwrite(
                                        str(crop_dir / f"trk{track_id:03d}_f{index:06d}.png"),
                                        frame[clipped[1]:clipped[3], clipped[0]:clipped[2]],
                                    )
                                finally:
                                    record_timing("image_write", step_started)
                                    logic_started = time.monotonic()
                        except OSError as error:
                            LOGGER.warning("event crop failed frame=%s: %s", index, error)
                record_timing("event_logic", logic_started)
                processing_fps = (index + 1) / max(time.monotonic() - frames_started, 1e-9)
                if pending_evidence:
                    step_started = time.monotonic()
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
                    record_timing("render_annotation", step_started)
                    evidence_dir = options.output_dir / "contraflow"
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    for evidence_track in pending_evidence:
                        step_started = time.monotonic()
                        try:
                            cv2.imwrite(
                                str(evidence_dir / f"trk{evidence_track:03d}_f{index:06d}.png"),
                                annotated,
                            )
                        finally:
                            record_timing("image_write", step_started)
                        summary["contraflow_evidence"] += 1
                    pending_evidence = []
                if writer is not None:
                    try:
                        step_started = time.monotonic()
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
                        record_timing("render_annotation", step_started)
                        step_started = time.monotonic()
                        writer.write(annotated)
                        record_timing("video_encode_write", step_started)
                        summary["video_written"] = True
                    except (OSError, cv2.error) as error:
                        writer.release()
                        writer = None
                        summary["warnings"].append(f"Video writer stopped: {error}")
                        summary["video_written"] = False
                step_started = time.monotonic()
                try:
                    flush_pending_events(
                        states, events, config, options.video, summary, plate_enabled,
                        current_frame=index,
                        timeout_frames=plate_timeout_frames,
                        plate_min_prob=options.plate_min_prob,
                        plate_single_read_prob=single_read_prob,
                    )
                finally:
                    record_timing("event_finalize_csv", step_started)
                index += 1
                if time.monotonic() - last_progress >= 30:
                    LOGGER.info("progress frames=%s video_time=%.1fs events=%s", index, timestamp, summary["events"])
                    last_progress = time.monotonic()
            else:
                summary["stop_reason"] = "max_frames"
            step_started = time.monotonic()
            try:
                flush_pending_events(
                    states, events, config, options.video, summary, plate_enabled,
                    plate_min_prob=options.plate_min_prob,
                    plate_single_read_prob=single_read_prob,
                )
            finally:
                record_timing("event_finalize_csv", step_started)
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
        step_started = time.monotonic()
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        record_timing("cleanup", step_started)
        finalize_summary(
            summary, config, states, polygons, measurements, options.output_dir, started
        )
    if failure is not None:
        raise failure
    return summary
