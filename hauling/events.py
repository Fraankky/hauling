"""Build and persist completed vehicle events."""

import logging
from pathlib import Path

from hauling.config import Config
from hauling.contraflow import STATUS_UNCONFIRMED, STATUS_VIOLATION
from hauling.output import EventWriter
from hauling.plate import disabled_plate_fields, plate_fields
from hauling.speed import calculate_speed
from hauling.state import TrackState

LOGGER = logging.getLogger(__name__)


def measurement(state: TrackState, config: Config) -> dict:
    """Calculate speed and overspeed status for one track."""
    return calculate_speed(
        state.start_timestamp,
        state.end_timestamp,
        config.distance_meter,
        config.speed_limit_kmh,
        state.speed_status,
    )


def build_event(
    state: TrackState,
    config: Config,
    video: Path,
    index: int,
    timestamp: float,
    event_number: int,
    speed: dict,
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
        "contraflow_coverage": (
            round(state.contraflow_coverage, 3)
            if state.contraflow_status in {STATUS_UNCONFIRMED, STATUS_VIOLATION}
            else None
        ),
        "contraflow_timestamp_s": state.contraflow_timestamp_s,
        **speed,
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


def flush_pending_events(
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
    """Write completed events after their tracks have timed out or the run ends."""
    written = 0
    for track_id, state in states.items():
        if current_frame is not None and current_frame - state.last_frame_index <= timeout_frames:
            continue
        if not state.flush_event(config.event_on_finish_fallback):
            continue
        speed = measurement(state, config)
        event = build_event(
            state,
            config,
            video,
            state.final_frame_index,
            state.final_timestamp if state.final_timestamp is not None else 0.0,
            summary["events"] + 1,
            speed,
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
        summary["speed_valid"] += int(speed["measurement_valid"])
        summary["overspeed"] += int(bool(speed["is_overspeed"]))
        if speed["measurement_valid"] and speed["speed_kmh"] is not None:
            summary["_speeds"].append(speed["speed_kmh"])
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
            LOGGER.info("track=%s event_from_finish (track ended before the final band)", track_id)
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
            speed["speed_kmh"],
            speed["speed_status"],
        )
    return written
