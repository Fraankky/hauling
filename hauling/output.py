"""Local output contracts for vehicle events."""

import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = (
    "event_id,source_video,track_id,event_timestamp_s,event_frame_index,travel_direction,"
    "vehicle_confidence,speed_start_timestamp_s,speed_end_timestamp_s,elapsed_seconds,"
    "distance_meter,speed_kmh,speed_status,speed_limit_kmh,is_overspeed,lead_track_id,"
    "headway_s,headway_status,unsafe_distance_limit_s,is_unsafe_distance,"
    "is_contraflow,contraflow_status,contraflow_coverage,contraflow_timestamp_s,"
    "plate_detected,"
    "plate_confidence,plate_bbox_xyxy,plate_timestamp_s,plate_text,measurement_valid,notes"
).split(",")


def prepare_output(path: Path) -> None:
    """Create an output directory without overwriting previous work."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory must be new or empty; choose another --output-dir")
    path.mkdir(parents=True, exist_ok=True)


class EventWriter:
    """Write the PRD CSV contract and reject duplicate track/event identifiers."""

    def __init__(self, path: Path) -> None:
        self.stream = path.open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=CSV_FIELDS)
        self.track_ids: set[int] = set()
        self.event_ids: set[str] = set()
        self.writer.writeheader()
        self.stream.flush()

    def write(self, event: dict[str, Any]) -> None:
        """Persist one event; missing values are empty, booleans lowercase."""
        track_id, event_id = event["track_id"], event["event_id"]
        if track_id in self.track_ids or event_id in self.event_ids:
            raise ValueError("Duplicate track or event ID")
        row = {}
        for key in CSV_FIELDS:
            value = event.get(key)
            if value is None:
                value = ""
            elif isinstance(value, bool):
                value = str(value).lower()
            elif key == "plate_bbox_xyxy" and not isinstance(value, str):
                value = json.dumps(value, separators=(",", ":"), allow_nan=False)
            row[key] = value
        self.writer.writerow(row)
        self.stream.flush()
        self.track_ids.add(track_id)
        self.event_ids.add(event_id)

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.stream.close()
