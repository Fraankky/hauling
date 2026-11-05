"""Build per-event plate evidence and compare two completed offline runs."""

import csv
import json
import shutil
from pathlib import Path

import cv2

COMPARE_FIELDS = (
    "event_timestamp_s",
    "travel_direction",
    "speed_status",
    "speed_kmh",
    "is_overspeed",
    "is_unsafe_distance",
    "is_contraflow",
    "plate_detected",
    "plate_text",
    "notes",
)


def _events(run_dir: Path) -> dict[int, dict[str, str]]:
    with (run_dir / "events.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    events = {}
    for row in rows:
        frame_index = int(row["event_frame_index"])
        if frame_index in events:
            # shortcut: match by crossing frame, use spatial matching if two events share a frame.
            raise ValueError(f"Multiple events at frame {frame_index}; match these manually")
        events[frame_index] = row
    return events


def save_plate_evidence(video: Path, run_dir: Path) -> dict[int, dict[str, str]]:
    """Use the OCR timestamp/box; fall back to a vehicle crop when no sign was found."""
    events = _events(run_dir)
    output = run_dir / "plates"
    output.mkdir(exist_ok=False)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open evidence video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        for frame_index, row in events.items():
            stem = f"f{frame_index:06d}_trk{int(row['track_id']):03d}"
            label = "read" if row["plate_text"] else "no_text"
            if not row["plate_bbox_xyxy"] or not row["plate_timestamp_s"]:
                source = run_dir / "events_crops" / f"trk{int(row['track_id']):03d}_f{frame_index:06d}.png"
                shutil.copyfile(source, output / f"{stem}_{label}_vehicle_only.png")
                continue

            evidence_frame = round(float(row["plate_timestamp_s"]) * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, evidence_frame)
            ok, frame = capture.read()
            if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != evidence_frame + 1:
                raise ValueError(f"Cannot decode evidence frame {evidence_frame}: {video}")
            x1, y1, x2, y2 = json.loads(row["plate_bbox_xyxy"])
            height, width = frame.shape[:2]
            pad_x, pad_y = (x2 - x1) * 0.6, (y2 - y1) * 0.6
            crop = frame[
                max(0, int(y1 - pad_y)) : min(height, int(y2 + pad_y)),
                max(0, int(x1 - pad_x)) : min(width, int(x2 + pad_x)),
            ]
            if crop.size == 0:
                raise ValueError(f"Empty plate box at event frame {frame_index}")
            if crop.shape[1] < 320:
                scale = 320 / crop.shape[1]
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            if not cv2.imwrite(str(output / f"{stem}_{label}_plate_f{evidence_frame:06d}.png"), crop):
                raise OSError(f"Cannot write plate evidence for frame {frame_index}")
    finally:
        capture.release()
    return events


def compare_runs(video: Path, pytorch_dir: Path, tensorrt_dir: Path, output: Path) -> dict:
    """Group differing events by crossing frame, ignoring runtime-specific track IDs."""
    events = {
        "pytorch": _events(pytorch_dir),
        "tensorrt": _events(tensorrt_dir),
    }
    dirs = {"pytorch": pytorch_dir, "tensorrt": tensorrt_dir}
    output.mkdir(parents=True, exist_ok=False)
    differing = []
    for frame_index in sorted(events["pytorch"].keys() | events["tensorrt"].keys()):
        a, b = events["pytorch"].get(frame_index), events["tensorrt"].get(frame_index)
        changes = {
            field: {"pytorch": a[field], "tensorrt": b[field]}
            for field in COMPARE_FIELDS
            if a and b and a[field] != b[field]
        }
        if not changes and a is not None and b is not None:
            continue
        folder = output / f"f{frame_index:06d}"
        folder.mkdir()
        (folder / "diff.json").write_text(
            json.dumps(
                {
                    "frame": frame_index,
                    "changes": changes,
                    "missing": [runtime for runtime in dirs if frame_index not in events[runtime]],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for runtime, row in (("pytorch", a), ("tensorrt", b)):
            if row is None:
                continue
            stem = f"f{frame_index:06d}_trk{int(row['track_id']):03d}"
            for crop in (dirs[runtime] / "plates").glob(f"{stem}_*.png"):
                shutil.copyfile(crop, folder / f"{runtime}_{crop.name}")
            vehicle = dirs[runtime] / "events_crops" / f"trk{int(row['track_id']):03d}_f{frame_index:06d}.png"
            shutil.copyfile(vehicle, folder / f"{runtime}_vehicle.png")
        differing.append(frame_index)
    report = {
        "video": video.name,
        "events": {key: len(value) for key, value in events.items()},
        "different_event_frames": differing,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
