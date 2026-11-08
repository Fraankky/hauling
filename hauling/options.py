"""Input and output controls for one pipeline run."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOptions:
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
