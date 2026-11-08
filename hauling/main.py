"""Application entry point for one offline hauling run."""

import json
import logging
import traceback
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from hauling.cli import parse_args
from hauling.config import Config, load_config
from hauling.options import RunOptions
from hauling.pipeline import run_pipeline


def _tracker_path(config_path: Path, config: Config, requested: Path | None) -> tuple[Path, bool]:
    tracker = requested or config_path.resolve().parent / config.tracker_config
    if tracker.is_file() or requested is not None:
        return tracker, False
    return Path(__file__).resolve().parent / "configs" / "bytetrack.yaml", True


def _run_options(args: Any, config: Config) -> RunOptions:
    plate_models = (
        args.plate_character_model,
        args.plate_letter_model,
        args.plate_digit_classifier,
        args.plate_letter_classifier,
    )
    if any(plate_models) and not all(plate_models):
        raise ValueError(
            "Plate reading requires --plate-character-model, --plate-letter-model, "
            "--plate-digit-classifier and --plate-letter-classifier"
        )
    tracker, fallback = _tracker_path(args.config, config, args.tracker_config)
    return RunOptions(
        video=args.video,
        vehicle_model=args.vehicle_model,
        tracker_config=tracker,
        output_dir=args.output_dir,
        device=args.device,
        max_frames=args.max_frames,
        save_video=not args.no_save_video,
        threads=args.threads,
        tracker_fallback=fallback,
        plate_character_model=args.plate_character_model,
        plate_letter_character_model=args.plate_letter_model,
        plate_digit_model=args.plate_digit_classifier,
        plate_letter_model=args.plate_letter_classifier,
        plate_conf=args.plate_conf,
        plate_min_prob=args.plate_min_prob,
        plate_single_read_prob=args.plate_single_read_prob,
        dump_plate_crops=args.dump_plate_crops,
    )


def run_command(
    argv: list[str] | None = None,
    pipeline_runner: Callable[[RunOptions, Config], dict] = run_pipeline,
) -> int:
    """Parse and run one command, returning a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.unsafe_distance_limit is not None:
            if not args.unsafe_distance_limit > 0:
                raise ValueError("--unsafe-distance-limit must be positive")
            config = replace(config, unsafe_distance_limit_s=args.unsafe_distance_limit)
        summary = pipeline_runner(_run_options(args, config), config)
    except KeyboardInterrupt:
        logging.warning("Run interrupted; partial outputs and summary retained")
        return 130
    except Exception as error:
        logging.error("%s: %s", type(error).__name__, error)
        logging.debug("Traceback:", exc_info=True)
        traceback.print_exc()
        return 1
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "total_frames",
                    "events",
                    "speed_valid",
                    "overspeed",
                    "unsafe_distance",
                    "unsafe_distance_limit_s",
                    "duration_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_command(argv)
