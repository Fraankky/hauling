"""Command-line interface for the offline hauling pipeline."""

import argparse
import json
import logging
import traceback
from dataclasses import replace
from pathlib import Path

from hauling.config import load_config
from hauling.pipeline import RunOptions, run_pipeline


def main(argv: list[str] | None = None) -> int:
    """Run one local video, returning a nonzero exit code on failure."""
    parser = argparse.ArgumentParser(description="Offline hauling vehicle tracking and speed measurement")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--vehicle-model", type=Path, required=True, help="Local vehicle model (.pt or .engine)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracker-config", type=Path)
    parser.add_argument("--device", help="cpu, cuda, or a CUDA index")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--unsafe-distance-limit",
        type=float,
        help=(
            "Override the config's unsafe-distance headway limit in seconds. "
            "Limits at or below the 0.9 s artifact floor can never report a violation."
        ),
    )
    parser.add_argument("--threads", type=int, default=8, help="CPU Torch threads (default: 8; 12+ oversubscribes)")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument(
        "--dump-plate-crops",
        action="store_true",
        help="Write plate OCR diagnostics to outputs/<run>/plate_debug/ (band, detector input, characters)",
    )
    parser.add_argument("--plate-character-model", type=Path, help="Digit character detector .pt/.engine; enables text reading")
    parser.add_argument("--plate-letter-model", type=Path, help="Letter character detector .pt/.engine; pairs with --plate-character-model")
    parser.add_argument("--plate-digit-classifier", type=Path, help="Digit classifier .pt/.engine (0-9)")
    parser.add_argument("--plate-letter-classifier", type=Path, help="Letter classifier .pt/.engine (A-Z)")
    parser.add_argument("--plate-conf", type=float, default=0.4, help="Plate/character confidence threshold (default: 0.4)")
    parser.add_argument("--plate-min-prob", type=float, default=0.0, help="Reject a final plate read below this classifier probability (default: 0.0 = off)")
    parser.add_argument("--plate-single-read-prob", type=float, default=None, help="Accept a lone read at or above this probability (default: off)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        plate_models = (args.plate_character_model, args.plate_letter_model, args.plate_digit_classifier, args.plate_letter_classifier)
        if any(plate_models) and not all(plate_models):
            raise ValueError("Plate reading requires --plate-character-model, --plate-letter-model, --plate-digit-classifier and --plate-letter-classifier")
        config = load_config(args.config)
        if args.unsafe_distance_limit is not None:
            if not args.unsafe_distance_limit > 0:
                raise ValueError("--unsafe-distance-limit must be positive")
            config = replace(config, unsafe_distance_limit_s=args.unsafe_distance_limit)
        tracker = args.tracker_config or args.config.resolve().parent / config.tracker_config
        fallback = False
        if not tracker.is_file() and args.tracker_config is None:
            tracker = Path(__file__).resolve().parent / "configs" / "bytetrack.yaml"
            fallback = True
        options = RunOptions(
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
        summary = run_pipeline(options, config)
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
