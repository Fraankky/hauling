"""Command-line arguments for the offline hauling pipeline."""

import argparse
from pathlib import Path

from hauling.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compatibility entry point for the installed ``hauling`` command."""
    from hauling.main import run_command

    return run_command(argv, pipeline_runner=run_pipeline)
