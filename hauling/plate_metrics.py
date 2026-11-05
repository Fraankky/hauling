"""Plate-detection scoring: confusion matrix and derived metrics.

Answers *how well does plate reading do?* — a measurement over a labelled
table, deliberately separate from the pipeline that produces the predictions.

**Two questions are scored separately**, because the pipeline ties them
together (``plate_detected`` is true only when text was read):

* **deteksi** — is there a readable cab sign in the frame? (prediction:
  ``plate_detected``; truth: ``papan_terlihat``)
* **teks** — when a sign exists, is the string right? (``plate_text`` vs
  ``teks_gt``)

A wrong string in front of a real sign is still a correct *detection*. Scoring
it as a miss would hide a text problem inside the detection metric, and a
detection miss would be invisible in the text metric.

Truth comes from a separate CSV keyed by ``track_id`` so a new run can be scored
against the same labels without re-labelling:

    track_id,papan_terlihat,teks_gt
    1,1,AEK 0083
    3,1,
    14,0,

``papan_terlihat`` is 1 when a cab sign is visible in the frame (readable or
not) and 0 when none is. ``teks_gt`` is left empty when the sign is visible but
its characters cannot be confirmed — such rows count for detection but are
excluded from text accuracy, so an illegible-by-eye sign never inflates or
deflates the text score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRIC_FIELDS = ("precision", "recall", "accuracy", "f1", "specificity")


def _flag(value: object) -> bool:
    """Read a CSV flag that may be ``true``/``1``/``yes``."""
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _ratio(numerator: int, denominator: int) -> float | None:
    """Rate, or ``None`` when the denominator is zero (undefined, not 0.0)."""
    return None if denominator == 0 else numerator / denominator


def load_truth(path: Path) -> dict[str, dict[str, str]]:
    """Ground truth keyed by ``track_id``."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    truth: dict[str, dict[str, str]] = {}
    for row in rows:
        track_id = str(row.get("track_id", "")).strip()
        if not track_id:
            raise ValueError(f"{path}: row without track_id")
        truth[track_id] = {key: str(value or "") for key, value in row.items()}
    return truth


def score(events: list[dict], truth: dict[str, dict[str, str]]) -> tuple[list[dict], dict]:
    """Join predictions with truth and count TP/FP/FN/TN.

    Rows are dropped, never guessed, in two cases: a label with no prediction
    (the track produced no event) and a prediction with no label. Both are
    reported as ``skipped`` so a shrinking denominator stays visible.
    """
    joined: list[dict] = []
    skipped = {"unlabelled": 0, "no_prediction": 0}
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    text_ok = text_wrong = text_unconfirmed = 0

    for row in events:
        track_id = str(row.get("track_id", "")).strip()
        label = truth.get(track_id)
        if label is None:
            skipped["unlabelled"] += 1
            continue
        sign_present = _flag(label.get("papan_terlihat"))
        detected = _flag(row.get("plate_detected"))
        if sign_present and detected:
            verdict = "TP"
        elif not sign_present and detected:
            verdict = "FP"
        elif sign_present and not detected:
            verdict = "FN"
        else:
            verdict = "TN"
        counts[verdict] += 1

        predicted = str(row.get("plate_text") or "").strip()
        expected = str(label.get("teks_gt") or "").strip()
        text_verdict = ""
        if verdict == "TP":
            if not expected:
                text_verdict = "tidak_terkonfirmasi"
                text_unconfirmed += 1
            elif predicted.upper() == expected.upper():
                text_verdict = "benar"
                text_ok += 1
            else:
                text_verdict = "salah"
                text_wrong += 1

        joined.append(
            {
                "track_id": track_id,
                "travel_direction": row.get("travel_direction", ""),
                "speed_kmh": row.get("speed_kmh", ""),
                "plate_detected": row.get("plate_detected", ""),
                "plate_text": predicted,
                "papan_terlihat": int(sign_present),
                "teks_gt": expected,
                "verdict": verdict,
                "verdict_teks": text_verdict,
                "bukti": label.get("bukti", ""),
                "catatan": label.get("catatan", ""),
            }
        )

    for track_id in truth:
        if not any(row["track_id"] == track_id for row in joined):
            skipped["no_prediction"] += 1

    tp, fp, fn, tn = (counts[key] for key in ("TP", "FP", "FN", "TN"))
    summary = {
        "total_scored": len(joined),
        "skipped": skipped,
        "counts": counts,
        "metrics": {
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "specificity": _ratio(tn, tn + fp),
        },
        "text": {
            "benar": text_ok,
            "salah": text_wrong,
            "tidak_terkonfirmasi": text_unconfirmed,
            "accuracy": _ratio(text_ok, text_ok + text_wrong),
        },
    }
    return joined, summary


def render(summary: dict) -> str:
    """Human-readable block for the CLI."""
    counts = summary["counts"]
    metrics = summary["metrics"]
    text = summary["text"]

    def rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"Confusion (n={summary['total_scored']}): "
        f"TP={counts['TP']} FP={counts['FP']} FN={counts['FN']} TN={counts['TN']}",
        "  " + "  ".join(f"{name}={rate(metrics[name])}" for name in METRIC_FIELDS),
        f"Text: benar={text['benar']} salah={text['salah']} "
        f"tidak_terkonfirmasi={text['tidak_terkonfirmasi']} accuracy={rate(text['accuracy'])}",
    ]
    skipped = summary["skipped"]
    if skipped["unlabelled"] or skipped["no_prediction"]:
        lines.append(
            f"Skipped: unlabelled={skipped['unlabelled']} "
            f"no_prediction={skipped['no_prediction']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: score an ``events.csv`` against a labelled ground-truth CSV."""
    parser = argparse.ArgumentParser(
        prog="python -m hauling.plate_metrics",
        description="Plate detection/text metrics against a ground-truth CSV (no re-inference).",
    )
    parser.add_argument("events", type=Path, help="events.csv from a run")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth CSV keyed by track_id")
    parser.add_argument("--output", type=Path, help="Write the joined per-track table here")
    parser.add_argument("--json", type=Path, help="Write the summary as JSON here")
    args = parser.parse_args(argv)

    with args.events.open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    truth = load_truth(args.gt)
    joined, summary = score(events, truth)

    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
            writer.writeheader()
            writer.writerows(joined)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(render(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
