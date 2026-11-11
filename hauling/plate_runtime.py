"""Connect plate OCR to pipeline frames, state, metrics, and diagnostics."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2

from hauling.config import Config
from hauling.geometry import clip_vehicle_box
from hauling.options import RunOptions
from hauling.plate import DEFAULT_HUE_WINDOWS, ColorSignLocalizer, LambungReader, ocr_gate
from hauling.state import TrackState

LOGGER = logging.getLogger(__name__)


def create_plate_reader(
    options: RunOptions,
    config: Config,
    device: str,
    runtime_dir: Path,
    reader_class: type | None = None,
) -> Any | None:
    if options.plate_character_model is None:
        return None
    reader_class = reader_class or LambungReader
    return reader_class(
        options.plate_character_model,
        options.plate_letter_character_model,
        options.plate_digit_model,
        options.plate_letter_model,
        runtime_dir,
        device,
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
        char_class_min_conf=config.plate_char_class_min_conf,
        preprocess=config.plate_preprocess,
        deskew=config.plate_deskew,
    )


def plate_calibration(config: Config, reader_class: type | None = None) -> dict:
    reader_class = reader_class or LambungReader
    return {
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
        "dup_band_px": reader_class.DUP_BAND_PX,
        "char_imgsz": config.plate_char_imgsz,
        "classifier_imgsz": config.plate_classifier_imgsz,
        "char_min_conf": config.plate_char_min_conf,
        "char_class_min_conf": config.plate_char_class_min_conf,
        "preprocess": config.plate_preprocess,
        "deskew": config.plate_deskew,
        "max_ocr_frames": config.plate_max_ocr_frames,
        "event_on_finish_fallback": config.event_on_finish_fallback,
        "pattern": config.plate_pattern,
        "roi": list(config.plate_roi) if config.plate_roi else None,
        "ocr_classes": list(config.plate_ocr_classes),
    }


def dump_plate_attempt(
    reader: Any,
    output_dir: Path,
    track_id: int,
    frame_index: int,
    timestamp: float,
    result: tuple[str, tuple[int, int, int, int]] | None,
    attempt: int,
    vehicle_confidence: float | None = None,
    vehicle_bbox_xyxy: tuple[float, float, float, float] | None = None,
) -> bool:
    """Persist one diagnostic OCR attempt."""
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
    boxes = debug.get("char_boxes", [])
    picked = debug.get("picked_indices", [])
    characters = [
        {
            "x": round(char[0], 1),
            "y": round(char[1], 1),
            "char": char[2],
            "channel": char[3],
            "det_conf": round(char[4], 4),
            "class_prob": round(char[5], 4),
            "bbox_xyxy": boxes[index] if index < len(boxes) else None,
            "selected": index in picked,
        }
        for index, char in enumerate(debug.get("characters") or [])
    ]
    if detector_input is not None:
        for index, char in enumerate(characters):
            if not char["selected"] or char["bbox_xyxy"] is None:
                continue
            x1, y1, x2, y2 = char["bbox_xyxy"]
            glyph = detector_input[max(0, y1):y2, max(0, x1):x2]
            if glyph.size:
                cv2.imwrite(str(directory / f"{stem}_char{index:02d}.png"), glyph)
    payload = {
        "track_id": track_id,
        "frame_index": frame_index,
        "timestamp_s": round(timestamp, 3),
        "attempt": attempt,
        "vehicle_confidence_ocr_frame": vehicle_confidence,
        "vehicle_bbox_xyxy_ocr_frame": vehicle_bbox_xyxy,
        "sign_band_bbox_xywh_in_vehicle": debug.get("band_bbox_xywh"),
        "sign_candidate_count": debug.get("candidate_count"),
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


def process_plate_observation(
    reader: Any,
    frame: Any,
    observation: Any,
    state: TrackState,
    areas: set[str],
    frame_index: int,
    timestamp: float,
    options: RunOptions,
    config: Config,
    summary: dict,
    dumped_counts: dict[int, int],
    model_formats: set[str],
) -> None:
    """Read one eligible vehicle crop and update its state and run counters."""
    class_ok = not config.plate_ocr_classes or state.vehicle_class in config.plate_ocr_classes
    if not class_ok or not (areas & TrackState.MEASUREMENT_AREAS):
        return
    ocr_cap = config.plate_max_ocr_frames
    within_cap = ocr_cap <= 0 or state.ocr_attempts < ocr_cap
    x1, y1, x2, y2 = observation.bbox
    if state.direction != "South" or not within_cap or not ocr_gate((x2 - x1) * (y2 - y1), state.max_bbox_area):
        return
    clipped = clip_vehicle_box(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
    if clipped is None:
        return
    cx1, cy1, cx2, cy2 = clipped
    crop = frame[cy1:cy2, cx1:cx2]
    try:
        step_started = time.monotonic()
        try:
            lambung = reader.read(crop)
        finally:
            summary["timing_seconds"]["plate_read"] += time.monotonic() - step_started
            summary["timing_calls"]["plate_read"] += 1
        summary["ocr_frames"] += 1
        state.ocr_attempts += 1
        summary["ocr_attempts"] += 1
        if reader.last_localized:
            summary["ocr_band_found"] += 1
            state.plate_localized = True
            band_conf = reader.last_band_probability
            if band_conf and (state.plate_confidence or 0.0) < band_conf:
                state.plate_confidence = band_conf
        if reader.last_debug.get("characters"):
            summary["ocr_chars_found"] += 1
        if lambung is not None:
            summary["ocr_decode_ok"] += 1
        track_id = observation.track_id
        if (
            options.dump_plate_crops
            and reader.last_debug
            and dumped_counts.get(track_id, 0) < config.plate_dump_max_per_track
            and dump_plate_attempt(
                reader,
                options.output_dir,
                track_id,
                frame_index,
                timestamp,
                lambung,
                dumped_counts.get(track_id, 0) + 1,
                observation.confidence,
                observation.bbox,
            )
        ):
            dumped_counts[track_id] = dumped_counts.get(track_id, 0) + 1
        if lambung is None:
            return
        text, sign_box = lambung
        frame_h, frame_w = frame.shape[:2]
        veh_w = max(1.0, x2 - x1)
        veh_h = max(1.0, y2 - y1)
        rel_w = sign_box[2] / veh_w
        rel_h = sign_box[3] / veh_h
        summary["plate_rel_w_max"] = max(summary["plate_rel_w_max"], rel_w)
        summary["plate_rel_h_max"] = max(summary["plate_rel_h_max"], rel_h)
        if rel_w > 0.30 or rel_h > 0.06:
            summary["plate_crop_suspect"] += 1
        sign_crop = crop[
            max(0, sign_box[1]):sign_box[1] + sign_box[3],
            max(0, sign_box[0]):sign_box[0] + sign_box[2],
        ]
        sharpness = (
            float(cv2.Laplacian(cv2.cvtColor(sign_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if sign_crop.size
            else 0.0
        )
        summary["plate_sharpness_max"] = max(summary["plate_sharpness_max"], sharpness)
        state.observe_lambung(
            text,
            reader.last_probability,
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
        LOGGER.warning("plate inference failed frame=%s: %s", frame_index, error)
    except Exception as error:
        if ".engine" in model_formats:
            raise
        LOGGER.warning("plate inference failed frame=%s: %s", frame_index, error)
