"""Color-sign localization and per-character reading for cab signs."""

import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hauling.detector import validate_model_path

DISABLED_NOTE = "plate_detection_disabled"
NO_CANDIDATE_NOTE = "no_plate_candidate"
LAMBUNG_NOTE = "lambung_read"

REASON_NOTES = {
    "low_agreement": "plate_low_agreement",
    "no_agreement": "plate_no_agreement",
    "single_rejected": "plate_single_read_rejected",
    "format_invalid": "plate_format_invalid",
    "low_confidence": "plate_low_confidence",
    "blurry": "plate_blurry",
}

MIN_CROP_SIZE = 16

PREPROCESS_MODES = ("none", "clahe", "unsharp", "clahe+unsharp")


def preprocess_band(image: np.ndarray, mode: str) -> np.ndarray:
    """Enhance a sign band for character detection.

    ``clahe`` equalises contrast per tile in the LAB lightness channel, which
    recovers glyphs washed out by haze without shifting colour. ``unsharp``
    sharpens edges with a light unsharp mask. Both are opt-in because they also
    amplify noise; the local evaluator decides per site which mode wins.
    """
    if mode in ("", "none"):
        return image
    out = image
    if "clahe" in mode:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        out = cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    if "unsharp" in mode:
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)
    return out


def deskew_band(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate a band so its glyph row is horizontal.

    A sign seen at a slight angle fragments into several rows during mask
    merging, which loses the band entirely. The rotation is estimated from the
    red glyph pixels themselves; angles below 1 degree are left alone (the
    interpolation would blur more than it fixes) and above ``max_angle`` are
    treated as a wrong estimate.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 60, 60), (15, 255, 255)) | cv2.inRange(
        hsv, (160, 60, 60), (180, 255, 255)
    )
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 20:
        return image
    angle = cv2.minAreaRect(points)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 1.0 or abs(angle) > max_angle:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# OCR runs only near closest approach: vehicle bbox area relative to the
# largest area observed for the track so far. Far frames waste CPU and yield
# unreadable ~50 px plates whose garbage text then wins by raw confidence.
OCR_AREA_RATIO = 0.75

# Calibrated red-sign hue windows, most specific first. Each window is a tuple
# of (h_start, h_end) ranges ORed into one mask with a fixed S/V floor; the
# first window that yields a band wins. Override per site via the optional
# ``plate_hue_windows`` config key (JSON), e.g. [[[0,8],[168,180]]].
DEFAULT_HUE_WINDOWS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 8), (168, 180)),  # strict red
    ((0, 12), (165, 180)),  # relaxed, for orange-red signs
)
_RED_SAT_FLOOR = 90
_RED_VAL_FLOOR = 80


def ocr_gate(area: float, max_area: float, ratio: float = OCR_AREA_RATIO) -> bool:
    """Whether this frame is close enough to justify character reading."""
    return max_area > 0 and area >= ratio * max_area


def disabled_plate_fields() -> dict[str, object]:
    """Keep the final CSV contract without pretending a plate model was run."""
    return {
        "plate_detected": False,
        "plate_confidence": None,
        "plate_bbox_xyxy": None,
        "plate_timestamp_s": None,
        "plate_text": None,
        "notes": DISABLED_NOTE,
    }


class _OfflineYolo:
    """Shared loader: local weights only, no network, no auto-install side effects."""

    def __init__(self, model_path: Path, runtime_dir: Path, device: str, threads: int) -> None:
        validate_model_path(model_path, device)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir.resolve())
        os.environ["YOLO_VERBOSE"] = "false"
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(threads)
        self.device = device
        self.model = YOLO(str(model_path.resolve()), task="detect")
        if model_path.suffix.lower() == ".engine":
            self.model.overrides["device"] = device
            _ = self.model.names  # Fail before processing frames if the engine cannot load.
        # CPU predictor setup resets Torch threads; reapply before every inference.
        self.model.add_callback("on_predict_start", lambda predictor: torch.set_num_threads(threads))

    def predict(self, image: np.ndarray, conf: float, imgsz: int):
        result = self.model.predict(
            source=image,
            conf=conf,
            iou=0.7,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            save=False,
            show=False,
            augment=False,
        )[0]
        return result.boxes.cpu().numpy()


def _bands_from_mask(
    red: np.ndarray,
    *,
    bright: np.ndarray | None = None,
    w_min: int = 30,
    h_min: int = 10,
    h_max: int = 30,
    aspect: tuple[float, float] = (2.2, 8.0),
    min_components: int = 1,
    density_min: float = 0.0,
    density_max: float = 1.0,
    bright_min: float = 0.0,
    gap_ratio: float = 1.5,
    max_glyphs: int = 7,
) -> list[tuple[float, int, tuple[int, int, int, int]]]:
    """Merge same-row red fragments into text bands; score by red pixels.

    Returns ``(red_pixels, glyph_components, band)``. ``glyph_components``
    counts separate blobs inside the band: a real sign is several glyphs
    (measured 2-3), while a solid red surface such as a truck body is a
    single blob (measured 1). ``min_components`` and the density bounds let the
    hazy fallback require a text-like band (several sparse glyphs) so a
    desaturated red body is not mistaken for a sign.

    ``gap_ratio`` bounds the horizontal gap allowed between two fragments
    before they are merged: glyphs on one sign are separated by roughly their
    own height, so a fragment further away belongs to something else. Without
    this bound a sign was merged with unrelated red paint 28 px away, turning a
    71 px sign into a 146 px band that still passed the aspect filter.
    ``max_glyphs`` rejects bands holding more components than a plate has
    characters (a truck body fragments into dozens).

    ``density_min``/``density_max`` bound the red fill inside the band. A real
    sign is sparse red text on a light board (measured 0.02-0.20); a painted
    truck body is dense (0.3+) and a blank panel is nearly empty (<0.02).
    ``bright_min`` requires a light board behind the text (measured 0.10-0.44 on
    real signs, 0.00-0.10 on body panels), which is what separates a sign from
    red paint.
    """
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours]
    rects = [r for r in rects if r[2] >= 6 and r[3] >= 4]
    bands: list[tuple[int, int, int, int]] = []
    for x, y, w, h in rects:
        for index, (bx, by, bw, bh) in enumerate(bands):
            vertical = min(y + h, by + bh) - max(y, by)
            if vertical <= 0.5 * min(h, bh):
                continue
            gap = max(0, max(x, bx) - min(x + w, bx + bw))
            if gap > gap_ratio * min(h, bh):
                continue
            end_x = max(x + w, bx + bw)
            bands[index] = (min(x, bx), min(y, by), end_x - min(x, bx), max(y + h, by + bh) - min(y, by))
            break
        else:
            bands.append((x, y, w, h))
    scored = []
    for x, y, w, h in bands:
        # Text bands are wide and short; beacon clusters are taller and squarer.
        if w < w_min or h < h_min or h > h_max or not aspect[0] <= w / h <= aspect[1]:
            continue
        red_pixels = float(red[y : y + h, x : x + w].sum() / 255)
        components, _ = cv2.connectedComponents(red[y : y + h, x : x + w], 8)
        glyphs = components - 1
        if glyphs < min_components or glyphs > max_glyphs:
            continue
        density = red_pixels / (w * h)
        if density < density_min or density > density_max:
            continue
        if bright is not None and bright_min > 0:
            bright_fraction = float(bright[y : y + h, x : x + w].sum() / 255) / (w * h)
            if bright_fraction < bright_min:
                continue
        scored.append((red_pixels, glyphs, (x, y, w, h)))
    # Text-like bands (several glyph blobs) first, then by red pixel mass.
    scored.sort(key=lambda item: (-(item[1] >= 2), -item[0]))
    return scored


def color_sign_bands(
    vehicle_crop: np.ndarray,
    max_candidates: int = 3,
    hue_windows: tuple[tuple[tuple[int, int], ...], ...] = DEFAULT_HUE_WINDOWS,
    sat_floors: tuple[int, ...] = (_RED_SAT_FLOOR,),
    val_floor: int = _RED_VAL_FLOOR,
    h_min: int = 10,
    h_max: int = 30,
    hazy_h_max: int = 90,
    roi: tuple[float, float, float, float] | None = None,
    gap_ratio: float = 1.5,
    max_glyphs: int = 7,
    density_min: float = 0.03,
    density_max: float = 0.60,
    bright_min: float = 0.12,
) -> list[tuple[int, int, int, int]]:
    """Candidate red text bands, most text-like first, adapting to haze.

    The cab sign competes with beacon lights, windshield banners and — on
    red-bodied trucks — the paint itself. ``sat_floors`` are tried in order
    (strictest first): the first floor that yields a band wins. The strict
    floor keeps calibrated behaviour unchanged; a lower fallback floor runs
    only when the strict one finds nothing, and there it additionally demands
    a text-like band (>=2 glyph blobs, not a near-solid red area) with a taller
    height allowance — the only way a desaturated sign can be told apart from
    the truck body a low floor also admits.

    ``roi`` is an optional ``(x_min, y_min, x_max, y_max)`` window as fractions
    of the vehicle crop; a band whose centre falls outside it is discarded. The
    cab sign sits in a narrow region (measured centre x 0.72-0.80, y 0.60-0.69
    of the vehicle box), so this rejects red paint elsewhere on the truck.

    ``density_min``/``density_max``/``bright_min`` describe what a sign looks
    like regardless of hue: sparse red text (0.02-0.20 fill) on a light board
    (>=0.12 bright pixels). Red paint on the body is denser or has no light
    board behind it, which is what these bounds reject.
    """
    hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
    height, width = vehicle_crop.shape[:2]
    # Light board behind the characters: low saturation, high value. A real sign
    # is printed on a white plate, so this is independent of the text colour.
    bright = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
    for order, sat_floor in enumerate(sat_floors):
        fallback = order > 0
        limit = hazy_h_max if fallback else h_max
        for window in hue_windows:
            red = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for h_start, h_end in window:
                red |= cv2.inRange(hsv, (h_start, sat_floor, val_floor), (h_end, 255, 255))
            red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            scored = _bands_from_mask(
                red,
                bright=bright,
                h_min=h_min,
                h_max=limit,
                min_components=2 if fallback else 1,
                density_min=density_min,
                density_max=density_max,
                bright_min=bright_min,
                gap_ratio=gap_ratio,
                max_glyphs=max_glyphs,
            )
            if roi is not None:
                x_min, y_min, x_max, y_max = roi
                scored = [
                    item for item in scored
                    if x_min <= (item[2][0] + item[2][2] / 2) / width <= x_max
                    and y_min <= (item[2][1] + item[2][3] / 2) / height <= y_max
                ]
            if scored:
                return [band for _px, _n, band in scored[:max_candidates]]
    return []


class ColorSignLocalizer:
    """HSV red-sign localization (fast, no model; only finds red signs)."""

    def __init__(
        self,
        hue_windows: tuple[tuple[tuple[int, int], ...], ...] = DEFAULT_HUE_WINDOWS,
        sat_floors: tuple[int, ...] = (_RED_SAT_FLOOR,),
        val_floor: int = _RED_VAL_FLOOR,
        band_h_min: int = 10,
        band_h_max: int = 30,
        hazy_h_max: int = 90,
        roi: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.hue_windows = hue_windows
        self.sat_floors = sat_floors
        self.val_floor = val_floor
        self.band_h_min = band_h_min
        self.band_h_max = band_h_max
        self.hazy_h_max = hazy_h_max
        self.roi = roi

    def localize(
        self, vehicle_crop: np.ndarray, max_candidates: int = 3
    ) -> list[tuple[int, int, int, int]]:
        return color_sign_bands(
            vehicle_crop, max_candidates, self.hue_windows, self.sat_floors, self.val_floor,
            self.band_h_min, self.band_h_max, self.hazy_h_max,
            self.roi,
        )


def plate_fields(
    state: Any,
    min_prob: float = 0.0,
    single_read_prob: float | None = None,
    full_digits: int = 4,
    min_agreement: int = 2,
    pattern: str | None = None,
    min_sharpness: float = 0.0,
    trusted_min_confidence: float = 0.90,
    trusted_min_sharpness: float = 1000.0,
) -> dict[str, object]:
    """Build the CSV plate columns from the track's resolved read.

    ``plate_detected`` follows the PRD: it is true when a sign *region* was
    located, whether or not its characters could be read. ``plate_text`` is
    filled only for an accepted read; otherwise it stays empty and ``notes``
    carries the reason. ``pattern`` (optional regex) rejects text that does not
    match the site's plate format, and ``min_prob``/``min_agreement``/
    ``min_sharpness`` gate the read — an empty cell is preferred over a wrong
    plate.

    A rejected read still reports its ``plate_bbox_xyxy`` **and**
    ``plate_timestamp_s``, so the audit crop can be taken from the frame the box
    was measured on. Previously the timestamp was dropped, and evidence crops
    fell back to the crossing frame, showing road instead of the sign.
    """
    localized = bool(getattr(state, "plate_localized", False))
    outcome = state.lambung_outcome(
        single_read_prob=single_read_prob,
        full_digits=full_digits,
        min_agreement=min_agreement,
        min_sharpness=min_sharpness,
        trusted_min_confidence=trusted_min_confidence,
        trusted_min_sharpness=trusted_min_sharpness,
    )
    # A read attempt means a sign region was found (the reader only reads
    # located bands), so a rejected read still counts as a detection.
    if outcome.reason != "no_read":
        localized = True
    text = outcome.text
    if text is not None and pattern and re.fullmatch(pattern, text) is None:
        outcome = type(outcome)(None, None, None, "format_invalid")
        text = None
    if text is not None and outcome.confidence is not None and outcome.confidence < min_prob:
        outcome = type(outcome)(None, None, None, "low_confidence")
        text = None

    if text is not None:
        bbox = list(getattr(state, "lambung_box", None) or ())
        return {
            "plate_detected": True,
            "plate_confidence": outcome.confidence,
            "plate_bbox_xyxy": bbox or None,
            "plate_timestamp_s": outcome.timestamp,
            "plate_text": text,
            "notes": LAMBUNG_NOTE,
        }
    if localized:
        # Sign located but not accepted as text: keep the detection and the
        # evidence frame, explain the empty text.
        return {
            "plate_detected": True,
            "plate_confidence": getattr(state, "plate_confidence", None),
            "plate_bbox_xyxy": list(getattr(state, "lambung_box", None) or ()) or None,
            "plate_timestamp_s": getattr(state, "lambung_box_ts", None),
            "plate_text": None,
            "notes": REASON_NOTES.get(outcome.reason, NO_CANDIDATE_NOTE),
        }
    return {
        "plate_detected": False,
        "plate_confidence": None,
        "plate_bbox_xyxy": None,
        "plate_timestamp_s": None,
        "plate_text": None,
        "notes": NO_CANDIDATE_NOTE,
    }


class LambungReader:
    """Read the cab sign (nomor lambung), e.g. RBT 4259, on a localized region.

    Localization is delegated to a pluggable localizer (:class:`ColorSignLocalizer`
    by default). Characters are read on a 4x upscale with the paired char
    detectors/classifiers; slot decoding clusters same-glyph duplicates, folds
    between-slots ghosts, then anchors the pick (3 letters left, 4 digits right).
    """

    UPSCALE = 4.0

    def __init__(
        self,
        digit_detect_model_path: Path,
        letter_detect_model_path: Path,
        digit_model_path: Path,
        letter_model_path: Path,
        runtime_dir: Path,
        device: str,
        conf: float,
        threads: int = 4,
        localizer: Any = None,
        letters: int = 3,
        digits: int = 4,
        min_digits: int = 3,
        band_pad: int = 4,
        upscale: float = 4.0,
        char_imgsz: int = 224,
        classifier_imgsz: int = 96,
        min_crop_size: int = MIN_CROP_SIZE,
        char_min_conf: float = 0.0,
        char_class_min_conf: float = 0.0,
        preprocess: str = "none",
        deskew: bool = False,
    ) -> None:
        for path in (digit_detect_model_path, letter_detect_model_path, digit_model_path, letter_model_path):
            validate_model_path(path, device)
        self.digit_detector = _OfflineYolo(digit_detect_model_path, runtime_dir, device, threads)
        self.letter_detector = _OfflineYolo(letter_detect_model_path, runtime_dir, device, threads)
        self.conf = conf
        self.localizer = localizer if localizer is not None else ColorSignLocalizer()
        self.LETTERS = letters
        self.DIGITS = digits
        self.MIN_DIGITS = min_digits
        self.band_pad = band_pad
        self.UPSCALE = upscale
        self.char_imgsz = char_imgsz
        self.classifier_imgsz = classifier_imgsz
        self.min_crop_size = min_crop_size
        self.char_min_conf = char_min_conf
        self.char_class_min_conf = char_class_min_conf
        self.preprocess = preprocess
        self.deskew = deskew
        from ultralytics import YOLO

        self.digit_model = YOLO(str(digit_model_path.resolve()), task="classify")
        self.letter_model = YOLO(str(letter_model_path.resolve()), task="classify")
        for model, path in ((self.digit_model, digit_model_path), (self.letter_model, letter_model_path)):
            if path.suffix.lower() == ".engine":
                model.overrides["device"] = device
                _ = model.names
        self.last_probability = 0.0
        # Whether the most recent read() located a candidate sign region (even
        # if its characters could not be decoded) — feeds ``plate_detected``.
        self.last_localized = False
        # Best classifier probability seen for a located band, even when no text
        # was produced. The PRD asks for the highest-confidence plate candidate
        # per track, so a rejected read still reports its confidence.
        self.last_band_probability = 0.0
        self.last_debug: dict[str, Any] = {}
        self._char_boxes: list[tuple[int, int, int, int]] = []

    @classmethod
    def find_signs(cls, vehicle_crop: np.ndarray, max_candidates: int = 3) -> list[tuple[int, int, int, int]]:
        """Color-path candidate text bands (kept for tests/simple callers)."""
        return color_sign_bands(vehicle_crop, max_candidates)

    @classmethod
    def find_sign(cls, vehicle_crop: np.ndarray) -> tuple[int, int, int, int] | None:
        """Best single color-path candidate (kept for tests/simple callers)."""
        candidates = cls.find_signs(vehicle_crop, max_candidates=1)
        return candidates[0] if candidates else None

    def read(self, vehicle_crop: np.ndarray) -> tuple[str, tuple[int, int, int, int]] | None:
        """Return (text, sign_box_in_crop) or None when no plausible sign exists.

        Localizes via the pluggable localizer, then scores the regions.
        See :meth:`read_regions`.
        """
        bands = self.localizer.localize(vehicle_crop)
        self.last_localized = bool(bands)
        return self.read_regions(vehicle_crop, bands)

    def read_regions(
        self, vehicle_crop: np.ndarray, bands: list[tuple[int, int, int, int]]
    ) -> tuple[str, tuple[int, int, int, int]] | None:
        """Classify/decode explicit candidate bands; the strongest digit evidence wins.

        Split from :meth:`read` so the deferred learned-localizer pass can score
        regions found by a different localizer without re-running localization.
        The mean classifier probability of the winning read is exposed as
        ``last_probability`` so the caller can weight its vote.
        """
        self.last_probability = 0.0
        self.last_band_probability = 0.0
        self.last_debug = {}
        best: tuple[int, str, tuple[int, int, int, int], float] | None = None
        for sign in bands:
            x, y, w, h = sign
            pad = self.band_pad
            sub = vehicle_crop[max(0, y - pad) : min(vehicle_crop.shape[0], y + h + pad), max(0, x - pad) : min(vehicle_crop.shape[1], x + w + pad)]
            if min(sub.shape[:2]) < self.min_crop_size:
                continue
            if self.deskew:
                sub = deskew_band(sub)
            sub = preprocess_band(sub, self.preprocess)
            big = cv2.resize(sub, None, fx=self.UPSCALE, fy=self.UPSCALE, interpolation=cv2.INTER_LANCZOS4)
            characters = self._classify(big)
            letters, digits, probability = self._decode(characters)
            self.last_band_probability = max(self.last_band_probability, probability)
            # Diagnostic snapshot: the exact pixels handed to the character
            # detectors and what they reported, so a rejected read can be
            # inspected instead of guessed about.
            self.last_debug = {
                # Raw band (no pad/preprocess/upscale): the local evaluator
                # re-applies those variants to it without needing the video.
                "raw": vehicle_crop[y : y + h, x : x + w].copy(),
                "band_bbox_xywh": sign,
                "candidate_count": len(bands),
                "band": sub,
                "input": big,
                "characters": characters,
                "char_boxes": self._char_boxes.copy(),
                "picked_indices": self._picked_indices,
                "letters": letters,
                "digits": digits,
                "probability": probability,
            }
            if len(digits) < self.MIN_DIGITS:
                continue
            # Prefer the band with the most digit evidence; classifier confidence
            # is miscalibrated in haze (confident garbage), so it must not
            # outrank glyph count. (Probability is still recorded for voting.)
            evidence = len(digits) * 10 + len(letters)
            if best is None or evidence > best[0]:
                best = (evidence, letters + " " + digits, sign, probability)
        if best is None:
            return None
        self.last_probability = best[3]
        return best[1], best[2]

    def _classify(self, big: np.ndarray) -> list[tuple[float, float, str, str, float, float]]:
        """(center_x, center_y, char, source, det_conf, class_prob) per detected character.

        ``det_conf`` is the detector's box confidence; ``class_prob`` is the
        classifier's probability for the chosen character — the latter is the
        one that separates a confident read from a guess.

        Glyph crops are classified in one batched call per channel instead of
        one call per glyph: a plate yields ~7 characters, so this turns a dozen
        sequential forward passes into two, which is the cheapest CPU win in the
        plate path.
        """
        characters = []
        self._char_boxes = []
        for detector, classifier, source in (
            (self.digit_detector, self.digit_model, "d"),
            (self.letter_detector, self.letter_model, "l"),
        ):
            boxes = detector.predict(big, self.conf, self.char_imgsz)
            if len(boxes) == 0:
                continue
            glyphs, positions = [], []
            for xyxy, box_conf in zip(boxes.xyxy, boxes.conf, strict=True):
                if float(box_conf) < self.char_min_conf:
                    continue
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                char_crop = big[max(0, y1) : y2, max(0, x1) : x2]
                if char_crop.size == 0:
                    continue
                glyphs.append(char_crop)
                positions.append(((x1 + x2) / 2, (y1 + y2) / 2, float(box_conf), (x1, y1, x2, y2)))
            if not glyphs:
                continue
            results = classifier.predict(glyphs, imgsz=self.classifier_imgsz, device=self.digit_detector.device, verbose=False)
            for (center_x, center_y, box_conf, box), result in zip(positions, results, strict=True):
                class_prob = float(result.probs.top1conf)
                if class_prob < self.char_class_min_conf:
                    continue
                characters.append(
                    (
                        center_x,
                        center_y,
                        result.names[int(result.probs.top1)],
                        source,
                        box_conf,
                        class_prob,
                    )
                )
                self._char_boxes.append(box)
        return characters

    def _decode(self, characters: list[tuple[float, float, str, str, float, float]]) -> tuple[str, str, float]:
        """Split by channel with anchored slot picking: letters left, digits right.

        Both detectors fire on the same glyphs with high confidence, so a
        position-based split cannot tell a real fourth letter from a digit the
        letter detector misread. Measured on the same footage:

        ``RBT 4245`` → letters R,B,T,**A**(x=304) and digits 4(x=306)
        ``GECL 9105`` → letters G,E,C,**L**(x=292) and digits 3(x=296)

        Both are "one glyph, two readings" at the same x, and both readings are
        confident — the difference is only what the plate really says. The
        decoder therefore honours ``plate_letters`` as a hard count, and the
        site decides: three letters is the calibrated default; a site whose
        plates really carry four letters sets ``plate_letters = 4`` knowing that
        its three-letter plates will then gain a spurious fourth character.

        The pick itself is geometry-driven. The old fixed-gap dedupe (30 px in
        upscale space) measured wrong against the dumps: real glyph gaps run
        19-48 px, so the fixed rule merged *real* digits and kept ghost boxes,
        which displaced the anchored right-4 window. Instead each channel is
        clustered into same-glyph groups (a duplicate box hugs its source),
        a between-slots ghost folds into its tighter neighbour, and the slots
        are then anchored — leftmost letters, rightmost digits. See
        :meth:`_pick_slots` and :meth:`_fold_ghost_slots`.

        Returns (letters, digits, mean class probability of picked chars).
        """
        letters = sorted((c for c in characters if c[3] == "l"), key=lambda c: c[0])
        digits = sorted((c for c in characters if c[3] == "d"), key=lambda c: c[0])
        picked_letters = self._pick_slots(letters, self.LETTERS, from_left=True)
        picked_digits = self._pick_slots(digits, self.DIGITS, from_left=False)
        letters_text = "".join(c[2] for c in picked_letters)
        digits_text = "".join(c[2] for c in picked_digits)
        picked = picked_letters + picked_digits
        self._picked_indices = [next(i for i, candidate in enumerate(characters) if candidate is char) for char in picked]
        probability = sum(c[5] for c in picked) / len(picked) if picked else 0.0
        return letters_text, digits_text, probability

    DUP_BAND_PX = 3.5  # same-glyph duplicates sit within ~2.75 band px (measured <=11 px at 4x)

    def _pick_slots(
        self, chars: list[tuple[float, float, str, str, float, float]], count: int, from_left: bool
    ) -> list[tuple[float, float, str, str, float, float]]:
        """Cluster same-glyph detections, then pick anchored slots.

        One glyph is often detected twice — the cross-channel echo (a letter
        shows up in the digit stream) or an NMS survivor — and the duplicate
        box hugs its source within ~3 band px, while real slot gaps measure
        >=19 band px. After clustering, the letter slots are the leftmost
        ``count`` clusters and the digit slots the rightmost: leading and
        trailing ghosts fall outside the anchored window. Each kept cluster is
        read by its strongest (class_prob, det_conf) glyph.
        """
        if not chars:
            return []
        clusters: list[list[tuple[float, float, str, str, float, float]]] = [[chars[0]]]
        for char in chars[1:]:
            if char[0] - clusters[-1][-1][0] < self.DUP_BAND_PX * self.UPSCALE:
                clusters[-1].append(char)
            else:
                clusters.append([char])
        clusters = self._fold_ghost_slots(clusters)
        slots = clusters[:count] if from_left else clusters[-count:]
        return [max(cluster, key=lambda c: (c[5], c[4])) for cluster in slots]

    def _fold_ghost_slots(
        self, clusters: list[list[tuple[float, float, str, str, float, float]]]
    ) -> list[list[tuple[float, float, str, str, float, float]]]:
        """Merge a between-slots ghost cluster into its tighter neighbour.

        A stray box that sits inside the glyph grid (measured: the ghost 6
        between 0 and 1 in ``RBT 4301`` lands 15/17 px from its neighbours at
        a ~34 px slot pitch) is squeezed below 0.65x the median gap on *both*
        sides — a real glyph never is. Real tight pairs (the double 0 of
        ``...0047``, gap 21 at a 27 px pitch) stay above the fold line.
        """
        if len(clusters) < 3:
            return clusters
        gaps = [right[0][0] - left[-1][0] for left, right in zip(clusters, clusters[1:])]
        tight = 0.65 * sorted(gaps)[len(gaps) // 2]
        keep = [True] * len(clusters)
        folded: dict[int, list[tuple[float, float, str, str, float, float]]] = {}
        for i in range(1, len(clusters) - 1):
            left = clusters[i][0][0] - clusters[i - 1][-1][0]
            right = clusters[i + 1][0][0] - clusters[i][-1][0]
            if left < tight and right < tight:
                keep[i] = False
                folded.setdefault(i - 1 if left <= right else i + 1, []).append(clusters[i])
        if not folded:
            return clusters
        out: list[list[tuple[float, float, str, str, float, float]]] = []
        for i, cluster in enumerate(clusters):
            if not keep[i]:
                continue
            group = cluster + [char for fold in folded.get(i, []) for char in fold]
            out.append(sorted(group, key=lambda c: c[0]))
        return out
