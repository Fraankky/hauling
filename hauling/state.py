"""Per-track crossing and event state without detector or output dependencies."""

import math
from collections import Counter
from dataclasses import dataclass, field

REASON_READ = "read"
REASON_NO_READ = "no_read"
REASON_LOW_AGREEMENT = "low_agreement"
REASON_NO_AGREEMENT = "no_agreement"
REASON_SINGLE_REJECTED = "single_rejected"
REASON_FORMAT_INVALID = "format_invalid"
REASON_BLURRY = "blurry"


@dataclass(frozen=True)
class PlateOutcome:
    """Result of resolving a track's cab-sign reads.

    ``text`` is ``None`` whenever the read was rejected, and ``reason`` says
    why — so a caller can report "sign seen but not readable" instead of
    pretending the sign was absent. ``confidence``/``timestamp`` describe the
    same frame as the winning read.
    """

    text: str | None
    confidence: float | None
    timestamp: float | None
    reason: str


@dataclass
class TrackState:
    """Remember first crossings and issue at most one final-area event per track."""

    track_id: int
    first_centroid: tuple[float, float] | None = None
    last_centroid: tuple[float, float] | None = None
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    event_written: bool = False
    event_pending: bool = False
    final_timestamp: float | None = None
    final_frame_index: int = -1
    # Finish-fallback rows have no final crossing to compare.
    lead_track_id: int | None = None
    headway_s: float | None = None
    headway_status: str = "no_final"
    is_unsafe_distance: bool = False
    # Keep the largest near-frame coverage so the verdict can be audited later.
    is_contraflow: bool = False
    contraflow_status: str = "inactive"
    contraflow_coverage: float = 0.0
    contraflow_timestamp_s: float | None = None
    contraflow_frame_index: int = -1
    final_not_crossed: bool = False
    last_frame_index: int = -1
    vehicle_confidence: float = 0.0
    max_bbox_area: float = 0.0
    # OCR is gated by class so stickers on other vehicles are not read as signs.
    vehicle_class: str | None = None
    # PRD plate_detected means localization, even when no text can be read.
    plate_localized: bool = False
    plate_confidence: float | None = None
    ocr_attempts: int = 0
    _finish_band: tuple[float, float] | None = None
    _finish_band_final: tuple[float, float] | None = None
    _lambung_reads: int = 0
    _lambung_best: tuple[str, float, float] | None = None
    _lambung_read_data: list = field(default_factory=list)
    lambung_reads: list[tuple[str, float]] = field(default_factory=list)
    lambung_box: tuple[float, float, float, float] | None = None
    lambung_box_ts: float | None = None

    # Bands extend beyond main; gating only on main loses valid crossings.
    MEASUREMENT_AREAS = frozenset({"main", "start", "finish", "final"})

    def update(
        self,
        reference: tuple[float, float],
        timestamp: float,
        frame_index: int,
        areas: set[str],
        confidence: float,
        bbox_area: float = 0.0,
    ) -> bool:
        """Observe a track; return true exactly once upon reaching main and final.

        ``reference`` is the point the caller tests against the bands — the
        ground contact by default, because the bands and the 47 m basis live on
        the road plane. Observations in none of :attr:`MEASUREMENT_AREAS` are
        ignored entirely, including ``last_frame_index``; pipeline's
        ``lost_before_final`` count relies on this. ``max_bbox_area`` tracks the
        largest vehicle area seen, used by the pipeline's near-camera OCR gate.
        """
        if not areas & self.MEASUREMENT_AREAS:
            return False
        if self.first_centroid is None:
            self.first_centroid = reference
        previous_y = self.last_centroid[1] if self.last_centroid else None
        self.last_centroid = reference
        self.last_frame_index = frame_index
        self.vehicle_confidence = confidence
        if bbox_area > self.max_bbox_area:
            self.max_bbox_area = bbox_area
        if self.event_written:
            return False
        if "finish" in areas or self._crossed(previous_y, reference[1], self._finish_band):
            if self.end_timestamp is None:
                self.end_timestamp = timestamp
        if "start" in areas and self.start_timestamp is None:
            self.start_timestamp = timestamp
        crossed_final = "final" in areas or self._crossed(previous_y, reference[1], self._finish_band_final)
        # Speed and cab-sign OCR are calibrated only for Southbound traffic.
        if crossed_final and self.direction == "South" and not self.event_written and not self.event_pending:
            # Defer writing until track end to use closer post-crossing OCR frames.
            self.event_pending = True
            self.final_timestamp = timestamp
            self.final_frame_index = frame_index
            return False
        return False

    def flush_event(self, allow_finish_fallback: bool = False) -> bool:
        """Promote a pending event to writable; the track is done.

        ``allow_finish_fallback`` is off by default because the PRD is explicit:
        a track lost before the final area must not be written, and
        ``event_timestamp_s`` must be the moment the final area was reached.
        Sites that prefer to keep the vehicle (speed is already complete at
        finish) can opt in through config, accepting that the timestamp then
        refers to the finish crossing.
        """
        if self.event_pending and not self.event_written:
            self.event_written = True
            return True
        if not allow_finish_fallback:
            return False
        # Finish is enough for speed, but without final this is an opt-in event.
        if (
            not self.event_written
            and self.direction == "South"
            and self.start_timestamp is not None
            and self.end_timestamp is not None
        ):
            self.event_pending = True
            self.final_not_crossed = True
            self.final_timestamp = self.end_timestamp
            self.final_frame_index = self.last_frame_index
            self.event_written = True
            return True
        return False

    @staticmethod
    def _crossed(previous_y: float | None, current_y: float, band: tuple[float, float] | None) -> bool:
        """Segment test: centroid moved across [lo, hi] between frames."""
        if previous_y is None or band is None:
            return False
        lo, hi = band
        return min(previous_y, current_y) <= hi and max(previous_y, current_y) >= lo

    def set_crossing_bands(self, finish: tuple[float, float], final: tuple[float, float]) -> None:
        """Install y-ranges used for interpolated crossing tests (frame coords)."""
        self._finish_band = finish
        self._finish_band_final = final

    @staticmethod
    def _split_read(text: str) -> tuple[str, str]:
        """Split a read into (letters, digits) regardless of spacing.

        Reads arrive as ``"AEK 0083"`` (from the reader) but tests and callers
        may pass ``"4259"`` or ``"AEK0083"``; taking the leading alphabetic run
        as letters and the rest as digits handles all three. Splitting on a
        space alone would classify a digit-only read as letters.
        """
        compact = text.replace(" ", "")
        index = 0
        while index < len(compact) and compact[index].isalpha():
            index += 1
        return compact[:index], compact[index:]

    def observe_lambung(
        self,
        text: str,
        confidence: float,
        timestamp: float,
        weight: int = 1,
        box_frame: tuple[float, float, float, float] | None = None,
        sharpness: float = 0.0,
    ) -> None:
        """Record one cab-sign read; every character position votes separately.

        ``weight`` is the sign band area in pixels (closer frames carry larger
        bands), ``confidence`` the classifier probability, and ``sharpness`` the
        Laplacian variance of the sign crop. The vote weight multiplies all
        three: a blurry frame is down-weighted even when the classifier sounded
        sure, because haze makes confident mistakes common. ``box_frame`` is the
        sign rectangle in frame coordinates and is kept even for reads that are
        later rejected, so the evidence crop can point at the frame that was
        actually measured.
        """
        vote = max(1, int(weight * max(0.0, confidence) * (1.0 + min(sharpness, 200.0) / 200.0)))
        self.lambung_reads.append((text, timestamp))
        letter_part, digit_part = self._split_read(text)
        self._lambung_read_data.append(
            (letter_part, digit_part, vote, confidence, timestamp, box_frame, sharpness)
        )
        if box_frame is not None:
            self.lambung_box = box_frame
            self.lambung_box_ts = timestamp
        self._lambung_reads += 1
        if self._lambung_best is None or confidence > self._lambung_best[1]:
            self._lambung_best = (text, confidence, timestamp)

    @staticmethod
    def _digit_mode(reads: list[tuple], minimum_readers: int) -> tuple[str, list[tuple]] | None:
        """Digit string with at least ``minimum_readers`` frames, and its frames.

        Among the strings that qualify, the one whose best read is most
        confident wins, with total vote weight and reader count as tiebreaks.
        Confidence leads because haze produces *many* plausible wrong readings:
        measured on km18 track 17, ``AEK 0150`` appeared three times at 0.9998
        while ``AEK 1504`` appeared four times at 0.98 — reader count alone
        picked the wrong one.
        """
        readers: Counter = Counter()
        weight: Counter = Counter()
        best_conf: dict[str, float] = {}
        for read in reads:
            digits = read[1]
            if not digits:
                continue
            readers[digits] += 1
            weight[digits] += read[2]
            best_conf[digits] = max(best_conf.get(digits, 0.0), read[3])
        qualified = [digits for digits in readers if readers[digits] >= minimum_readers]
        if not qualified:
            return None
        digits = max(
            qualified,
            key=lambda key: (best_conf[key], weight[key], readers[key]),
        )
        return digits, [read for read in reads if read[1] == digits]

    def _evidence_read(self, letters: str, digits: str) -> tuple:
        """Read whose frame best matches the resolved text (for bbox/timestamp).

        The CSV must point at a frame that was actually measured, so pick the
        read sharing the most characters with the final text; ties fall back to
        the highest vote weight.
        """
        def score(read: tuple) -> tuple[int, int]:
            matches = sum(1 for a, b in zip(read[0], letters) if a == b)
            matches += sum(1 for a, b in zip(read[1], digits) if a == b)
            return matches, read[2]

        return max(self._lambung_read_data, key=score)

    def read_digest(self, limit: int = 50) -> list[dict]:
        """Top reads by vote weight, for run diagnostics.

        Enough to see *why* a plate was rejected (which characters repeated,
        how sharp each frame was) without dumping every OCR frame.
        """
        ranked = sorted(self._lambung_read_data, key=lambda read: read[2], reverse=True)[:limit]
        return [
            {
                "text": f"{read[0]} {read[1]}".strip(),
                "letters": read[0],
                "digits": read[1],
                "vote": read[2],
                "confidence": round(read[3], 4),
                "timestamp_s": round(read[4], 3),
                "sharpness": round(read[6], 1),
            }
            for read in ranked
        ]

    def _best_read(self) -> tuple:
        """Highest-weight read, used as evidence even when the text is rejected."""
        return max(self._lambung_read_data, key=lambda read: read[2])

    def _remember_box(self, read: tuple) -> None:
        """Pin the evidence box/timestamp to the chosen read's frame."""
        if read[5] is not None:
            self.lambung_box = read[5]
            self.lambung_box_ts = read[4]

    def lambung_outcome(
        self,
        *,
        single_read_prob: float | None = None,
        full_digits: int = 4,
        min_agreement: int = 2,
        min_sharpness: float = 0.0,
        trusted_min_confidence: float = 0.90,
        trusted_min_sharpness: float = 1000.0,
    ) -> PlateOutcome:
        """Resolve the track's reads into a plate outcome, preferring precision.

        The plate is chosen from whole reads, never assembled from parts, so the
        reported text is always something one frame actually saw. The steps, in
        order:

        1. Drop tracks whose every read is a smear (``min_sharpness``).
        2. Take the digit count from the **most confident** read. Using the
           most frequent count let four junk three-digit reads outvote the one
           confident four-digit read (measured: km18 track 11 read ``BBS 6019``
           once at 0.997, plus six junk reads, and produced nothing).
        3. Take the digit string with the **most readers**, weight as tiebreak.
        4. Take the letters from the most confident read that carries those
           digits, so a single differing glyph does not discard a plate whose
           digits are certain (measured: km18 track 8, ``BBS 6061`` twice plus
           ``IBS 6061``, previously discarded entirely).
        5. When no digits reach agreement, fall back to the single highest-vote
           read only if it is both confident (``trusted_min_confidence``) and
           sharp (``trusted_min_sharpness``).

        Rejections still pin the evidence box and timestamp so the audit crop
        shows the frame that was measured rather than the crossing frame.
        """
        reads = self._lambung_read_data
        if not reads:
            return PlateOutcome(None, None, None, REASON_NO_READ)

        if max(read[6] for read in reads) < min_sharpness:
            self._remember_box(self._best_read())
            return PlateOutcome(None, None, None, REASON_BLURRY)

        if self._lambung_reads < min_agreement:
            single = self._single_read_result(single_read_prob, full_digits)
            if single is not None:
                return PlateOutcome(single[0], single[1], single[2], REASON_READ)
            self._remember_box(self._best_read())
            reason = REASON_SINGLE_REJECTED if self._lambung_reads == 1 else REASON_LOW_AGREEMENT
            return PlateOutcome(None, None, None, reason)

        # Prefer the most confident digit length, then the number of supporting frames.
        by_length: dict[int, tuple[float, int]] = {}
        for read in reads:
            length = len(read[1])
            best_conf, best_readers = by_length.get(length, (0.0, 0))
            by_length[length] = (max(best_conf, read[3]), best_readers + 1)
        digit_length = max(by_length, key=lambda length: by_length[length])
        if digit_length < 3:
            self._remember_box(self._best_read())
            return PlateOutcome(None, None, None, REASON_NO_AGREEMENT)

        digit_pool = [read for read in reads if len(read[1]) == digit_length]
        resolved = self._digit_mode(digit_pool, min_agreement)
        if resolved is not None:
            digits, matching = resolved
        else:
            # Without agreement, require both high confidence and sharpness.
            trusted = max(reads, key=lambda read: (read[2], read[3]))
            if (
                trusted[3] < trusted_min_confidence
                or trusted[6] < trusted_min_sharpness
                or len(trusted[1]) != digit_length
            ):
                self._remember_box(self._best_read())
                return PlateOutcome(None, None, None, REASON_NO_AGREEMENT)
            digits = trusted[1]
            matching = [trusted]

        letters = ""
        letter_candidates = [read for read in matching if read[0]]
        if letter_candidates:
            letters = max(letter_candidates, key=lambda read: (read[3], read[2]))[0]

        text = f"{letters} {digits}".strip()
        # Never assemble a plate that did not occur in a single frame.
        if not any(f"{read[0]} {read[1]}".strip() == text for read in reads):
            self._remember_box(self._best_read())
            return PlateOutcome(None, None, None, REASON_NO_AGREEMENT)

        best = self._evidence_read(letters, digits)
        self._remember_box(best)
        # Confidence, timestamp and box must come from the same frame.
        return PlateOutcome(text, best[3], best[4], REASON_READ)

    def lambung_result(
        self, single_read_prob: float | None = None, full_digits: int = 4
    ) -> tuple[str, float, float] | None:
        """Voted text as ``(text, confidence, timestamp)`` or ``None``.

        Thin wrapper over :meth:`lambung_outcome` for callers that only need the
        text (render, tests).
        """
        outcome = self.lambung_outcome(
            single_read_prob=single_read_prob, full_digits=full_digits
        )
        if outcome.text is None:
            return None
        return outcome.text, outcome.confidence or 0.0, outcome.timestamp or 0.0

    def _single_read_result(self, threshold: float | None, full_digits: int = 4) -> tuple[str, float, float] | None:
        """Accept a lone read only when confident and a full-length digit run.

        Voting needs two reads; a vehicle seen for only a few close frames may
        produce just one. With a high probability threshold that lone read is
        still usable evidence, so it is accepted (opt-in via the threshold).
        """
        if threshold is None:
            return None
        letters, digits, _vote, confidence, timestamp, box, _sharpness = self._lambung_read_data[0]
        if len(digits) != full_digits or confidence < threshold:
            return None
        self.lambung_box = box
        self.lambung_box_ts = timestamp
        return f"{letters} {digits}".strip(), confidence, timestamp

    @property
    def direction(self) -> str:
        """Infer vertical direction from first and latest centroid, allowing jitter."""
        if self.first_centroid is None or self.last_centroid is None:
            return "Unknown"
        delta = self.last_centroid[1] - self.first_centroid[1]
        if abs(delta) < 3:
            return "Unknown"
        return "South" if delta > 0 else "North"

    @property
    def speed_status(self) -> str:
        """Describe crossing validity without calculating a numerical speed.

        The corridor is calibrated for the Southbound flow, so a negative
        interval (finish crossed before start) is a genuine order anomaly.
        """
        if self.start_timestamp is None:
            return "not_crossed_start"
        if self.end_timestamp is None:
            return "not_crossed_finish"
        elapsed = self.end_timestamp - self.start_timestamp
        if not math.isfinite(elapsed) or elapsed < 0:
            return "wrong_order"
        if elapsed == 0:
            return "invalid_elapsed_time"
        return "valid"
