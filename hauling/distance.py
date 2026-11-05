"""Cross-track time headway at the final area and the unsafe-distance flag."""

import math
from dataclasses import dataclass

# Headways at or below this are treated as tracking artifacts (an id switch or
# two boxes on one vehicle), not as two real vehicles. Not configurable: the
# task treats it as an artifact floor, not a tuning knob.
MIN_HEADWAY_S = 0.9


@dataclass(frozen=True)
class HeadwayOutcome:
    """Headway of one final crossing against the previous crossing.

    ``status`` is one of ``no_lead``, ``normal``, ``unsafe`` or ``artifact``.
    ``lead_track_id``/``headway_s`` are ``None`` for the first vehicle.
    """

    lead_track_id: int | None
    headway_s: float | None
    status: str
    is_unsafe: bool


class FinalCrossingLedger:
    """Remember the last final crossing so the next one can be measured.

    Only one crossing is retained: timestamps are monotonic and each track
    crosses the final band once. An artifact crossing does **not** replace the
    lead, so the next vehicle is still compared against the last plausible one
    instead of an id-switch duplicate.
    """

    def __init__(self, limit_s: float, min_headway_s: float = MIN_HEADWAY_S) -> None:
        if not math.isfinite(limit_s) or limit_s <= 0:
            raise ValueError("Headway limit must be finite and positive")
        self.limit_s = limit_s
        self.min_headway_s = min_headway_s
        self._last: tuple[int, float] | None = None

    def register(self, track_id: int, timestamp: float) -> HeadwayOutcome:
        """Record a final crossing and measure its headway against the lead."""
        if self._last is None:
            self._last = (track_id, timestamp)
            return HeadwayOutcome(None, None, "no_lead", False)
        lead_track_id, lead_timestamp = self._last
        headway = timestamp - lead_timestamp
        if not math.isfinite(headway) or headway <= self.min_headway_s:
            # Keep the previous lead: a duplicate id-switch must not drag the
            # reference timestamp forward and mask the next real vehicle.
            return HeadwayOutcome(lead_track_id, headway, "artifact", False)
        self._last = (track_id, timestamp)
        # The limit is inclusive; isclose only forgives floating-point noise at
        # the boundary (3.0000000000001 s is "at the limit", not "over").
        unsafe = headway < self.limit_s or math.isclose(headway, self.limit_s, rel_tol=1e-12, abs_tol=1e-12)
        return HeadwayOutcome(lead_track_id, headway, "unsafe" if unsafe else "normal", unsafe)
