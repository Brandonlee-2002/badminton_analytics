from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneDecision:
    scene_id: int
    cut: bool
    gameplay: bool
    reason: str
    cut_score: float
    court_score: float


class SceneGate:
    """Detect camera cuts and accept frames matching a calibrated court view."""

    def __init__(
        self,
        polygon: np.ndarray | None,
        fps: float,
        enabled: bool = False,
        cut_threshold: float = 0.55,
        min_court_fraction: float = 0.20,
        court_hsv_lower: list[int] | tuple[int, int, int] = (30, 25, 25),
        court_hsv_upper: list[int] | tuple[int, int, int] = (100, 255, 255),
        warmup_frames: int = 2,
        manual_skip_ranges: list[list[float]] | None = None,
    ) -> None:
        self.polygon = polygon
        self.fps = fps
        self.enabled = enabled
        self.cut_threshold = cut_threshold
        self.min_court_fraction = min_court_fraction
        self.lower = np.asarray(court_hsv_lower, dtype=np.uint8)
        self.upper = np.asarray(court_hsv_upper, dtype=np.uint8)
        self.warmup_frames = max(0, warmup_frames)
        self.manual_skip_ranges = manual_skip_ranges or []
        self.scene_id = 0
        self.previous_histogram: np.ndarray | None = None
        self.frames_since_cut = self.warmup_frames
        self._court_mask: np.ndarray | None = None
        self._mask_shape: tuple[int, int] | None = None

    @staticmethod
    def _histogram(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        return cv2.normalize(histogram, histogram).flatten()

    def _court_fraction(self, frame: np.ndarray) -> float:
        if self.polygon is None:
            return 1.0

        shape = frame.shape[:2]
        if self._court_mask is None or self._mask_shape != shape:
            self._court_mask = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(self._court_mask, [self.polygon], 255)
            self._mask_shape = shape

        polygon_pixels = cv2.countNonZero(self._court_mask)
        if polygon_pixels == 0:
            return 0.0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        court_color = cv2.inRange(hsv, self.lower, self.upper)
        court_color = cv2.bitwise_and(court_color, self._court_mask)
        return cv2.countNonZero(court_color) / polygon_pixels

    def _manually_skipped(self, frame_number: int) -> bool:
        timestamp = frame_number / self.fps
        return any(start <= timestamp <= end for start, end in self.manual_skip_ranges)

    def update(self, frame: np.ndarray, frame_number: int) -> SceneDecision:
        histogram = self._histogram(frame)
        cut_score = 0.0
        cut = False

        if self.previous_histogram is not None:
            cut_score = float(
                cv2.compareHist(
                    self.previous_histogram,
                    histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
            cut = cut_score >= self.cut_threshold

        self.previous_histogram = histogram

        if cut:
            self.scene_id += 1
            self.frames_since_cut = 0
        else:
            self.frames_since_cut += 1

        court_score = self._court_fraction(frame)

        if not self.enabled:
            gameplay = True
            reason = "scene_filter_disabled"
        elif self._manually_skipped(frame_number):
            gameplay = False
            reason = "manual_skip"
        elif self.frames_since_cut < self.warmup_frames:
            gameplay = False
            reason = "camera_cut_warmup"
        elif court_score < self.min_court_fraction:
            gameplay = False
            reason = "court_not_visible"
        else:
            gameplay = True
            reason = "gameplay"

        return SceneDecision(
            scene_id=self.scene_id,
            cut=cut,
            gameplay=gameplay,
            reason=reason,
            cut_score=round(cut_score, 4),
            court_score=round(court_score, 4),
        )


def append_scene_frame(segments: list[dict], decision: SceneDecision, frame: int) -> None:
    """Compact consecutive decisions into scene segments."""
    key = (decision.scene_id, decision.gameplay, decision.reason)
    if segments:
        previous = segments[-1]
        previous_key = (
            previous["scene_id"],
            previous["gameplay"],
            previous["reason"],
        )
        if previous_key == key and previous["end_frame"] == frame - 1:
            previous["end_frame"] = frame
            previous["max_cut_score"] = max(
                previous["max_cut_score"], decision.cut_score
            )
            previous["max_court_score"] = max(
                previous["max_court_score"], decision.court_score
            )
            return

    segments.append(
        {
            "scene_id": decision.scene_id,
            "start_frame": frame,
            "end_frame": frame,
            "gameplay": decision.gameplay,
            "reason": decision.reason,
            "max_cut_score": decision.cut_score,
            "max_court_score": decision.court_score,
        }
    )
