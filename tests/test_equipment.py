import numpy as np

from badminton_analysis.equipment import (
    ObjectObservation,
    ShuttleTrajectory,
    associate_rackets,
)


def observation(x: float, y: float, confidence: float = 0.9) -> ObjectObservation:
    return ObjectObservation(
        frame=0,
        timestamp=0.0,
        label="shuttle",
        confidence=confidence,
        bbox=(x - 1, y - 1, x + 1, y + 1),
        x=x,
        y=y,
        scene_id=0,
    )


def test_racket_is_associated_with_nearest_player():
    racket = observation(82, 50)
    racket.label = "racket"
    players = [
        (1, np.array([0, 0, 40, 100], dtype=np.float32)),
        (2, np.array([60, 0, 100, 100], dtype=np.float32)),
    ]

    associate_rackets([racket], players)

    assert racket.player_track_id == 2


def test_shuttle_trajectory_predicts_short_gap_and_resets_on_scene_change():
    tracker = ShuttleTrajectory(alpha=1.0, beta=1.0, max_gap=2)

    first = tracker.update([observation(10, 10)], 0, 0.0, scene_id=0)
    second_detection = observation(12, 10)
    second_detection.frame = 1
    second = tracker.update([second_detection], 1, 1 / 30, scene_id=0)
    predicted = tracker.update([], 2, 2 / 30, scene_id=0)

    assert first is not None
    assert second is not None
    assert predicted is not None
    assert not predicted.observed
    assert predicted.x == 14.0

    assert tracker.update([], 3, 3 / 30, scene_id=1) is None
