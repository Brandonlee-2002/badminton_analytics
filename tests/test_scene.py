import cv2
import numpy as np

from badminton_analysis.scene import SceneGate, append_scene_frame


def solid_hsv(hue: int) -> np.ndarray:
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = 200
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_scene_gate_accepts_court_color_and_rejects_other_views():
    polygon = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.int32)
    gate = SceneGate(
        polygon=polygon,
        fps=30,
        enabled=True,
        cut_threshold=2.0,
        min_court_fraction=0.5,
        court_hsv_lower=[30, 100, 100],
        court_hsv_upper=[90, 255, 255],
        warmup_frames=0,
    )

    assert gate.update(solid_hsv(60), 0).gameplay
    decision = gate.update(solid_hsv(5), 1)
    assert not decision.gameplay
    assert decision.reason == "court_not_visible"


def test_scene_gate_detects_cut_and_compacts_segments():
    gate = SceneGate(
        polygon=None,
        fps=30,
        enabled=True,
        cut_threshold=0.2,
        warmup_frames=1,
    )
    segments = []

    first = gate.update(solid_hsv(0), 0)
    append_scene_frame(segments, first, 0)
    second = gate.update(solid_hsv(0), 1)
    append_scene_frame(segments, second, 1)
    cut = gate.update(solid_hsv(60), 2)
    append_scene_frame(segments, cut, 2)

    assert cut.cut
    assert cut.scene_id == 1
    assert not cut.gameplay
    assert cut.reason == "camera_cut_warmup"
    assert len(segments) == 2
    assert segments[0]["start_frame"] == 0
    assert segments[0]["end_frame"] == 1
