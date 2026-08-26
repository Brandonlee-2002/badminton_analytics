import numpy as np

from badminton_analysis.tracking import court_box_indices


def test_court_box_indices_uses_bottom_center_of_box():
    polygon = np.array(
        [
            [10, 10],
            [90, 10],
            [90, 90],
            [10, 90],
        ],
        dtype=np.int32,
    )
    boxes = np.array(
        [
            [20, 20, 40, 60],  # Foot point (30, 60) is inside.
            [0, 20, 10, 60],  # Foot point (5, 60) is outside.
            [80, 20, 100, 90],  # Foot point (90, 90) is on the boundary.
        ],
        dtype=np.float32,
    )

    assert court_box_indices(boxes, polygon) == [0, 2]
