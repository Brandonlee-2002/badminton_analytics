from pathlib import Path

import cv2
import numpy as np
import yaml


def load_homography(path: str | None) -> np.ndarray | None:
    if not path:
        return None

    values = yaml.safe_load(Path(path).read_text())
    matrix = values.get("homography") if isinstance(values, dict) else values
    homography = np.asarray(matrix, dtype=np.float32)
    if homography.shape != (3, 3):
        raise ValueError("Homography must be a 3x3 matrix")
    return homography


def project_point(x: float, y: float, homography: np.ndarray | None) -> tuple[float, float] | None:
    if homography is None:
        return None

    source = np.array([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(source, homography)[0, 0]
    return float(projected[0]), float(projected[1])
