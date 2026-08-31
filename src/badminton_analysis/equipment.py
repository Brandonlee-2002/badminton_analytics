from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class ObjectObservation:
    frame: int
    timestamp: float
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    x: float
    y: float
    scene_id: int
    player_track_id: int | None = None
    observed: bool = True

    def to_dict(self) -> dict:
        result = asdict(self)
        result["bbox"] = [round(value, 2) for value in self.bbox]
        result["x"] = round(self.x, 2)
        result["y"] = round(self.y, 2)
        result["confidence"] = round(self.confidence, 4)
        return result


class YoloObjectDetector:
    """Optional detector kept separate from the player tracking model."""

    def __init__(self, label: str, config: dict | None) -> None:
        config = config or {}
        self.label = label
        self.enabled = bool(config.get("enabled", False))
        self.stride = max(1, int(config.get("stride", 1)))
        self.confidence = float(config.get("confidence", 0.20))
        self.image_size = int(config.get("image_size", 1280))
        self.classes = config.get("classes")
        self.model = None

        if self.enabled:
            model_path = config.get("model")
            if not model_path:
                raise ValueError(f"{label}_detector.model is required when enabled")
            if Path(model_path).suffix == ".pt" and "/" in str(model_path):
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"Missing {label} model: {model_path}")
            self.model = YOLO(model_path)

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        scene_id: int,
        polygon: np.ndarray | None = None,
    ) -> list[ObjectObservation]:
        if not self.enabled or frame_number % self.stride:
            return []

        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=self.classes,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        observations = []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().tolist()

        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = map(float, box)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            if polygon is not None:
                if cv2.pointPolygonTest(polygon, (x, y), False) < 0:
                    continue
            observations.append(
                ObjectObservation(
                    frame=frame_number,
                    timestamp=timestamp,
                    label=self.label,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                    x=x,
                    y=y,
                    scene_id=scene_id,
                )
            )

        return observations


def associate_rackets(
    rackets: list[ObjectObservation],
    players: list[tuple[int, np.ndarray]],
) -> None:
    """Associate each racket with the closest on-court player box."""
    for racket in rackets:
        candidates = []
        for track_id, box in players:
            x1, y1, x2, y2 = map(float, box)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            distance = float(np.hypot(racket.x - center_x, racket.y - center_y))
            inside_expanded = (
                x1 - (x2 - x1) * 0.35 <= racket.x <= x2 + (x2 - x1) * 0.35
                and y1 - (y2 - y1) * 0.20 <= racket.y <= y2 + (y2 - y1) * 0.20
            )
            candidates.append((not inside_expanded, distance, track_id))
        if candidates:
            racket.player_track_id = min(candidates)[2]


class ShuttleTrajectory:
    """Alpha-beta temporal smoothing for custom shuttle detections."""

    def __init__(self, alpha: float = 0.75, beta: float = 0.20, max_gap: int = 4):
        self.alpha = alpha
        self.beta = beta
        self.max_gap = max_gap
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_frame: int | None = None
        self.gap = 0
        self.scene_id: int | None = None

    def reset(self) -> None:
        self.position = None
        self.velocity[:] = 0
        self.last_frame = None
        self.gap = 0
        self.scene_id = None

    def update(
        self,
        detections: list[ObjectObservation],
        frame_number: int,
        timestamp: float,
        scene_id: int,
    ) -> ObjectObservation | None:
        if self.scene_id is not None and scene_id != self.scene_id:
            self.reset()
        self.scene_id = scene_id

        if self.position is None:
            if not detections:
                return None
            selected = max(detections, key=lambda item: item.confidence)
            self.position = np.array([selected.x, selected.y], dtype=np.float32)
            self.last_frame = frame_number
            return selected

        delta = max(1, frame_number - (self.last_frame or frame_number - 1))
        predicted = self.position + self.velocity * delta

        if detections:
            selected = min(
                detections,
                key=lambda item: float(np.hypot(item.x - predicted[0], item.y - predicted[1])),
            )
            measured = np.array([selected.x, selected.y], dtype=np.float32)
            residual = measured - predicted
            self.position = predicted + self.alpha * residual
            self.velocity = self.velocity + self.beta * residual / delta
            self.gap = 0
            selected.x = float(self.position[0])
            selected.y = float(self.position[1])
            self.last_frame = frame_number
            return selected

        self.gap += delta
        if self.gap > self.max_gap:
            self.reset()
            return None

        self.position = predicted
        self.last_frame = frame_number
        x, y = map(float, self.position)
        return ObjectObservation(
            frame=frame_number,
            timestamp=timestamp,
            label="shuttle",
            confidence=0.0,
            bbox=(x, y, x, y),
            x=x,
            y=y,
            scene_id=scene_id,
            observed=False,
        )


def draw_observations(frame: np.ndarray, observations: list[ObjectObservation]) -> None:
    colors = {"racket": (0, 165, 255), "shuttle": (255, 0, 255)}
    for observation in observations:
        color = colors.get(observation.label, (255, 255, 0))
        if observation.label == "shuttle":
            cv2.circle(frame, (round(observation.x), round(observation.y)), 5, color, 2)
        else:
            x1, y1, x2, y2 = map(round, observation.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                observation.label,
                (x1, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
