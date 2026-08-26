from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from .calibration import project_point
from .types import PlayerTrack, TrackPoint


def court_box_indices(boxes: np.ndarray, polygon: np.ndarray) -> list[int]:
    """Return indices of boxes whose bottom-center point is on the court."""
    keep = []

    for index, box in enumerate(boxes):
        x1, _, x2, y2 = box
        foot = (float((x1 + x2) / 2), float(y2))

        if cv2.pointPolygonTest(polygon, foot, False) >= 0:
            keep.append(index)

    return keep


class ProgressBar:
    def __init__(self, total: int | None, description: str) -> None:
        self.total = total
        self.description = description
        self.completed = 0

    def update(self, amount: int = 1) -> None:
        self.completed += amount
        if self.total:
            percent = min(self.completed / self.total, 1.0)
            bar_width = 28
            filled = int(percent * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            message = f"\r{self.description}: [{bar}] {percent:6.1%} ({self.completed}/{self.total} frames)"
        else:
            message = f"\r{self.description}: {self.completed} frames"
        sys.stderr.write(message)
        sys.stderr.flush()

    def close(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()


def analyze_video(
    video_path: str,
    output_path: str,
    model_path: str = "yolo26s.pt",
    tracker_config: str = "bytetrack.yaml",
    confidence: float = 0.25,
    image_size: int = 960,
    classes: list[int] | None = None,
    court_polygon: list[list[int]] | None = None,
    max_track_history: int = 45,
    homography=None,
) -> tuple[dict[int, PlayerTrack], float]:
    source = cv2.VideoCapture(video_path)
    if not source.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = source.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        source.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    model = YOLO(model_path)
    polygon = None

    if court_polygon:
        polygon = np.asarray(court_polygon, dtype=np.int32)

        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError("court_polygon must contain at least three [x, y] points")

        outside_frame = (
            (polygon[:, 0] < 0)
            | (polygon[:, 0] >= width)
            | (polygon[:, 1] < 0)
            | (polygon[:, 1] >= height)
        )
        if outside_frame.any():
            raise ValueError(
                f"court_polygon points must be inside the {width}x{height} video frame"
            )

    tracks: dict[int, PlayerTrack] = defaultdict(lambda: PlayerTrack(track_id=-1))
    frame_number = 0
    progress = ProgressBar(total_frames, "Analyzing video")

    try:
        while True:
            success, frame = source.read()
            if not success:
                break

            result = model.track(
                frame,
                persist=True,
                tracker=tracker_config,
                conf=confidence,
                imgsz=image_size,
                classes=classes,
                verbose=False,
            )[0]

            if polygon is not None and result.boxes is not None:
                detected_boxes = result.boxes.xyxy.cpu().numpy()
                keep = court_box_indices(detected_boxes, polygon)
                result.boxes = result.boxes[keep]

            annotated = result.plot()

            if polygon is not None:
                cv2.polylines(
                    annotated,
                    [polygon],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=2,
                )

            if result.boxes is not None and result.boxes.is_track:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                timestamp = frame_number / fps

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    x = float((x1 + x2) / 2)
                    y = float(y2)
                    projected = project_point(x, y, homography)
                    track = tracks[track_id]
                    track.track_id = track_id
                    track.add(
                        TrackPoint(
                            frame=frame_number,
                            timestamp=timestamp,
                            x=x,
                            y=y,
                            court_x=projected[0] if projected else None,
                            court_y=projected[1] if projected else None,
                        ),
                        max_track_history,
                    )

            writer.write(annotated)
            frame_number += 1
            progress.update(1)
    finally:
        progress.close()
        source.release()
        writer.release()
        cv2.destroyAllWindows()

    return dict(tracks), fps
