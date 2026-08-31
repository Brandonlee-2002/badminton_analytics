from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from .calibration import project_point
from .equipment import (
    ShuttleTrajectory,
    YoloObjectDetector,
    associate_rackets,
    draw_observations,
)
from .scene import SceneGate, append_scene_frame
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
    scene_filter: dict | None = None,
    racket_detector: dict | None = None,
    shuttle_detector: dict | None = None,
    analysis_log: dict | None = None,
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

    scene_gate = SceneGate(polygon=polygon, fps=fps, **(scene_filter or {}))
    racket_model = YoloObjectDetector("racket", racket_detector)
    shuttle_model = YoloObjectDetector("shuttle", shuttle_detector)
    shuttle_trajectory = ShuttleTrajectory(
        alpha=float((shuttle_detector or {}).get("alpha", 0.75)),
        beta=float((shuttle_detector or {}).get("beta", 0.20)),
        max_gap=int((shuttle_detector or {}).get("max_gap", 4)),
    )
    event_data = analysis_log if analysis_log is not None else {}
    scene_segments: list[dict] = []
    racket_events: list[dict] = []
    shuttle_events: list[dict] = []
    tracks: dict[int, PlayerTrack] = defaultdict(lambda: PlayerTrack(track_id=-1))
    frame_number = 0
    progress = ProgressBar(total_frames, "Analyzing video")

    try:
        while True:
            success, frame = source.read()
            if not success:
                break

            timestamp = frame_number / fps
            decision = scene_gate.update(frame, frame_number)
            append_scene_frame(scene_segments, decision, frame_number)

            if decision.cut:
                model.predictor = None
                shuttle_trajectory.reset()

            if not decision.gameplay:
                annotated = frame.copy()
                cv2.putText(
                    annotated,
                    f"SKIPPED: {decision.reason}",
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                frame_number += 1
                progress.update(1)
                continue

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

            player_boxes: list[tuple[int, np.ndarray]] = []
            if result.boxes is not None and result.boxes.is_track:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    global_track_id = track_id + decision.scene_id * 100000
                    player_boxes.append((global_track_id, box))
                    x1, y1, x2, y2 = box
                    x = float((x1 + x2) / 2)
                    y = float(y2)
                    projected = project_point(x, y, homography)
                    track = tracks[global_track_id]
                    track.track_id = global_track_id
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

            rackets = racket_model.detect(
                frame, frame_number, timestamp, decision.scene_id, polygon
            )
            associate_rackets(rackets, player_boxes)
            racket_events.extend(item.to_dict() for item in rackets)

            shuttle_candidates = shuttle_model.detect(
                frame, frame_number, timestamp, decision.scene_id, polygon
            )
            shuttle = shuttle_trajectory.update(
                shuttle_candidates, frame_number, timestamp, decision.scene_id
            )
            equipment = list(rackets)
            if shuttle is not None:
                equipment.append(shuttle)
                shuttle_events.append(shuttle.to_dict())
            draw_observations(annotated, equipment)

            cv2.putText(
                annotated,
                f"scene {decision.scene_id} | gameplay | court {decision.court_score:.0%}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)
            frame_number += 1
            progress.update(1)
    finally:
        progress.close()
        source.release()
        writer.release()
        cv2.destroyAllWindows()

    event_data.update(
        {
            "scenes": scene_segments,
            "rackets": racket_events,
            "shuttle": shuttle_events,
            "processed_frames": frame_number,
            "gameplay_frames": sum(
                item["end_frame"] - item["start_frame"] + 1
                for item in scene_segments
                if item["gameplay"]
            ),
        }
    )
    return dict(tracks), fps
