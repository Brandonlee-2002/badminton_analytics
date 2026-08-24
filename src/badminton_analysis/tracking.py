from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm import tqdm

from .calibration import project_point
from .types import PlayerTrack, TrackPoint


def analyze_video(
    video_path: str,
    output_path: str,
    model_path: str = "yolo26s.pt",
    tracker_config: str = "bytetrack.yaml",
    confidence: float = 0.25,
    image_size: int = 960,
    classes: list[int] | None = None,
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
    tracks: dict[int, PlayerTrack] = defaultdict(lambda: PlayerTrack(track_id=-1))
    frame_number = 0
    progress = tqdm(
        total=total_frames,
        desc="Analyzing video",
        unit="frame",
        dynamic_ncols=True,
    )

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

            annotated = result.plot()
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
