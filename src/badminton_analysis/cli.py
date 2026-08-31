import argparse
import json
from pathlib import Path

import yaml

from .calibration import load_homography
from .metrics import summarize_tracks
from .tracking import analyze_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze badminton video with YOLO26 tracking")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--config", default="config/default.yaml", help="YAML configuration path")
    parser.add_argument("--output-video", help="Override annotated video output path")
    parser.add_argument("--output-metrics", help="Override JSON metrics output path")
    parser.add_argument("--output-events", help="Override scene/equipment events output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_video = args.output_video or config["output_video"]
    output_metrics = args.output_metrics or config["output_metrics"]
    output_events = args.output_events or config.get("output_events", "data/processed/events.json")
    homography = load_homography(config.get("court_homography"))
    analysis_log: dict = {}

    tracks, fps = analyze_video(
        video_path=args.video,
        output_path=output_video,
        model_path=config["model"],
        tracker_config=config["tracker"],
        confidence=config["confidence"],
        image_size=config["image_size"],
        classes=config.get("classes"),
        court_polygon=config.get("court_polygon"),
        max_track_history=config["max_track_history"],
        scene_filter=config.get("scene_filter"),
        racket_detector=config.get("racket_detector"),
        shuttle_detector=config.get("shuttle_detector"),
        analysis_log=analysis_log,
        homography=homography,
    )
    metrics = {
        "video": args.video,
        "model": config["model"],
        "calibrated": homography is not None,
        "players": summarize_tracks(tracks.values(), fps, calibrated=homography is not None),
        "summary": {
            "scene_segments": len(analysis_log.get("scenes", [])),
            "gameplay_frames": analysis_log.get("gameplay_frames", 0),
            "processed_frames": analysis_log.get("processed_frames", 0),
            "racket_observations": len(analysis_log.get("rackets", [])),
            "shuttle_observations": len(analysis_log.get("shuttle", [])),
        },
    }
    metrics_path = Path(output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    events_path = Path(output_events)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(json.dumps(analysis_log, indent=2))
    print(json.dumps(metrics, indent=2))
    print("Analysis complete.")
    print(f"Annotated video: {output_video}")
    print(f"Metrics: {output_metrics}")


if __name__ == "__main__":
    main()
