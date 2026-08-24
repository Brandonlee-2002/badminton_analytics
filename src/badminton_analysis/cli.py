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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_video = args.output_video or config["output_video"]
    output_metrics = args.output_metrics or config["output_metrics"]
    homography = load_homography(config.get("court_homography"))

    tracks, fps = analyze_video(
        video_path=args.video,
        output_path=output_video,
        model_path=config["model"],
        tracker_config=config["tracker"],
        confidence=config["confidence"],
        image_size=config["image_size"],
        classes=config.get("classes"),
        max_track_history=config["max_track_history"],
        homography=homography,
    )
    metrics = {
        "video": args.video,
        "model": config["model"],
        "calibrated": homography is not None,
        "players": summarize_tracks(tracks.values(), fps, calibrated=homography is not None),
    }
    metrics_path = Path(output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("Analysis complete.")
    print(f"Annotated video: {output_video}")
    print(f"Metrics: {output_metrics}")


if __name__ == "__main__":
    main()
