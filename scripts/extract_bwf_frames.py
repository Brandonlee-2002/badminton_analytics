#!/usr/bin/env python3
"""Extract a diverse, time-separated frame set from a badminton broadcast."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Candidate:
    frame_number: int
    timestamp_seconds: float
    image: np.ndarray
    fingerprint: np.ndarray
    court_score: float
    category: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically extract diverse court and other broadcast frames for "
            "racket annotation."
        )
    )
    parser.add_argument("--video", required=True, help="Source BWF video")
    parser.add_argument(
        "--output",
        default="datasets/bwf_rackets",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--split",
        choices=("both", "train", "val"),
        default="both",
        help="Destination split: mixed train/val, training only, or validation only",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Project YAML containing court_polygon and scene_filter",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=400,
        help="Target number of extracted frames",
    )
    parser.add_argument(
        "--court-ratio",
        type=float,
        default=0.75,
        help="Target fraction of frames showing the calibrated court",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation fraction when --split both is used",
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=4.0,
        help="Number of candidate frames examined per requested output frame",
    )
    parser.add_argument(
        "--duplicate-distance",
        type=int,
        default=5,
        help="Reject dHash fingerprints with Hamming distance below this value",
    )
    parser.add_argument(
        "--split-block-seconds",
        type=float,
        default=30.0,
        help="Keep all selected frames in each time block in the same split",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not 0.0 <= args.court_ratio <= 1.0:
        raise ValueError("--court-ratio must be between 0 and 1")
    if not 0.0 <= args.val_ratio <= 1.0:
        raise ValueError("--val-ratio must be between 0 and 1 inclusive")
    if args.oversample < 1.0:
        raise ValueError("--oversample must be at least 1")
    if args.duplicate_distance < 0:
        raise ValueError("--duplicate-distance cannot be negative")
    if args.split_block_seconds <= 0:
        raise ValueError("--split-block-seconds must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")


def load_scene_config(
    config_path: Path,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, float]:
    config = yaml.safe_load(config_path.read_text()) or {}
    polygon_values = config.get("court_polygon")
    polygon = (
        np.asarray(polygon_values, dtype=np.int32)
        if polygon_values is not None
        else None
    )
    if polygon is not None and (polygon.ndim != 2 or polygon.shape[1] != 2):
        raise ValueError("court_polygon must contain [x, y] points")

    scene = config.get("scene_filter") or {}
    lower = np.asarray(scene.get("court_hsv_lower", [30, 25, 25]), dtype=np.uint8)
    upper = np.asarray(
        scene.get("court_hsv_upper", [100, 255, 255]), dtype=np.uint8
    )
    threshold = float(scene.get("min_court_fraction", 0.50))
    return polygon, lower, upper, threshold


def make_court_mask(
    frame_shape: tuple[int, ...], polygon: np.ndarray | None
) -> np.ndarray | None:
    if polygon is None:
        return None
    height, width = frame_shape[:2]
    if (
        np.any(polygon[:, 0] < 0)
        or np.any(polygon[:, 0] >= width)
        or np.any(polygon[:, 1] < 0)
        or np.any(polygon[:, 1] >= height)
    ):
        raise ValueError(
            "court_polygon extends outside the video frame. Calibrate the polygon "
            f"for this {width}x{height} video before extracting frames."
        )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    if cv2.countNonZero(mask) == 0:
        raise ValueError("court_polygon has zero area")
    return mask


def calculate_court_score(
    frame: np.ndarray,
    mask: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    if mask is None:
        return 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    court_color = cv2.inRange(hsv, lower, upper)
    matching = cv2.countNonZero(cv2.bitwise_and(court_color, mask))
    return matching / cv2.countNonZero(mask)


def difference_hash(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).reshape(-1)


def hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def read_candidates(
    video_path: Path,
    count: int,
    oversample: float,
    polygon: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
    court_threshold: float,
) -> tuple[list[Candidate], float, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video metadata is incomplete or invalid")

    candidate_count = min(frame_count, max(count, int(round(count * oversample))))
    positions = np.linspace(0, frame_count - 1, candidate_count, dtype=np.int64)
    positions = np.unique(positions)
    court_mask: np.ndarray | None = None
    candidates: list[Candidate] = []
    progress_step = max(1, len(positions) // 20)

    print(
        f"Video: {width}x{height}, {fps:.3f} FPS, "
        f"{frame_count} frames ({frame_count / fps:.1f}s)"
    )
    print(f"Examining {len(positions)} evenly spaced candidate frames...")

    for position_index, frame_number_value in enumerate(positions, start=1):
        frame_number = int(frame_number_value)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        if court_mask is None and polygon is not None:
            court_mask = make_court_mask(frame.shape, polygon)

        court_score = calculate_court_score(frame, court_mask, lower, upper)
        category = (
            "court_view" if polygon is not None and court_score >= court_threshold
            else "other_broadcast_view"
        )
        candidates.append(
            Candidate(
                frame_number=frame_number,
                timestamp_seconds=frame_number / fps,
                image=frame,
                fingerprint=difference_hash(frame),
                court_score=court_score,
                category=category,
            )
        )
        if position_index % progress_step == 0 or position_index == len(positions):
            percent = 100.0 * position_index / len(positions)
            print(
                f"Candidate scan: {position_index}/{len(positions)} "
                f"({percent:.0f}%)",
                flush=True,
            )

    capture.release()
    return candidates, fps, width, height


def select_diverse(
    candidates: list[Candidate],
    limit: int,
    duplicate_distance: int,
    existing: list[Candidate] | None = None,
) -> list[Candidate]:
    if limit <= 0 or not candidates:
        return []

    selected = list(existing or [])
    new_selection: list[Candidate] = []
    group_count = min(limit, len(candidates))
    groups = np.array_split(np.asarray(candidates, dtype=object), group_count)
    deferred: list[Candidate] = []

    for group in groups:
        chosen: Candidate | None = None
        for value in group:
            candidate = value
            if all(
                hamming_distance(candidate.fingerprint, item.fingerprint)
                >= duplicate_distance
                for item in selected
            ):
                chosen = candidate
                break
            deferred.append(candidate)
        if chosen is not None:
            selected.append(chosen)
            new_selection.append(chosen)
        if len(new_selection) >= limit:
            break

    if len(new_selection) < limit:
        chosen_frames = {item.frame_number for item in selected}
        remaining = deferred + [
            item for item in candidates if item.frame_number not in chosen_frames
        ]
        for candidate in remaining:
            if candidate.frame_number in chosen_frames:
                continue
            if all(
                hamming_distance(candidate.fingerprint, item.fingerprint)
                >= duplicate_distance
                for item in selected
            ):
                selected.append(candidate)
                new_selection.append(candidate)
                chosen_frames.add(candidate.frame_number)
            if len(new_selection) >= limit:
                break

    return new_selection


def choose_frames(
    candidates: list[Candidate],
    count: int,
    court_ratio: float,
    duplicate_distance: int,
) -> list[Candidate]:
    court = [item for item in candidates if item.category == "court_view"]
    other = [item for item in candidates if item.category != "court_view"]
    court_target = round(count * court_ratio)
    other_target = count - court_target

    selected = select_diverse(court, court_target, duplicate_distance)
    selected.extend(
        select_diverse(other, other_target, duplicate_distance, existing=selected)
    )

    if len(selected) < count:
        chosen_frames = {item.frame_number for item in selected}
        remaining = [
            item for item in candidates if item.frame_number not in chosen_frames
        ]
        selected.extend(
            select_diverse(
                remaining,
                count - len(selected),
                duplicate_distance,
                existing=selected,
            )
        )

    return sorted(selected, key=lambda item: item.frame_number)


def validation_blocks(
    selected: list[Candidate],
    val_ratio: float,
    block_seconds: float,
    seed: int,
) -> set[int]:
    if val_ratio == 0.0 or not selected:
        return set()
    block_counts: dict[int, int] = {}
    for candidate in selected:
        block = int(candidate.timestamp_seconds // block_seconds)
        block_counts[block] = block_counts.get(block, 0) + 1

    blocks = list(block_counts)
    random.Random(seed).shuffle(blocks)
    target = max(1, round(len(selected) * val_ratio))
    result: set[int] = set()
    current = 0
    for block in blocks:
        before_difference = abs(target - current)
        after_difference = abs(target - (current + block_counts[block]))
        if current < target or after_difference < before_difference:
            result.add(block)
            current += block_counts[block]
        if current >= target:
            break
    return result


def ensure_empty_output(output: Path) -> None:
    manifest = output / "frame_manifest.csv"
    existing_images = list(output.glob("images/**/*.jpg"))
    if manifest.exists() or existing_images:
        raise FileExistsError(
            f"{output} already contains an extraction. Choose another --output "
            "directory to avoid overwriting it."
        )


def write_dataset(
    selected: list[Candidate],
    video_path: Path,
    output: Path,
    val_blocks: set[int],
    block_seconds: float,
    jpeg_quality: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    ensure_empty_output(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    split_counts = {"train": 0, "val": 0}
    for candidate in selected:
        block = int(candidate.timestamp_seconds // block_seconds)
        split = "val" if block in val_blocks else "train"
        filename = (
            f"{video_path.stem}_f{candidate.frame_number:09d}_"
            f"t{candidate.timestamp_seconds:010.3f}.jpg"
        )
        destination = output / "images" / split / filename
        written = cv2.imwrite(
            str(destination),
            candidate.image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not written:
            raise RuntimeError(f"Failed to write {destination}")
        split_counts[split] += 1
        rows.append(
            {
                "image": destination.relative_to(output).as_posix(),
                "source_video": str(video_path),
                "frame_number": candidate.frame_number,
                "timestamp_seconds": f"{candidate.timestamp_seconds:.3f}",
                "split": split,
                "time_block": block,
                "category": candidate.category,
                "court_score": f"{candidate.court_score:.4f}",
                "width": width,
                "height": height,
                "annotation_status": "unlabeled",
            }
        )

    manifest_path = output / "frame_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dataset_config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "racket"},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(dataset_config, sort_keys=False)
    )
    return split_counts["train"], split_counts["val"]


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    video_path = Path(args.video)
    config_path = Path(args.config)
    output = Path(args.output)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    polygon, lower, upper, threshold = load_scene_config(config_path)
    candidates, _, width, height = read_candidates(
        video_path=video_path,
        count=args.count,
        oversample=args.oversample,
        polygon=polygon,
        lower=lower,
        upper=upper,
        court_threshold=threshold,
    )
    selected = choose_frames(
        candidates,
        count=args.count,
        court_ratio=args.court_ratio,
        duplicate_distance=args.duplicate_distance,
    )
    if not selected:
        raise RuntimeError("No readable frames were selected")

    if args.split == "train":
        val_blocks = set()
    elif args.split == "val":
        val_blocks = {
            int(candidate.timestamp_seconds // args.split_block_seconds)
            for candidate in selected
        }
    else:
        val_blocks = validation_blocks(
            selected,
            val_ratio=args.val_ratio,
            block_seconds=args.split_block_seconds,
            seed=args.seed,
        )
    train_count, val_count = write_dataset(
        selected=selected,
        video_path=video_path,
        output=output,
        val_blocks=val_blocks,
        block_seconds=args.split_block_seconds,
        jpeg_quality=args.jpeg_quality,
        width=width,
        height=height,
    )
    court_count = sum(item.category == "court_view" for item in selected)

    print()
    print("Extraction complete.")
    print(f"Destination mode: {args.split}")
    print(f"Selected: {len(selected)} frames")
    print(f"Court views: {court_count}")
    print(f"Other broadcast views: {len(selected) - court_count}")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Output: {output}")
    if len(selected) < args.count:
        print(
            f"Warning: requested {args.count}, but only {len(selected)} sufficiently "
            "different frames were available. Reduce --duplicate-distance or increase "
            "--oversample to collect more."
        )
    print(
        "Labels were not fabricated. Review/annotate the images before using "
        f"{output / 'data.yaml'} for training."
    )


if __name__ == "__main__":
    main()
