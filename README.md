# Badminton Analysis

YOLO26-based badminton video analytics MVP. The current pipeline tracks players in a video, renders persistent track IDs, optionally maps their foot positions into court coordinates, and exports basic movement metrics.

## What It Does

- Detects and tracks people with Ultralytics YOLO26.
- Writes an annotated MP4 with player boxes and track IDs.
- Stores per-player observations and summary metrics as JSON.
- Supports a 3x3 court homography for metre-based distance and speed.
- Keeps the model and tracker configurable through YAML.

The pretrained COCO weights do not detect shuttlecocks, rackets, court lines, or shot types reliably. Those should be added as custom models after the player-tracking baseline is working.

## Setup

Requires Python 3.10 or newer. Create an environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

The first YOLO26 run downloads the selected model weights. On Apple Silicon, Ultralytics can use MPS for training; video inference can start on the default device and be configured later if needed. Video analysis displays a live frame progress bar using only the Python standard library and prints `Analysis complete.` when the output files are ready.

## Analyze A Video

Put a video in `data/raw/`, then run:

```bash
python -m badminton_analysis data/raw/match.mp4
```

The default outputs are:

- `data/processed/annotated.mp4`
- `data/processed/metrics.json`

You can override the configuration or output paths:

```bash
python -m badminton_analysis data/raw/match.mp4 \
	--config config/default.yaml \
	--output-video data/processed/match_annotated.mp4 \
	--output-metrics data/processed/match_metrics.json
```

The equivalent installed command is:

```bash
badminton-analyze data/raw/match.mp4
```

For a static camera, start with `bytetrack.yaml`. For moving-camera footage, try `botsort.yaml` in `config/default.yaml`.

## Court Calibration

Distance is reported in pixels unless a homography is configured. Create a matrix from four or more known image-to-court point pairs and save it using the format in `config/court_homography.example.yaml`.

Then update `config/default.yaml`:

```yaml
court_homography: config/court_homography.yaml
```

The homography should map image pixels to court coordinates in metres. For a full singles court, a practical coordinate system is approximately 6.10 m wide by 13.40 m long. Use the actual court geometry and the camera view when calculating the matrix; the identity matrix in the example is only a schema example.

## Project Layout

```text
config/                 YAML model and calibration settings
data/raw/               input videos, ignored by git
data/processed/         annotated videos and JSON metrics, ignored by git
src/badminton_analysis/ tracking, calibration, metrics, and CLI code
tests/                   focused unit tests
```

## Development Checks

```bash
python -m compileall -q src tests
python -m pytest -q
```

## Next Analytics Modules

The recommended next steps are:

1. Train a custom detector for `shuttle` and `racket` using frames from the target camera setup.
2. Add court-line or court-keypoint detection and automate homography estimation.
3. Combine shuttle position, racket proximity, wrist velocity, and direction changes into hit events.
4. Train a temporal shot classifier for clears, drops, smashes, drives, lifts, and net shots.
5. Add rally segmentation and a dashboard over the exported event JSON.

Keep train/validation splits separated by video rather than by adjacent frames to avoid leakage.

## Licensing

Ultralytics is available under AGPL-3.0 with separate enterprise licensing options. Review the dependency license before distributing a closed-source or commercial deployment. This project does not include model weights or third-party video assets.
