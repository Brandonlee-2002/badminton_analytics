import math
from collections.abc import Iterable

from .types import PlayerTrack


def distance_travelled(track: PlayerTrack, court_coordinates: bool = True) -> float:
    total = 0.0
    points = track.points
    for previous, current in zip(points, points[1:]):
        if court_coordinates and previous.court_x is not None and current.court_x is not None:
            dx = current.court_x - previous.court_x
            dy = current.court_y - previous.court_y
        else:
            dx = current.x - previous.x
            dy = current.y - previous.y
        total += math.hypot(dx, dy)
    return total


def summarize_tracks(tracks: Iterable[PlayerTrack], fps: float, calibrated: bool) -> list[dict]:
    summaries = []
    for track in tracks:
        if not track.points:
            continue
        duration = max(track.points[-1].timestamp - track.points[0].timestamp, 0.0)
        distance = distance_travelled(track, court_coordinates=calibrated)
        summaries.append({
            "track_id": track.track_id,
            "observed_frames": len(track.points),
            "duration_seconds": round(duration, 3),
            "distance": round(distance, 3),
            "distance_unit": "meters" if calibrated else "pixels",
            "average_speed": round(distance / duration, 3) if duration else 0.0,
            "fps": fps,
        })
    return summaries
