from dataclasses import dataclass, field


@dataclass
class TrackPoint:
    frame: int
    timestamp: float
    x: float
    y: float
    court_x: float | None = None
    court_y: float | None = None


@dataclass
class PlayerTrack:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)

    def add(self, point: TrackPoint, max_points: int) -> None:
        self.points.append(point)
        if len(self.points) > max_points:
            del self.points[:-max_points]
