from badminton_analysis.metrics import distance_travelled
from badminton_analysis.types import PlayerTrack, TrackPoint


def test_distance_travelled_uses_court_coordinates_when_available():
    track = PlayerTrack(track_id=1)
    track.add(TrackPoint(0, 0.0, 100, 100, 0.0, 0.0), 45)
    track.add(TrackPoint(1, 0.1, 110, 110, 3.0, 4.0), 45)

    assert distance_travelled(track) == 5.0
