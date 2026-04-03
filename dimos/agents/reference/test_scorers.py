import numpy as np

from dimos.agents.reference.scorers import score_selectors
from dimos.agents.reference.types import QuerySelector, ReferenceCandidate
from dimos.msgs.sensor_msgs import Image
from dimos.perception.detection.type import Detection2DBBox


def _candidate(track_id: int, bbox: tuple[float, float, float, float]) -> ReferenceCandidate:
    image = Image(data=np.zeros((1000, 1000, 3), dtype=np.uint8))
    detection = Detection2DBBox(
        bbox=bbox,
        track_id=track_id,
        class_id=0,
        confidence=1.0,
        name="chair",
        ts=0.0,
        image=image,
    )
    return ReferenceCandidate(detection=detection)


def test_ordinal_scoring_prefers_second_from_right() -> None:
    candidates = [
        _candidate(0, (100, 100, 200, 300)),
        _candidate(1, (300, 100, 400, 300)),
        _candidate(2, (500, 100, 600, 300)),
        _candidate(3, (700, 100, 800, 300)),
    ]
    selectors = [
        QuerySelector(kind="side", value="right", frame="image"),
        QuerySelector(kind="ordinal", value=2),
        QuerySelector(kind="axis_order", axis="right_to_left"),
    ]
    ranked = score_selectors(candidates, selectors)
    ranked = sorted(ranked, key=lambda item: item.total_score, reverse=True)
    assert ranked[0].detection.track_id == 2


def test_closest_scoring_prefers_lower_objects_in_image() -> None:
    candidates = [
        _candidate(0, (100, 50, 200, 150)),
        _candidate(1, (100, 150, 200, 250)),
        _candidate(2, (100, 250, 200, 350)),
    ]
    ranked = score_selectors(candidates, [QuerySelector(kind="closest")])
    ranked = sorted(ranked, key=lambda item: item.total_score, reverse=True)
    assert ranked[0].detection.track_id == 2
