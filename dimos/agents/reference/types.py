from dataclasses import dataclass, field
from typing import Any

from dimos.perception.detection.type import Detection2DBBox


@dataclass
class QueryAttribute:
    kind: str
    value: str


@dataclass
class QueryRelation:
    kind: str
    object_name: str


@dataclass
class QuerySelector:
    kind: str
    value: Any = None
    axis: str | None = None
    frame: str | None = None


@dataclass
class ReferenceQuery:
    raw_query: str
    noun: str
    attributes: list[QueryAttribute] = field(default_factory=list)
    relations: list[QueryRelation] = field(default_factory=list)
    selectors: list[QuerySelector] = field(default_factory=list)


@dataclass
class CandidateScore:
    name: str
    value: float
    detail: str = ""


@dataclass
class ReferenceCandidate:
    detection: Detection2DBBox
    scores: list[CandidateScore] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return sum(score.value for score in self.scores)

    @property
    def center_x(self) -> float:
        return self.detection.get_bbox_center()[0]

    @property
    def center_y(self) -> float:
        return self.detection.get_bbox_center()[1]


@dataclass
class ReferenceGroundingResult:
    query: ReferenceQuery
    candidates: list[ReferenceCandidate]
    selected: ReferenceCandidate | None
    explanation: str
