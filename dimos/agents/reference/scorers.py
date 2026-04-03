from __future__ import annotations

from statistics import mean

import cv2
import numpy as np

from dimos.agents.reference.types import (
    CandidateScore,
    QueryAttribute,
    QuerySelector,
    ReferenceCandidate,
)


def _bbox_area(candidate: ReferenceCandidate) -> float:
    return candidate.detection.bbox_2d_volume()


def _normalized_rank(position: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 - (position / max(total - 1, 1))


def score_selectors(
    candidates: list[ReferenceCandidate],
    selectors: list[QuerySelector],
) -> list[ReferenceCandidate]:
    if not candidates:
        return candidates

    for selector in selectors:
        if selector.kind == "side":
            ordered = sorted(candidates, key=lambda item: item.center_x)
            if selector.value == "right":
                ordered = list(reversed(ordered))
            for rank, candidate in enumerate(ordered):
                candidate.scores.append(
                    CandidateScore(
                        name=f"side:{selector.value}",
                        value=_normalized_rank(rank, len(ordered)),
                        detail=f"rank={rank + 1}/{len(ordered)}",
                    )
                )
        elif selector.kind == "ordinal":
            axis = next(
                (
                    item.axis
                    for item in selectors
                    if item.kind == "axis_order" and item.axis is not None
                ),
                "left_to_right",
            )
            ordered = sorted(candidates, key=lambda item: item.center_x)
            if axis == "right_to_left":
                ordered = list(reversed(ordered))
            target_index = max(int(selector.value or 1) - 1, 0)
            for rank, candidate in enumerate(ordered):
                distance = abs(rank - target_index)
                score = 1.0 / (1.0 + distance)
                candidate.scores.append(
                    CandidateScore(
                        name=f"ordinal:{selector.value}",
                        value=score,
                        detail=f"rank={rank + 1}/{len(ordered)} axis={axis}",
                    )
                )
        elif selector.kind == "largest":
            ordered = sorted(candidates, key=_bbox_area, reverse=True)
            for rank, candidate in enumerate(ordered):
                candidate.scores.append(
                    CandidateScore(
                        name="largest",
                        value=_normalized_rank(rank, len(ordered)),
                        detail=f"rank={rank + 1}/{len(ordered)}",
                    )
                )
        elif selector.kind == "closest":
            ordered = sorted(candidates, key=lambda item: item.center_y, reverse=True)
            for rank, candidate in enumerate(ordered):
                candidate.scores.append(
                    CandidateScore(
                        name="closest",
                        value=_normalized_rank(rank, len(ordered)),
                        detail=f"rank={rank + 1}/{len(ordered)}",
                    )
                )
    return candidates


def score_attributes(
    candidates: list[ReferenceCandidate],
    attributes: list[QueryAttribute],
) -> list[ReferenceCandidate]:
    for attribute in attributes:
        for candidate in candidates:
            score = 0.0
            detail = "unsupported"
            if attribute.kind == "color":
                score, detail = _score_color_attribute(candidate, attribute.value)
            candidate.scores.append(
                CandidateScore(
                    name=f"{attribute.kind}:{attribute.value}",
                    value=score,
                    detail=detail,
                )
            )
    return candidates


def _score_color_attribute(candidate: ReferenceCandidate, value: str) -> tuple[float, str]:
    crop = candidate.detection.cropped_image(padding=0).to_opencv()
    if crop.size == 0:
        return 0.0, "empty crop"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    if value == "white":
        ratio = float(np.mean((s < 40) & (v > 150)))
        return ratio, f"white_ratio={ratio:.3f}"
    if value == "red":
        ratio = float(np.mean(((h < 10) | (h > 170)) & (s > 80) & (v > 60)))
        return ratio, f"red_ratio={ratio:.3f}"
    if value == "yellow":
        ratio = float(np.mean((h > 18) & (h < 40) & (s > 80) & (v > 80)))
        return ratio, f"yellow_ratio={ratio:.3f}"
    if value == "blue":
        ratio = float(np.mean((h > 90) & (h < 135) & (s > 70) & (v > 50)))
        return ratio, f"blue_ratio={ratio:.3f}"
    if value == "green":
        ratio = float(np.mean((h > 40) & (h < 90) & (s > 60) & (v > 50)))
        return ratio, f"green_ratio={ratio:.3f}"
    if value == "black":
        ratio = float(np.mean(v < 50))
        return ratio, f"black_ratio={ratio:.3f}"

    return 0.0, "unknown color"


def explain_candidate(candidate: ReferenceCandidate) -> str:
    if not candidate.scores:
        return "no scores"
    detail = ", ".join(f"{score.name}={score.value:.3f}" for score in candidate.scores)
    return f"total={candidate.total_score:.3f}; {detail}"


def average_score(candidate: ReferenceCandidate) -> float:
    if not candidate.scores:
        return 0.0
    return mean(score.value for score in candidate.scores)
