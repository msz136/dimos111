from __future__ import annotations

import json
from typing import Any

from dimos.agents.reference.parser import AceBrainReferenceParser
from dimos.agents.reference.scorers import explain_candidate, score_attributes, score_selectors
from dimos.agents.reference.torchvision_detector import TorchvisionCocoDetector
from dimos.agents.reference.types import (
    CandidateScore,
    QueryAttribute,
    ReferenceCandidate,
    ReferenceGroundingResult,
)
from dimos.msgs.sensor_msgs import Image
from dimos.models.vl.acebrain import AceBrainVlModel
from dimos.perception.detection.detectors.types import Detector
from dimos.perception.detection.detectors.yolo import Yolo2DDetector
from dimos.perception.detection.detectors.yoloe import Yoloe2DDetector, YoloePromptMode
from dimos.perception.detection.type import Detection2DBBox, ImageDetections2D


class ReferenceGroundingPipeline:
    def __init__(
        self,
        parser: AceBrainReferenceParser | None = None,
        detector: Detector | None = None,
        vlm: AceBrainVlModel | None = None,
        use_remote_parser: bool = False,
        use_remote_attribute_judge: bool = False,
        detector_device: str | None = None,
        detector_backend: str = "yoloe",
        use_torchvision_fallback: bool = True,
    ) -> None:
        self.vlm = vlm or AceBrainVlModel()
        self.detector_backend = detector_backend
        self.use_remote_attribute_judge = use_remote_attribute_judge
        self.use_torchvision_fallback = use_torchvision_fallback
        self.torchvision_detector: TorchvisionCocoDetector | None = None
        self.parser = parser or AceBrainReferenceParser(
            model=self.vlm,
            use_remote=use_remote_parser,
        )
        if detector is not None:
            self.detector = detector
        elif detector_backend == "yolo":
            # Use a lower confidence threshold here than the generic detector path so
            # referring-expression grounding can keep borderline candidates and let
            # later attribute/selector scoring decide among them.
            self.detector = Yolo2DDetector(
                device=detector_device,
                conf=0.15,
                iou=0.5,
            )
        else:
            self.detector = Yoloe2DDetector(
                prompt_mode=YoloePromptMode.PROMPT,
                device=detector_device,
            )

    def detect(self, image: Image, query: str) -> ReferenceGroundingResult:
        parsed = self.parser.parse(query)
        detections = self._detect_candidates(image, parsed.noun)
        candidates = [ReferenceCandidate(detection=det) for det in detections.detections]

        score_attributes(candidates, parsed.attributes)
        if self.use_remote_attribute_judge:
            self._score_remote_attributes(candidates, parsed.attributes)
        score_selectors(candidates, parsed.selectors)
        ranked = sorted(candidates, key=lambda item: item.total_score, reverse=True)
        selected = ranked[0] if ranked else None

        explanation = "No candidates found."
        if selected is not None:
            explanation = explain_candidate(selected)

        return ReferenceGroundingResult(
            query=parsed,
            candidates=ranked,
            selected=selected,
            explanation=explanation,
        )

    def stop(self) -> None:
        self.detector.stop()
        if self.torchvision_detector is not None:
            self.torchvision_detector.stop()
        self.vlm.stop()

    @staticmethod
    def selected_bbox(result: ReferenceGroundingResult) -> tuple[float, float, float, float] | None:
        if result.selected is None:
            return None
        return result.selected.detection.bbox

    def _score_remote_attributes(
        self,
        candidates: list[ReferenceCandidate],
        attributes: list[QueryAttribute],
    ) -> None:
        for attribute in attributes:
            for candidate in candidates:
                try:
                    prompt = self._attribute_prompt(attribute)
                    raw = self.vlm.query(candidate.detection.cropped_image(padding=0), prompt)
                    parsed = self._extract_json(raw)
                    score = float(parsed.get("confidence", 0.0)) if parsed.get("match") else 0.0
                    candidate.scores.append(
                        CandidateScore(
                            name=f"remote:{attribute.kind}:{attribute.value}",
                            value=score,
                            detail=parsed.get("reason", ""),
                        )
                    )
                except Exception:
                    continue

    @staticmethod
    def _attribute_prompt(attribute: QueryAttribute) -> str:
        return (
            "Judge whether the cropped object matches the requested attribute. "
            "Return only JSON with keys match (bool), confidence (0-1), reason (string). "
            f"Attribute kind: {attribute.kind}. Attribute value: {attribute.value}."
        )

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        raise ValueError("Expected dict JSON response")

    def _detect_candidates(self, image: Image, noun: str) -> Any:
        if self.detector_backend == "yolo":
            return self._detect_candidates_with_tiling(image, noun)

        if hasattr(self.detector, "set_prompts"):
            prompts = sorted(_noun_aliases(noun))
            getattr(self.detector, "set_prompts")(text=prompts)
            return self.detector.process_image(image)

        detections = self.detector.process_image(image)
        detections.detections = [
            det
            for det in detections.detections
            if _matches_noun(det.name, noun)
        ]
        return detections

    def _detect_candidates_with_tiling(self, image: Image, noun: str) -> ImageDetections2D:
        merged: list[Detection2DBBox] = []
        full_image_detections = self.detector.process_image(image)
        merged.extend(
            det for det in full_image_detections.detections if _matches_noun(det.name, noun)
        )

        for x, y, width, height in _tile_regions(image.width, image.height):
            tile = image.crop(x, y, width, height)
            tile_detections = self.detector.process_image(tile)
            for det in tile_detections.detections:
                if not _matches_noun(det.name, noun):
                    continue
                merged.append(_remap_detection_to_image(det, image, x, y))

        if self.use_torchvision_fallback and (not merged or noun == "table"):
            merged.extend(self._detect_torchvision_candidates(image, noun))

        return ImageDetections2D(image=image, detections=_dedupe_detections(merged))

    def _detect_torchvision_candidates(self, image: Image, noun: str) -> list[Detection2DBBox]:
        if noun not in {"chair", "table", "person"}:
            return []
        if self.torchvision_detector is None:
            self.torchvision_detector = TorchvisionCocoDetector()
        return self.torchvision_detector.detect(image, noun)


def _noun_aliases(noun: str) -> set[str]:
    normalized = noun.strip().lower()
    aliases: dict[str, set[str]] = {
        "chair": {"chair", "bench", "stool", "couch", "sofa", "seat"},
        "table": {"table", "dining table", "desk", "coffee table"},
        "person": {"person", "man", "woman", "people"},
        "helmet": {"helmet", "hard hat"},
        "forklift": {"forklift"},
        "door": {"door", "doorway", "gate"},
    }
    return aliases.get(normalized, {normalized})


def _matches_noun(detection_name: str, noun: str) -> bool:
    candidate = detection_name.strip().lower()
    aliases = _noun_aliases(noun)
    return any(alias in candidate or candidate in alias for alias in aliases)


def _tile_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    tile_regions: list[tuple[int, int, int, int]] = []

    quad_width = max(int(width * 0.65), 1)
    quad_height = max(int(height * 0.7), 1)
    x_positions = sorted({0, max(width - quad_width, 0)})
    y_positions = sorted({0, max(height - quad_height, 0)})
    for x in x_positions:
        for y in y_positions:
            tile_regions.append((x, y, quad_width, quad_height))

    strip_width = max(int(width * 0.55), 1)
    strip_x_positions = sorted({0, max((width - strip_width) // 2, 0), max(width - strip_width, 0)})
    for x in strip_x_positions:
        tile_regions.append((x, 0, strip_width, height))

    # Preserve order while removing duplicates.
    deduped: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for region in tile_regions:
        if region not in seen:
            seen.add(region)
            deduped.append(region)
    return deduped


def _remap_detection_to_image(
    detection: Detection2DBBox,
    image: Image,
    offset_x: int,
    offset_y: int,
) -> Detection2DBBox:
    x1, y1, x2, y2 = detection.bbox
    return Detection2DBBox(
        bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
        track_id=detection.track_id,
        class_id=detection.class_id,
        confidence=detection.confidence,
        name=detection.name,
        ts=image.ts,
        image=image,
    )


def _dedupe_detections(
    detections: list[Detection2DBBox],
    iou_threshold: float = 0.6,
) -> list[Detection2DBBox]:
    kept: list[Detection2DBBox] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(
            _bbox_iou(detection.bbox, existing.bbox) >= iou_threshold and detection.name == existing.name
            for existing in kept
        ):
            continue
        kept.append(detection)
    return kept


def _bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
