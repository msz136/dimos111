from __future__ import annotations

import json
from typing import Any

from dimos.agents.reference.parser import AceBrainReferenceParser
from dimos.agents.reference.scorers import explain_candidate, score_attributes, score_selectors
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
    ) -> None:
        self.vlm = vlm or AceBrainVlModel()
        self.use_remote_attribute_judge = use_remote_attribute_judge
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
