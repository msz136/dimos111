from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

from dimos.msgs.sensor_msgs import Image
from dimos.perception.detection.type import Detection2DBBox
from dimos.utils.gpu_utils import is_cuda_available


@dataclass
class TorchvisionDetectorConfig:
    score_threshold: float = 0.2
    tiled: bool = True


class TorchvisionCocoDetector:
    def __init__(self, config: TorchvisionDetectorConfig | None = None) -> None:
        self.config = config or TorchvisionDetectorConfig()
        self.device = torch.device("cuda" if is_cuda_available() else "cpu")  # type: ignore[no-untyped-call]
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()

    def stop(self) -> None:
        del self.model

    def detect(self, image: Image, noun: str) -> list[Detection2DBBox]:
        aliases = _torchvision_aliases(noun)
        if not aliases:
            return []

        regions = [(0, 0, image.width, image.height)]
        if self.config.tiled:
            regions.extend(_tile_regions(image.width, image.height))

        detections: list[Detection2DBBox] = []
        for x, y, width, height in regions:
            crop = image.crop(x, y, width, height)
            tensor = to_tensor(crop.to_rgb().as_numpy()).to(self.device)
            with torch.inference_mode():
                prediction = self.model([tensor])[0]

            for box, label, score in zip(
                prediction["boxes"], prediction["labels"], prediction["scores"], strict=False
            ):
                score_value = float(score)
                if score_value < self.config.score_threshold:
                    continue
                name = self.categories[int(label)]
                if name not in aliases:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                detections.append(
                    Detection2DBBox(
                        bbox=(x1 + x, y1 + y, x2 + x, y2 + y),
                        track_id=-1,
                        class_id=int(label),
                        confidence=score_value,
                        name=name,
                        ts=image.ts,
                        image=image,
                    )
                )

        return _dedupe_torchvision(detections)


def _torchvision_aliases(noun: str) -> set[str]:
    normalized = noun.strip().lower()
    aliases: dict[str, set[str]] = {
        "chair": {"chair", "bench", "couch"},
        "table": {"dining table"},
        "person": {"person"},
    }
    return aliases.get(normalized, set())


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

    deduped: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for region in tile_regions:
        if region not in seen:
            seen.add(region)
            deduped.append(region)
    return deduped


def _dedupe_torchvision(
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
