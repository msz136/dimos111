from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image as PilImage, ImageDraw

from dimos.agents.reference.pipeline import ReferenceGroundingPipeline
from dimos.msgs.sensor_msgs import Image


def _find_primary_image(case_dir: Path) -> Path:
    preferred = sorted(
        path
        for path in case_dir.glob("*.jpg")
        if "_axes" not in path.name and "_boxed" not in path.name and "qwen" not in path.name
    )
    if preferred:
        return preferred[0]
    return sorted(case_dir.glob("*.jpg"))[0]


def _extract_query(case_dir: Path) -> str:
    prompt = (case_dir / "prompt.txt").read_text(encoding="utf-8")
    marker = "find the '"
    if marker in prompt:
        rest = prompt.split(marker, 1)[1]
        return rest.split("'", 1)[0]
    return prompt.strip()


def _load_expected_bbox(case_dir: Path) -> list[float] | None:
    for name in ["bbox_meta.json", "qwen3vl8b_bbox_meta.json"]:
        path = case_dir / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            bbox = data.get("clamped_bbox") or data.get("raw_bbox")
            if bbox:
                return [float(value) for value in bbox]
    return None


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1.0)


def _draw_boxes(image_path: Path, predicted: list[float] | None, expected: list[float] | None, out_path: Path) -> None:
    image = PilImage.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if expected:
        draw.rectangle(expected, outline=(0, 255, 0), width=4)
    if predicted:
        draw.rectangle(predicted, outline=(255, 0, 0), width=4)
    image.save(out_path)


def run_eval(
    cases_dir: Path,
    output_json: Path,
    use_remote_parser: bool = False,
    use_remote_attribute_judge: bool = False,
    detector_backend: str = "yolo",
) -> list[dict[str, Any]]:
    pipeline = ReferenceGroundingPipeline(
        use_remote_parser=use_remote_parser,
        use_remote_attribute_judge=use_remote_attribute_judge,
        detector_backend=detector_backend,
    )
    results: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
        image_path = _find_primary_image(case_dir)
        query = _extract_query(case_dir)
        image = Image.from_file(image_path)
        result = pipeline.detect(image, query)
        predicted = list(result.selected.detection.bbox) if result.selected else None
        expected = _load_expected_bbox(case_dir)
        iou = _iou(predicted, expected) if predicted and expected else None
        viz_path = case_dir / "reference_pipeline_eval.jpg"
        _draw_boxes(image_path, predicted, expected, viz_path)
        results.append(
            {
                "case": case_dir.name,
                "query": query,
                "noun": result.query.noun,
                "attributes": [attr.__dict__ for attr in result.query.attributes],
                "selectors": [selector.__dict__ for selector in result.query.selectors],
                "predicted_bbox": predicted,
                "expected_bbox": expected,
                "iou": iou,
                "explanation": result.explanation,
                "viz": str(viz_path),
            }
        )
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pipeline.stop()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reference_pipeline_summary.json"))
    parser.add_argument("--use-remote-parser", action="store_true")
    parser.add_argument("--use-remote-attribute-judge", action="store_true")
    parser.add_argument("--detector-backend", choices=["yolo", "yoloe"], default="yolo")
    args = parser.parse_args()
    run_eval(
        args.cases_dir,
        args.output,
        use_remote_parser=args.use_remote_parser,
        use_remote_attribute_judge=args.use_remote_attribute_judge,
        detector_backend=args.detector_backend,
    )


if __name__ == "__main__":
    main()
