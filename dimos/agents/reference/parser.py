from __future__ import annotations

import json
import re
from typing import Any

from dimos.agents.reference.types import QueryAttribute, QuerySelector, ReferenceQuery
from dimos.models.vl.acebrain import AceBrainVlModel


_COLOR_SYNONYMS = {
    "white": ["white", "白", "白色"],
    "red": ["red", "红", "红色"],
    "yellow": ["yellow", "黄", "黄色"],
    "blue": ["blue", "蓝", "蓝色"],
    "green": ["green", "绿", "绿色"],
    "black": ["black", "黑", "黑色"],
}

_STATE_SYNONYMS = {
    "open": ["open", "opened", "打开", "开的"],
    "closed": ["closed", "close", "关闭", "关着"],
    "broken": ["broken", "damaged", "破损", "坏的"],
    "wet": ["wet", "潮湿", "湿的"],
}

_NOUN_SYNONYMS = {
    "chair": ["chair", "椅子"],
    "table": ["table", "desk", "桌子"],
    "door": ["door", "门"],
    "person": ["person", "human", "人"],
    "helmet": ["helmet", "安全帽", "头盔"],
    "forklift": ["forklift", "叉车"],
}


def _contains_any(text: str, values: list[str]) -> bool:
    lowered = text.lower()
    return any(value.lower() in lowered for value in values)


def _extract_noun(text: str) -> str | None:
    for noun, values in _NOUN_SYNONYMS.items():
        if _contains_any(text, values):
            return noun
    return None


def _extract_attributes(text: str) -> list[QueryAttribute]:
    attributes: list[QueryAttribute] = []
    for color, synonyms in _COLOR_SYNONYMS.items():
        if _contains_any(text, synonyms):
            attributes.append(QueryAttribute(kind="color", value=color))
    for state, synonyms in _STATE_SYNONYMS.items():
        if _contains_any(text, synonyms):
            attributes.append(QueryAttribute(kind="state", value=state))
    return attributes


def _extract_selectors(text: str) -> list[QuerySelector]:
    selectors: list[QuerySelector] = []
    lowered = text.lower()

    if any(token in text for token in ["左边", "左侧"]) or "left" in lowered:
        selectors.append(QuerySelector(kind="side", value="left", frame="image"))
    if any(token in text for token in ["右边", "右侧"]) or "right" in lowered:
        selectors.append(QuerySelector(kind="side", value="right", frame="image"))

    if "closest" in lowered or "最近" in text:
        selectors.append(QuerySelector(kind="closest"))
    if "largest" in lowered or "最大" in text:
        selectors.append(QuerySelector(kind="largest"))

    ordinals = [
        (2, [r"\bsecond\b", "第二"]),
        (1, [r"\bfirst\b", "第一"]),
        (3, [r"\bthird\b", "第三"]),
    ]
    for value, patterns in ordinals:
        for pattern in patterns:
            if pattern.startswith(r"\b"):
                if re.search(pattern, lowered):
                    selectors.append(QuerySelector(kind="ordinal", value=value))
                    break
            elif pattern in text:
                selectors.append(QuerySelector(kind="ordinal", value=value))
                break

    if "from the right" in lowered or "从右" in text:
        selectors.append(QuerySelector(kind="axis_order", axis="right_to_left"))
    elif "from the left" in lowered or "从左" in text:
        selectors.append(QuerySelector(kind="axis_order", axis="left_to_right"))

    return selectors


def _fallback_parse(query: str) -> ReferenceQuery:
    noun = _extract_noun(query) or "object"
    return ReferenceQuery(
        raw_query=query,
        noun=noun,
        attributes=_extract_attributes(query),
        selectors=_extract_selectors(query),
    )


class AceBrainReferenceParser:
    def __init__(self, model: AceBrainVlModel | None = None, use_remote: bool = True) -> None:
        self._model = model or AceBrainVlModel()
        self._use_remote = use_remote

    def parse(self, query: str) -> ReferenceQuery:
        if not self._use_remote:
            return _fallback_parse(query)

        prompt = (
            "Convert the user reference query into JSON. "
            "Return only JSON with keys noun, attributes, selectors, relations. "
            "attributes is a list of {'kind': str, 'value': str}. "
            "selectors is a list of {'kind': str, 'value': any, 'axis': str|null, 'frame': str|null}. "
            "relations is a list of {'kind': str, 'object_name': str}. "
            f"Query: {query}"
        )
        try:
            raw = self._model.query_text(prompt)
            parsed = json.loads(self._extract_json(raw))
            noun = parsed.get("noun") or _extract_noun(query) or "object"
            attributes = [
                QueryAttribute(kind=item["kind"], value=item["value"])
                for item in parsed.get("attributes", [])
                if isinstance(item, dict) and item.get("kind") and item.get("value")
            ]
            selectors = [
                QuerySelector(
                    kind=item["kind"],
                    value=item.get("value"),
                    axis=item.get("axis"),
                    frame=item.get("frame"),
                )
                for item in parsed.get("selectors", [])
                if isinstance(item, dict) and item.get("kind")
            ]
            if not attributes:
                attributes = _extract_attributes(query)
            if not selectors:
                selectors = _extract_selectors(query)
            return ReferenceQuery(
                raw_query=query,
                noun=noun,
                attributes=attributes,
                selectors=selectors,
            )
        except Exception:
            return _fallback_parse(query)

    @staticmethod
    def _extract_json(raw: str) -> str:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return raw[start : end + 1]
        return raw
