from dimos.agents.reference.parser import AceBrainReferenceParser


def test_parser_extracts_english_ordinal_and_axis_order() -> None:
    parser = AceBrainReferenceParser(use_remote=False)
    result = parser.parse("the second chair from the right")

    assert result.noun == "chair"
    assert any(selector.kind == "ordinal" and selector.value == 2 for selector in result.selectors)
    assert any(
        selector.kind == "axis_order" and selector.axis == "right_to_left"
        for selector in result.selectors
    )
    assert not any(selector.kind == "side" for selector in result.selectors)


def test_parser_extracts_chinese_color_and_ordinal() -> None:
    parser = AceBrainReferenceParser(use_remote=False)
    result = parser.parse("右边第一个白色桌子")

    assert result.noun == "table"
    assert any(attr.kind == "color" and attr.value == "white" for attr in result.attributes)
    assert any(selector.kind == "ordinal" and selector.value == 1 for selector in result.selectors)
    assert any(selector.kind == "side" and selector.value == "right" for selector in result.selectors)
