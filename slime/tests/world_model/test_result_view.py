from __future__ import annotations

import pytest

from slime.world_model.result_view import (
    parse_result_only_view,
    parse_tool_result_bundle,
    render_result_only_view,
    result_only_view,
)


def test_result_only_view_preserves_order_and_removes_names() -> None:
    feedback = (
        "<tool_result name=shell_exec>\nalpha\n</tool_result>\n\n"
        '<tool_result name="python">\nbeta\nline 2\n</tool_result>'
    )
    rendered = result_only_view(feedback)
    assert rendered == (
        "<result index=0>\nalpha\n</result>\n\n"
        "<result index=1>\nbeta\nline 2\n</result>"
    )
    assert "shell_exec" not in rendered
    assert "python" not in rendered
    assert "<tool_result" not in rendered
    assert parse_result_only_view(rendered) == ("alpha", "beta\nline 2")


def test_result_view_rejects_unparsed_text_and_empty_values() -> None:
    with pytest.raises(ValueError, match="outside"):
        parse_tool_result_bundle("prefix <tool_result name=x>\na\n</tool_result>")
    with pytest.raises(ValueError, match="non-empty"):
        parse_tool_result_bundle(" ")
    with pytest.raises(ValueError, match="at least one"):
        render_result_only_view([])
    with pytest.raises(ValueError, match="indices"):
        parse_result_only_view("<result index=2>\na\n</result>")
