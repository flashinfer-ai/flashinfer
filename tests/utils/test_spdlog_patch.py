import pytest

from flashinfer.jit.spdlog_patch import (
    ROUNDING_THRESHOLDS,
    ROUNDING_THRESHOLDS_WINDOWS,
    patch_bundled_format_h,
)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_patch_rewrites_the_literal_and_keeps_line_endings(newline):
    original = (
        "// head"
        + newline
        + ROUNDING_THRESHOLDS.replace("\n", newline)
        + newline
        + "// tail"
        + newline
    )
    patched = patch_bundled_format_h(original)
    assert ROUNDING_THRESHOLDS.replace("\n", newline) not in patched
    assert ROUNDING_THRESHOLDS_WINDOWS.replace("\n", newline) in patched
    assert patched.startswith("// head" + newline) and patched.endswith(
        "// tail" + newline
    )
    assert patch_bundled_format_h(patched) == patched


def test_patch_leaves_other_text_alone():
    text = "// a newer bundled fmt without the literal\n"
    assert patch_bundled_format_h(text) == text
