"""The bundled-fmt text literal that MSVC-hosted nvcc rejects, and its Windows form.

spdlog bundles fmt 12.1, whose ``rounding_thresholds`` helper returns one element
of a ``char32_t`` string literal.  Under nvcc with the Visual Studio host compiler
the literal is "out of range", so every TensorRT-LLM module that includes the
spdlog logger fails to build.  fmt 12.2 fixed the literal upstream; until spdlog
bumps its bundled copy, the wheel build (build_backend.py) and the JIT include
overlay (jit/cpp_ext.py) rewrite the two lines with the same strings from here.
"""

from __future__ import annotations

from pathlib import Path

SPDLOG_FORMAT_HEADER = Path("spdlog/fmt/bundled/format.h")

ROUNDING_THRESHOLDS = (
    '  return U"\\x9999999a\\x828f5c29\\x80418938\\x80068db9\\x8000a7c6\\x800010c7"\n'
    '         U"\\x800001ae\\x8000002b"[index];'
)
ROUNDING_THRESHOLDS_WINDOWS = (
    "  return static_cast<uint32_t>(\n"
    '             u"\\x9999\\x828f\\x8041\\x8006\\x8000\\x8000\\x8000\\x8000"[index])\n'
    "             << 16u |\n"
    "         static_cast<uint32_t>(\n"
    '             u"\\x999a\\x5c29\\x8938\\x8db9\\xa7c6\\x10c7\\x01ae\\x002b"[index]);'
)


def patch_bundled_format_h(content: str) -> str:
    """Return ``format.h`` with the char32_t literal split into two char16_t halves.

    The content keeps its own line endings.  Text without the literal (a newer
    bundled fmt) is returned unchanged.
    """
    newline = "\r\n" if "\r\n" in content else "\n"
    return content.replace(
        ROUNDING_THRESHOLDS.replace("\n", newline),
        ROUNDING_THRESHOLDS_WINDOWS.replace("\n", newline),
    )
