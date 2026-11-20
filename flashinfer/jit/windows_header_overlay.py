"""Windows-only overlay of the trtllm-gen export headers.

The exported ``trtllm/gen/CommonUtils.h`` declares ``XLargeN`` as ``unsigned
long``, which is 32 bits wide under the MSVC host compiler (LLP64), so
``1UL << 35`` does not compile. Until a republished export carries the fixed
header, the Windows build compiles against a patched copy of the export headers
that sits in front of the artifact directory on the include path.
"""

from pathlib import Path
from typing import Optional, Sequence

from . import env as jit_env
from .cpp_ext import _write_if_changed, is_windows

_OLD_DECLARATION = "constexpr unsigned long XLargeN = 1UL << 35;"
_NEW_DECLARATION = "constexpr unsigned long long XLargeN = 1ULL << 35;"
_PATCHED_HEADER = "trtllm/gen/CommonUtils.h"


def get_windows_header_overlay(
    source_root: Path,
    module_name: str,
    headers: Sequence[str],
    export_subdir: Sequence[str],
) -> Optional[Path]:
    """Return an include root holding the patched export headers, None off Windows.

    ``export_subdir`` is the include prefix the sources use for the export, for
    example ``("flashinfer", "trtllm", "gemm", "trtllmGen_gemm_export")``.
    """
    if not is_windows:
        return None

    overlay_root = (
        jit_env.FLASHINFER_GEN_SRC_DIR / module_name / "windows_header_overlay"
    )
    for header in headers:
        source = source_root / header
        # Skip stubbed downloads (CPU tests); verify_symlinked_headers flags real gaps.
        if not source.is_file():
            continue
        content = source.read_bytes().decode("utf-8")
        if header == _PATCHED_HEADER:
            old_count = content.count(_OLD_DECLARATION)
            new_count = content.count(_NEW_DECLARATION)
            if old_count == 1 and new_count == 0:
                content = content.replace(_OLD_DECLARATION, _NEW_DECLARATION)
            elif old_count != 0 or new_count != 1:
                raise RuntimeError(f"Unexpected XLargeN declaration in {source}")
        destination = overlay_root.joinpath(*export_subdir, header)
        # A temporary file plus os.replace: a process that compiles the module
        # while another one regenerates the overlay reads a whole header, never
        # a truncated one.
        _write_if_changed(destination, content.encode("utf-8"))

    return overlay_root
