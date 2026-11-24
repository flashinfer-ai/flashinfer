"""Build a host shim with embedded cubins: ``tvm_ffi.cpp.load_inline`` on POSIX, FlashInfer's own JIT on Windows.

``tvm_ffi.cpp.load_inline(embed_cubin=...)`` embeds each cubin with GNU ``ld`` and ``objcopy`` and raises
``NotImplementedError`` on Windows. FlashInfer's ninja build already embeds cubins on Windows (a COFF object per
cubin, see ``embed_cubin_coff.py``), so there the shim goes through ``gen_jit_spec(embedded_cubin_factory=...)``.
"""

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from . import env as jit_env
from .cpp_ext import is_windows
from .utils import write_if_different

# GNU-style flags with no counterpart for the MSVC host: optimisation levels (the JIT sets its own) and
# ``-l<lib>`` (the Windows link already carries cuda.lib and the runtime libraries).
_DROPPED_ON_WINDOWS = ("-O", "-l")


def _write_bytes_if_different(path: Path, content: bytes) -> None:
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def load_inline_with_cubin(
    name: str,
    *,
    cpp_sources: str,
    embed_cubin: Mapping[str, bytes],
    extra_include_paths: Optional[Sequence[Union[str, Path]]] = None,
    extra_cflags: Optional[Sequence[str]] = None,
    extra_ldflags: Optional[Sequence[str]] = None,
    build_directory: Optional[Union[str, Path]] = None,
) -> Any:
    """Compile ``cpp_sources`` with ``embed_cubin`` embedded and load the module.

    ``embed_cubin`` maps each identifier the source's ``TVM_FFI_EMBED_CUBIN`` names to its cubin bytes. The
    module is rebuilt only when the source, the cubins or the flags change; ``name`` must identify them, as
    for ``cpp.load_inline``.
    """
    if not is_windows:
        from tvm_ffi import cpp

        kwargs: dict[str, Any] = {}
        if extra_include_paths is not None:
            kwargs["extra_include_paths"] = extra_include_paths
        if extra_cflags is not None:
            kwargs["extra_cflags"] = extra_cflags
        if extra_ldflags is not None:
            kwargs["extra_ldflags"] = extra_ldflags
        if build_directory is not None:
            kwargs["build_directory"] = str(build_directory)
        return cpp.load_inline(
            name, cpp_sources=cpp_sources, embed_cubin=embed_cubin, **kwargs
        )

    from .core import gen_jit_spec

    source_dir = jit_env.FLASHINFER_GEN_SRC_DIR / name
    source = source_dir / f"{name}.cc"
    write_if_different(source, cpp_sources)
    cubin_paths: dict[str, Path] = {}
    for ident, data in embed_cubin.items():
        cubin_paths[ident] = source_dir / f"{ident}.cubin"
        _write_bytes_if_different(cubin_paths[ident], data)
    cflags = [f for f in (extra_cflags or ()) if not f.startswith(_DROPPED_ON_WINDOWS)]
    spec = gen_jit_spec(
        name,
        [source],
        extra_cflags=cflags,
        extra_include_paths=list(extra_include_paths or ()),
        embedded_cubin_factory=lambda _build_dir: dict(cubin_paths),
    )
    return spec.build_and_load()
