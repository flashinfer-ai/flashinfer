"""Persistent CuTe-DSL artifacts for GDN prefill and WY HMMA kernels."""

import errno
import hashlib
import os

from cutlass.cutlass_dsl import CuTeDSL

from ..jit.core import MissingJITCacheError, logger
from ..jit.cute_dsl_core import (
    build_and_load_cute_dsl_kernel,
    cute_dsl_cache_disabled,
)
from .cute_dsl_cache_naming import make_kernel_name


def _kernel_name(kind: str, specialization: tuple, args, compiler_key: str) -> str:
    # CuTe's DLPack descriptors key the element type, alignment and static
    # layout, omitting the addresses and values of dynamic launch arguments.
    signature = tuple((type(arg), getattr(arg, "__cache_key__", None)) for arg in args)
    identity = repr((specialization, signature, compiler_key)).encode()
    return make_kernel_name(kind, hashlib.sha256(identity).hexdigest())


def build_and_load_gdn_kernel(
    module_name: str,
    kind: str,
    specialization: tuple,
    args,
    compile_fn,
    *,
    target_arch: str,
    extra_key_files: tuple[str, ...],
    enable_tvm_ffi: bool,
):
    """Load a GDN specialization or compile it using the caller's options."""
    if cute_dsl_cache_disabled():
        return compile_fn()

    env = CuTeDSL._get_dsl().envar
    # Diagnostic modes need to run the compiler. Overrides can change the
    # target or calling convention supplied by the caller.
    if (
        env.dryrun
        or env.keep_tokens
        or env.print_ir
        or env.print_after_preprocessor
        or env.no_cache
        or env.compiler_opt
        or (env.enable_tvm_ffi and not enable_tvm_ffi)
    ):
        if os.environ.get("FLASHINFER_DISABLE_JIT"):
            raise MissingJITCacheError(
                "JIT compilation is disabled via FLASHINFER_DISABLE_JIT, but "
                "the CuTe-DSL compiler settings require compiling this GDN kernel."
            )
        return compile_fn()

    name = _kernel_name(kind, specialization, args, env.cache_key_str())
    compile_started = False

    def compile_for_cache():
        nonlocal compile_started
        compile_started = True
        return compile_fn()

    try:
        return build_and_load_cute_dsl_kernel(
            module_name,
            name,
            compile_for_cache,
            extra_key_files=extra_key_files,
            enable_tvm_ffi=enable_tvm_ffi,
            target_arch=target_arch,
        )
    except OSError as exc:
        # A permission error after compilation starts can come from the
        # compiler or artifact loader. Preserve those failures.
        if (
            compile_started
            or exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS)
            or os.environ.get("FLASHINFER_DISABLE_JIT")
        ):
            raise
        logger.warning(
            f"Cannot access the CuTe-DSL cache for {module_name}: {exc}. "
            "Compiling in process."
        )
        return compile_fn()
