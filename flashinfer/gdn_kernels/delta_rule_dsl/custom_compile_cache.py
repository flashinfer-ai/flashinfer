import functools
import glob
import hashlib
import os
import sys
import tempfile
import warnings
from collections.abc import Hashable
from functools import lru_cache

import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import DumpDir


_in_mem_compile_cache: dict = {}
_DISK_CACHE_MODULE = "gdn_delta_rule_dsl"


@lru_cache(maxsize=8)
def _gpu_arch_cached(
    device_type: str, device_index: int | None, family: int, suffix: str
) -> cute.GPUArch:
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    major, minor = torch.cuda.get_device_capability(device)
    if major != family:
        raise RuntimeError(
            f"sm{family}x_gpu_arch requires an SM{family}x device, got compute "
            f"capability {major}.{minor}"
        )
    return cute.GPUArch(f"sm_{major}{minor}{suffix}")


def _gpu_arch(device: str | torch.device, family: int, suffix: str) -> cute.GPUArch:
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return _gpu_arch_cached(d.type, d.index, family, suffix)


def sm12x_gpu_arch(device: str | torch.device = "cuda") -> cute.GPUArch:
    """Return the on-device CuteDSL arch for SM12x (``sm_120a`` / ``sm_121a``)."""
    return _gpu_arch(device, 12, "a")


def sm12x_compile_options(
    device: str | torch.device = "cuda",
) -> tuple[cute.GPUArch]:
    return (sm12x_gpu_arch(device),)


def sm8x_compile_options(
    device: str | torch.device = "cuda",
) -> tuple[cute.GPUArch]:
    # No arch-specific ("a") target exists before SM90.
    return (_gpu_arch(device, 8, ""),)


_TMA_CLUSTER_LOAD = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
    "mbarrier::complete_tx::bytes.L2::cache_hint"
)


def _as_options_tuple(options):
    if options is None:
        return ()
    if isinstance(options, tuple):
        return options
    return (options,)


class KeyedCompileMixin:
    def manual_cache_key(self, *attr_names):
        collected_attrs = tuple(
            (attr_name, getattr(self, attr_name)) for attr_name in attr_names
        )
        compile_key = (type(self).__mro__,) + collected_attrs
        hash(compile_key)
        setattr(self, "_KeyedCompileMixin_compile_key", compile_key)  # noqa: B010

    def _get_compile_key(self):
        compile_key = getattr(self, "_KeyedCompileMixin_compile_key", None)
        if compile_key is None:
            warnings.warn(
                f"{type(self).__name__} is using automatic DSL compile-cache key generation; "
                "call manual_cache_key(...) at the end of __init__ to avoid host launch overhead.",
                RuntimeWarning,
                stacklevel=2,
            )
            collected_attrs = tuple(
                (attr_name, attr_value)
                for attr_name, attr_value in sorted(self.__dict__.items())
                if not attr_name.startswith("_")
            )
            compile_key = (str(type(self).__mro__),) + tuple(collected_attrs)
            try:
                hash(compile_key)
            except TypeError:
                collected_attrs = tuple(
                    (attr_name, attr_value)
                    for attr_name, attr_value in collected_attrs
                    if isinstance(attr_value, Hashable)
                )
                compile_key = (str(type(self).__mro__),) + tuple(collected_attrs)
            setattr(self, "_KeyedCompileMixin_compile_key", compile_key)  # noqa: B010

        return compile_key


@functools.cache
def _compile_options_tuple_key(options):
    return tuple((type(option), option.value) for option in options)


def _compile_options_key(options):
    if options is None:
        return None
    return _compile_options_tuple_key(_as_options_tuple(options))


def _option_value(options, option_type):
    for option in _as_options_tuple(options):
        if isinstance(option, option_type):
            return option.value
    return None


def _has_option_value(options, option_type, option_value):
    for option in _as_options_tuple(options):
        if isinstance(option, option_type) and option.value == option_value:
            return True
    return False


def _needs_sm12x_ptx_check(options):
    # Arch-specific SM12x targets share the same cluster-TMA restriction.
    return _has_option_value(options, cute.GPUArch, "sm_120a") or _has_option_value(
        options, cute.GPUArch, "sm_121a"
    )


def _ptx_check_compile_options(options, dump_dir):
    options = _as_options_tuple(options)

    has_keep_ptx = _has_option_value(options, cute.KeepPTX, True)
    has_dump_dir = _option_value(options, DumpDir)
    extras = []
    if not has_keep_ptx:
        extras.append(cute.KeepPTX(True))
    if has_dump_dir:
        dump_dir = _option_value(options, DumpDir)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
    else:
        extras.append(DumpDir(dump_dir))
    return options + tuple(extras)


def _read_ptx_text(ptx_artifact):
    if isinstance(ptx_artifact, str) and os.path.exists(ptx_artifact):
        with open(ptx_artifact, "r", encoding="utf-8") as f:
            return f.read()
    return ptx_artifact


def _sm12x_ptx_check(compiled_fn):
    ptx = _read_ptx_text(getattr(compiled_fn, "__ptx__", None))
    if not ptx:
        raise RuntimeError("Unable to inspect generated PTX for the SM12x GDN kernel")
    if _TMA_CLUSTER_LOAD in ptx:
        raise RuntimeError(
            "SM12x GDN kernel compilation produced unsupported cluster-scoped TMA loads. "
            "Install the CUDA 13 CUTLASS DSL compiler with "
            "`pip install --upgrade --force-reinstall 'nvidia-cutlass-dsl[cu13]'`. "
            "See https://github.com/NVIDIA/cutlass/issues/3170."
        )


def _compile_uncached(func, *args, compile_options=None, **kwargs):
    """Compile ``func`` with CuTe-DSL, running the SM12x PTX check when needed."""
    if not _needs_sm12x_ptx_check(compile_options):
        return cute.compile[compile_options](func, *args, **kwargs)
    with tempfile.TemporaryDirectory(prefix="cutedsl_ptx_check_") as tempdir:
        compile_options = _ptx_check_compile_options(compile_options, tempdir)
        compiled_fn = cute.compile[compile_options](func, *args, **kwargs)
        _sm12x_ptx_check(compiled_fn)
    return compiled_fn


def _disk_kernel_name(func, cache_key) -> str | None:
    """On-disk name: class name + truncated SHA-256 of ``repr(cache_key)``.

    None when the repr carries an object address (it would never hit across
    processes); this guards that pattern only, not every unstable repr.
    """
    key_repr = repr(cache_key)
    if " at 0x" in key_repr:
        return None
    digest = hashlib.sha256(key_repr.encode()).hexdigest()[:24]
    return f"{type(func).__name__}_{digest}"


def _key_source_dirs(func) -> tuple[str, ...]:
    """``delta_rule_dsl`` plus the directory of every FlashInfer class in the MRO.

    Subclasses such as the Blackwell CP kernels live outside ``delta_rule_dsl``.
    """
    dirs = {os.path.dirname(os.path.abspath(__file__))}
    for cls in type(func).__mro__:
        path = getattr(sys.modules.get(cls.__module__), "__file__", None)
        if path and cls.__module__.startswith("flashinfer."):
            dirs.add(os.path.dirname(os.path.abspath(path)))
    return tuple(sorted(dirs))


def _disk_module(func) -> tuple[str, tuple[str, ...]]:
    """Disk-cache module name and key files for ``func``.

    Each set of source directories gets its own module, so families keyed on
    different sources never wipe each other's artifacts.
    """
    dirs = _key_source_dirs(func)
    here = os.path.dirname(os.path.abspath(__file__))
    extra = [os.path.basename(d) for d in dirs if d != here]
    module = "_".join([_DISK_CACHE_MODULE, *extra])
    files = tuple(sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.py"))))
    return module, files


def cached_compile(func, *args, compile_options=None, **kwargs):
    """Return the compiled ``func``, from memory, the disk cache or a fresh compile.

    TVM-FFI compiles are persisted with ``build_and_load_cute_dsl_kernel`` so
    later processes reload them; other compiles are cached in memory only.
    """
    cache_key = (func._get_compile_key(), _compile_options_key(compile_options))
    compiled_fn = _in_mem_compile_cache.get(cache_key)
    if compiled_fn is not None:
        return compiled_fn

    def compile_fn():
        return _compile_uncached(func, *args, compile_options=compile_options, **kwargs)

    # TVM-FFI kernels (SM80/SM90/SM12x paths) are exported and reloaded through
    # the CuTe-DSL disk cache, so later processes skip the compilation.
    kernel_name = _disk_kernel_name(func, cache_key)
    if kernel_name is not None and _has_option_value(
        compile_options, cute.EnableTVMFFI, True
    ):
        from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

        module, key_files = _disk_module(func)
        compiled_fn = build_and_load_cute_dsl_kernel(
            module, kernel_name, compile_fn, key_files
        )
    else:
        compiled_fn = compile_fn()
    _in_mem_compile_cache[cache_key] = compiled_fn
    return compiled_fn


def get_cached_compile(func, compile_options=None):
    cache_key = (func._get_compile_key(), _compile_options_key(compile_options))
    return _in_mem_compile_cache.get(cache_key)
