import os
import subprocess
import tempfile
import types
import warnings
from collections.abc import Hashable

import cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir


_in_mem_compile_cache: dict = {}


_TMA_CLUSTER_LOAD = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
    "mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_CTA_LOAD = (
    "cp.async.bulk.tensor.3d.shared::cta.global.tile."
    "mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_TILE_STORE = (
    "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
)
_TMA_CTA_STORE = "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group"


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


def _compile_options_key(options):
    if options is None:
        return None
    options = _as_options_tuple(options)
    return tuple((type(option), option.value) for option in options)


def _option_value(options, option_type):
    for option in _as_options_tuple(options):
        if isinstance(option, option_type):
            return option.value
    return None


def _needs_sm120a_tma_patch(options):
    return _option_value(options, cute.GPUArch) == "sm_120a"


def _patched_compile_options(options):
    options = _as_options_tuple(options)
    if not _needs_sm120a_tma_patch(options):
        return options

    has_keep_ptx = any(isinstance(option, cute.KeepPTX) for option in options)
    has_dump_dir = any(isinstance(option, DumpDir) for option in options)
    extras = []
    if not has_keep_ptx:
        extras.append(cute.KeepPTX(True))
    if has_dump_dir:
        dump_dir = _option_value(options, DumpDir)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
    else:
        dump_dir = os.environ.get(
            "FLASHINFER_DSL_TMA_PATCH_DIR", "/tmp/flashinfer_dsl_tma_patch"
        )
        os.makedirs(dump_dir, exist_ok=True)
        extras.append(DumpDir(dump_dir))
    return options + tuple(extras)


def _read_ptx_text(ptx_artifact):
    if isinstance(ptx_artifact, str) and os.path.exists(ptx_artifact):
        with open(ptx_artifact, "r", encoding="utf-8") as f:
            return f.read()
    return ptx_artifact


def _patch_sm120a_tma_ptx(ptx_text: str) -> str:
    patched = ptx_text.replace(_TMA_CLUSTER_LOAD, _TMA_CTA_LOAD)
    if _TMA_TILE_STORE in patched:
        lines = []
        for line in patched.splitlines(keepends=True):
            if _TMA_TILE_STORE not in line:
                lines.append(line)
                continue

            line_body = line.rstrip("\r\n")
            line_end = line[len(line_body) :]
            line_body = line_body.replace(_TMA_TILE_STORE, _TMA_CTA_STORE)
            if line_body.endswith(";"):
                operands, separator, _cache_policy = line_body[:-1].rpartition(", %rd")
                if separator:
                    line_body = operands + ";"
            lines.append(line_body + line_end)
        patched = "".join(lines)

    if patched == ptx_text:
        return ptx_text
    return patched.replace("\x00", "")


def _assemble_sm120a_cubin(ptx: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="flashinfer_dsl_tma_patch_") as tmp_dir:
        ptx_path = os.path.join(tmp_dir, "kernel.ptx")
        cubin_path = os.path.join(tmp_dir, "kernel.cubin")
        with open(ptx_path, "w", encoding="utf-8") as f:
            f.write(ptx)

        cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        ptxas = os.path.join(cuda_path, "bin", "ptxas")
        cmd = [ptxas, "-arch=sm_120a", ptx_path, "-o", cubin_path]
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                "failed to assemble patched sm120a TMA PTX\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        with open(cubin_path, "rb") as f:
            return f.read()


def _install_patched_cubin_loader(compiled_fn, patched_cubin: bytes):
    import ctypes

    import cuda.bindings.runtime as cuda_runtime

    from cutlass.base_dsl.common import DSLRuntimeError
    from cutlass.base_dsl.runtime.cuda import checkCudaErrors

    def _load_cuda_library(self):
        if self.engine is None:
            raise DSLRuntimeError("CUDA JIT engine is not available")

        cuda_load_to_device = self.engine.raw_lookup("cuda_load_to_device")
        if cuda_load_to_device is None:
            raise DSLRuntimeError("cuda_load_to_device not found")
        cuda_load_to_device = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)(
            cuda_load_to_device
        )

        library_obj = checkCudaErrors(
            cuda_runtime.cudaLibraryLoadData(
                patched_cubin, None, None, 0, None, None, 0
            )
        )
        library = ctypes.c_void_p(int(library_obj))
        pointer_to_library = ctypes.pointer(library)
        pointer_to_pointer_to_library = ctypes.pointer(pointer_to_library)
        err = ctypes.c_int32(0)
        pointer_to_err = ctypes.pointer(err)
        device_id = ctypes.c_int32(0)
        pointer_to_device_id = ctypes.pointer(device_id)

        cuda_load_args = [
            pointer_to_pointer_to_library,
            pointer_to_device_id,
            pointer_to_err,
        ]
        packed_args = (ctypes.c_void_p * len(cuda_load_args))()
        for i, arg in enumerate(cuda_load_args):
            packed_args[i] = ctypes.cast(arg, ctypes.c_void_p)

        for dev in range(self.num_devices):
            device_id.value = dev
            cuda_load_to_device(packed_args)
            checkCudaErrors((cuda_runtime.cudaError_t(err.value),))

        return [library_obj]

    compiled_fn._flat_patched_cubin = patched_cubin
    compiled_fn._load_cuda_library = types.MethodType(_load_cuda_library, compiled_fn)
    if compiled_fn.artifacts is not None:
        compiled_fn.artifacts.CUBIN = patched_cubin


def _maybe_patch_sm120a_tma(compiled_fn, options):
    if not _needs_sm120a_tma_patch(options):
        return compiled_fn
    ptx = _read_ptx_text(getattr(compiled_fn, "__ptx__", None))
    if not ptx:
        return compiled_fn
    patched_ptx = _patch_sm120a_tma_ptx(ptx)
    if patched_ptx == ptx:
        return compiled_fn
    patched_cubin = _assemble_sm120a_cubin(patched_ptx)
    _install_patched_cubin_loader(compiled_fn, patched_cubin)
    return compiled_fn


def cached_compile(func, *args, compile_options=None, **kwargs):
    cache_key = (func._get_compile_key(), _compile_options_key(compile_options))
    compiled_fn = _in_mem_compile_cache.get(cache_key)

    if compiled_fn is None:
        compiler = cute.compile
        effective_compile_options = _patched_compile_options(compile_options)
        if effective_compile_options:
            compiler = cute.compile[effective_compile_options]
        compiled_fn = compiler(func, *args, **kwargs)
        compiled_fn = _maybe_patch_sm120a_tma(compiled_fn, effective_compile_options)
        _in_mem_compile_cache[cache_key] = compiled_fn

    return compiled_fn
