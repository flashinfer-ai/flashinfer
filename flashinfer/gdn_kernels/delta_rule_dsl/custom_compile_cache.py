import os
import tempfile
import types
import typing

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
    def _get_compile_key(self):
        compile_key = getattr(self, "_KeyedCompileMixin_compile_key", None)
        if compile_key is None:
            collected_attrs = []
            for attr_name in sorted(dir(self)):
                attr_value = getattr(self, attr_name)
                if attr_name.startswith("__"):
                    continue
                if isinstance(
                    attr_value,
                    (
                        types.MethodType,
                        types.BuiltinMethodType,
                        types.MethodWrapperType,
                    ),
                ):
                    continue

                if isinstance(attr_value, typing.Hashable):
                    collected_attrs.append((attr_name, attr_value))

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


def _has_option_value(options, option_type, option_value):
    for option in _as_options_tuple(options):
        if isinstance(option, option_type) and option.value == option_value:
            return True
    return False


def _needs_sm120a_tma_patch(options):
    return _has_option_value(options, cute.GPUArch, "sm_120a")


def _patched_compile_options(options, dump_dir):
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

    return patched


def _install_patched_ptx_loader(compiled_fn, patched_ptx: str):
    import ctypes

    import cuda.bindings.driver as cuda_driver
    import cuda.bindings.runtime as cuda_runtime

    from cutlass.base_dsl.common import DSLRuntimeError
    from cutlass.base_dsl.runtime.cuda import checkCudaErrors

    jit_options = [cuda_driver.CUjit_option.CU_JIT_TARGET]
    jit_option_values = [cuda_driver.CUjit_target.CU_TARGET_COMPUTE_120A]

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
            cuda_driver.cuLibraryLoadData(
                patched_ptx.encode("utf-8"),
                jit_options,
                jit_option_values,
                len(jit_options),
                None,
                None,
                0,
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

    compiled_fn._flat_patched_ptx = patched_ptx
    compiled_fn._flat_patched_ptx_jit_options = tuple(jit_options)
    compiled_fn._flat_patched_ptx_jit_option_values = tuple(jit_option_values)
    compiled_fn._load_cuda_library = types.MethodType(_load_cuda_library, compiled_fn)


def _patch_sm120a_tma(compiled_fn, options):
    ptx = _read_ptx_text(getattr(compiled_fn, "__ptx__", None))
    if not ptx:
        return compiled_fn
    patched_ptx = _patch_sm120a_tma_ptx(ptx)
    if patched_ptx == ptx:
        return compiled_fn
    _install_patched_ptx_loader(compiled_fn, patched_ptx)
    return compiled_fn


def cached_compile(func, *args, compile_options=None, **kwargs):
    cache_key = (func._get_compile_key(), _compile_options_key(compile_options))
    compiled_fn = _in_mem_compile_cache.get(cache_key)

    if compiled_fn is None:
        _needs_patch = _needs_sm120a_tma_patch(compile_options)
        if not _needs_patch:
            compiled_fn = cute.compile[compile_options](func, *args, **kwargs)
        else:
            with tempfile.TemporaryDirectory(prefix="cutedsl_patch_") as tempdir:
                compile_options = _patched_compile_options(compile_options, tempdir)
                compiled_fn = cute.compile[compile_options](func, *args, **kwargs)
                compiled_fn = _patch_sm120a_tma(compiled_fn, compile_options)

        _in_mem_compile_cache[cache_key] = compiled_fn

    return compiled_fn
