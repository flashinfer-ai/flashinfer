# flashinfer/jit/ep.py
#
# JIT compilation entry point for the unified Expert Parallelism module.
#
# On first call to gen_ep_module().build_and_load(), this compiles
# csrc/ep/bindings.cu via ninja + nvcc, linking against NCCL for the
# alltoall P2P communication. The resulting .so is loaded through tvm_ffi.
#
# Exported TVM FFI functions:
#   ep_create_group, ep_destroy_group, ep_get_dispatch_layout,
#   ep_dispatch, ep_combine,
#   ep_create_handle, ep_destroy_handle, ep_handle_get_num_recv,
#   ep_handle_invoke_deferred, ep_layout_scatter_2d_to_3d,
#   ep_layout_gather_3d_to_2d

import os
from .core import gen_jit_spec, JitSpec
from . import env as jit_env


def _find_nccl():
    """Find NCCL include and lib paths.

    Searches (in order):
      1. NCCL_HOME / NCCL_DIR environment variable
      2. PyTorch's bundled NCCL (torch.utils.cpp_extension)
      3. System paths (/usr/include, /usr/lib)
    """
    # 1. Explicit env var
    for env_key in ("NCCL_HOME", "NCCL_DIR"):
        nccl_home = os.environ.get(env_key, "")
        if nccl_home:
            inc = os.path.join(nccl_home, "include")
            lib = os.path.join(nccl_home, "lib")
            if os.path.isfile(os.path.join(inc, "nccl.h")):
                return inc, lib

    # 2. PyTorch's bundled NCCL
    try:
        import torch
        torch_lib = os.path.dirname(torch.__file__)
        # PyTorch bundles nccl.h under torch/include/nccl.h or
        # in the CUDA toolkit that PyTorch was compiled against
        candidates = [
            os.path.join(torch_lib, "include"),
            os.path.join(torch_lib, "include", "nccl"),
        ]
        for inc in candidates:
            if os.path.isfile(os.path.join(inc, "nccl.h")):
                return inc, os.path.join(torch_lib, "lib")
    except ImportError:
        pass

    # 3. System paths
    for inc in ("/usr/include", "/usr/local/include",
                "/usr/include/nccl", "/usr/local/cuda/include"):
        if os.path.isfile(os.path.join(inc, "nccl.h")):
            lib = inc.replace("include", "lib")
            return inc, lib

    # Fallback — let the compiler find it via default search paths
    return None, None


def gen_ep_module() -> JitSpec:
    """Generate a JitSpec for the core EP module.

    Compiles:
      - csrc/ep/bindings.cu       — full dispatch/combine with NCCL P2P alltoall,
                                     group/handle lifecycle, layout norm kernels
      - csrc/ep/layout_normalize.cu — standalone 2D↔3D kernels

    Links against NCCL (-lnccl) for ncclSend/ncclRecv/ncclGroupStart/End.
    """
    nccl_inc, nccl_lib = _find_nccl()

    extra_includes = []
    extra_ldflags = ["-lnccl"]

    if nccl_inc:
        extra_includes.append(nccl_inc)
    if nccl_lib:
        extra_ldflags.append(f"-L{nccl_lib}")

    return gen_jit_spec(
        "ep",
        [
            jit_env.FLASHINFER_CSRC_DIR / "ep" / "bindings.cu",
            jit_env.FLASHINFER_CSRC_DIR / "ep" / "layout_normalize.cu",
        ],
        extra_include_paths=extra_includes if extra_includes else None,
        extra_ldflags=extra_ldflags,
    )


def gen_ep_deepep_module() -> JitSpec:
    """Generate a JitSpec for the DeepEP backend.

    Compiles csrc/ep/deepep_backend.cu which depends on:
      - deep_ep/buffer.h (from the DeepEP package)
      - NVSHMEM headers

    Only call this if DeepEP is installed.
    """
    import deep_ep

    deep_ep_root = os.path.dirname(deep_ep.__file__)
    deep_ep_include = os.path.join(deep_ep_root, "include")
    if not os.path.isdir(deep_ep_include):
        deep_ep_include = deep_ep_root

    nvshmem_dir = os.environ.get("NVSHMEM_DIR", "")
    nvshmem_include = os.path.join(nvshmem_dir, "include") if nvshmem_dir else ""

    extra_includes = [deep_ep_include]
    if nvshmem_include and os.path.isdir(nvshmem_include):
        extra_includes.append(nvshmem_include)

    return gen_jit_spec(
        "ep_deepep",
        [
            jit_env.FLASHINFER_CSRC_DIR / "ep" / "deepep_backend.cu",
        ],
        extra_include_paths=extra_includes,
    )


def gen_ep_nccl_module() -> JitSpec:
    """Generate a JitSpec for the NCCL-EP Device API backend.

    Compiles csrc/ep/nccl_ep_backend.cu which depends on:
      - nccl_ep.h (from NCCL >= 2.28)

    Only call this if NCCL-EP headers are available.
    """
    nccl_include = os.environ.get("NCCL_INCLUDE_DIR", "")
    extra_includes = []
    if nccl_include and os.path.isdir(nccl_include):
        extra_includes.append(nccl_include)

    return gen_jit_spec(
        "ep_nccl",
        [
            jit_env.FLASHINFER_CSRC_DIR / "ep" / "nccl_ep_backend.cu",
        ],
        extra_include_paths=extra_includes if extra_includes else None,
    )
