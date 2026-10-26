"""JIT loader for the split-sequence affine KDA prefill helpers.

One module holds the fused epilogue (``run``) and the per-launch index
preparation (``index_prep``) of the composite.
"""

from functools import cache
from pathlib import Path

import torch

from . import env as jit_env
from .core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags


def _source_path(relative):
    installed = jit_env.FLASHINFER_CSRC_DIR / Path(relative).relative_to("csrc")
    if installed.is_file():
        return installed
    return Path(__file__).resolve().parents[2] / relative


def arch_for(device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (10, 0):
        return "sm_100a"
    if (major, minor) == (10, 3):
        return "sm_103a"
    raise NotImplementedError(
        f"the fused affine epilogue is built for sm_100a / sm_103a, got sm_{major}{minor}"
    )


@cache
def load(arch: str):
    flags = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]
    source = _source_path("csrc/kda/cake_kda_affine_epilogue.cu")
    include = jit_env.FLASHINFER_INCLUDE_DIR
    if not include.is_dir():
        include = Path(__file__).resolve().parents[2] / "include"
    module = gen_jit_spec(
        name=f"cake_kda_affine_epilogue_{arch}",
        sources=[source],
        extra_cuda_cflags=list(flags),
        extra_include_paths=[_source_path("csrc/tvm_ffi_utils.h").parent, include],
        # The epilogue must reproduce torch's fp32 adds bit for bit; -use_fast_math
        # implies --ftz=true and flushes subnormal sums that torch keeps.
        use_fast_math=False,
    ).build_and_load()
    return module


def load_for_device(device):
    """The loaded module for ``device``: ``.run`` (fused epilogue) and ``.index_prep``."""
    return load(arch_for(device))
