"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated chunked LM-head + loss
# programs.  One record per architecture.  A record carries ``arch``, the host
# binding profile ``abi`` (the keyword set its kernels expect, see
# ``cake_backend``), the list of kernel ``stages`` it registers, the
# ``geometry`` the kernels were built for (see ``cake_backend.Geometry``: the
# vocabulary columns per row-statistics partial, the GEMM row tile and CTA
# pair, the K block of the weight-gradient GEMM, the element vector of the
# cast, the divisibility ``V`` / ``H`` / the row stride of ``X`` must satisfy
# and the label element type) and one physical entry per stage (translation
# units, compile flags, FFI entry, argument plan, grid rule, launch geometry
# and closure identity).  Populated verbatim by the generated-program export;
# do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {}

# Kernel stages of one token chunk of the training step, in launch order.
#
# ``gemm_logits``          z_c = bf16(X_c @ W^T) for the chunk's rows plus the
#                          per-(row, vocabulary tile) online (max, sum-exp)
#                          partials of the bf16-rounded logits.
# ``gemm_logits_nostats``  the same GEMM without the statistics (the
#                          recompute of the log-probability backward).
# ``row_finalize``         merges the partials into ``lse`` (every row),
#                          gathers the selected logit, writes ``logp`` (0 on
#                          ignored rows) and, per objective, the per-row
#                          logit-gradient scale ``d`` and loss term.
# ``loss_reduce``          fixed-order sum of the chunk's loss terms into the
#                          loss accumulator (chunk order); the last chunk
#                          writes the finished loss.
# ``row_grad``             dz_c = d_t * (1[v = y_t] - exp(z - lse_t)) in bf16,
#                          in place over ``z_c``; ignored rows become zero.
# ``gemm_dx``              dX_acc[rows] = fp32(dz_c @ W).
# ``gemm_dx_s2`` / ``_s3``  the same GEMM as 2 / 3 K-slice work items per output
#                          tile (slice 0 writes dX_acc, slices >= 1 write FP32
#                          workspace slabs the host adds in fixed order); the
#                          host picks the slice count per chunk from its row
#                          count and the SM count.  Optional (a contiguous
#                          prefix may be registered).
# ``gemm_dw_acc``          dW_acc (=|+=) fp32(dz_c^T @ X_c): store on the first
#                          chunk, accumulate afterwards (chunk order = the
#                          reduction order, no atomics).
# ``scale_cast_bf16``      out = bf16(g * acc) over a flat fp32 accumulator
# ``scale_cast_f32``       out = g * acc (fp32) -- the single output cast of
#                          ``dW`` (either) and ``dX`` (bf16) in the backward.
#
# A record registers the subset its program uses; the host refuses an entry
# point whose stages are missing.
STAGES = (
    "gemm_logits",
    "gemm_logits_nostats",
    "row_finalize",
    "loss_reduce",
    "row_grad",
    "gemm_dx",
    "gemm_dx_s2",
    "gemm_dx_s3",
    "gemm_dw_acc",
    "scale_cast_bf16",
    "scale_cast_f32",
)
GEMM_STAGES = ("gemm_logits", "gemm_logits_nostats", "gemm_dx", "gemm_dx_s2", "gemm_dx_s3", "gemm_dw_acc")
ROW_STAGES = ("row_finalize", "loss_reduce", "row_grad", "scale_cast_bf16", "scale_cast_f32")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?  (SM100 / SM103 only.)"""
    return arch in ARCH_NVCC_FLAGS


def select_module(arch: str) -> str:
    """Return the registered module name for ``arch``."""
    names = [name for name, record in MODULES.items() if record["arch"] == arch]
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one chunked LM-head program: {names}"
        )
    if not names:
        raise NotImplementedError(
            f"The generated chunked LM-head + loss program for {arch} is not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#5680)"
        )
    return names[0]


def registered_stages(name: str) -> tuple[str, ...]:
    """Stages a record registers, in launch order."""
    present = tuple(stage for stage in STAGES if stage in MODULES[name])
    declared = tuple(MODULES[name].get("stages", present))
    if tuple(s for s in STAGES if s in declared) != present:
        raise ValueError(
            f"registry record {name!r} declares stages {declared} but carries {present}"
        )
    return present


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_cake_lm_head_loss_module(name: str, stage: str):
    record = MODULES[name]
    if not toolchain_supports(record["arch"]):
        raise RuntimeError(
            f"generated chunked LM-head program {name!r} targets {record['arch']}, "
            "which this checkout cannot compile"
        )
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{name}_{stage}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *physical["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_lm_head_loss_module(name: str, stage: str):
    return gen_cake_lm_head_loss_module(name, stage).build_and_load()
