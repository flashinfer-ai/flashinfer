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
import os
import re
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
from ...jit.utils import write_if_different

# Explicit target-owned registration of the generated program.  One record per
# (architecture, dispatch arm, block_m); each record carries the single
# physical stage (the fused router kernel of that arm) with its own
# translation units, compile flags, FFI entry, argument plan and launch
# resources (block, cluster, cooperative flag, dynamic shared memory).
# Populated verbatim by the generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {}

STAGES = ("main",)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}

# Dispatch arms of the generated program.  The per-shape route tables in
# ``cake_backend`` name one arm per (architecture, num_tokens, block_m):
#   L   : one-join plan builder, num_tokens <= 512, at least 128 CTAs launched
#   LC  : one cluster of num_tokens CTAs (2, 4 or 8) exchanging selected ids
#         through distributed shared memory; one kernel per row count,
#         non-cooperative cluster launch
#   M   : one-join bitmap plan builder, num_tokens = 256
#   Q4S : arm M's cp.async ID stream in 4-CTA clusters (cooperative cluster
#         launch), compiled with __launch_bounds__(224, 4): four CTAs per SM,
#         512 <= num_tokens <= 2048
#   G   : two-join persistent kernel for the largest batches, compiled with
#         per-architecture launch bounds (4 CTAs/SM on SM100, 6 on SM103)
ARMS = ("L", "LC", "M", "Q4S", "G")
# Arms registered per row count (one kernel per num_tokens); every other arm
# registers one module per (arch, block_m) and serves all of its rows.
PER_ROW_COUNT_ARMS = ("LC",)


def module_num_tokens(arm: str, num_tokens: int):
    """``num_tokens`` key of the module serving ``arm`` (``None`` for shared kernels)."""
    return int(num_tokens) if arm in PER_ROW_COUNT_ARMS else None


def select_module(arch: str, arm: str, block_m: int, num_tokens=None) -> str:
    """Registered module name for ``arch``, dispatch ``arm``, ``block_m`` (and, for
    per-row-count arms, ``num_tokens``)."""
    for name, record in MODULES.items():
        if (
            record["arch"] == arch
            and record["arm"] == arm
            and int(record["block_m"]) == int(block_m)
            and record.get("num_tokens") == num_tokens
        ):
            return name
    rows = "" if num_tokens is None else f", num_tokens {num_tokens}"
    raise NotImplementedError(
        f"The generated Kimi-K3 fused router program (arm {arm}, block_m {block_m}{rows}) "
        f"for {arch} is not registered in this checkout yet"
    )


def registered_programs(arch: str) -> set[tuple[str, int, Any]]:
    """``{(arm, block_m, num_tokens_or_None)}`` registered for ``arch``."""
    return {
        (record["arm"], int(record["block_m"]), record.get("num_tokens"))
        for record in MODULES.values()
        if record["arch"] == arch
    }


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


# ---------------------------------------------------------------------------
# Occupancy helper for cluster-launched arms
# ---------------------------------------------------------------------------
#
# The generated binding launches the kernel (cooperative and cluster attributes
# included) but exposes no occupancy query.  Arm Q's persistent grid must be
# bounded by the number of co-resident 4-CTA clusters the driver admits (GPC
# placement, not just CTAs per SM), so this package adds one small translation
# unit per cluster module that forwards the kernel's host stub to
# ``cudaOccupancyMaxActiveClusters`` with the same one-cluster configuration the
# launch uses (block, cluster dims, dynamic shared memory, cooperative flag).

_KERNEL_DECLARATION = re.compile(
    r'^extern "C" __global__ void (?P<symbol>kernel_[A-Za-z0-9_]+)\((?P<params>[^;]*)\);\s*$',
    re.MULTILINE,
)

_OCCUPANCY_TEMPLATE = """\
// Occupancy query for the generated cluster kernel {symbol}; rendered by
// cake_jit.py from the kernel declaration of the generated binding.
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <cstdint>

{declaration}

namespace {{

int64_t MaxActiveClusters(int64_t device, int64_t block_x, int64_t block_y, int64_t block_z,
                          int64_t cluster_x, int64_t cluster_y, int64_t cluster_z,
                          int64_t dynamic_smem_bytes, bool cooperative) {{
  int previous_device = -1;
  TVM_FFI_CHECK(cudaGetDevice(&previous_device) == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed";
  if (static_cast<int64_t>(previous_device) != device) {{
    TVM_FFI_CHECK(cudaSetDevice(static_cast<int>(device)) == cudaSuccess, RuntimeError)
        << "cudaSetDevice failed";
  }}
  const void* function = reinterpret_cast<const void*>(&{symbol});
  if (dynamic_smem_bytes > 48 * 1024) {{
    TVM_FFI_CHECK(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(dynamic_smem_bytes)) == cudaSuccess,
                  RuntimeError)
        << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed for {symbol}";
  }}
  cudaLaunchAttribute attrs[2]{{}};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = static_cast<unsigned>(cluster_x);
  attrs[n].val.clusterDim.y = static_cast<unsigned>(cluster_y);
  attrs[n].val.clusterDim.z = static_cast<unsigned>(cluster_z);
  ++n;
  if (cooperative) {{
    attrs[n].id = cudaLaunchAttributeCooperative;
    attrs[n].val.cooperative = 1;
    ++n;
  }}
  cudaLaunchConfig_t config{{}};
  config.gridDim = dim3(static_cast<unsigned>(cluster_x), static_cast<unsigned>(cluster_y),
                        static_cast<unsigned>(cluster_z));
  config.blockDim = dim3(static_cast<unsigned>(block_x), static_cast<unsigned>(block_y),
                         static_cast<unsigned>(block_z));
  config.dynamicSmemBytes = static_cast<size_t>(dynamic_smem_bytes);
  config.attrs = attrs;
  config.numAttrs = static_cast<unsigned>(n);
  int clusters = 0;
  cudaError_t status = cudaOccupancyMaxActiveClusters(&clusters, function, &config);
  if (static_cast<int64_t>(previous_device) != device) {{
    cudaSetDevice(previous_device);
  }}
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaOccupancyMaxActiveClusters for {symbol} failed: " << cudaGetErrorString(status);
  return static_cast<int64_t>(clusters);
}}

}}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(max_active_clusters, MaxActiveClusters);
"""


def _kernel_declaration(binding_path: Path) -> tuple[str, str]:
    """``(symbol, declaration)`` of the kernel declared by a generated binding."""
    matches = _KERNEL_DECLARATION.findall(binding_path.read_text())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one generated kernel declaration in {binding_path}, "
            f"found {len(matches)}"
        )
    symbol, params = matches[0]
    return symbol, f'extern "C" __global__ void {symbol}({params});'


def uses_cluster_launch(record: dict[str, Any]) -> bool:
    launch = record["main"]["launch"]
    return any(int(dim) > 1 for dim in launch["cluster"])


def _occupancy_source(spec_name: str, binding_path: Path) -> Path:
    symbol, declaration = _kernel_declaration(binding_path)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / spec_name
    os.makedirs(gen_directory, exist_ok=True)
    path = gen_directory / f"{symbol}_occupancy.cu"
    write_if_different(
        path, _OCCUPANCY_TEMPLATE.format(symbol=symbol, declaration=declaration)
    )
    return path


@functools.cache
def gen_kimi_k3_fused_router_module(name: str, stage: str):
    record = MODULES[name]
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    spec_name = f"{name}_{stage}_" + physical["closure_sha256"][:20]
    if uses_cluster_launch(record):
        binding = next(path for path in sources if path.name.endswith("_binding.cu"))
        sources.append(_occupancy_source(spec_name, binding))
    return gen_jit_spec(
        name=spec_name,
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
def load_kimi_k3_fused_router_module(name: str, stage: str):
    return gen_kimi_k3_fused_router_module(name, stage).build_and_load()
