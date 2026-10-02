"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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
from .jit import _header_dirs

PROGRAM = "cake_nvfp4_sparse_mla_decode"
ARCHES = ("sm_100a", "sm_103a")
# Thread-block cluster sizes (CTAs per query token) the generated program is
# rendered for: one physical module per (architecture, cluster size).  7 is not
# a variant of the program (see ``cake_backend.PLAN_CTAS``).
CLUSTER_SIZES = (1, 2, 3, 4, 5, 6, 8)

# Exact argument plan of every module's ``run`` entry: the kernel's buffers and
# scalar parameters in signature order, then the launch grid.  ``q`` is the
# E4M3 ``[T, 16, 576]`` query viewed as int32 words, ``kv`` the flat uint8 row
# pool, ``indices`` int32 ``[T, topk]``, ``out`` BF16 ``[T, 16, 512]``;
# ``qk_scale`` is ``bmm1_scale * log2(e)`` and ``out_scale`` is ``bmm2_scale``.
ARG_PLAN: list[list[str]] = [
    ["buffer", "q"],
    ["buffer", "kv"],
    ["buffer", "indices"],
    ["buffer", "out"],
    ["parameter", "topk"],
    ["parameter", "keys_per_cta"],
    ["parameter", "qk_scale"],
    ["parameter", "out_scale"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]

# Explicit target-owned registration of the generated program.  One record per
# (architecture, cluster size); each record carries the single physical stage
# (the decode kernel of that cluster size) with its translation units, compile
# flags, FFI entry, argument plan and launch resources (block, cluster,
# cooperative flag, dynamic shared memory).  ``module`` / ``sources`` /
# ``closure_sha256`` are ``None`` / empty until the generated-program export
# fills them in verbatim (see ``csrc/cake_nvfp4_sparse_mla_decode/*/MANIFEST.todo``);
# such a record is not registered.  Do not edit the exported values by hand.
#
# Launch resources are generated-kernel constants: 448 threads (8 math warps,
# 4 softmax warps, 2 loader warps), cluster ``(1, C, 1)`` over the grid's y
# axis, 192256 bytes of dynamic shared memory for ``C = 1`` and 226816 for
# ``C > 1`` (the ``C = 1`` footprint plus the 34560-byte cluster landing zone).
MODULES: dict[str, dict[str, Any]] = {
    "cake_nvfp4_sparse_mla_decode_c1_sm_100a": {
        "arch": "sm_100a",
        "cluster": 1,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 192256,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c2_sm_100a": {
        "arch": "sm_100a",
        "cluster": 2,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 2, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c3_sm_100a": {
        "arch": "sm_100a",
        "cluster": 3,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 3, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c4_sm_100a": {
        "arch": "sm_100a",
        "cluster": 4,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 4, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c5_sm_100a": {
        "arch": "sm_100a",
        "cluster": 5,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 5, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c6_sm_100a": {
        "arch": "sm_100a",
        "cluster": 6,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 6, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c8_sm_100a": {
        "arch": "sm_100a",
        "cluster": 8,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 8, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c1_sm_103a": {
        "arch": "sm_103a",
        "cluster": 1,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 192256,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c2_sm_103a": {
        "arch": "sm_103a",
        "cluster": 2,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 2, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c3_sm_103a": {
        "arch": "sm_103a",
        "cluster": 3,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 3, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c4_sm_103a": {
        "arch": "sm_103a",
        "cluster": 4,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 4, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c5_sm_103a": {
        "arch": "sm_103a",
        "cluster": 5,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 5, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c6_sm_103a": {
        "arch": "sm_103a",
        "cluster": 6,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 6, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
    "cake_nvfp4_sparse_mla_decode_c8_sm_103a": {
        "arch": "sm_103a",
        "cluster": 8,
        "main": {
            "module": None,
            "sources": [],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": ARG_PLAN,
            "closure_sha256": None,
            "tma_workspace_bytes": 0,
            "launch": {
                "block": [448, 1, 1],
                "cluster": [1, 8, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 226816,
            },
        },
        "closure_sha256": None,
    },
}

STAGES = ("main",)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def is_registered(record: dict[str, Any]) -> bool:
    """True once the export filled in the record's physical module and sources."""
    physical = record["main"]
    return bool(physical["module"]) and bool(physical["sources"])


def module_name(arch: str, cluster: int) -> str:
    return f"{PROGRAM}_c{int(cluster)}_{arch}"


def select_module(arch: str, cluster: int) -> str:
    """Registered module name for ``arch`` and ``cluster`` CTAs per token."""
    name = module_name(arch, cluster)
    record = MODULES.get(name)
    if record is None or not is_registered(record):
        raise NotImplementedError(
            f"The generated NVFP4 sparse MLA decode program (cluster size {cluster}) "
            f"for {arch} is not registered in this checkout yet"
        )
    return name


def registered_clusters(arch: str) -> set[int]:
    """Cluster sizes whose generated program is registered for ``arch``."""
    return {
        int(record["cluster"])
        for record in MODULES.values()
        if record["arch"] == arch and is_registered(record)
    }


# ---------------------------------------------------------------------------
# Occupancy helper
# ---------------------------------------------------------------------------
#
# The generated binding launches the kernel (cluster attribute included) but
# exposes no occupancy query.  The planner picks the cluster size whose
# ``num_tokens`` clusters are co-resident in one wave, which the driver decides
# from GPC placement (``cudaOccupancyMaxActiveClusters``), not from the SM count
# alone.  This package adds one small translation unit per module that forwards
# the kernel's host stub to that query with the same configuration the launch
# uses (block, cluster dims, dynamic shared memory, cooperative flag).

_KERNEL_DECLARATION = re.compile(
    r'^extern "C" __global__ void (?P<symbol>kernel_[A-Za-z0-9_]+)\((?P<params>[^;]*)\);\s*$',
    re.MULTILINE,
)

_OCCUPANCY_TEMPLATE = """\
// Occupancy query for the generated cluster kernel {symbol}; rendered by
// cake_jit.py from the kernel declaration of the generated binding.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
def gen_nvfp4_sparse_mla_decode_cake_module(name: str, stage: str):
    record = MODULES[name]
    if not is_registered(record):
        raise NotImplementedError(
            f"{name} is not registered in this checkout yet (no generated sources)"
        )
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    spec_name = f"{name}_{stage}_" + physical["closure_sha256"][:20]
    # Every module (the cluster-size-1 one included) gets the occupancy query so
    # the planner can ask any registered variant for its one-wave capacity.
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
def load_nvfp4_sparse_mla_decode_cake_module(name: str, stage: str):
    return gen_nvfp4_sparse_mla_decode_cake_module(name, stage).build_and_load()
