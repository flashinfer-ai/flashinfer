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

# Backend of flashinfer.dsa_indexer.dsa_indexer_topk (LiteTopK, SM100).

import functools
import importlib.util
from pathlib import Path

import torch

from ...jit.core import gen_jit_spec, sm100a_nvcc_flags
from ...utils import get_compute_capability

_CSRC = Path(__file__).resolve().parent / "csrc"
_NUM_BUCKETS = 256
# DeepGEMM SM100 helpers used by the kernels (DeepGEMM 891d57b4 layout).
_DEEP_GEMM_HEADERS = (
    "common/math.cuh",
    "common/tma_copy.cuh",
    "mma/sm100.cuh",
    "ptx/tcgen05.cuh",
)


def deep_gemm_include() -> Path:
    spec = importlib.util.find_spec("deep_gemm")
    include = (
        None
        if spec is None or spec.origin is None
        else Path(spec.origin).parent / "include"
    )
    if include is None or not all(
        (include / "deep_gemm" / h).is_file() for h in _DEEP_GEMM_HEADERS
    ):
        raise ImportError(
            "dsa_indexer_topk requires the SM100 headers of a recent deep_gemm package"
        )
    return include


@functools.cache
def _module():
    return gen_jit_spec(
        "dsa_indexer_litetopk_sm100a",
        [_CSRC / "dsa_indexer.cu"],
        extra_cuda_cflags=[*sm100a_nvcc_flags, "--expt-extended-lambda"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[_CSRC, deep_gemm_include()],
        # The seed and scan bucket arithmetic must match bit for bit.
        use_fast_math=False,
    ).build_and_load()


def run(q, kv, kv_scales, weights, prefix_logits, cu_end, top_k, cand_cap, out, status):
    # Shapes, dtypes and limits are checked by the launcher.
    if not q.is_cuda or get_compute_capability(q.device) != (10, 0):
        raise RuntimeError(
            "dsa_indexer_topk requires an SM100 (compute capability 10.0) GPU"
        )
    if top_k != 2048:
        raise ValueError("only top_k == 2048 is supported")
    num_q, prefix_len = q.shape[0], prefix_logits.shape[-1]
    i32 = dict(dtype=torch.int32, device=q.device)
    if out is None:
        out = torch.empty(num_q, top_k, **i32)
    if status is None:
        status = torch.empty(num_q, **i32)
    _module().dsa_indexer_topk(
        q,
        kv,
        kv_scales,
        weights,
        prefix_logits,
        torch.full((num_q,), prefix_len, **i32),  # scan start: just past the prefix
        cu_end,
        torch.empty(num_q, dtype=torch.float32, device=q.device),  # bucket origin
        torch.empty(num_q, dtype=torch.float32, device=q.device),  # bucket scale
        torch.empty(num_q, **i32),  # gate bucket
        torch.empty(num_q, _NUM_BUCKETS, **i32),  # prefix histogram
        torch.empty(
            num_q, cand_cap, dtype=torch.int16, device=q.device
        ),  # score low bits
        torch.empty(num_q, cand_cap, **i32),  # KV index and score high bits
        torch.empty(num_q, **i32),  # candidate count
        out,
        status,
    )
    return out, status
