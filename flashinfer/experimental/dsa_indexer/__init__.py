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

# Adapted from vLLM PR #65920 (833266447780a74713c574e67f3b3bf9caa868eb).

import functools
import importlib.util
from pathlib import Path

import torch
from packaging.version import Version

from ...jit.core import gen_jit_spec, sm100f_nvcc_flags
from ...jit.cpp_ext import get_cuda_version
from ...utils import get_compute_capability

_CSRC = Path(__file__).resolve().parent / "csrc"
_WINDOW_ROWS = 4096
_CODES_PER_CTA = 32768
_RECOVERY_FLOATS = 1 << 24  # 64 MiB, independent of the total query count.
_MAX_KEYS = 1 << 20
_DEEP_GEMM_HEADERS = (
    "mma/sm100.cuh",
    "ptx/ld_st.cuh",
    "ptx/tcgen05.cuh",
    "ptx/utils.cuh",
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


def _jit_spec():
    version = get_cuda_version()
    if version < Version("12.9"):
        raise RuntimeError("dsa_indexer_topk requires CUDA 12.9 or newer")
    flags = [*sm100f_nvcc_flags, "-std=c++20", "--expt-extended-lambda"]
    producer_flags = list(flags)
    if version < Version("13.5"):
        # Match the source producer: NVVM rematerialization spills its scan loops.
        producer_flags += ["-Xcicc", "-disable-new-nvvm-remat"]
    return gen_jit_spec(
        "dsa_indexer_litetopk_sampled_sm100f",
        [_CSRC / "dsa_indexer.cu", _CSRC / "fp8_indexer.cu"],
        extra_cuda_cflags=flags,
        extra_cuda_cflags_by_source={_CSRC / "fp8_indexer.cu": producer_flags},
        extra_include_paths=[_CSRC, deep_gemm_include()],
        use_fast_math=False,
    )


@functools.cache
def _module():
    return _jit_spec().build_and_load()


@functools.cache
def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _check_dimensions(num_q, num_heads, seq_kv, cand_cap):
    if not isinstance(num_heads, int) or num_heads not in (32, 64):
        raise ValueError("num_heads must be 32 or 64")
    if not isinstance(num_q, int) or not 0 <= num_q * num_heads <= 2**31 - 1:
        raise ValueError("num_q must be nonnegative and num_q * num_heads < 2**31")
    if not isinstance(seq_kv, int) or not 1 <= seq_kv <= _MAX_KEYS:
        raise ValueError("seq_kv must be in [1, 2**20]")
    if not isinstance(cand_cap, int) or not (
        4096 <= cand_cap <= 12288 and cand_cap % 4 == 0
    ):
        raise ValueError("cand_cap must be in [4096, 12288] and a multiple of 4")


def _check_device(device):
    if device.type != "cuda" or get_compute_capability(device)[0] != 10:
        raise RuntimeError("dsa_indexer_topk requires an SM100-family GPU")


def _check_tensor(tensor, name, device, dtype, shape=None, aligned=False):
    if tensor.device != device or tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype} on {device}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if aligned and tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def allocate_workspace(device, num_q, num_heads, seq_kv, cand_cap):
    _check_dimensions(num_q, num_heads, seq_kv, cand_cap)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _check_device(device)
    rows = min(num_q, _WINDOW_ROWS)
    blocks = (rows + 256 // num_heads - 1) // (256 // num_heads)
    ctas = min(_sm_count(device), 32 * blocks) if rows else 0
    # Eight-row slots also cover the four-row blocks of the 64-head kernel.
    slots = min((rows + 7) // 8, 32, _RECOVERY_FLOATS // (8 * seq_kv))
    empty = functools.partial(torch.empty, device=device)
    return (
        torch.zeros(2 * blocks, dtype=torch.int32, device=device),
        empty(ctas * _CODES_PER_CTA, dtype=torch.int32),
        empty((rows, cand_cap), dtype=torch.float32),
        empty((rows, cand_cap), dtype=torch.int32),
        empty((3, rows), dtype=torch.int32),
        empty(slots * 8 * seq_kv, dtype=torch.float32),
    )


def _check_workspace(workspace, device, rows, heads, keys, cand_cap):
    if not isinstance(workspace, (tuple, list)) or len(workspace) != 6:
        raise ValueError(
            "workspace must contain the six dsa_indexer_topk workspace tensors"
        )
    counter, codes, values, indices, row_state, recovery = workspace
    names = ("counter", "codes", "values", "indices", "row_state", "recovery")
    dtypes = (
        torch.int32,
        torch.int32,
        torch.float32,
        torch.int32,
        torch.int32,
        torch.float32,
    )
    for tensor, name, dtype in zip(workspace, names, dtypes, strict=False):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"workspace {name} must be a tensor")
        _check_tensor(
            tensor,
            f"workspace {name}",
            device,
            dtype,
            aligned=name in ("codes", "values", "indices"),
        )
    blocks = (rows + 256 // heads - 1) // (256 // heads)
    ctas = min(_sm_count(device), 32 * blocks) if rows else 0
    if counter.ndim != 1 or counter.numel() < 2 * blocks:
        raise ValueError("workspace counter is too small or not one-dimensional")
    if codes.ndim != 1 or codes.numel() < ctas * _CODES_PER_CTA:
        raise ValueError("workspace codes is too small or not one-dimensional")
    for tensor, name in ((values, "values"), (indices, "indices")):
        if tensor.ndim != 2 or tensor.shape[0] < rows or tensor.shape[1] != cand_cap:
            raise ValueError(
                f"workspace {name} must have at least {rows} rows and {cand_cap} columns"
            )
    if row_state.ndim != 2 or row_state.shape[0] != 3 or row_state.shape[1] < rows:
        raise ValueError(f"workspace row_state must have shape [3, at least {rows}]")
    if recovery.ndim != 1 or (rows and recovery.numel() < (256 // heads) * keys):
        raise ValueError("workspace recovery is too small or not one-dimensional")


def run(q, kv, kv_scales, weights, cu_end, top_k, cu_start, cand_cap, out, workspace):
    if q.ndim != 3 or q.shape[-1] != 128:
        raise ValueError("q must have shape [num_q, 32 or 64, 128]")
    if kv.ndim != 2 or kv.shape[1] != 128:
        raise ValueError("kv must have shape [seq_kv, 128]")
    rows, heads, keys = q.shape[0], q.shape[1], kv.shape[0]
    _check_dimensions(rows, heads, keys, cand_cap)
    if not isinstance(top_k, int) or top_k not in (512, 1024, 2048):
        raise ValueError("top_k must be 512, 1024 or 2048")
    device = q.device
    _check_device(device)
    _check_tensor(q, "q", device, torch.float8_e4m3fn, aligned=True)
    _check_tensor(kv, "kv", device, torch.float8_e4m3fn, aligned=True)
    _check_tensor(kv_scales, "kv_scales", device, torch.float32, (keys,), aligned=True)
    _check_tensor(
        weights, "weights", device, torch.float32, (rows, heads), aligned=True
    )
    _check_tensor(cu_end, "cu_end", device, torch.int32, (rows,))
    if cu_start is not None:
        _check_tensor(cu_start, "cu_start", device, torch.int32, (rows,))
    if out is not None:
        _check_tensor(out, "out", device, torch.int32, (rows, top_k), aligned=True)
    if workspace is not None:
        _check_workspace(
            workspace, device, min(rows, _WINDOW_ROWS), heads, keys, cand_cap
        )
    if out is None:
        out = torch.empty((rows, top_k), device=device, dtype=torch.int32)
    if not rows:
        return out
    if workspace is None:
        workspace = allocate_workspace(device, rows, heads, keys, cand_cap)
    if cu_start is None:
        cu_start = torch.zeros(rows, device=device, dtype=torch.int32)
    module = _module()
    counter, codes, values, indices, row_state, recovery = workspace
    # Each op resets the counters. Reuse scratch in stream order, including the
    # final partial window, whose row_state must have a compact 3 x n layout.
    for start in range(0, rows, _WINDOW_ROWS):
        n = min(_WINDOW_ROWS, rows - start)
        window = slice(start, start + n)
        module.dsa_indexer_topk(
            q[window],
            kv,
            kv_scales,
            weights[window],
            cu_start[window],
            cu_end[window],
            out[window],
            counter,
            codes,
            values[:n],
            indices[:n],
            row_state.view(-1)[: 3 * n].view(3, n),
            recovery,
        )
    return out
