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

# Launch adapter for the Cake-generated CuTe DSL programs under csrc/: DSA
# indexer decode logits over an FP8 paged key cache whose per-token scales are
# UE8M0 exponents. One program per (next_n, architecture), compiled for the
# device's exact architecture through FlashInfer's CuTe DSL JIT cache. The
# programs import only CUTLASS's CuTe DSL and cuda-python.

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch
import tvm_ffi

from ...jit import cute_dsl_core
from ...utils import get_compute_capability, get_device_sm_count
from ..cake_fp8_ue8m0_paged_mqa_logits_common import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    check_launcher,
    import_program,
    on_device,
    program_arch,
    program_files,
)

HEADS = 32
HEAD_DIM = 128
PAGE = 64  # tokens per KV block
TOKEN_BYTES = HEAD_DIM + 4  # a token's FP8 values plus its 4-byte scale slot
BLOCK_BYTES = PAGE * TOKEN_BYTES
NEXT_NS = (1, 2, 4)  # the generated FP8 MMA tiles are N = 32, 64 and 128
_MODULE = "cake_fp8_ue8m0_paged_mqa_logits_decode"
_CSRC = Path(__file__).resolve().parent / "csrc"
# The generated launcher's parameters, in the order `_launch` binds them; its
# stream is the TVM-FFI environment stream.
_LAUNCH_PARAMETERS = (
    "q_tma",
    "_loom_tma_q_tma_dim_0",
    "_loom_tma_q_tma_dim_1",
    "_loom_tma_q_tma_stride16_0",
    "kv_tma",
    "_loom_tma_kv_tma_dim_0",
    "_loom_tma_kv_tma_dim_1",
    "_loom_tma_kv_tma_dim_2",
    "_loom_tma_kv_tma_stride16_0",
    "_loom_tma_kv_tma_stride16_1",
    "sf_tma",
    "_loom_tma_sf_tma_dim_0",
    "_loom_tma_sf_tma_dim_1",
    "_loom_tma_sf_tma_stride16_0",
    "weights",
    "block_table",
    "seq_lens",
    "out",
    "block_table_stride",
    "out_cols",
    "grid_x",
    "grid_y",
    "grid_z",
    "stream",
)


def _source(next_n: int, arch: str) -> Path:
    return _CSRC / f"dsa_indexer_decode_logits_n{next_n}_{arch}.py"


@functools.cache
def _kernel(device_index: int, next_n: int) -> Any:
    path = _source(next_n, program_arch(device_index))
    module = import_program(path, __name__)
    check_launcher(path, module.launch_dsa_indexer_decode_logits, _LAUNCH_PARAMETERS)
    with torch.cuda.device(device_index):
        return cute_dsl_core.build_and_load_cute_dsl_kernel(
            _MODULE,
            f"dsa_indexer_decode_logits_n{next_n}",
            module.compile_program,
            extra_key_files=program_files(_CSRC, "dsa_indexer_decode_logits_*.py"),
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"fp8_ue8m0_paged_mqa_logits_decode: {message}")


def check_inputs(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: torch.Tensor | None,
) -> None:
    """Raise ``ValueError`` unless the generated programs serve this call.

    Host metadata only (shapes, dtypes, strides, devices, alignment): nothing
    here reads device memory, so the check is safe under CUDA-graph capture.
    """
    _require(
        q.dim() == 4 and tuple(q.shape[2:]) == (HEADS, HEAD_DIM),
        f"q must be [batch_size, next_n, {HEADS}, {HEAD_DIM}]; got {tuple(q.shape)}",
    )
    batch, next_n = q.shape[:2]
    _require(
        next_n in NEXT_NS, f"next_n = q.shape[1] must be one of {NEXT_NS}; got {next_n}"
    )
    _require(
        q.dtype == torch.float8_e4m3fn and q.is_contiguous(),
        f"q must be contiguous float8_e4m3fn; got {q.dtype}",
    )
    _require(q.is_cuda, "q must be on a CUDA device")
    capability = get_compute_capability(q.device)
    _require(
        capability in SUPPORTED_COMPUTE_CAPABILITIES,
        f"needs compute capability in {SUPPORTED_COMPUTE_CAPABILITIES}; "
        f"{q.device} has {capability}",
    )
    _require(
        kv_fused.dtype == torch.uint8
        and kv_fused.dim() == 4
        and tuple(kv_fused.shape[1:]) == (PAGE, 1, TOKEN_BYTES)
        and kv_fused.is_contiguous(),
        f"kv_fused must be contiguous uint8 [num_blocks, {PAGE}, 1, {TOKEN_BYTES}]; "
        f"got {kv_fused.dtype} {tuple(kv_fused.shape)}",
    )
    _require(kv_fused.shape[0] > 0, "kv_fused must hold at least one block")
    _require(
        weights.dtype == torch.float32
        and tuple(weights.shape) == (batch * next_n, HEADS)
        and weights.is_contiguous(),
        f"weights must be contiguous float32 [{batch * next_n}, {HEADS}]; got "
        f"{weights.dtype} {tuple(weights.shape)}",
    )
    _require(
        block_tables.dtype == torch.int32
        and block_tables.dim() == 2
        and block_tables.shape[0] == batch
        and block_tables.stride(1) == 1,
        f"block_tables must be int32 [{batch}, max_blocks_per_seq] with unit column "
        f"stride; got {block_tables.dtype} {tuple(block_tables.shape)} strides "
        f"{block_tables.stride()}",
    )
    _require(
        batch == 0
        or (batch - 1) * block_tables.stride(0) + block_tables.shape[1] < 2**31,
        "block_tables spans 2**31 or more entries; the programs index it with int32",
    )
    _require(
        seq_lens.dtype == torch.int32
        and tuple(seq_lens.shape) == (batch,)
        and seq_lens.is_contiguous(),
        f"seq_lens must be contiguous int32 [{batch}]; got {seq_lens.dtype} "
        f"{tuple(seq_lens.shape)}",
    )
    _require(
        isinstance(max_seq_len, int) and 0 <= max_seq_len < 2**31,
        f"max_seq_len must be an int in [0, 2**31); got {max_seq_len!r}",
    )
    tensors = [q, kv_fused, weights, block_tables, seq_lens]
    if out is not None:
        _require(
            out.dtype == torch.float32
            and out.dim() == 2
            and out.is_contiguous()
            and out.shape[0] >= batch * next_n
            and max_seq_len <= out.shape[1] < 2**31,
            f"out must be contiguous float32 with at least [{batch * next_n}, "
            f"{max_seq_len}] elements; got {out.dtype} {tuple(out.shape)} strides "
            f"{out.stride()}",
        )
        tensors.append(out)
    _require(
        all(tensor.device == q.device for tensor in tensors),
        f"every tensor must be on {q.device}",
    )
    _require(
        all(tensor.data_ptr() % 16 == 0 for tensor in tensors),
        "every tensor's data must be 16-byte aligned",
    )


def paged_mqa_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """See :func:`flashinfer.attn_scores.fp8_ue8m0_paged_mqa_logits_decode`."""
    check_inputs(q, kv_fused, weights, block_tables, seq_lens, max_seq_len, out)
    batch, next_n = q.shape[:2]
    rows = batch * next_n
    if out is None:
        out = torch.empty(rows, max_seq_len, dtype=torch.float32, device=q.device)
    logits = out[:rows, :max_seq_len]
    if batch == 0 or max_seq_len == 0:
        return logits
    device_index = q.device.index
    num_blocks = kv_fused.shape[0]
    kv_bytes = kv_fused.view(-1)
    # Request b's pages start at entry b * stride(0) of this flat view.
    table = block_tables.as_strided(
        ((batch - 1) * block_tables.stride(0) + block_tables.shape[1],), (1,)
    )
    with on_device(device_index):
        kernel = _kernel(device_index, next_n)
        with tvm_ffi.use_torch_stream():
            kernel(
                # Q as bytes [rows * HEADS, HEAD_DIM]; a box is one request's rows.
                q.view(torch.uint8).view(-1),
                HEAD_DIM,
                rows * HEADS,
                HEAD_DIM // 16,
                # A block's values as [num_blocks, PAGE, HEAD_DIM] bytes.
                kv_bytes,
                HEAD_DIM,
                PAGE,
                num_blocks,
                HEAD_DIM // 16,
                BLOCK_BYTES // 16,
                # Blocks as byte rows: the scale slots are a box inside each row.
                kv_bytes,
                BLOCK_BYTES,
                num_blocks,
                BLOCK_BYTES // 16,
                weights.view(-1),
                table,
                seq_lens,
                out.view(-1),
                block_tables.stride(0),
                out.shape[1],
                # One request per grid row; its CTAs split its tiles on the device.
                batch,
                max(1, get_device_sm_count(q.device) // batch),
                1,
            )
    return logits
