"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Gated Delta Net Chunked Prefill - Blackwell SM100 Adapter
==========================================================

Bridges FlashInfer's PyTorch-based ``chunk_gated_delta_rule()`` API to the
CuTe DSL chunked GDN kernel for SM100 (Blackwell).

Follows the same compile-once-cache-and-replay pattern used by the decode
kernels in ``gdn_decode_pretranspose.py``.

State layout: ``[N, H, V, K]``.
"""

import functools
from typing import Optional

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flashinfer.cute_dsl.utils import get_num_sm

from .gated_delta_net_chunked import GatedDeltaNetChunkedKernel


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------


# Keyed on static kernel configuration. Head counts (HQ, HV) are part of
# the key because the tile scheduler and GQA reshape logic bake them in.
@functools.cache
def _get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def _cutlass_io_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    elif torch_dtype == torch.float16:
        return cutlass.Float16
    else:
        raise ValueError(
            f"Unsupported dtype {torch_dtype}, expected bfloat16 or float16"
        )


def _cutlass_state_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.float32:
        return cutlass.Float32
    elif torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    else:
        raise ValueError(
            f"Unsupported state dtype {torch_dtype}, expected float32 or bfloat16"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    scale: float,
    checkpoint_every_n_tokens: int = 0,
    cu_checkpoints: Optional[torch.Tensor] = None,
    output_checkpoints: Optional[torch.Tensor] = None,
) -> None:
    """Execute the Blackwell chunked GDN prefill kernel.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DK)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, forget gate
        beta: ``(total_tokens, HO)`` float32, update gate
        output: ``(total_tokens, HO, DK)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        output_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        scale: attention scale factor (must not be 0)
        checkpoint_every_n_tokens: store intermediate state every N tokens (0 = disabled)
        cu_checkpoints: ``(num_seqs + 1,)`` int32, cumulative checkpoint counts
        output_checkpoints: ``(total_checkpoints, HO, DK, DK)`` float32/bfloat16, or None
    """
    HQ = q.size(1)
    HV = v.size(1)
    DK = q.size(2)
    is_GQA = HQ >= HV
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    enable_checkpoints = checkpoint_every_n_tokens > 0
    io_dtype = _cutlass_io_dtype(q.dtype)

    # Auto-detect state dtype from initial_state, default to float32
    if initial_state is not None:
        state_torch_dtype = initial_state.dtype
    elif output_state is not None:
        state_torch_dtype = output_state.dtype
    else:
        state_torch_dtype = torch.float32
    state_dtype = _cutlass_state_dtype(state_torch_dtype)

    _initial_state = initial_state if use_initial_state else None
    B = cu_seqlens.size(0) - 1
    _output_state = output_state if store_final_state else None

    cache = _get_compiled_cache(
        str(q.dtype),
        str(state_torch_dtype),
        HQ,
        HV,
        is_GQA,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
    )

    if "compiled" not in cache:
        # --- First call: compile the kernel ---
        num_sm = get_num_sm(q.device)
        max_active_clusters = num_sm

        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=io_dtype,
            acc_dtype=cutlass.Float32,
            state_dtype=state_dtype,
            mma_tiler_qk=(64, 64, 128),
            mma_tiler_qs=(128, 64, 128),
            mma_tiler_qkv=(128, 64, 64),
            mma_tiler_kv=(128, 128, 64),
            max_active_clusters=max_active_clusters,
            num_sm=num_sm,
            is_GQA=is_GQA,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            enable_checkpoints=enable_checkpoints,
            is_persistent=True,
        )

        # Convert PyTorch tensors to CuTe tensors for compilation.
        # Token dimension (dim 0) must be dynamic to handle varying seq lengths.
        # Head and head_dim dimensions stay static (part of cache key).
        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        beta_cute = from_dlpack(beta, assumed_align=16)
        beta_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        o_cute = from_dlpack(output, assumed_align=16)
        o_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()

        s_in_cute = None
        if use_initial_state:
            s_in_cute = from_dlpack(_initial_state, assumed_align=16)
            s_in_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )

        s_out_cute = None
        if store_final_state:
            s_out_cute = from_dlpack(_output_state, assumed_align=16)
            s_out_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )

        s_checkpoints_cute = None
        cu_checkpoints_cute = None
        if enable_checkpoints:
            s_checkpoints_cute = from_dlpack(output_checkpoints, assumed_align=16)
            s_checkpoints_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )
            cu_checkpoints_cute = from_dlpack(
                cu_checkpoints, assumed_align=4
            ).mark_layout_dynamic()

        workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
            num_sm, B, HQ, HV, True
        )
        workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
        workspace_cute = from_dlpack(workspace, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)

        compiled = cute.compile(
            gdn,
            q_cute,
            k_cute,
            v_cute,
            gate_cute,
            beta_cute,
            o_cute,
            cu_seqlens_cute,
            s_in_cute,
            s_out_cute,
            s_checkpoints_cute,
            cu_checkpoints_cute,
            checkpoint_every_n_tokens,
            scale,
            workspace_cute,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        )

        cache["compiled"] = compiled
        cache["num_sm"] = num_sm

    # --- Execute ---
    compiled = cache["compiled"]
    num_sm = cache["num_sm"]

    workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
        num_sm, B, HQ, HV, True
    )
    ws_key = f"workspace_{q.device.index}"
    if ws_key not in cache or cache[ws_key].size(0) < workspace_size:
        cache[ws_key] = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
    workspace = cache[ws_key]

    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        _initial_state,
        _output_state,
        output_checkpoints,
        cu_checkpoints,
        checkpoint_every_n_tokens,
        scale,
        workspace,
        stream,
    )
