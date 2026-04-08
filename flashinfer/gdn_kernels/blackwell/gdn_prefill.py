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
import cutlass.utils as cutlass_utils
from cutlass.cute.runtime import from_dlpack

from .gated_delta_net_chunked import GatedDeltaNetChunkedKernel


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------


# Keyed on static kernel configuration. Head counts (HQ, HV) are part of
# the key because the tile scheduler and GQA reshape logic bake them in.
@functools.cache
def _get_compiled_cache(
    io_dtype_str: str,
    HQ: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
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
) -> None:
    """Execute the Blackwell chunked GDN prefill kernel.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, 128)`` float16/bfloat16
        k: ``(total_tokens, HK, 128)`` float16/bfloat16
        v: ``(total_tokens, HV, 128)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, forget gate
        beta: ``(total_tokens, HO)`` float32, update gate
        output: ``(total_tokens, HO, 128)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, 128, 128)`` float32, or None
        output_state: ``(num_seqs, HO, 128, 128)`` float32, or None
        scale: attention scale factor (must not be 0)
    """
    HQ = q.size(1)
    HV = v.size(1)
    is_GQA = HQ >= HV
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    io_dtype = _cutlass_io_dtype(q.dtype)

    # Pass states through directly
    _initial_state = initial_state if use_initial_state else None
    B = cu_seqlens.size(0) - 1
    if store_final_state:
        _output_state = output_state
    else:
        _output_state = None

    cache = _get_compiled_cache(
        str(q.dtype), HQ, HV, is_GQA, use_initial_state, store_final_state
    )

    if "compiled" not in cache:
        # --- First call: compile the kernel ---
        hardware_info = cutlass_utils.HardwareInfo()
        num_sm = hardware_info.get_max_active_clusters(1)
        max_active_clusters = hardware_info.get_max_active_clusters(1)

        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=io_dtype,
            acc_dtype=cutlass.Float32,
            mma_tiler_qk=(128, 128, 128),
            mma_tiler_qs=(128, 128, 128),
            mma_tiler_qkv=(128, 128, 128),
            mma_tiler_kv=(128, 128, 128),
            max_active_clusters=max_active_clusters,
            num_sm=num_sm,
            is_GQA=is_GQA,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
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
            s_in_cute.mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2, 3), divisibility=1
            )

        s_out_cute = None
        if store_final_state:
            s_out_cute = from_dlpack(_output_state, assumed_align=16)
            s_out_cute.mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2, 3), divisibility=1
            )

        workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
            num_sm, B, HQ, HV, True
        )
        workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
        workspace_cute = from_dlpack(workspace, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

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
    if "workspace" not in cache or cache["workspace"].size(0) < workspace_size:
        cache["workspace"] = torch.empty(
            workspace_size, dtype=torch.int8, device=q.device
        )
    workspace = cache["workspace"]

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
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
        scale,
        workspace,
        stream,
    )
