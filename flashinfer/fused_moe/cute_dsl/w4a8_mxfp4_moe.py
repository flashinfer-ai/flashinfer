# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full W4A8 MXFP4 MoE on Hopper SM90 -- a ``cutlass_fused_moe``-aligned wrapper.

This wraps the lower-level ``w4a8_mxfp4_grouped_gemm`` building block into a complete MoE
whose forward signature mirrors ``flashinfer.fused_moe.cutlass_fused_moe`` (the W4A16
path sglang already uses, e.g. ``Mxfp4FlashinferCutlassMoEMethod``), so a serving layer
can swap to W4A8 with a near-identical call:

    flashinfer.fused_moe.w4a8_mxfp4_moe(
        input=x, token_selected_experts=topk_ids.int(), token_final_scales=topk_weights,
        fc1_expert_weights=w13, fc2_expert_weights=w2, output_dtype=torch.bfloat16,
        quant_scales=[fc1_scale, fc2_scale], output=out)

Internally:
    permute (by expert) -> GEMM1 (gate/up) + fused SwiGLU -> BF16
                        -> per-token FP8 requant -> GEMM2 (down)
                        -> finalize (scatter / moe_reduce) -> output

Differences from the W4A16 cutlass path (W4A8 quantizes the activation to FP8):

- **Input activation**: ``input`` (bf16/fp16) is quantized to FP8 e4m3 with a *per-token*
  scale (``x_fp8 = x / s_t``, ``s_t = row amax / 448``), and GEMM1's fused epilogue
  multiplies ``s_t`` back per row before the SwiGLU math (the scale cannot fold into the
  routing weight across a nonlinearity). This keeps outlier features exact (a direct cast
  hard-clips |x| > 448) and keeps small hidden values out of e4m3's subnormal band. A
  SmoothQuant-style per-channel scale remains a possible refinement for channel outliers.
- **Intermediate requant**: the GEMM1 SwiGLU output is produced in BF16, then requantized
  to FP8 with a *per-token* scale before GEMM2 (matching the reference routed-MoE
  ``fc2_input_scale``). The scale is constant along the GEMM2 contraction, so it folds into
  the routing weight and GEMM2 applies it for free. (For clamped-SwiGLU models the
  intermediate is already bounded, but the per-token scale still recovers the FP8 precision
  an unscaled cast would lose.)
- **GEMM1 weight (``fc1``)** ``[E, 2I, H/2]`` packed MXFP4 with **interleaved gate/up**
  rows (row ``2j`` = gate_j, ``2j+1`` = up_j) so each (gate,up) pair lands in one thread's
  adjacent WGMMA C registers for the fused SwiGLU. (cutlass W4A16 instead uses stacked
  ``[up; gate]`` + its own ``interleave_moe_weights_for_sm90_mixed_gemm`` byte layout, so
  the load-time weight prep differs -- see ``interleave_w4a8_fc1_gate_up`` below.)
- **GEMM2 weight (``fc2``)** ``[E, H, I/2]`` packed MXFP4 (down).
- **Scales**: UE8M0 ``[E, 2I, H/32]`` / ``[E, H, I/32]`` Uint8, ``quant_scales=[s1, s2]``.

Clamped SwiGLU (GPT-OSS / DeepSeek-V4) is supported via ``swiglu_alpha/beta/limit`` (a
uniform per-expert scalar each, matching cutlass "SwiGLUBias"); ``swiglu_limit=None`` keeps
plain ``silu(gate)*up``. Serving layers should pass these as plain Python floats
(extracted once at weight-load time) -- the tensor form costs two device->host syncs per
parameter per call. Not yet supported (raise rather than silently mis-compute): expert
biases; non-uniform per-expert SwiGLU params; TP/EP sharding (caller shards).

Host path: the forward is SYNC-FREE and its per-call host cost is independent of E. All
routing-dependent state -- per-group sizes, operand pointers, and the persistent
scheduler's cluster totals -- is written by one device kernel (``fill_w4a8_moe_meta``)
into a persistent metadata workspace whose cute wrappers are created once, and the
grouped GEMM reads the cluster total from device memory. The host never calls
``.item()``/``.tolist()`` and never loops over experts, so the GEMM launches stream
back-to-back (and the whole forward is CUDA-graph-capturable in principle). The
underlying grouped GEMM compiles once per config (num_groups == E is static; empty
experts are legal M=0 groups).
"""

from typing import List, Optional, Union

import torch

from ...api_logging import flashinfer_api
from ...trace.templates.moe import w4a8_mxfp4_moe_trace
from .w4a8_mxfp4_grouped_gemm_sm90 import (
    w4a8_grouped_gemm_premeta,
    w4a8_select_tile_mn,
)
from .moe_reduce_triton import fill_w4a8_moe_meta, moe_reduce, per_token_quant_fp8

try:
    import cutlass
    import cutlass.torch as cutlass_torch

    _ACC = cutlass.Float32
    _CUTLASS_DTYPE = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
except ImportError:  # pragma: no cover
    cutlass = None
    cutlass_torch = None
    _ACC = None
    _CUTLASS_DTYPE = {}


# Bare @flashinfer_api (no trace=), matching the W4A16 analog
# interleave_moe_weights_for_sm90_mixed_gemm: a one-time load-time weight reshape, not a
# benchmarkable kernel, so it carries the logging decorator but no trace template.
@flashinfer_api
def interleave_w4a8_fc1_gate_up(
    fc1_stacked: torch.Tensor, fc1_scale_stacked: torch.Tensor
) -> "tuple[torch.Tensor, torch.Tensor]":
    """One-time load-time prep: convert a stacked ``[E, 2I, *]`` gate/up weight (rows
    ``[0:I]`` = gate, ``[I:2I]`` = up) into the **interleaved** layout the fused SwiGLU
    needs (row ``2j`` = gate_j, ``2j+1`` = up_j). Applies to both the packed weight
    ``[E, 2I, H/2]`` and its scale ``[E, 2I, H/32]`` (same row permutation)."""
    two_i = fc1_stacked.shape[1]
    assert two_i % 2 == 0
    i = two_i // 2
    perm = torch.empty(two_i, dtype=torch.long, device=fc1_stacked.device)
    perm[0::2] = torch.arange(i, device=fc1_stacked.device)  # gate -> even rows
    perm[1::2] = torch.arange(i, 2 * i, device=fc1_stacked.device)  # up   -> odd rows
    return fc1_stacked[:, perm].contiguous(), fc1_scale_stacked[:, perm].contiguous()


def _check_unsupported(fc1_expert_biases, fc2_expert_biases):
    if fc1_expert_biases is not None or fc2_expert_biases is not None:
        raise NotImplementedError("w4a8_mxfp4_moe does not support expert biases yet")


def _uniform_scalar(t, name):
    """Extract a single scalar from a per-expert SwiGLU param. In practice these are
    uniform across experts (GPT-OSS / DeepSeek-V4 broadcast one value to all experts),
    and the fused epilogue bakes each as one compile-time const, so require uniformity
    rather than silently using expert 0's value.

    A plain Python float/int passes straight through: serving layers should extract
    the scalars ONCE at weight-load time and pass floats, because the tensor path's
    uniformity check + ``.item()`` are device->host syncs -- per MoE call and per
    parameter (3 params x 2 syncs), they were the dominant host stall of the
    otherwise sync-free forward. The tensor path stays for convenience/back-compat
    and is deliberately *not* memoized: caching a content-derived value keyed on the
    buffer address would go stale if the buffer is reused -- a silent wrong result,
    a worse failure mode than the sync."""
    if t is None or isinstance(t, (int, float)):
        return None if t is None else float(t)
    flat = t.reshape(-1)
    if not bool(torch.all(flat == flat[0])):
        raise NotImplementedError(
            f"w4a8_mxfp4_moe requires a uniform per-expert {name} (got non-uniform values)"
        )
    return float(flat[0].item())


# Persistent grouped-GEMM metadata workspaces, keyed on the MoE config (device, E,
# per-GEMM N/K, scatter-or-not). Each holds the sizes/strides/ptrs/cluster-total
# device tensors AND their cute wrappers, both created once; every call rewrites only
# the routing-dependent device CONTENTS (one fill_w4a8_moe_meta launch) and reuses the
# wrappers, so the per-call host cost is independent of E and the launch path issues
# no host->device copies and no syncs. Reusing one workspace across layers is safe on
# a single stream: the fill of call N+1 is stream-ordered after the GEMMs of call N
# (same constraint as the kernel-cache tensormap workspace; concurrent multi-stream
# callers would need per-stream workspaces).
_W4A8_MOE_WS: dict = {}


def _gemm_meta_set(E: int, n: int, k: int, n_ops: int) -> dict:
    """Build one grouped-GEMM metadata set: persistent device tensors + cute wrappers.

    Everything routing-independent is prefilled here once -- the N/K/L size columns
    and the full stride matrix (all groups share (n, k), only M varies). The per-call
    meta-fill kernel rewrites just the M column and the pointer matrix."""
    sizes_cpu = torch.zeros((E, 4), dtype=torch.int32)
    sizes_cpu[:, 1] = n
    sizes_cpu[:, 2] = k
    sizes_cpu[:, 3] = 1
    # Per-operand (row, col) element strides: A [M,K] k-major, B packed Uint8
    # [N,K/2], C [M,N] n-major, scale [N,K/32] (+ per-row [M] vector operands:
    # GEMM1's A-row scale at col 4, or GEMM2's scatter route map / weights).
    stride_row = [(k, 1), (k // 2, 1), (n, 1), (k // 32, 1)]
    stride_row += [(1, 0)] * (n_ops - 4)
    strides_cpu = torch.tensor(stride_row, dtype=torch.int32).repeat(E, 1, 1)
    sizes_cute, sizes_dev = cutlass_torch.cute_tensor_like(
        sizes_cpu, cutlass.Int32, is_dynamic_layout=False, assumed_align=16
    )
    strides_cute, _ = cutlass_torch.cute_tensor_like(
        strides_cpu, cutlass.Int32, is_dynamic_layout=False, assumed_align=16
    )
    ptrs_cute, ptrs_dev = cutlass_torch.cute_tensor_like(
        torch.zeros((E, n_ops), dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    clusters_cute, clusters_dev = cutlass_torch.cute_tensor_like(
        torch.zeros(1, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    return {
        "sizes_cute": sizes_cute,
        "sizes_dev": sizes_dev,
        "strides_cute": strides_cute,
        "ptrs_cute": ptrs_cute,
        "ptrs_dev": ptrs_dev,
        "clusters_cute": clusters_cute,
        "clusters_dev": clusters_dev,
    }


def _moe_meta_ws(
    device: torch.device, E: int, n1: int, k1: int, n2: int, k2: int, n_ops2: int
) -> "tuple[dict, dict]":
    key = (device.index, E, n1, k1, n2, k2, n_ops2)
    ws = _W4A8_MOE_WS.get(key)
    if ws is None:
        with torch.cuda.device(device):
            # GEMM1 carries the per-row A-scale column (n_ops=5, swiglu epilogue).
            ws = (_gemm_meta_set(E, n1, k1, 5), _gemm_meta_set(E, n2, k2, n_ops2))
        _W4A8_MOE_WS[key] = ws
    return ws


@flashinfer_api(trace=w4a8_mxfp4_moe_trace)
def w4a8_mxfp4_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    fc1_expert_biases: Optional[torch.Tensor] = None,
    fc2_expert_biases: Optional[torch.Tensor] = None,
    swiglu_alpha: Optional[Union[torch.Tensor, float]] = None,
    swiglu_beta: Optional[Union[torch.Tensor, float]] = None,
    swiglu_limit: Optional[Union[torch.Tensor, float]] = None,
    dequant_exp_bias: int = 0,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full W4A8 MXFP4 MoE: ``out[t] = sum_k scales[t,k] * down_e(silu(gate_e(x))*up_e(x))``.

    Signature mirrors ``cutlass_fused_moe``. ``input`` is ``[num_tokens, H]`` (bf16/fp16,
    cast to FP8 internally), ``token_selected_experts``/``token_final_scales`` are
    ``[num_tokens, top_k]``, ``fc1_expert_weights`` is ``[E, 2I, H/2]`` packed MXFP4 with
    interleaved gate/up rows (see ``interleave_w4a8_fc1_gate_up``), ``fc2_expert_weights``
    is ``[E, H, I/2]``, and ``quant_scales=[fc1_scale, fc2_scale]`` are UE8M0 Uint8.
    Returns (and optionally writes ``output``) ``[num_tokens, H]`` in ``output_dtype``.
    """
    if cutlass is None:
        raise RuntimeError("w4a8_mxfp4_moe requires the cutlass (CuTe DSL) package")
    c_cutlass = _CUTLASS_DTYPE.get(output_dtype)
    if c_cutlass is None:
        raise ValueError(f"unsupported output_dtype {output_dtype}")
    _check_unsupported(fc1_expert_biases, fc2_expert_biases)
    # Clamped SwiGLU (GPT-OSS / DeepSeek-V4): a uniform per-expert (alpha, beta, limit)
    # passed straight into the fused GEMM1 epilogue. None limit => plain silu(gate)*up.
    alpha = _uniform_scalar(swiglu_alpha, "swiglu_alpha")
    beta = _uniform_scalar(swiglu_beta, "swiglu_beta")
    limit = _uniform_scalar(swiglu_limit, "swiglu_limit")

    dev = input.device
    T, H = input.shape
    # Empty batch (no tokens): skip routing / grouped GEMM / finalize entirely. They all
    # assume at least one active expert (route_maps[0], a 0-group GEMM, and the scatter
    # would each fail), so return an empty [0, H] output directly.
    if T == 0:
        if output is None:
            output = torch.empty(0, H, device=dev, dtype=output_dtype)
        return output
    E = fc1_expert_weights.shape[0]
    I = fc2_expert_weights.shape[2] * 2  # fc2 is [E, H, I/2] packed
    top_k = token_selected_experts.shape[1]
    fc1_scale, fc2_scale = quant_scales

    TK = T * top_k

    # 1. Activation -> FP8 with a PER-TOKEN scale (x_fp8 = x / s_t, s_t = row amax /
    # 448). Replaces the earlier direct saturating cast, which had two real error
    # sources on DSv4: outlier features beyond the e4m3 max were hard-clipped
    # (directional error on exactly the channels that matter), and small hidden
    # values fell into e4m3's subnormal band where mantissa bits drop off (the same
    # failure mode dequant_exp_bias fixes on the weight side). The scale cannot fold
    # into the routing weight (GEMM1 feeds the nonlinear SwiGLU), so GEMM1's epilogue
    # multiplies it back per row BEFORE the SwiGLU math (a_row_scale operand below).
    x_fp8 = torch.empty(T, H, device=dev, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(T, device=dev, dtype=torch.float32)
    per_token_quant_fp8(input, out=x_fp8, scale_out=x_scale)

    # 2. Routing -> permute order (group the (token, slot) pairs by expert), entirely
    # on device. NOTE: counts deliberately avoids torch.bincount, whose CUDA kernel
    # sizes its output via input.max().item() -- a hidden device->host sync. From here
    # to the final output the host never learns anything routing-dependent: the
    # grouped GEMM metadata AND the persistent scheduler's cluster totals are written
    # by a device kernel (fill_w4a8_moe_meta) and read by the GEMM from device memory,
    # so the whole forward is sync-free (and thus CUDA-graph-capturable).
    exp_e = token_selected_experts.reshape(-1).long()  # [TK]
    order = torch.argsort(exp_e, stable=True)  # group (token, slot) pairs by expert
    sorted_tok = order // top_k  # routed row -> source token (flat index // top_k)
    sorted_w = token_final_scales.reshape(-1).float()[order]
    counts = torch.zeros(E, dtype=torch.int32, device=dev)
    counts.index_add_(0, exp_e, torch.ones(TK, dtype=torch.int32, device=dev))

    # One coalesced permute of the activation (and its per-token scale) into routed
    # (group) order; a_scale_perm is GEMM1's per-row scale operand.
    a_perm = x_fp8[sorted_tok]  # [TK, H] FP8
    a_scale_perm = x_scale[sorted_tok]  # [TK] f32

    # 3. All per-call buffers up front -- their base addresses feed the meta-fill
    # kernel below. GEMM1 writes the SwiGLU intermediate in BF16 (not FP8) so it can
    # be requantized per-token before GEMM2 (see the requant step below); c1_fp8 /
    # a_scale / wt_buf are preallocated so the requant + weight-fold write into
    # buffers whose pointers are already in the metadata.
    c1_buf = torch.empty(TK, I, device=dev, dtype=torch.bfloat16)
    c1_fp8 = torch.empty(TK, I, device=dev, dtype=torch.float8_e4m3fn)
    a_scale = torch.empty(TK, device=dev, dtype=torch.float32)
    wt_buf = torch.empty(TK, device=dev, dtype=torch.float32)
    if output is None:
        output = torch.empty(T, H, device=dev, dtype=output_dtype)
    scatter = top_k == 1
    if scatter:
        # Each token written once -> vectorized fused scatter (no atomicAdd) into a
        # plain f32 buffer, cast to output_dtype at the end.
        out_f32 = torch.empty(T, H, device=dev, dtype=torch.float32)
        c2_buf = out_f32
        c2_row_bytes = 0  # shared C base: every group scatters into out_f32
        route_i32 = sorted_tok.to(torch.int32)
        r2_base, w2_base = route_i32.data_ptr(), wt_buf.data_ptr()
    else:
        # top_k >= 2: GEMM2 -> per-expert C rows (routed order) + non-atomic reduce.
        c2_buf = torch.empty(TK, H, device=dev, dtype=torch.float16)
        c2_row_bytes = c2_buf.stride(0) * c2_buf.element_size()
        r2_base = w2_base = 0

    # 4. Metadata: ALL experts are always passed, including empty ones (M=0
    # contributes zero tiles; the scheduler never touches them), keeping
    # num_groups == E static -- ONE compile per config regardless of routing. A
    # single device kernel rewrites the M column, the operand pointers, and the
    # cluster totals of BOTH GEMMs inside the persistent metadata workspace.
    g1, g2 = _moe_meta_ws(dev, E, 2 * I, H, H, I, 6 if scatter else 4)
    tile1 = w4a8_select_tile_mn([2 * I])
    tile2 = w4a8_select_tile_mn([H])
    fill_w4a8_moe_meta(
        counts,
        g1["sizes_dev"],
        g1["ptrs_dev"],
        g1["clusters_dev"],
        g2["sizes_dev"],
        g2["ptrs_dev"],
        g2["clusters_dev"],
        g1_bases=(
            a_perm.data_ptr(),
            fc1_expert_weights.data_ptr(),
            c1_buf.data_ptr(),
            fc1_scale.data_ptr(),
            a_scale_perm.data_ptr(),
        ),
        g1_rows=(
            a_perm.stride(0) * a_perm.element_size(),
            c1_buf.stride(0) * c1_buf.element_size(),
        ),
        g1_experts=(
            fc1_expert_weights.stride(0) * fc1_expert_weights.element_size(),
            fc1_scale.stride(0) * fc1_scale.element_size(),
        ),
        g2_bases=(
            c1_fp8.data_ptr(),
            fc2_expert_weights.data_ptr(),
            c2_buf.data_ptr(),
            fc2_scale.data_ptr(),
            r2_base,
            w2_base,
        ),
        g2_rows=(c1_fp8.stride(0) * c1_fp8.element_size(), c2_row_bytes),
        g2_experts=(
            fc2_expert_weights.stride(0) * fc2_expert_weights.element_size(),
            fc2_scale.stride(0) * fc2_scale.element_size(),
        ),
        tile_m1=tile1[0],
        nn1=(2 * I + tile1[1] - 1) // tile1[1],
        tile_m2=tile2[0],
        nn2=(H + tile2[1] - 1) // tile2[1],
    )

    # 5. GEMM1 + fused SwiGLU -> BF16 [cnt, I] (gate/up interleaved in fc1).
    w4a8_grouped_gemm_premeta(
        E,
        g1["sizes_cute"],
        g1["strides_cute"],
        g1["ptrs_cute"],
        g1["clusters_cute"],
        c_dtype=cutlass.BFloat16,
        acc_dtype=_ACC,
        tile_shape_mn=tile1,
        swiglu=True,
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
        a_row_scale=True,
        dequant_exp_bias=dequant_exp_bias,
    )

    # 5b. Requant the BF16 intermediate to FP8 with a per-token (per-routed-row) scale
    # for the FP8 GEMM2. The scale is constant along the GEMM2 contraction dim, so it
    # pulls out to a per-output-row factor -- fold it straight into the routing weight
    # (sorted_w, same routed order as c1_buf) and GEMM2's scatter / moe_reduce applies
    # it for free, no GEMM2 change. (Matches the reference routed-MoE
    # `fc2_input_scale`; for clamped-SwiGLU models the intermediate is already
    # bounded, but the per-token scale recovers the FP8 precision an unscaled cast
    # would lose.)
    per_token_quant_fp8(c1_buf, out=c1_fp8, scale_out=a_scale)
    torch.mul(sorted_w, a_scale, out=wt_buf)

    # 6. GEMM2 (down) + finalize. GEMM2 input is the requantized FP8 intermediate;
    # the per-token scale is already folded into wt_buf.
    w4a8_grouped_gemm_premeta(
        E,
        g2["sizes_cute"],
        g2["strides_cute"],
        g2["ptrs_cute"],
        g2["clusters_cute"],
        c_dtype=c_cutlass if scatter else cutlass.Float16,
        acc_dtype=_ACC,
        tile_shape_mn=tile2,
        scatter=scatter,
        no_accumulate=scatter,
        dequant_exp_bias=dequant_exp_bias,
    )
    if scatter:
        output.copy_(out_f32)
    else:
        # Group each token's top_k routed rows: argsort(sorted_tok) IS the permuted
        # row index list of build_reduce_index (the concatenated per-group route maps
        # are exactly sorted_tok), computed without materializing per-group lists.
        order2 = torch.argsort(sorted_tok, stable=True)
        permuted_idx = order2.reshape(T, top_k).to(torch.int32)
        topk_scales = wt_buf[order2].reshape(T, top_k)
        moe_reduce(c2_buf, output, permuted_idx, topk_scales, top_k)

    return output
