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

- **Input activation**: ``input`` (bf16/fp16) is cast straight to FP8 e4m3 -- *no* per-token
  input scale. A direct cast is exact for in-range (post-norm, O(1)) inputs; outlier-heavy
  models would want a SmoothQuant-style per-channel scale (a follow-up).
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
plain ``silu(gate)*up``. Not yet supported (raise rather than silently mis-compute): expert
biases; non-uniform per-expert SwiGLU params; TP/EP sharding (caller shards).

Perf caveat: per-expert token counts vary with the routing, so the per-call problem sizes
are dynamic and the underlying grouped GEMM recompiles when they change. For static-shape
serving, pad each expert to a fixed capacity (a follow-up); correctness is unaffected.
"""

from typing import List, Optional

import torch

from ...api_logging import flashinfer_api
from ...trace.templates.moe import w4a8_mxfp4_moe_trace
from .w4a8_mxfp4_grouped_gemm_sm90 import w4a8_mxfp4_grouped_gemm
from .moe_reduce_triton import moe_reduce, build_reduce_index, per_token_quant_fp8

try:
    import cutlass

    _ACC = cutlass.Float32
    _CUTLASS_DTYPE = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
except ImportError:  # pragma: no cover
    cutlass = None
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
    """Extract a single scalar from a per-expert SwiGLU param tensor. In practice these
    are uniform across experts (GPT-OSS / DeepSeek-V4 broadcast one value to all experts),
    and the fused epilogue bakes each as one compile-time const, so require uniformity
    rather than silently using expert 0's value.

    Deliberately *not* memoized: the uniformity check + ``.item()`` force a small
    device->host sync, but it only runs on the clamped-SwiGLU path and is dwarfed by the
    routing-side syncs (``bincount(...).tolist()``, ``argsort``). Caching the extracted
    *value* keyed on the buffer address would go stale if the buffer's contents change or
    the allocator reuses the storage -- a silent wrong result, a worse failure mode than
    the tiny sync. (Unlike ``_W4A8_META_CACHE``, which caches pointer wrappers whose data
    is re-read at launch, this would cache a content-derived value.)"""
    if t is None:
        return None
    flat = t.reshape(-1)
    if not bool(torch.all(flat == flat[0])):
        raise NotImplementedError(
            f"w4a8_mxfp4_moe requires a uniform per-expert {name} (got non-uniform values)"
        )
    return float(flat[0].item())


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
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
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

    # 1. Activation -> FP8 (no per-token scale; see module docstring).
    x_fp8 = input.to(torch.float8_e4m3fn)

    # 2. Routing -> permute order (group the (token, slot) pairs by expert).
    exp_e = token_selected_experts.reshape(-1).long()  # [T*top_k]
    exp_tok = torch.arange(T, device=dev).repeat_interleave(top_k)  # [T*top_k]
    exp_w = token_final_scales.reshape(-1).float()  # [T*top_k]
    order = torch.argsort(exp_e, stable=True)  # group by expert
    sorted_tok = exp_tok[order]
    sorted_w = exp_w[order]
    counts = torch.bincount(exp_e, minlength=E).tolist()  # tokens per expert
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)

    # One coalesced permute of the activation into routed (group) order.
    a_perm = x_fp8[sorted_tok]  # [T*top_k, H] FP8

    active = [e for e in range(E) if counts[e] > 0]
    a_list, w1_list, s1_list, ps1, c1_list = [], [], [], [], []
    # GEMM1 writes the SwiGLU intermediate in BF16 (not FP8) so it can be requantized
    # per-token before GEMM2 (see the requant step below).
    c1_buf = torch.empty(T * top_k, I, device=dev, dtype=torch.bfloat16)
    for e in active:
        off, cnt = offsets[e], counts[e]
        a_list.append(a_perm[off : off + cnt])
        w1_list.append(fc1_expert_weights[e])
        s1_list.append(fc1_scale[e])
        c1_list.append(c1_buf[off : off + cnt])
        ps1.append((cnt, 2 * I, H, 1))

    # 3. GEMM1 + fused SwiGLU -> BF16 [cnt, I] (gate/up interleaved in fc1).
    w4a8_mxfp4_grouped_gemm(
        a_list,
        w1_list,
        s1_list,
        c1_list,
        ps1,
        acc_dtype=_ACC,
        c_dtype=cutlass.BFloat16,
        swiglu=True,
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
    )

    # 3b. Requant the BF16 intermediate to FP8 with a per-token (per-routed-row) scale
    # for the FP8 GEMM2. The scale is constant along the GEMM2 contraction dim, so it
    # pulls out to a per-output-row factor -- fold it straight into the routing weight
    # (sorted_w, same routed order as c1_buf) and GEMM2's scatter / moe_reduce applies it
    # for free, no GEMM2 change. (Matches the reference routed-MoE `fc2_input_scale`;
    # for clamped-SwiGLU models the intermediate is already bounded, but the per-token
    # scale recovers the FP8 precision an unscaled cast would lose.)
    c1_fp8, a_scale = per_token_quant_fp8(c1_buf)  # [T*top_k, I] fp8, [T*top_k] f32
    sorted_w = sorted_w * a_scale

    # 4. GEMM2 (down) + finalize. GEMM2 input is the requantized FP8 intermediate; the
    # per-token scale is already folded into sorted_w (-> wt_list below).
    c1q_list = [c1_fp8[offsets[e] : offsets[e] + counts[e]] for e in active]
    w2_list = [fc2_expert_weights[e] for e in active]
    s2_list = [fc2_scale[e] for e in active]
    ps2 = [(counts[e], H, I, 1) for e in active]
    route_list = [
        sorted_tok[offsets[e] : offsets[e] + counts[e]].to(torch.int32) for e in active
    ]
    wt_list = [
        sorted_w[offsets[e] : offsets[e] + counts[e]].contiguous() for e in active
    ]

    if output is None:
        output = torch.empty(T, H, device=dev, dtype=output_dtype)

    if top_k == 1:
        # Each token written once -> vectorized fused scatter (no atomicAdd).
        out_f32 = torch.empty(T, H, device=dev, dtype=torch.float32)
        w4a8_mxfp4_grouped_gemm(
            c1q_list,
            w2_list,
            s2_list,
            None,
            ps2,
            acc_dtype=_ACC,
            c_dtype=c_cutlass,
            route_maps=route_list,
            weights=wt_list,
            output=out_f32,
            no_accumulate=True,
        )
        output.copy_(out_f32)
    else:
        # top_k >= 2: GEMM2 -> per-expert C (routed order) + non-atomic moe_reduce.
        c2_buf = torch.empty(T * top_k, H, device=dev, dtype=torch.float16)
        c2_list = [c2_buf[offsets[e] : offsets[e] + counts[e]] for e in active]
        w4a8_mxfp4_grouped_gemm(
            c1q_list,
            w2_list,
            s2_list,
            c2_list,
            ps2,
            acc_dtype=_ACC,
            c_dtype=cutlass.Float16,
        )
        permuted_idx, topk_scales = build_reduce_index(route_list, wt_list, T, top_k)
        moe_reduce(c2_buf, output, permuted_idx, topk_scales, top_k)

    return output
