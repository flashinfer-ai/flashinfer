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

Tests for the SM90 MoE GEMM1 (gather + grouped GEMM + SiLU gating).
"""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.utils import get_compute_capability

cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="Requires cute-dsl (nvidia-cutlass-dsl)"
)


def is_sm90():
    if not torch.cuda.is_available():
        return False
    return get_compute_capability(torch.device("cuda"))[0] == 9


sm90_required = pytest.mark.skipif(not is_sm90(), reason="Requires SM90 (Hopper) GPU")


def make_random_topk_ids(num_experts, num_tokens, top_k, device="cuda"):
    return torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).int()


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize(
    "n2,k,tile_shape_mn",
    [
        # Qwen3-30B-A3B GEMM1: N2 = 2*768/tp interleaved up/gate, K = 2048
        (1536, 2048, (128, 128)),  # tp=1
        (1536, 2048, (128, 256)),  # tp=1, 2 MMA warpgroups (reg-budget fixed)
        (768, 2048, (64, 128)),  # tp=2, tile_m=64
        (1536, 2048, (64, 256)),  # tp=1, tile_m=64 with a 256-wide N tile
        (192, 2048, (128, 192)),  # tp=8, 2-WG small-N tile
        (192, 2048, (128, 64)),  # tp=8 (tile N=64)
        (384, 2048, (128, 128)),  # tp=4
    ],
)
@pytest.mark.parametrize("num_tokens", [3, 777])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_bf16_gather_grouped_gemm(n2, k, tile_shape_mn, num_tokens, dtype):
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
        sm90_contiguous_gather_grouped_gemm_act_fusion,
    )

    torch.manual_seed(0)
    num_experts, top_k = 128, 8
    tile_m = tile_shape_mn[0]
    inter = n2 // 2

    ids = make_random_topk_ids(num_experts, num_tokens, top_k)
    scales = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _total_padded,
        num_non_exiting_tiles,
        _,
    ) = moe_sort(
        ids, scales, num_experts=num_experts, top_k=top_k, tile_tokens_dim=tile_m
    )
    permuted_m = tile_idx_to_expert_idx.numel() * tile_m

    x = torch.randn(num_tokens, k, device="cuda", dtype=dtype) / (k**0.25)
    # vLLM-style [gate; up] concatenated weights -> 32-col interleave.
    w_gate_up = torch.randn(num_experts, n2, k, device="cuda", dtype=dtype) / (k**0.25)
    w1 = interleave_up_gate_sm90(w_gate_up)

    # NaN-poisoned output: a valid row the kernel fails to write is caught.
    out = torch.full((permuted_m, inter), float("nan"), device="cuda", dtype=dtype)
    out = sm90_contiguous_gather_grouped_gemm_act_fusion(
        x,
        w1,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        out=out,
        topk=top_k,
        permuted_m=permuted_m,
        tile_shape_mn=tile_shape_mn,
    )
    assert out.shape == (permuted_m, inter)

    # Reference on valid rows only: silu(x @ gate.T) * (x @ up.T).
    n_tiles = int(num_non_exiting_tiles.item())
    rows = torch.arange(permuted_m, device="cuda")
    row_tile = rows // tile_m
    row_valid = (row_tile < n_tiles) & (rows < tile_idx_to_mn_limit.long()[row_tile])

    w_gate = w_gate_up[:, :inter].float()
    w_up = w_gate_up[:, inter:].float()
    expert_of_tile = tile_idx_to_expert_idx.long()
    token_of_expanded = torch.arange(num_tokens, device="cuda").repeat_interleave(top_k)
    token_of_row = torch.zeros(permuted_m, dtype=torch.long, device="cuda")
    perm_rows = expanded_idx_to_permuted_idx.flatten().long()
    token_of_row[perm_rows] = token_of_expanded

    ref = torch.zeros(permuted_m, inter, device="cuda", dtype=torch.float32)
    for e in torch.unique(expert_of_tile[:n_tiles]).tolist():
        rows_e = rows[row_valid & (expert_of_tile[row_tile] == e)]
        if rows_e.numel():
            xe = x[token_of_row[rows_e]].float()
            gate = xe @ w_gate[e].T
            up = xe @ w_up[e].T
            ref[rows_e] = torch.nn.functional.silu(gate) * up

    torch.testing.assert_close(
        out[row_valid].float(), ref[row_valid], atol=2e-1, rtol=3e-2
    )


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("swizzle_size", [8, 16])
def test_cute_dsl_bf16_gather_grouped_gemm_swizzled_walk(swizzle_size):
    """L2-blocked persistent walk must be numerically identical to the
    default order at a wide-N geometry (the shape class where dispatch
    enables it)."""
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
        sm90_contiguous_gather_grouped_gemm_act_fusion,
    )

    torch.manual_seed(5)
    num_experts, top_k, n2, k, num_tokens = 8, 2, 8192, 2048, 777
    tile_shape_mn = (128, 256)

    ids = make_random_topk_ids(num_experts, num_tokens, top_k)
    scales = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        _expanded_to_perm,
        permuted_idx_to_expanded_idx,
        _total_padded,
        num_non_exiting_tiles,
    ) = moe_sort(ids, scales, num_experts=num_experts, top_k=top_k, tile_tokens_dim=128)
    permuted_m = tile_idx_to_expert_idx.numel() * 128

    x = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16) / (k**0.25)
    w1 = interleave_up_gate_sm90(
        torch.randn(num_experts, n2, k, device="cuda", dtype=torch.bfloat16) / (k**0.25)
    )

    def run(swizzle):
        return sm90_contiguous_gather_grouped_gemm_act_fusion(
            x,
            w1,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            topk=top_k,
            permuted_m=permuted_m,
            tile_shape_mn=tile_shape_mn,
            swizzle_size=swizzle,
        )

    baseline = run(1)
    swizzled = run(swizzle_size)
    # Pure schedule reorder: bitwise-identical outputs on valid rows.
    n_tiles = int(num_non_exiting_tiles.item())
    rows = torch.arange(permuted_m, device="cuda")
    row_valid = (rows // 128 < n_tiles) & (
        rows < tile_idx_to_mn_limit.long()[rows // 128]
    )
    assert torch.equal(baseline[row_valid], swizzled[row_valid])


def _activation_cases():
    """Activation configs exercised by the gather GEMM epilogue tests."""
    from flashinfer.fused_moe import GeGLUTanh, ReLU2, SiTU, SwiGLU

    return [
        pytest.param(SwiGLU(alpha=1.702, beta=1.0, limit=7.0), id="swiglu_oai"),
        pytest.param(SiTU(gate_scale=4.0, linear_scale=None), id="situ"),
        pytest.param(SiTU(gate_scale=4.0, linear_scale=25.0), id="situ_linear"),
        pytest.param(GeGLUTanh(), id="geglu_tanh"),
        pytest.param(ReLU2(), id="relu2"),
    ]


@cute_dsl_available
@sm90_required
def test_cute_dsl_bf16_gather_grouped_gemm_out_handling():
    """Without ``out`` the gather GEMM allocates ``[permuted_m, I]`` in the
    activation dtype and matches an explicit-buffer run bitwise; a wrong-shaped
    ``out`` is rejected before any launch."""
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
        sm90_contiguous_gather_grouped_gemm_act_fusion,
    )

    torch.manual_seed(5)
    dtype = torch.bfloat16
    num_experts, top_k, k, inter, num_tokens, tile_m = 8, 2, 256, 64, 37, 64
    ids = make_random_topk_ids(num_experts, num_tokens, top_k)
    scales = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        _,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
    ) = moe_sort(
        ids, scales, num_experts=num_experts, top_k=top_k, tile_tokens_dim=tile_m
    )
    permuted_m = tile_idx_to_expert_idx.numel() * tile_m
    x = torch.randn(num_tokens, k, device="cuda", dtype=dtype) / (k**0.25)
    w1 = interleave_up_gate_sm90(
        torch.randn(num_experts, 2 * inter, k, device="cuda", dtype=dtype) / (k**0.25)
    )
    args = (
        x,
        w1,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
    )
    kwargs = dict(topk=top_k, permuted_m=permuted_m, tile_shape_mn=(tile_m, 64))

    allocated = sm90_contiguous_gather_grouped_gemm_act_fusion(*args, **kwargs)
    assert allocated.shape == (permuted_m, inter) and allocated.dtype == dtype
    explicit = torch.zeros(permuted_m, inter, device="cuda", dtype=dtype)
    sm90_contiguous_gather_grouped_gemm_act_fusion(*args, out=explicit, **kwargs)
    n_valid_tiles = int(num_non_exiting_tiles.item())
    valid = torch.arange(permuted_m, device="cuda") < n_valid_tiles * tile_m
    torch.testing.assert_close(allocated[valid], explicit[valid], atol=0, rtol=0)

    with pytest.raises(ValueError, match="out shape"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(
            *args,
            out=torch.empty(permuted_m, 2 * inter, device="cuda", dtype=dtype),
            **kwargs,
        )


@cute_dsl_available
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_situ_f32_constant_and_runtime_beta():
    """``situ_f32`` folds ``1 / beta`` for a trace-time Python beta and divides
    for a runtime ``cutlass.Float32`` beta; both paths match the float32
    reference ``beta * tanh(x / beta) * sigmoid(x)`` and each other."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from flashinfer.fused_moe.cute_dsl.common.kernel_utils import situ_f32

    @cute.kernel
    def kernel(
        x_ptr: cute.Pointer,
        beta_ptr: cute.Pointer,
        out_const_ptr: cute.Pointer,
        out_runtime_ptr: cute.Pointer,
        beta_const: cutlass.Constexpr[float],
    ):
        idx, _, _ = cute.arch.block_idx()
        x = cutlass.Float32(x_ptr[idx])
        out_const_ptr[idx] = situ_f32(x, beta_const)
        out_runtime_ptr[idx] = situ_f32(x, cutlass.Float32(beta_ptr[0]))

    @cute.jit
    def launch(
        x_ptr: cute.Pointer,
        beta_ptr: cute.Pointer,
        out_const_ptr: cute.Pointer,
        out_runtime_ptr: cute.Pointer,
        beta_const: cutlass.Constexpr[float],
        count: cutlass.Constexpr[int],
    ):
        kernel(x_ptr, beta_ptr, out_const_ptr, out_runtime_ptr, beta_const).launch(
            grid=(count, 1, 1), block=(1, 1, 1)
        )

    beta = 4.0
    x = torch.linspace(-12.0, 12.0, 257, device="cuda", dtype=torch.float32)
    beta_dev = torch.full((1,), beta, device="cuda", dtype=torch.float32)
    out_const = torch.empty_like(x)
    out_runtime = torch.empty_like(x)

    def ptr(t):
        return make_ptr(
            cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )

    launch(ptr(x), ptr(beta_dev), ptr(out_const), ptr(out_runtime), beta, x.numel())
    torch.cuda.synchronize()

    ref = beta * torch.tanh(x / beta) * torch.sigmoid(x)
    torch.testing.assert_close(out_const, ref, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(out_runtime, ref, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(out_runtime, out_const, atol=2e-6, rtol=1e-6)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("activation", _activation_cases())
@pytest.mark.parametrize(
    "inter,tile_shape_mn",
    [
        (768, (128, 128)),
        (192, (128, 192)),  # 2-WG tile at M=128; N == 2I (gated) or I (relu2)
        (768, (64, 256)),
    ],
)
def test_cute_dsl_bf16_gather_grouped_gemm_activations(
    activation, inter, tile_shape_mn
):
    """Every fused activation matches the shared float32 reference on the
    valid rows.

    Gated activations read the 32-column up/gate interleave and write I
    columns; ReLU2 reads a plain [E, I, K] projection (no interleave).
    """
    from tests.moe.utils import compute_reference_activation
    from flashinfer.fused_moe.runners import _cute_dsl_activation_kwargs
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
        sm90_contiguous_gather_grouped_gemm_act_fusion,
    )

    torch.manual_seed(3)
    num_experts, top_k, k, num_tokens = 32, 4, 1024, 333
    dtype = torch.bfloat16
    tile_m = tile_shape_mn[0]
    gated = activation.is_gated
    act = _cute_dsl_activation_kwargs(activation)
    if not gated and tile_shape_mn[1] > inter:
        pytest.skip("N tile wider than the non-gated projection")

    ids = make_random_topk_ids(num_experts, num_tokens, top_k)
    scales = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _total_padded,
        num_non_exiting_tiles,
    ) = moe_sort(
        ids, scales, num_experts=num_experts, top_k=top_k, tile_tokens_dim=tile_m
    )
    permuted_m = tile_idx_to_expert_idx.numel() * tile_m

    x = torch.randn(num_tokens, k, device="cuda", dtype=dtype) / (k**0.25)
    if gated:
        w_gate_up = torch.randn(num_experts, 2 * inter, k, device="cuda", dtype=dtype)
        w_gate_up = w_gate_up / (k**0.25)
        w1 = interleave_up_gate_sm90(w_gate_up)
    else:
        w1 = torch.randn(num_experts, inter, k, device="cuda", dtype=dtype) / (k**0.25)

    out = torch.full((permuted_m, inter), float("nan"), device="cuda", dtype=dtype)
    out = sm90_contiguous_gather_grouped_gemm_act_fusion(
        x,
        w1,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        out=out,
        topk=top_k,
        permuted_m=permuted_m,
        tile_shape_mn=tile_shape_mn,
        **act,
    )
    assert out.shape == (permuted_m, inter)

    n_tiles = int(num_non_exiting_tiles.item())
    rows = torch.arange(permuted_m, device="cuda")
    row_tile = rows // tile_m
    row_valid = (row_tile < n_tiles) & (rows < tile_idx_to_mn_limit.long()[row_tile])
    expert_of_tile = tile_idx_to_expert_idx.long()
    token_of_expanded = torch.arange(num_tokens, device="cuda").repeat_interleave(top_k)
    token_of_row = torch.zeros(permuted_m, dtype=torch.long, device="cuda")
    token_of_row[expanded_idx_to_permuted_idx.flatten().long()] = token_of_expanded

    ref = torch.zeros(permuted_m, inter, device="cuda", dtype=torch.float32)
    for e in torch.unique(expert_of_tile[:n_tiles]).tolist():
        rows_e = rows[row_valid & (expert_of_tile[row_tile] == e)]
        if not rows_e.numel():
            continue
        xe = x[token_of_row[rows_e]].float()
        if gated:
            # The shared reference takes [up | gate]; the model pack is [gate; up].
            gate = xe @ w_gate_up[e, :inter].float().T
            up = xe @ w_gate_up[e, inter:].float().T
            values = torch.cat((up, gate), dim=-1)
        else:
            values = xe @ w1[e].float().T
        ref[rows_e] = compute_reference_activation(
            values, activation, inter, dtype
        ).float()

    torch.testing.assert_close(
        out[row_valid].float(), ref[row_valid], atol=2e-1, rtol=3e-2
    )


@cute_dsl_available
def test_cute_dsl_bf16_gather_grouped_gemm_rejects_bad_activation_config():
    """Unsupported activation types, bad SwiGLU constants and inconsistent SiTU
    parameters are rejected when the kernel is configured, before any
    compilation, checked in that order."""
    import cutlass

    from flashinfer.fused_moe.cute_dsl.hopper.contiguous_gather_grouped_gemm_act_fusion import (
        Sm90ContiguousGatherGroupedGemmActFusionKernel,
    )
    from flashinfer.tllm_enums import ActivationType

    def make(**kwargs):
        return Sm90ContiguousGatherGroupedGemmActFusionKernel(
            cutlass.Float32, (128, 64), topk=2, **kwargs
        )

    for bad in (ActivationType.Geglu, ActivationType.Silu, ActivationType.SwigluStep):
        with pytest.raises(ValueError, match="Unsupported activation_type"):
            make(activation_type=bad.value)
    with pytest.raises(ValueError, match="swiglu_alpha must be finite"):
        make(swiglu_alpha=float("nan"))
    with pytest.raises(ValueError, match="swiglu_beta must be finite"):
        make(swiglu_beta=float("inf"))
    with pytest.raises(ValueError, match="swiglu_limit must be positive"):
        make(swiglu_limit=0.0)
    with pytest.raises(ValueError, match="requires situ_beta"):
        make(situ_linear_beta=25.0)
    with pytest.raises(ValueError, match="require ActivationType.Swiglu"):
        make(activation_type=ActivationType.GegluTanh.value, situ_beta=4.0)
    with pytest.raises(ValueError, match="swiglu_limit must be positive"):
        make(swiglu_limit=0.0, situ_linear_beta=25.0)
    assert make(activation_type=ActivationType.Relu2.value).out_n_factor == 1
    assert make(activation_type=ActivationType.GegluTanh.value).out_n_factor == 2
