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

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported
from flashinfer.tllm_enums import ActivationType

from .utils import create_moe_tensors


def _is_sm100_family() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = [
    pytest.mark.skipif(
        not _is_sm100_family(), reason="requires the SM100/SM103 GEMM1 kernel"
    ),
    pytest.mark.skipif(not is_cute_dsl_available(), reason="CuTe DSL unavailable"),
    pytest.mark.skipif(
        torch.cuda.is_available()
        and not is_cute_dsl_arch_supported(
            *torch.cuda.get_device_capability(), native_only=True
        ),
        reason="installed CuTe DSL cannot target the current GPU",
    ),
]


def _set_quant_mode(monkeypatch: pytest.MonkeyPatch, deterministic: bool) -> None:
    values = {
        "FLASHINFER_NVFP4_4OVER6": "1" if deterministic else "0",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1" if deterministic else "0",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1" if deterministic else "0",
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1" if deterministic else "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _unpack_blocked8(input_amax: torch.Tensor) -> torch.Tensor:
    """Return the logical [row, GEMM-N-tile] view of the aux buffer."""
    return (
        input_amax.permute(0, 2, 1)
        .reshape(input_amax.shape[0] * 8, input_amax.shape[1])
        .contiguous()
    )


def _make_gemm1_output(
    *,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    tile_size: int,
    intermediate_size: int,
) -> torch.Tensor:
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        get_max_num_permuted_tokens,
    )

    max_num_permuted_tokens = get_max_num_permuted_tokens(
        num_tokens,
        top_k,
        num_local_experts,
        tile_size,
    )
    # A finite sentinel makes the deliberately unused routing-buffer tail
    # deterministic. GEMM1 must not touch it and GEMM2 must not consume it.
    return torch.full(
        (max_num_permuted_tokens, intermediate_size),
        17.0,
        dtype=torch.bfloat16,
        device="cuda",
    )


def _make_core_kwargs(
    tensors: dict,
    *,
    num_tokens: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int,
    tile_size: int,
    gemm1_n: int,
    activation_type: ActivationType,
) -> dict:
    return {
        "x": tensors["x"],
        "x_sf": tensors["x_sf"],
        "token_selected_experts": tensors["token_selected_experts"],
        "token_final_scales": tensors["token_final_scales"],
        "w1_weight": tensors["w1_weight"],
        "w1_weight_sf": tensors["w1_weight_sf"],
        "w1_alpha": tensors["w1_alpha"],
        "fc2_input_scale": tensors["fc2_input_scale"],
        "w2_weight": tensors["w2_weight"],
        "w2_weight_sf": tensors["w2_weight_sf"],
        "w2_alpha": tensors["w2_alpha"],
        "num_experts": num_experts,
        "top_k": top_k,
        "num_local_experts": num_local_experts,
        "local_expert_offset": local_expert_offset,
        "tile_size": tile_size,
        "gemm1_mma_tiler_mn": (tile_size, gemm1_n),
        "gemm1_cluster_shape_mn": (tile_size // 128, 1),
        "gemm2_mma_tiler_mn": (tile_size, 128),
        "gemm2_cluster_shape_mn": (tile_size // 128, 1),
        "use_async_memset": False,
        "use_fused_finalize": False,
        "enable_pdl": True,
        "activation_type": activation_type.value,
        "per_token_scale": tensors["x_per_token_scale"],
        "gemm1_out": _make_gemm1_output(
            num_tokens=num_tokens,
            top_k=top_k,
            num_local_experts=num_local_experts,
            tile_size=tile_size,
            intermediate_size=intermediate_size,
        ),
    }


def _assert_legacy_and_aux_paths_are_bitwise_equal(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict,
    *,
    tile_size: int,
    gemm1_n: int,
    gated: bool,
    expect_unused_rows: bool,
    expect_masked_routes: bool = False,
    expect_padded_rows: bool = False,
) -> None:
    """Compare every defined intermediate row and the final MoE output."""
    import flashinfer.fused_moe.cute_dsl.fused_moe as fused_moe_module

    original_sort = fused_moe_module.moe_sort
    sort_results = []

    def checked_sort(*args, **sort_kwargs):
        result = original_sort(*args, **sort_kwargs)
        sort_results.append(result)
        return result

    monkeypatch.setattr(fused_moe_module, "moe_sort", checked_sort)
    original_gemm1 = (
        fused_moe_module.blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4
    )
    original_quantize = fused_moe_module.nvfp4_quantize_per_token_cute_dsl
    legacy_gemm1_calls = 0
    legacy_quantize_calls = 0

    def legacy_gemm1(*args, **call_kwargs):
        nonlocal legacy_gemm1_calls
        assert call_kwargs.get("out_amax") is not None
        call_kwargs = dict(call_kwargs)
        call_kwargs["out_amax"] = None
        legacy_gemm1_calls += 1
        return original_gemm1(*args, **call_kwargs)

    def legacy_quantize(*args, **call_kwargs):
        nonlocal legacy_quantize_calls
        assert call_kwargs.get("input_amax") is not None
        assert call_kwargs.get("input_amax_valid_rows") is not None
        call_kwargs = dict(call_kwargs)
        call_kwargs.pop("input_amax")
        call_kwargs.pop("input_amax_valid_rows")
        legacy_quantize_calls += 1
        return original_quantize(*args, **call_kwargs)

    with monkeypatch.context() as legacy_patch:
        legacy_patch.setattr(
            fused_moe_module,
            "blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4",
            legacy_gemm1,
        )
        legacy_patch.setattr(
            fused_moe_module,
            "nvfp4_quantize_per_token_cute_dsl",
            legacy_quantize,
        )
        legacy_output = fused_moe_module._moe_core_impl(**kwargs)

    assert legacy_gemm1_calls == 1
    assert legacy_quantize_calls == 1
    calls_checked = 0

    def checked_quantize(
        input: torch.Tensor,
        global_scale_inv: torch.Tensor,
        sf_layout: int,
        enable_pdl: bool,
        input_amax: torch.Tensor | None = None,
        input_amax_valid_rows: torch.Tensor | None = None,
    ):
        nonlocal calls_checked
        assert input_amax is not None

        sort_result = sort_results[-1]
        permuted_idx_to_expanded_idx = sort_result[3]
        num_non_exiting_tiles = int(sort_result[5].item())
        active_rows = num_non_exiting_tiles * tile_size
        assert input.shape[0] == permuted_idx_to_expanded_idx.shape[0]
        assert active_rows <= input.shape[0]
        assert input_amax_valid_rows is not None
        assert input_amax_valid_rows.dtype == torch.int32
        assert input_amax_valid_rows.shape == (1,)
        assert int(input_amax_valid_rows.item()) == active_rows

        output_tile_n = gemm1_n // (2 if gated else 1)
        expected_amax = (
            input[:active_rows]
            .float()
            .abs()
            .reshape(
                active_rows,
                input.shape[1] // output_tile_n,
                output_tile_n,
            )
            .amax(dim=2)
            .to(input.dtype)
        )
        # Only scheduled epilogue rows are defined. Rows in the allocation
        # tail are deliberately not synchronized or initialized.
        logical_input_amax = _unpack_blocked8(input_amax)
        assert input_amax.dtype == input.dtype
        assert torch.equal(logical_input_amax[:active_rows], expected_amax)

        legacy_quantized = original_quantize(
            input,
            global_scale_inv,
            sf_layout=sf_layout,
            enable_pdl=False,
        )
        accelerated_quantized = original_quantize(
            input,
            global_scale_inv,
            sf_layout=sf_layout,
            enable_pdl=enable_pdl,
            input_amax=input_amax,
            input_amax_valid_rows=input_amax_valid_rows,
        )
        # FP4 codes, swizzled block scales, and per-token scales are compared
        # bitwise for every row GEMM1 actually produced. active_rows is a
        # multiple of 128, so it also covers whole 128x4 scale-layout tiles.
        for accelerated, legacy in zip(
            accelerated_quantized, legacy_quantized, strict=True
        ):
            assert torch.equal(accelerated[:active_rows], legacy[:active_rows])
        calls_checked += 1
        return accelerated_quantized

    monkeypatch.setattr(
        fused_moe_module,
        "nvfp4_quantize_per_token_cute_dsl",
        checked_quantize,
    )
    accelerated_output = fused_moe_module._moe_core_impl(**kwargs)

    assert calls_checked == 1
    assert len(sort_results) == 2
    # Within-expert permutation order is not part of moe_sort's contract, but
    # both launches must agree on the padded and scheduled extents.
    assert torch.equal(sort_results[0][4], sort_results[1][4])
    assert torch.equal(sort_results[0][5], sort_results[1][5])

    active_rows = int(sort_results[1][5].item()) * tile_size
    expanded_idx_to_permuted_idx = sort_results[1][2]
    gemm1_out = kwargs["gemm1_out"]
    if expect_unused_rows:
        assert active_rows < gemm1_out.shape[0]
        assert torch.equal(
            gemm1_out[active_rows:],
            torch.full_like(gemm1_out[active_rows:], 17.0),
        )
    else:
        assert active_rows == gemm1_out.shape[0]
    if expect_masked_routes:
        assert (expanded_idx_to_permuted_idx < 0).any()
        assert (expanded_idx_to_permuted_idx >= 0).any()
    if expect_padded_rows:
        # The sort kernel does not promise a sentinel value in its padded
        # inverse-map entries. Prove that padding exists from the number of
        # valid local routes instead of reading those deliberately undefined
        # entries.
        num_valid_local_routes = int((expanded_idx_to_permuted_idx >= 0).sum().item())
        assert num_valid_local_routes < active_rows
    assert torch.equal(accelerated_output, legacy_output)


@pytest.mark.parametrize(
    ("num_tokens", "expect_aux"),
    [
        pytest.param(2, False, id="tokens2-legacy"),
        pytest.param(4, True, id="tokens4-aux"),
    ],
)
def test_per_token_aux_amax_dispatch_boundary(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    expect_aux: bool,
):
    """The host-static token threshold selects the intended exact path."""
    import flashinfer.fused_moe.cute_dsl.fused_moe as fused_moe_module

    _set_quant_mode(monkeypatch, deterministic=False)

    top_k, num_experts = 2, 16
    hidden_size, intermediate_size = 256, 512
    tile_size, gemm1_n = 128, 128
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
        gated=True,
        use_per_token_activation=True,
    )
    kwargs = _make_core_kwargs(
        tensors,
        num_tokens=num_tokens,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        gemm1_n=gemm1_n,
        activation_type=ActivationType.Swiglu,
    )

    original_gemm1 = (
        fused_moe_module.blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4
    )
    original_quantize = fused_moe_module.nvfp4_quantize_per_token_cute_dsl
    producer_uses_aux = []
    quantizer_aux_args = []

    def checked_gemm1(*args, **call_kwargs):
        producer_uses_aux.append(call_kwargs.get("out_amax") is not None)
        return original_gemm1(*args, **call_kwargs)

    def checked_quantize(*args, **call_kwargs):
        quantizer_aux_args.append(
            (
                call_kwargs.get("input_amax") is not None,
                call_kwargs.get("input_amax_valid_rows") is not None,
            )
        )
        return original_quantize(*args, **call_kwargs)

    monkeypatch.setattr(
        fused_moe_module,
        "blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4",
        checked_gemm1,
    )
    monkeypatch.setattr(
        fused_moe_module,
        "nvfp4_quantize_per_token_cute_dsl",
        checked_quantize,
    )

    output = fused_moe_module._moe_core_impl(**kwargs)
    torch.cuda.synchronize()

    assert output.shape == (num_tokens, hidden_size)
    assert producer_uses_aux == [expect_aux]
    assert quantizer_aux_args == [(expect_aux, expect_aux)]


def test_fp16_gemm1_aux_amax_handoff_is_bitwise_equal(
    monkeypatch: pytest.MonkeyPatch,
):
    """Exercise the real FP16 producer/consumer boundary without GEMM2."""
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
    )
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        SF_LAYOUT_128x4,
        nvfp4_quantize_per_token_cute_dsl,
    )

    _set_quant_mode(monkeypatch, deterministic=False)

    num_tokens = 128
    hidden_size, intermediate_size = 256, 512
    tile_size, gemm1_n = 128, 256
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=1,
        num_local_experts=1,
        top_k=1,
        gated=True,
        use_per_token_activation=True,
    )

    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        _,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = moe_sort(
        tensors["token_selected_experts"],
        tensors["token_final_scales"],
        num_experts=1,
        top_k=1,
        num_local_experts=1,
        tile_tokens_dim=tile_size,
        enable_pdl=True,
    )

    num_permuted_rows = permuted_idx_to_expanded_idx.shape[0]
    output_tile_n = gemm1_n // 2
    num_output_tiles = intermediate_size // output_tile_n
    intermediate = torch.empty(
        (num_permuted_rows, intermediate_size),
        dtype=torch.float16,
        device="cuda",
    )
    intermediate_amax = torch.empty(
        (num_permuted_rows // 8, num_output_tiles, 8),
        dtype=torch.float16,
        device="cuda",
    )

    produced_intermediate, produced_sf = (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4(
            a=tensors["x"],
            b=tensors["w1_weight"],
            a_scale=tensors["x_sf"],
            b_scale=tensors["w1_weight_sf"],
            alpha=tensors["w1_alpha"],
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            token_id_mapping=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            out=intermediate,
            a_per_token_scale=tensors["x_per_token_scale"],
            out_amax=intermediate_amax,
            topk=1,
            c_dtype="float16",
            mma_tiler_mn=(tile_size, gemm1_n),
            cluster_shape_mn=(1, 1),
            enable_pdl=True,
            activation_type=ActivationType.Swiglu.value,
            gated=True,
        )
    )
    assert produced_intermediate is intermediate
    assert produced_sf is None

    # Keep this launch immediately after GEMM1 so the test exercises the real
    # producer-to-consumer PDL chain without a host synchronization in between.
    accelerated_quantized = nvfp4_quantize_per_token_cute_dsl(
        intermediate,
        tensors["fc2_input_scale"],
        sf_layout=SF_LAYOUT_128x4,
        enable_pdl=True,
        input_amax=intermediate_amax,
        input_amax_valid_rows=total_num_padded_tokens,
    )
    legacy_quantized = nvfp4_quantize_per_token_cute_dsl(
        intermediate,
        tensors["fc2_input_scale"],
        sf_layout=SF_LAYOUT_128x4,
        enable_pdl=False,
    )

    assert int(total_num_padded_tokens.item()) == num_tokens
    assert int(num_non_exiting_tiles.item()) == 1
    assert intermediate.dtype == intermediate_amax.dtype == torch.float16
    expected_amax = (
        intermediate.float()
        .abs()
        .reshape(num_tokens, num_output_tiles, output_tile_n)
        .amax(dim=2)
        .to(torch.float16)
    )
    assert torch.equal(_unpack_blocked8(intermediate_amax), expected_amax)
    for accelerated, legacy in zip(
        accelerated_quantized, legacy_quantized, strict=True
    ):
        assert torch.equal(accelerated, legacy)


@pytest.mark.parametrize("deterministic_quant", [False, True])
@pytest.mark.parametrize(
    ("tile_size", "gemm1_n", "activation_type", "gated"),
    [
        pytest.param(128, 128, ActivationType.Swiglu, True, id="m128-n128-swiglu"),
        pytest.param(128, 256, ActivationType.Swiglu, True, id="m128-n256-swiglu"),
        pytest.param(
            256,
            256,
            ActivationType.Swiglu,
            True,
            id="m256-2cta-n256-swiglu",
        ),
        pytest.param(256, 128, ActivationType.Relu2, False, id="m256-n128-relu2"),
        pytest.param(256, 256, ActivationType.Relu2, False, id="m256-n256-relu2"),
    ],
)
def test_gemm1_aux_amax_and_per_token_output_are_bitwise_equal(
    monkeypatch: pytest.MonkeyPatch,
    tile_size: int,
    gemm1_n: int,
    activation_type: ActivationType,
    gated: bool,
    deterministic_quant: bool,
):
    """Check producer values and the exact legacy/accelerated MoE boundary."""
    _set_quant_mode(monkeypatch, deterministic_quant)

    num_tokens, top_k, num_experts = 8, 2, 16
    hidden_size, intermediate_size = 256, 512
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
        gated=gated,
        use_per_token_activation=True,
    )

    # Give every expert exactly one routed row. The actual and maximum tile
    # counts are then identical, so every materialized row and every aux cell
    # is written and can be checked without consulting internal sort metadata.
    tensors["token_selected_experts"].copy_(
        torch.arange(
            num_tokens * top_k,
            device="cuda",
            dtype=torch.int32,
        ).reshape(num_tokens, top_k)
    )
    tensors["token_final_scales"].copy_(
        torch.tensor([0.375, 0.625], device="cuda", dtype=torch.float32).repeat(
            num_tokens, 1
        )
    )

    kwargs = _make_core_kwargs(
        tensors,
        num_tokens=num_tokens,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        gemm1_n=gemm1_n,
        activation_type=activation_type,
    )

    _assert_legacy_and_aux_paths_are_bitwise_equal(
        monkeypatch,
        kwargs,
        tile_size=tile_size,
        gemm1_n=gemm1_n,
        gated=gated,
        expect_unused_rows=False,
    )


@pytest.mark.parametrize("deterministic_quant", [False, True])
def test_local_routing_with_padded_unused_rows_is_bitwise_equal(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_quant: bool,
):
    """EP-local routes ignore both in-tile padding and allocation-tail rows."""
    _set_quant_mode(monkeypatch, deterministic_quant)

    num_tokens, top_k, num_experts = 9, 2, 16
    num_local_experts, local_expert_offset = 4, 4
    hidden_size, intermediate_size = 256, 512
    tile_size, gemm1_n = 128, 128
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        top_k=top_k,
        gated=True,
        use_per_token_activation=True,
    )

    # Experts [4, 8) are local. Only experts 4 and 5 receive local routes, so
    # moe_sort emits two heavily padded tiles into a four-tile allocation.
    tensors["token_selected_experts"].copy_(
        torch.tensor(
            [
                [4, 0],
                [5, 15],
                [4, 1],
                [5, 14],
                [4, 2],
                [5, 13],
                [4, 3],
                [5, 12],
                [4, 8],
            ],
            device="cuda",
            dtype=torch.int32,
        )
    )
    tensors["token_final_scales"].copy_(
        torch.tensor([0.75, 0.25], device="cuda", dtype=torch.float32).repeat(
            num_tokens, 1
        )
    )

    kwargs = _make_core_kwargs(
        tensors,
        num_tokens=num_tokens,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        tile_size=tile_size,
        gemm1_n=gemm1_n,
        activation_type=ActivationType.Swiglu,
    )
    _assert_legacy_and_aux_paths_are_bitwise_equal(
        monkeypatch,
        kwargs,
        tile_size=tile_size,
        gemm1_n=gemm1_n,
        gated=True,
        expect_unused_rows=True,
        expect_masked_routes=True,
        expect_padded_rows=True,
    )


@pytest.mark.parametrize("deterministic_quant", [False, True])
def test_large_token_moe_aux_path_is_bitwise_equal(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_quant: bool,
):
    """Exercise a 1K-token benchmark shape without adding a new kernel tactic."""
    _set_quant_mode(monkeypatch, deterministic_quant)

    num_tokens, top_k, num_experts = 1024, 2, 16
    hidden_size, intermediate_size = 256, 512
    tile_size, gemm1_n = 128, 128
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
        gated=True,
        use_per_token_activation=True,
    )

    # Two dense experts produce 16 scheduled M tiles. The tight worst-case
    # allocation has 31 tiles, which simultaneously exercises a long unused
    # tail while keeping the weight and compilation footprint small.
    tensors["token_selected_experts"].copy_(
        torch.tensor([0, 1], device="cuda", dtype=torch.int32).repeat(num_tokens, 1)
    )
    tensors["token_final_scales"].copy_(
        torch.tensor([0.375, 0.625], device="cuda", dtype=torch.float32).repeat(
            num_tokens, 1
        )
    )

    kwargs = _make_core_kwargs(
        tensors,
        num_tokens=num_tokens,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_experts,
        local_expert_offset=0,
        tile_size=tile_size,
        gemm1_n=gemm1_n,
        activation_type=ActivationType.Swiglu,
    )
    _assert_legacy_and_aux_paths_are_bitwise_equal(
        monkeypatch,
        kwargs,
        tile_size=tile_size,
        gemm1_n=gemm1_n,
        gated=True,
        expect_unused_rows=True,
    )
