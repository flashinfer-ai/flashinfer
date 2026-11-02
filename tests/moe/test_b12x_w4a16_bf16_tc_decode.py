# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded BF16 fused-sum decode with live routes at T=5 through T=8."""

import pytest
import torch
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_w4a16_kernel as kernel
from tests.moe.test_b12x_fused_moe import (
    cute_dsl_available,
    sm120_required,
    cuda_13_required,
)


@cute_dsl_available
@pytest.mark.parametrize(
    "arch,dtype,layout,scale,h,i,k,expected",
    [
        (120, "bf16", "packed", "e4m3_k16", 2048, 512, 10, 8),
        (120, "bf16", "packed", "e4m3_k16", 256, 256, 3, 8),
        (121, "bf16", "packed", "e4m3_k16", 2048, 512, 10, 4),
        (120, "fp16", "packed", "e4m3_k16", 2048, 512, 10, 4),
        (120, "bf16", "nf3_2p1", "e4m3_k16", 2048, 512, 10, 4),
        (120, "bf16", "packed", "e8m0_k32", 2048, 512, 10, 4),
        (120, "bf16", "modelopt", "e4m3_k16", 2048, 512, 10, 4),
        (120, "bf16", "packed", "e4m3_k16", 4096, 512, 10, 4),
        (120, "bf16", "packed", "e4m3_k16", 2048, 1024, 10, 4),
        (120, "bf16", "packed", "e4m3_k16", 2048, 512, 16, 4),
        (120, "bf16", "packed", "e4m3_k16", 7168, 2048, 8, 4),
    ],
)
def test_bf16_tc8_cap_preserves_unqualified_regions(
    arch, dtype, layout, scale, h, i, k, expected
):
    assert kernel._tc_decode_m_cap(arch, dtype, layout, scale, h, i, k) == expected


@cute_dsl_available
@sm120_required
@cuda_13_required
@pytest.mark.parametrize("tokens", [5, 6, 7, 8])
@pytest.mark.parametrize("activation", ["silu"])
@pytest.mark.parametrize("hidden,intermediate,top_k", [(256, 256, 3), (2048, 512, 10)])
@pytest.mark.parametrize("workspace_kind", ["dispatch", "packed"])
def test_bf16_tc8_matches_packed_routes_and_live_graph(
    tokens,
    activation,
    hidden,
    intermediate,
    top_k,
    workspace_kind,
    monkeypatch,
    experts=16,
):
    from flashinfer.utils import get_compute_capability

    if get_compute_capability(torch.device("cuda")) != (12, 0):
        pytest.skip("The expanded BF16 TC-decode region is qualified on SM120 only")

    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        prepare_w4a16_packed_weights,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        allocate_sm120_moe_workspace,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        make_w4a16_packed_buffers,
        select_route_block_size_m,
    )
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    torch.manual_seed(7190 + tokens)
    e, h, i, k = experts, hidden, intermediate, top_k
    branches = 2 if activation == "silu" else 1
    x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.1
    ids = (
        torch.arange(tokens * k, device="cuda", dtype=torch.int32).reshape(tokens, k)
        % e
    )
    scores = torch.rand(tokens, k, device="cuda")
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")

    def weights(rows, cols):
        q = torch.randint(
            0, 256, (e, rows, cols // 2), device="cuda", dtype=torch.uint8
        )
        sf = torch.full((e, rows, cols // 16), 0.0625, device="cuda").to(
            torch.float8_e4m3fn
        )
        swizzled = torch.stack(
            [block_scale_interleave(v.view(torch.uint8)) for v in sf]
        ).reshape_as(sf)
        codes = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).long()
        decoded = lut[codes & 7] * (1 - 2 * ((codes & 8) != 0).float()) * 0.0625
        return q, swizzled, decoded.double()

    w1, s1, dense1 = weights(branches * i, h)
    w2, s2, dense2 = weights(h, i)
    prepared = prepare_w4a16_packed_weights(
        w1,
        s1,
        torch.ones(e, device="cuda"),
        w2,
        s2,
        torch.ones(e, device="cuda"),
        activation=activation,
        params_dtype=torch.bfloat16,
    )
    props = torch.cuda.get_device_properties(x.device)
    block = select_route_block_size_m(tokens, k, e)
    calls = []
    for direct in (False, True):
        if workspace_kind == "dispatch":
            work = allocate_sm120_moe_workspace(
                state_E=e,
                weight_E=e,
                k=h,
                n=i,
                num_topk=k,
                device=x.device,
                routed_rows=tokens * k,
                quant_mode="w4a16",
                activation=activation,
            )
        else:
            work = make_w4a16_packed_buffers(
                prepared, m=tokens, topk=k, dtype=x.dtype, device=x.device
            )
        output = torch.empty_like(x)
        compiled = kernel.compile_w4a16_fused_moe(
            size_m=tokens,
            hidden_size=h,
            intermediate_size=i,
            num_experts=e,
            top_k=k,
            activation=activation,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block,
            max_m_blocks=tokens * k if direct else work.block_expert_ids.numel(),
            element_dtype="bf16",
            sms=props.multi_processor_count,
            max_shared_mem=props.shared_memory_per_block_optin,
            direct_topk_routes=direct,
            tc_decode_fused_sum=direct,
        )
        assert compiled.direct_topk_routes == direct

        def call(work=work, output=output, compiled=compiled):
            return kernel.run_w4a16_moe(
                x,
                prepared,
                scores,
                ids,
                activation=activation,
                intermediate_cache13=work.intermediate_cache13.view(torch.bfloat16),
                intermediate_cache2=work.intermediate_cache2.view(torch.bfloat16),
                output=output,
                fc1_c_tmp=work.fc1_c_tmp,
                fc2_c_tmp=work.fc2_c_tmp,
                packed_route_indices=work.packed_route_indices,
                block_expert_ids=work.block_expert_ids,
                packed_route_count=work.packed_route_count,
                expert_offsets=work.expert_offsets,
                fused_launch=compiled,
            )

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        calls.append((call, graph, captured, work))

    def reference():
        result = torch.zeros_like(x, dtype=torch.float32)
        for token in range(tokens):
            for slot, expert in enumerate(ids[token].tolist()):
                if expert < 0:
                    continue
                pre = (dense1[expert] @ x[token].double()).float()
                if activation == "silu":
                    up, gate = pre.chunk(2)
                    mid = (up * torch.nn.functional.silu(gate)).bfloat16()
                else:
                    mid = torch.relu(pre).square().bfloat16()
                value = (dense2[expert] @ mid.double()).float()
                result[token] += value * scores[token, slot]
        return result.bfloat16()

    def check(actual, expected):
        delta = actual.double() - expected.double()
        assert torch.isfinite(actual).all()
        assert delta.norm() / expected.double().norm().clamp_min(1e-20) < 0.01
        assert delta.abs().max() / expected.double().abs().max().clamp_min(1e-20) < 0.03

    for mode in ("disjoint", "shared", "permuted", "negative"):
        x.copy_(torch.randn_like(x) * 0.1)
        ids.copy_(torch.arange(tokens * k, device=x.device).reshape(tokens, k) % e)
        if mode != "disjoint":
            ids.remainder_(k)
        if mode == "permuted":
            for token in range(tokens):
                ids[token].copy_(torch.roll(ids[token], token))
        if mode == "negative":
            ids[::2, 0] = -1
        scores.copy_(torch.rand_like(scores))
        if mode == "negative":
            scores[::2, 0] = float("nan")
        expected = reference()
        for _call, graph, output, _work in calls:
            output.fill_(float("nan"))
            graph.replay()
            check(output, expected)
        check(calls[1][2], calls[0][2])

        # The non-fused preplanned route is the explicit fixed-order combine
        # alternative. Check identical inputs eagerly and under the retained
        # graph; tolerance-based comparison alone cannot test repeatability.
        fixed_call, fixed_graph, fixed_output, fixed_work = calls[0]
        expected_bits = fixed_output.view(torch.int16).clone()
        for execute in (fixed_call, fixed_graph.replay):
            for _ in range(8):
                fixed_output.fill_(float("nan"))
                for name in (
                    "intermediate_cache13",
                    "intermediate_cache2",
                    "fc1_c_tmp",
                    "fc2_c_tmp",
                ):
                    getattr(fixed_work, name).view(torch.bfloat16).fill_(float("nan"))
                for name in (
                    "packed_route_indices",
                    "block_expert_ids",
                    "packed_route_count",
                    "expert_offsets",
                ):
                    getattr(fixed_work, name).fill_(-1)
                execute()
                torch.testing.assert_close(
                    fixed_output.view(torch.int16), expected_bits, atol=0, rtol=0
                )

    # The direct launch must not silently fall back to route packing.
    def forbidden(*args, **kwargs):
        raise AssertionError("fused-sum BF16 decode unexpectedly packed routes")

    monkeypatch.setattr(kernel, "pack_topk_routes_by_expert", forbidden)
    check(calls[1][0](), reference())
    check(calls[1][0](compiled=None), reference())
