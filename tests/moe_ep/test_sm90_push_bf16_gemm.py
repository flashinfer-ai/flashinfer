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

Direct oracle tests for the SM90 push BF16 grouped GEMM adapter.
"""

from __future__ import annotations

import pytest
import torch


def _sm90_cuda_12_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least
        from flashinfer.utils import is_sm90a_supported

        return is_cuda_version_at_least("12.0") and is_sm90a_supported(
            torch.device("cuda")
        )
    except Exception:
        return False


requires_sm90 = pytest.mark.skipif(
    not _sm90_cuda_12_available(),
    reason="requires SM90 and CUDA Toolkit 12.0+",
)

EXPERT_ROWS = (0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 257)


def _reference_grouped_bf16(
    activation: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply one FP32-accumulating BF16 GEMM to each expert interval."""
    output = torch.zeros(
        activation.shape[0],
        weights.shape[1],
        dtype=torch.bfloat16,
        device=activation.device,
    )
    host_offsets = offsets.cpu().tolist()
    for expert, (start, end) in enumerate(
        zip(host_offsets[:-1], host_offsets[1:], strict=True)
    ):
        if start != end:
            output[start:end] = (
                activation[start:end].float() @ weights[expert].float().T
            ).to(torch.bfloat16)
    return output


@requires_sm90
@pytest.mark.parametrize("active_expert", range(4))
def test_direct_grouped_bf16_exact_m_boundaries(active_expert: int) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        create_sm90_push_bf16_gemm_runner,
    )

    device = torch.device("cuda", 0)
    num_experts, k, n = 4, 256, 384
    generator = torch.Generator(device="cpu").manual_seed(301 + active_expert)
    weights = (torch.randn(num_experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    runner = create_sm90_push_bf16_gemm_runner(
        max_rows=max(EXPERT_ROWS),
        num_experts=num_experts,
        n=n,
        k=k,
        device=device,
    )

    for rows in EXPERT_ROWS:
        activation = torch.randn(rows, k, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        counts = [0] * num_experts
        counts[active_expert] = rows
        offsets = torch.tensor(
            [0, *torch.tensor(counts).cumsum(0).tolist()],
            dtype=torch.int64,
            device=device,
        )
        output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)

        result = runner.run(output, activation, weights, offsets)
        torch.cuda.synchronize()
        reference = _reference_grouped_bf16(activation, weights, offsets)

        assert result is output
        torch.testing.assert_close(output, reference, rtol=0.125, atol=0.125)


@requires_sm90
def test_direct_grouped_bf16_empty_and_uneven_experts() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        create_sm90_push_bf16_gemm_runner,
    )

    device = torch.device("cuda", 0)
    num_experts, k, n = 6, 256, 128
    counts = [0, 1, 65, 0, 129, 7]
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    rows = sum(counts)
    generator = torch.Generator(device="cpu").manual_seed(401)
    activation = torch.randn(rows, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (torch.randn(num_experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    runner = create_sm90_push_bf16_gemm_runner(
        max_rows=rows,
        num_experts=num_experts,
        n=n,
        k=k,
        device=device,
    )

    runner.run(output, activation, weights, offsets)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output,
        _reference_grouped_bf16(activation, weights, offsets),
        rtol=0.125,
        atol=0.125,
    )
    assert runner.workspace.dtype == torch.uint8
    assert runner.workspace.numel() >= runner.workspace_size


@requires_sm90
def test_direct_grouped_bf16_nonempty_capacity_with_all_experts_empty() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        create_sm90_push_bf16_gemm_runner,
    )

    device = torch.device("cuda", 0)
    capacity, num_experts, k, n = 64, 4, 256, 128
    activation = torch.empty(capacity, k, dtype=torch.bfloat16, device=device)
    weights = torch.empty(num_experts, n, k, dtype=torch.bfloat16, device=device)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    output = torch.full((capacity, n), 7.0, dtype=torch.bfloat16, device=device)
    runner = create_sm90_push_bf16_gemm_runner(
        max_rows=capacity,
        num_experts=num_experts,
        n=n,
        k=k,
        device=device,
    )

    runner.run(output, activation, weights, offsets)
    torch.cuda.synchronize()

    assert torch.all(output == 7.0)


@requires_sm90
def test_direct_grouped_bf16_reuses_fc1_schedule_for_fc2() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        Bf16GemmFamilyTactic,
        Bf16GemmTactic,
        create_sm90_push_bf16_gemm_runner,
    )

    device = torch.device("cuda", 0)
    counts = [0, 1, 64, 65, 127, 128, 129, 257]
    experts = len(counts)
    rows = sum(counts)
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(509)
    fc1_k, fc1_n, fc2_n = 128, 256, 128
    activation = torch.randn(rows, fc1_k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    w13 = (torch.randn(experts, fc1_n, fc1_k, generator=generator) * fc1_k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    w2 = (torch.randn(experts, fc2_n, fc1_n, generator=generator) * fc1_n**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    hidden = torch.empty(rows, fc1_n, dtype=torch.bfloat16, device=device)
    output = torch.empty(rows, fc2_n, dtype=torch.bfloat16, device=device)
    fc1_tactic = Bf16GemmTactic(
        "dual",
        m64=Bf16GemmFamilyTactic(64, 128, 128, 3, 1, "pingpong"),
        m128=Bf16GemmFamilyTactic(128, 128, 128, 3, 1, "pingpong"),
    )
    fc2_tactic = Bf16GemmTactic(
        "m64",
        m64=Bf16GemmFamilyTactic(64, 64, 64, 2, 1, "pingpong"),
        swap_ab=True,
    )
    fc1 = create_sm90_push_bf16_gemm_runner(
        max_rows=rows,
        num_experts=experts,
        n=fc1_n,
        k=fc1_k,
        device=device,
        tactic=fc1_tactic,
    )
    fc2 = create_sm90_push_bf16_gemm_runner(
        max_rows=rows,
        num_experts=experts,
        n=fc2_n,
        k=fc1_n,
        device=device,
        shared_schedule_workspace=fc1.schedule_workspace,
        tactic=fc2_tactic,
    )

    fc1.run(hidden, activation, w13, offsets, prepare_schedule=True)
    fc2.run(output, hidden, w2, offsets, prepare_schedule=False)
    torch.cuda.synchronize()

    hidden_reference = _reference_grouped_bf16(activation, w13, offsets)
    output_reference = _reference_grouped_bf16(hidden_reference, w2, offsets)
    torch.testing.assert_close(hidden, hidden_reference, rtol=0.125, atol=0.125)
    torch.testing.assert_close(output, output_reference, rtol=0.125, atol=0.125)
    provenance = fc1.tactic_provenance()
    assert provenance["selector"] == "forced"
    assert provenance["zero_m_descriptors_supported"] is True
    assert set(provenance["families"]) == {family.tag for family in fc1.tactic.families}
    assert fc1.tactic != fc2.tactic


@requires_sm90
def test_direct_grouped_bf16_prepared_schedule_is_one_shot() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        create_sm90_push_bf16_gemm_runner,
    )

    device = torch.device("cuda", 0)
    rows, experts, k, n = 8, 2, 128, 128
    activation = torch.empty(rows, k, dtype=torch.bfloat16, device=device)
    weights = torch.empty(experts, n, k, dtype=torch.bfloat16, device=device)
    offsets = torch.tensor([0, 0, rows], dtype=torch.int64, device=device)
    output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    first = create_sm90_push_bf16_gemm_runner(
        max_rows=rows, num_experts=experts, n=n, k=k, device=device
    )
    second = create_sm90_push_bf16_gemm_runner(
        max_rows=rows,
        num_experts=experts,
        n=n,
        k=k,
        device=device,
        shared_schedule_workspace=first.schedule_workspace,
    )

    first.run(output, activation, weights, offsets, prepare_schedule=True)
    second.run(output, activation, weights, offsets, prepare_schedule=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        second.run(output, activation, weights, offsets, prepare_schedule=False)


@requires_sm90
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("offset_dtype", "int64"),
        ("offset_length", "offsets shape mismatch"),
        ("weight_dtype", "bfloat16"),
        ("output_shape", "output N mismatch"),
    ],
)
def test_direct_grouped_bf16_rejects_invalid_contract(
    mutation: str, error: str
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        create_sm90_push_bf16_gemm_runner,
    )

    device = torch.device("cuda", 0)
    rows, experts, k, n = 8, 2, 256, 128
    activation = torch.empty(rows, k, dtype=torch.bfloat16, device=device)
    weights = torch.empty(experts, n, k, dtype=torch.bfloat16, device=device)
    offsets = torch.tensor([0, 3, rows], dtype=torch.int64, device=device)
    output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    if mutation == "offset_dtype":
        offsets = offsets.int()
    elif mutation == "offset_length":
        offsets = offsets[:-1]
    elif mutation == "weight_dtype":
        weights = weights.half()
    else:
        output = torch.empty(rows, n - 1, dtype=torch.bfloat16, device=device)
    runner = create_sm90_push_bf16_gemm_runner(
        max_rows=rows,
        num_experts=experts,
        n=n,
        k=k,
        device=device,
    )

    with pytest.raises((ValueError, RuntimeError), match=error):
        runner.run(output, activation, weights, offsets)


@requires_sm90
def test_direct_grouped_bf16_core_forced_tactics_match_oracle() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        CORE_BF16_GEMM_TACTICS,
        create_sm90_push_bf16_gemm_runner,
    )

    device = torch.device("cuda", 0)
    counts = [0, 1, 63, 64, 65, 127, 128, 129]
    experts, n, k = len(counts), 256, 256
    rows = sum(counts)
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(1701)
    activation = torch.randn(rows, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (torch.randn(experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    reference = _reference_grouped_bf16(activation, weights, offsets)

    seen_tags: set[str] = set()
    for tactic in CORE_BF16_GEMM_TACTICS:
        assert tactic.tag not in seen_tags
        seen_tags.add(tactic.tag)
        output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
        runner = create_sm90_push_bf16_gemm_runner(
            max_rows=rows,
            num_experts=experts,
            n=n,
            k=k,
            device=device,
            tactic=tactic,
        )

        runner.run(output, activation, weights, offsets)
        torch.cuda.synchronize()

        torch.testing.assert_close(output, reference, rtol=0.125, atol=0.125)
        provenance = runner.tactic_provenance()
        assert provenance["selected_tactic"]["tag"] == tactic.tag
        assert provenance["selection_reason"] == "forced internal BF16 GEMM tactic"
        assert provenance["module_uri"]
        assert provenance["families"]


@requires_sm90
@pytest.mark.parametrize("rows", EXPERT_ROWS)
def test_direct_grouped_bf16_auto_selector_m_boundaries(rows: int) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        create_sm90_push_bf16_gemm_runner,
        select_sm90_push_bf16_gemm_tactic,
    )

    device = torch.device("cuda", 0)
    experts, n, k = 4, 256, 256
    counts = [rows, 0, 0, 0]
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(1901 + rows)
    activation = torch.randn(rows, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (torch.randn(experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    runner = create_sm90_push_bf16_gemm_runner(
        max_rows=max(rows, 1),
        num_experts=experts,
        n=n,
        k=k,
        device=device,
        tactic="auto",
        expected_m=rows / experts,
    )
    expected, reason = select_sm90_push_bf16_gemm_tactic(
        expected_m=rows / experts,
        n=n,
        k=k,
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
    )

    runner.run(output, activation, weights, offsets)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output,
        _reference_grouped_bf16(activation, weights, offsets),
        rtol=0.125,
        atol=0.125,
    )
    provenance = runner.tactic_provenance()
    assert provenance["selected_tactic"]["tag"] == expected.tag
    assert provenance["selection_reason"] == reason
