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

Direct oracle tests for the persistent-offset SM90 BF16 grouped GEMM.
"""

from __future__ import annotations

import os

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


pytestmark = pytest.mark.skipif(
    os.environ.get("SM90_PUSH_BF16_ENABLE_ARCHIVED") != "1",
    reason="archived engine (see bf16_single_gpu_20260817)",
)
requires_sm90 = pytest.mark.skipif(
    not _sm90_cuda_12_available(),
    reason="requires an SM90 GPU and CUDA Toolkit 12.0+",
)

EXPERT_ROWS = (0, 1, 63, 64, 65, 127, 128, 129)


def _reference_grouped_bf16(
    activation: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
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


def _make_tactic(
    *,
    block_m: int,
    block_n: int = 128,
    block_k: int = 128,
    stages: int = 3,
    cluster_m: int = 1,
    schedule: str = "pingpong",
    swap_ab: bool = False,
):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
        Bf16GemmFamilyTactic,
        Bf16GemmTactic,
        normalize_bf16_gemm_tactic,
    )

    family = Bf16GemmFamilyTactic(
        block_m, block_n, block_k, stages, cluster_m, schedule
    )
    mode = "m64" if block_m == 64 else "m128"
    return normalize_bf16_gemm_tactic(
        Bf16GemmTactic(mode, **{mode: family}, swap_ab=swap_ab)
    )


def _required_tactics():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
        CORE_BF16_GEMM_TACTICS,
    )

    cluster_swap = _make_tactic(block_m=64, cluster_m=2, swap_ab=True)
    m128_n64_swap = _make_tactic(block_m=128, block_n=64, swap_ab=True)
    return tuple(dict.fromkeys((*CORE_BF16_GEMM_TACTICS, cluster_swap, m128_n64_swap)))


@requires_sm90
@pytest.mark.parametrize("active_expert", range(4))
def test_persistent_offsets_exact_m_boundaries(active_expert: int) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        create_sm90_push_bf16_persistent_gemm_runner,
    )

    device = torch.device("cuda", 0)
    experts, n, k = 4, 256, 256
    generator = torch.Generator(device="cpu").manual_seed(2101 + active_expert)
    weights = (torch.randn(experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    runner = create_sm90_push_bf16_persistent_gemm_runner(
        max_rows=max(EXPERT_ROWS),
        num_experts=experts,
        n=n,
        k=k,
        device=device,
        tactic=_make_tactic(block_m=64),
    )

    for rows in EXPERT_ROWS:
        activation = torch.randn(rows, k, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        counts = [0] * experts
        counts[active_expert] = rows
        offsets = torch.tensor(
            [0, *torch.tensor(counts).cumsum(0).tolist()],
            dtype=torch.int64,
            device=device,
        )
        output = torch.full((rows, n), 7.0, dtype=torch.bfloat16, device=device)

        result = runner.run(output, activation, weights, offsets)
        torch.cuda.synchronize()

        assert result is output
        torch.testing.assert_close(
            output,
            _reference_grouped_bf16(activation, weights, offsets),
            rtol=0.125,
            atol=0.125,
        )


@requires_sm90
def test_persistent_offsets_nonempty_capacity_with_all_experts_empty() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        create_sm90_push_bf16_persistent_gemm_runner,
    )

    device = torch.device("cuda", 0)
    capacity, experts, n, k = 129, 8, 256, 256
    activation = torch.empty(capacity, k, dtype=torch.bfloat16, device=device)
    weights = torch.empty(experts, n, k, dtype=torch.bfloat16, device=device)
    offsets = torch.zeros(experts + 1, dtype=torch.int64, device=device)
    output = torch.full((capacity, n), 7.0, dtype=torch.bfloat16, device=device)
    runner = create_sm90_push_bf16_persistent_gemm_runner(
        max_rows=capacity,
        num_experts=experts,
        n=n,
        k=k,
        device=device,
        tactic=_make_tactic(block_m=64),
    )

    runner.run(output, activation, weights, offsets)
    torch.cuda.synchronize()

    assert torch.all(output == 7.0)


@requires_sm90
def test_persistent_offsets_uneven_experts_match_oracle() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        create_sm90_push_bf16_persistent_gemm_runner,
    )

    device = torch.device("cuda", 0)
    counts = list(EXPERT_ROWS)
    experts, n, k = len(counts), 256, 256
    rows = sum(counts)
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(2201)
    activation = torch.randn(rows, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (torch.randn(experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    runner = create_sm90_push_bf16_persistent_gemm_runner(
        max_rows=rows,
        num_experts=experts,
        n=n,
        k=k,
        device=device,
        tactic=_make_tactic(block_m=64),
    )

    runner.run(output, activation, weights, offsets)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output,
        _reference_grouped_bf16(activation, weights, offsets),
        rtol=0.125,
        atol=0.125,
    )


@requires_sm90
def test_persistent_offsets_n64_k64_shape_matches_oracle() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        create_sm90_push_bf16_persistent_gemm_runner,
    )

    device = torch.device("cuda", 0)
    counts = [0, 1, 63, 65]
    experts, n, k = len(counts), 64, 64
    rows = sum(counts)
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(2251)
    activation = torch.randn(rows, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (torch.randn(experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    tactic = _make_tactic(block_m=64, block_n=64, block_k=64, stages=2)
    runner = create_sm90_push_bf16_persistent_gemm_runner(
        max_rows=rows,
        num_experts=experts,
        n=n,
        k=k,
        device=device,
        tactic=tactic,
    )

    runner.run(output, activation, weights, offsets)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output,
        _reference_grouped_bf16(activation, weights, offsets),
        rtol=0.125,
        atol=0.125,
    )


@requires_sm90
def test_persistent_offsets_core_dimensions_and_schedules_match_oracle() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        create_sm90_push_bf16_persistent_gemm_runner,
    )

    device = torch.device("cuda", 0)
    counts = list(EXPERT_ROWS)
    experts, n, k = len(counts), 256, 256
    rows = sum(counts)
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(2301)
    activation = torch.randn(rows, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (torch.randn(experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )
    reference = _reference_grouped_bf16(activation, weights, offsets)

    tactics = _required_tactics()
    families = [family for tactic in tactics for family in tactic.families]
    assert {family.block_n for family in families} == {64, 128}
    assert {family.block_k for family in families} == {64, 128}
    assert {family.stages for family in families} == {2, 3, 4}
    assert {family.cluster_m for family in families} == {1, 2}
    assert {tactic.swap_ab for tactic in tactics} == {False, True}
    assert any(
        tactic.swap_ab and tactic.families[0].cluster_m == 2 for tactic in tactics
    )
    assert {tactic.family_mode for tactic in tactics} == {"m64", "m128", "dual"}

    for tactic in tactics:
        output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
        runner = create_sm90_push_bf16_persistent_gemm_runner(
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
        assert provenance["implementation"] == "persistent_offsets"
        assert provenance["selected_tactic"]["tag"] == tactic.tag
        assert provenance["module_uri"] == runner.module_uri
        assert provenance["families"]


@requires_sm90
def test_persistent_and_cutlass_same_tactic_match_randomized() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        create_sm90_push_bf16_gemm_runner,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_persistent_gemm import (
        create_sm90_push_bf16_persistent_gemm_runner,
    )

    device = torch.device("cuda", 0)
    rows, experts, n, k = 777, 8, 256, 256
    generator = torch.Generator(device="cpu").manual_seed(2401)
    assignments = torch.randint(experts, (rows,), generator=generator)
    counts = torch.bincount(assignments, minlength=experts)
    offsets = torch.tensor(
        [0, *counts.cumsum(0).tolist()], dtype=torch.int64, device=device
    )
    activation = torch.randn(rows, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (torch.randn(experts, n, k, generator=generator) * k**-0.5).to(
        device=device, dtype=torch.bfloat16
    )

    for tactic in _required_tactics():
        cutlass_output = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
        persistent_output = torch.empty_like(cutlass_output)
        cutlass = create_sm90_push_bf16_gemm_runner(
            max_rows=rows,
            num_experts=experts,
            n=n,
            k=k,
            device=device,
            tactic=tactic,
        )
        persistent = create_sm90_push_bf16_persistent_gemm_runner(
            max_rows=rows,
            num_experts=experts,
            n=n,
            k=k,
            device=device,
            tactic=tactic,
        )

        cutlass.run(cutlass_output, activation, weights, offsets)
        persistent.run(persistent_output, activation, weights, offsets)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            persistent_output, cutlass_output, rtol=0.125, atol=0.125
        )
