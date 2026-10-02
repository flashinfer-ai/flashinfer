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

Contracts and direct oracles for the SM90 push BF16 fused FC1 tactic.
"""

from __future__ import annotations

import os
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest
import torch


_PACKAGE_NAME = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = (
    _PROJECT_ROOT
    / "flashinfer"
    / "moe_ep"
    / "kernel_src"
    / "sm90"
    / "push_style_megamoe"
)


def _package_text(*parts: str) -> str:
    source_tree = _PACKAGE_ROOT.joinpath(*parts)
    if source_tree.is_file():
        return source_tree.read_text(encoding="utf-8")

    resource = resources.files(_PACKAGE_NAME)
    for part in parts:
        resource = resource / part
    return resource.read_text(encoding="utf-8")


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
    not _sm90_cuda_12_available()
    or os.environ.get("SM90_PUSH_BF16_ENABLE_ARCHIVED") != "1",
    reason=(
        "requires an SM90 GPU, CUDA Toolkit 12.0+, and the archived-engine opt-in "
        "(see bf16_single_gpu_20260817)"
    ),
)

M_BOUNDARIES = (0, 1, 7, 8, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129)


def test_archived_fused_fc1_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_fc1_fused import (
        _check_archived_gate,
    )

    monkeypatch.delenv("SM90_PUSH_BF16_ENABLE_ARCHIVED", raising=False)
    with pytest.raises(RuntimeError, match="bf16_single_gpu_20260817"):
        _check_archived_gate()
    monkeypatch.setenv("SM90_PUSH_BF16_ENABLE_ARCHIVED", "1")
    _check_archived_gate()


def _offsets(counts: list[int], device: torch.device) -> torch.Tensor:
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)
    return torch.tensor(prefix, dtype=torch.int64, device=device)


def _fp32_fused_oracle(
    activation: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    intermediate_size = weights.shape[1] // 2
    result = torch.empty(
        activation.shape[0],
        intermediate_size,
        dtype=torch.bfloat16,
        device=activation.device,
    )
    host_activation = activation.float().cpu()
    host_weights = weights.float().cpu()
    host_offsets = offsets.cpu().tolist()
    for expert, (start, end) in enumerate(
        zip(host_offsets[:-1], host_offsets[1:], strict=True)
    ):
        if start == end:
            continue
        gate = host_activation[start:end] @ host_weights[expert, :intermediate_size].T
        up = host_activation[start:end] @ host_weights[expert, intermediate_size:].T
        fused = gate / (1.0 + torch.exp(-gate)) * up
        result[start:end].copy_(fused.to(torch.bfloat16).to(activation.device))
    return result


def _bf16_unfused_oracle(projected: torch.Tensor) -> torch.Tensor:
    intermediate_size = projected.shape[1] // 2
    host = projected.cpu()
    gate = host[:, :intermediate_size].float()
    up = host[:, intermediate_size:].float()
    result = gate / (1.0 + torch.exp(-gate)) * up
    return result.to(torch.bfloat16).to(projected.device)


def test_fused_fc1_source_contract() -> None:
    kernel = _package_text("src", "bf16_fc1_fused", "bf16_fc1_fused.cuh")
    binding = _package_text("src", "bf16_fc1_fused", "bf16_fc1_fused_binding.cu")

    assert kernel.count("wmma::mma_sync") == 2
    assert kernel.count("__shared__ __align__(32)") == 2
    assert "float const gate = gate_accumulator.x[element]" in kernel
    assert "* up_accumulator.x[element]" in kernel
    assert "__float2bfloat16_rn(output_tile[element])" in kernel
    assert "bf16_fc1_unfused_epilogue_kernel" in kernel
    assert "weights.size(1), static_cast<int64_t>(intermediate_size_) * 2" in binding
    assert "output.size(1), intermediate_size_" in binding
    assert 'name == "run_unfused_epilogue"' in binding


def test_fused_fc1_sources_are_packaged_but_not_stable_exports() -> None:
    pyproject_path = _PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        pytest.skip("pyproject.toml is only available in source-tree test runs")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    package = _package_text("shim", "__init__.py")

    assert '"src/bf16_fc1_fused/*.cu"' in pyproject
    assert '"src/bf16_fc1_fused/*.cuh"' in pyproject
    assert "create_sm90_push_bf16_fused_fc1_runner" not in package
    assert "gen_sm90_push_bf16_fused_fc1_module" not in package


def test_fused_fc1_digest_covers_kernel_binding_and_flags() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        bf16_fc1_fused,
    )

    snapshot = bf16_fc1_fused._capture_source_snapshot()
    baseline = bf16_fc1_fused._source_digest(snapshot)
    for index in range(len(snapshot.sources)):
        name, content = snapshot.sources[index]
        changed_sources = list(snapshot.sources)
        changed_sources[index] = (name, content + b"\n")
        changed = replace(snapshot, sources=tuple(changed_sources))
        assert bf16_fc1_fused._source_digest(changed) != baseline

    original_flags = bf16_fc1_fused._cuda_flags
    try:
        bf16_fc1_fused._cuda_flags = lambda: (*original_flags(), "-DFUSED_FC1_TEST=1")
        assert bf16_fc1_fused._source_digest(snapshot) != baseline
    finally:
        bf16_fc1_fused._cuda_flags = original_flags


def test_fused_fc1_rejects_old_cuda_before_materialization(monkeypatch) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        bf16_fc1_fused,
    )

    monkeypatch.setattr(bf16_fc1_fused, "is_cuda_version_at_least", lambda _: False)
    with pytest.raises(RuntimeError, match="CUDA 12.0"):
        bf16_fc1_fused._make_jit_spec()


@requires_sm90
@pytest.mark.parametrize("active_expert", range(3))
def test_fused_fc1_matches_fp32_accumulation_oracle(active_expert: int) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_fc1_fused import (
        create_sm90_push_bf16_fused_fc1_runner,
    )

    device = torch.device("cuda", 0)
    experts, intermediate_size, k = 3, 48, 64
    generator = torch.Generator(device="cpu").manual_seed(1701 + active_expert)
    weights = (
        torch.randn(experts, 2 * intermediate_size, k, generator=generator) * k**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    runner = create_sm90_push_bf16_fused_fc1_runner(
        max_rows=max(M_BOUNDARIES),
        num_experts=experts,
        intermediate_size=intermediate_size,
        k=k,
        device=device,
    )

    for rows in M_BOUNDARIES:
        activation = torch.randn(rows, k, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        counts = [0] * experts
        counts[active_expert] = rows
        offsets = _offsets(counts, device)
        output = torch.empty(
            rows, intermediate_size, dtype=torch.bfloat16, device=device
        )

        result = runner.run(output, activation, weights, offsets)
        torch.cuda.synchronize()
        reference = _fp32_fused_oracle(activation, weights, offsets)

        assert result is output
        torch.testing.assert_close(output, reference, rtol=0.15, atol=0.15)


@requires_sm90
def test_fused_fc1_handles_empty_and_uneven_experts() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_fc1_fused import (
        create_sm90_push_bf16_fused_fc1_runner,
    )

    device = torch.device("cuda", 0)
    counts = [0, 1, 17, 0, 65, 129]
    rows = sum(counts)
    experts, intermediate_size, k = len(counts), 64, 80
    generator = torch.Generator(device="cpu").manual_seed(1801)
    activation = torch.randn(rows, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (
        torch.randn(experts, 2 * intermediate_size, k, generator=generator) * k**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    offsets = _offsets(counts, device)
    output = torch.empty(rows, intermediate_size, dtype=torch.bfloat16, device=device)
    runner = create_sm90_push_bf16_fused_fc1_runner(
        max_rows=rows,
        num_experts=experts,
        intermediate_size=intermediate_size,
        k=k,
        device=device,
    )

    runner.run(output, activation, weights, offsets)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output,
        _fp32_fused_oracle(activation, weights, offsets),
        rtol=0.15,
        atol=0.15,
    )


@requires_sm90
def test_fused_fc1_accepts_active_prefix_below_capacity() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_fc1_fused import (
        create_sm90_push_bf16_fused_fc1_runner,
    )

    device = torch.device("cuda", 0)
    capacity, active_rows = 64, 23
    experts, intermediate_size, k = 3, 32, 32
    generator = torch.Generator(device="cpu").manual_seed(1851)
    activation = torch.randn(capacity, k, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    weights = (
        torch.randn(experts, 2 * intermediate_size, k, generator=generator) * k**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    offsets = _offsets([0, 7, 16], device)
    sentinel = torch.tensor(-13.0, dtype=torch.bfloat16, device=device)
    output = torch.full(
        (capacity, intermediate_size), -13.0, dtype=torch.bfloat16, device=device
    )
    runner = create_sm90_push_bf16_fused_fc1_runner(
        max_rows=capacity,
        num_experts=experts,
        intermediate_size=intermediate_size,
        k=k,
        device=device,
    )

    runner.run(output, activation, weights, offsets)
    torch.cuda.synchronize()

    reference = _fp32_fused_oracle(activation[:active_rows], weights, offsets)
    torch.testing.assert_close(output[:active_rows], reference, rtol=0.15, atol=0.15)
    assert torch.equal(output[active_rows:], sentinel.expand_as(output[active_rows:]))


@requires_sm90
def test_unfused_epilogue_matches_materialized_bf16_oracle() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_fc1_fused import (
        create_sm90_push_bf16_fused_fc1_runner,
    )

    device = torch.device("cuda", 0)
    capacity, active_rows = 37, 23
    experts, intermediate_size, k = 2, 64, 64
    generator = torch.Generator(device="cpu").manual_seed(1901)
    projected = torch.randn(capacity, 2 * intermediate_size, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    output = torch.full(
        (capacity, intermediate_size), -17.0, dtype=torch.bfloat16, device=device
    )
    runner = create_sm90_push_bf16_fused_fc1_runner(
        max_rows=capacity,
        num_experts=experts,
        intermediate_size=intermediate_size,
        k=k,
        device=device,
    )

    offsets = _offsets([0, active_rows], device)
    runner.run_unfused_epilogue(output, projected, offsets)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output[:active_rows],
        _bf16_unfused_oracle(projected[:active_rows]),
        rtol=0.015,
        atol=0.015,
    )
    assert torch.equal(
        output[active_rows:],
        torch.full_like(output[active_rows:], -17.0),
    )


@requires_sm90
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("activation_dtype", "bfloat16"),
        ("weight_width", "paired-output"),
        ("weight_alignment", "32-byte aligned"),
        ("offset_dtype", "int64"),
        ("offset_length", "offsets shape mismatch"),
        ("output_width", "intermediate dimension"),
    ],
)
def test_fused_fc1_rejects_invalid_contract(mutation: str, error: str) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_fc1_fused import (
        create_sm90_push_bf16_fused_fc1_runner,
    )

    device = torch.device("cuda", 0)
    rows, experts, intermediate_size, k = 8, 2, 32, 32
    activation = torch.empty(rows, k, dtype=torch.bfloat16, device=device)
    weights = torch.empty(
        experts, 2 * intermediate_size, k, dtype=torch.bfloat16, device=device
    )
    offsets = torch.tensor([0, 3, rows], dtype=torch.int64, device=device)
    output = torch.empty(rows, intermediate_size, dtype=torch.bfloat16, device=device)
    if mutation == "activation_dtype":
        activation = activation.half()
    elif mutation == "weight_width":
        weights = weights[:, :-1, :].contiguous()
    elif mutation == "weight_alignment":
        weight_elements = experts * 2 * intermediate_size * k
        weight_storage = torch.empty(
            weight_elements + 8, dtype=torch.bfloat16, device=device
        )
        weights = weight_storage[8:].view(experts, 2 * intermediate_size, k)
        assert weights.is_contiguous()
        assert weights.data_ptr() % 32 == 16
    elif mutation == "offset_dtype":
        offsets = offsets.int()
    elif mutation == "offset_length":
        offsets = offsets[:-1]
    else:
        output = torch.empty(
            rows,
            intermediate_size - 1,
            dtype=torch.bfloat16,
            device=device,
        )
    runner = create_sm90_push_bf16_fused_fc1_runner(
        max_rows=rows,
        num_experts=experts,
        intermediate_size=intermediate_size,
        k=k,
        device=device,
    )

    with pytest.raises((ValueError, RuntimeError), match=error):
        runner.run(output, activation, weights, offsets)
