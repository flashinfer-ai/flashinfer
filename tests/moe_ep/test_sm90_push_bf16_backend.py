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

Correctness gates for the public SM90 push BF16 mega-MoE backend.
"""

from __future__ import annotations

import gc
import os
import socket
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F


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


_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
requires_sm90 = pytest.mark.skipif(
    not _sm90_cuda_12_available() or _WORLD > 1,
    reason="requires one SM90 GPU and CUDA Toolkit 12.0+ outside torchrun",
)
requires_dist = pytest.mark.skipif(
    _WORLD < 2 or not _sm90_cuda_12_available(),
    reason="requires torchrun with at least two SM90 GPUs and CUDA Toolkit 12.0+",
)
archived_engine = pytest.mark.skipif(
    os.environ.get("SM90_PUSH_BF16_ENABLE_ARCHIVED") != "1",
    reason="archived engine (see bf16_single_gpu_20260817)",
)

HIDDEN = 256
INTERMEDIATE = 384
LOCAL_EXPERTS = 4
TOP_K = 2
TOKEN_CAPACITY = 257

_KEEP_ALIVE: list[object] = []


def _subprocess_env_with_free_port() -> dict[str, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    return env


def _assert_device_trap(result: subprocess.CompletedProcess[str], marker: str) -> None:
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"expected a device trap, process passed: {combined[-1500:]!r}"
    )
    assert "UNEXPECTED-SURVIVAL" not in result.stdout, combined[-1500:]
    assert marker in combined, f"trap marker {marker!r} missing: {combined[-1500:]!r}"
    for bad_marker in ("ImportError", "ModuleNotFoundError"):
        assert bad_marker not in combined, combined[-1500:]


def _make_weights(
    num_experts: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (
        torch.randn(
            num_experts,
            2 * INTERMEDIATE,
            HIDDEN,
            generator=generator,
        )
        * HIDDEN**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    w2 = (
        torch.randn(
            num_experts,
            HIDDEN,
            INTERMEDIATE,
            generator=generator,
        )
        * INTERMEDIATE**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    return w13, w2


def _make_inputs(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    seed: int,
    device: torch.device,
    *,
    mode: str = "random",
    rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, HIDDEN, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    if mode == "hot":
        ids = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    else:
        logits = torch.randn(num_tokens, num_experts, generator=generator)
        if mode == "all_remote" and num_experts > LOCAL_EXPERTS:
            local_start = rank * LOCAL_EXPERTS
            logits[:, local_start : local_start + LOCAL_EXPERTS] = float("-inf")
        ids = logits.topk(top_k, dim=1).indices.to(torch.int32)
    weights = torch.rand(num_tokens, top_k, generator=generator) + 0.1
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (
        hidden_states,
        ids.to(device),
        weights.to(device=device, dtype=torch.float32),
    )


def _reference_bf16_staged(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the BF16 handoffs independently of the backend implementation."""
    num_tokens, hidden = hidden_states.shape
    intermediate = w13.shape[1] // 2
    partials = torch.zeros(
        num_tokens,
        topk_ids.shape[1],
        hidden,
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    for expert in range(w13.shape[0]):
        selected = topk_ids == expert
        if not bool(selected.any()):
            continue
        token_indices, slot_indices = selected.nonzero(as_tuple=True)
        fc1 = (hidden_states[token_indices].float() @ w13[expert].float().T).to(
            torch.bfloat16
        )
        gate = F.silu(fc1[:, :intermediate].float())
        activation = (gate * fc1[:, intermediate:].float()).to(torch.bfloat16)
        fc2 = (activation.float() @ w2[expert].float().T).to(torch.bfloat16)
        weighted = (fc2.float() * topk_weights[token_indices, slot_indices, None]).to(
            torch.bfloat16
        )
        partials[token_indices, slot_indices] = weighted
    return partials.float().sum(dim=1).to(torch.bfloat16)


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    norm = reference.float().square().mean().sqrt().clamp_min(1e-6)
    error = (output.float() - reference.float()).square().mean().sqrt()
    return float(error / norm)


def _cosine(output: torch.Tensor, reference: torch.Tensor) -> float:
    if output.numel() == 0:
        return 1.0
    return float(
        F.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        )
    )


def _assert_matches_reference(output: torch.Tensor, reference: torch.Tensor) -> None:
    assert output.shape == reference.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    if output.numel():
        assert _cosine(output, reference) > 0.999
        assert _relative_error(output, reference) < 0.035


def _build_layer(
    world_size: int,
    rank: int,
    device: torch.device,
    *,
    top_k: int = TOP_K,
    dedup_dispatch: bool = True,
    capacity_factor: float = 1.0,
    wave_schedule: str = "mono",
    grouped_combine: bool = False,
    fuse_fc1_epilogue: bool = False,
    transformed_weights: object | None = None,
    seed: int = 7,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
    )

    total_experts = LOCAL_EXPERTS * world_size
    w13, w2 = _make_weights(total_experts, seed, device)
    local_start = rank * LOCAL_EXPERTS
    local_end = local_start + LOCAL_EXPERTS
    process_group = None
    if world_size > 1:
        import torch.distributed as dist

        process_group = dist.group.WORLD
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            process_group=process_group,
        ),
        fleet_params=FleetParams(
            num_experts=total_experts,
            max_tokens_per_rank=TOKEN_CAPACITY,
            token_hidden_size=HIDDEN,
        ),
        weights=(
            None
            if transformed_weights is not None
            else MoEWeightPack(
                w13=w13[local_start:local_end].contiguous(),
                w2=w2[local_start:local_end].contiguous(),
            )
        ),
        backend=MegaConfig(
            megakernel=Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=top_k,
                capacity_factor=capacity_factor,
                dedup_dispatch=dedup_dispatch,
                wave_schedule=wave_schedule,
                grouped_combine=grouped_combine,
                fuse_fc1_epilogue=fuse_fc1_epilogue,
            ),
            quantize_input=True,
            preprocess_weights=transformed_weights is None,
            transformed_weights=transformed_weights,
        ),
    )
    _KEEP_ALIVE.append(layer)
    return layer, w13, w2


def _forward(
    layer,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    from flashinfer.moe_ep import MoEEpTensors

    return layer(
        MoEEpTensors(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
    )


def _install_persistent_gemm_runner(layer) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_runner import (
        Sm90PushBf16MoERunner,
    )

    workspace = layer._ensure_workspace()
    original_runner = workspace.runner
    workspace.runner = Sm90PushBf16MoERunner(
        workspace.pipe,
        workspace.transformed_weights,
        gemm_implementation="persistent_offsets",
    )
    _KEEP_ALIVE.append(original_runner)


@requires_sm90
@pytest.mark.parametrize("dedup_dispatch", [False, True])
@pytest.mark.parametrize("grouped_combine", [False, True])
def test_public_ep1_forward_matches_staged_oracle(
    dedup_dispatch: bool,
    grouped_combine: bool,
) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1,
        0,
        device,
        dedup_dispatch=dedup_dispatch,
        grouped_combine=grouped_combine,
    )
    hidden_states, ids, weights = _make_inputs(73, LOCAL_EXPERTS, TOP_K, 11, device)

    output = _forward(layer, hidden_states, ids, weights)
    torch.cuda.synchronize()
    reference = _reference_bf16_staged(hidden_states, w13, w2, ids, weights)

    _assert_matches_reference(output, reference)


@requires_sm90
@archived_engine
@pytest.mark.parametrize("num_tokens", [0, 73])
def test_internal_persistent_ep1_matches_staged_oracle(num_tokens: int) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device)
    _install_persistent_gemm_runner(layer)
    hidden_states, ids, weights = _make_inputs(
        num_tokens, LOCAL_EXPERTS, TOP_K, 271, device
    )

    output = _forward(layer, hidden_states, ids, weights)
    torch.cuda.synchronize()

    _assert_matches_reference(
        output,
        _reference_bf16_staged(hidden_states, w13, w2, ids, weights),
    )


@requires_sm90
@archived_engine
def test_internal_persistent_ep1_graph_replay_after_eager_warmup() -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device)
    _install_persistent_gemm_runner(layer)
    first = _make_inputs(81, LOCAL_EXPERTS, TOP_K, 277, device)
    second = _make_inputs(81, LOCAL_EXPERTS, TOP_K, 281, device)
    static_hidden = first[0].clone()
    static_ids = first[1].clone()
    static_weights = first[2].clone()

    for _ in range(2):
        _forward(layer, static_hidden, static_ids, static_weights)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _forward(layer, static_hidden, static_ids, static_weights)

    for inputs in (first, second):
        static_hidden.copy_(inputs[0])
        static_ids.copy_(inputs[1])
        static_weights.copy_(inputs[2])
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches_reference(
            static_output,
            _reference_bf16_staged(inputs[0], w13, w2, inputs[1], inputs[2]),
        )


@requires_sm90
@pytest.mark.parametrize("wave_schedule", ["serial2", "pipe2"])
@pytest.mark.parametrize("num_tokens", [0, 1, 64, 65, 257])
def test_public_ep1_two_wave_matches_staged_oracle(
    wave_schedule: str, num_tokens: int
) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1,
        0,
        device,
        wave_schedule=wave_schedule,
    )
    hidden_states, ids, weights = _make_inputs(
        num_tokens,
        LOCAL_EXPERTS,
        TOP_K,
        710 + num_tokens,
        device,
    )

    output = _forward(layer, hidden_states, ids, weights)
    torch.cuda.synchronize()
    reference = _reference_bf16_staged(hidden_states, w13, w2, ids, weights)

    _assert_matches_reference(output, reference)


@requires_sm90
@archived_engine
def test_public_ep1_fused_fc1_matches_staged_oracle() -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1,
        0,
        device,
        fuse_fc1_epilogue=True,
    )
    hidden_states, ids, weights = _make_inputs(73, LOCAL_EXPERTS, TOP_K, 811, device)

    output = _forward(layer, hidden_states, ids, weights)
    torch.cuda.synchronize()
    reference = _reference_bf16_staged(hidden_states, w13, w2, ids, weights)

    _assert_matches_reference(output, reference)


@requires_sm90
@pytest.mark.parametrize(
    "num_tokens",
    [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 257],
)
def test_public_ep1_exact_expert_m_boundaries(num_tokens: int) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device, top_k=1)
    hidden_states, ids, weights = _make_inputs(
        num_tokens, LOCAL_EXPERTS, 1, 101 + num_tokens, device, mode="hot"
    )

    output = _forward(layer, hidden_states, ids, weights)
    torch.cuda.synchronize()
    reference = _reference_bf16_staged(hidden_states, w13, w2, ids, weights)

    _assert_matches_reference(output, reference)


@requires_sm90
@pytest.mark.parametrize("case", ["masked", "hot", "empty"])
def test_public_ep1_route_edges_and_recovery(case: str) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device)
    num_tokens = 0 if case == "empty" else 97
    hidden_states, ids, weights = _make_inputs(
        num_tokens,
        LOCAL_EXPERTS,
        TOP_K,
        211,
        device,
        mode="hot" if case == "hot" else "random",
    )
    if case == "masked":
        ids = ids.clone()
        ids[::3, 0] = -1

    output = _forward(layer, hidden_states, ids, weights)
    torch.cuda.synchronize()
    reference = _reference_bf16_staged(hidden_states, w13, w2, ids, weights)
    _assert_matches_reference(output, reference)

    recovery = _make_inputs(83, LOCAL_EXPERTS, TOP_K, 223, device)
    recovered = _forward(layer, *recovery)
    torch.cuda.synchronize()
    _assert_matches_reference(
        recovered,
        _reference_bf16_staged(recovery[0], w13, w2, recovery[1], recovery[2]),
    )


@requires_sm90
def test_public_ep1_soak() -> None:
    rounds = int(os.environ.get("SM90_PUSH_BF16_SOAK_ROUNDS", "60"))
    assert rounds >= 1
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device)
    token_choices = (0, 1, 7, 64, TOKEN_CAPACITY)
    for round_index in range(rounds):
        num_tokens = token_choices[(round_index * 7 + 3) % len(token_choices)]
        inputs = _make_inputs(
            num_tokens,
            LOCAL_EXPERTS,
            TOP_K,
            3000 + round_index,
            device,
            mode="hot" if round_index % 3 == 2 else "random",
        )
        output = _forward(layer, *inputs)
        torch.cuda.synchronize()
        _assert_matches_reference(
            output,
            _reference_bf16_staged(inputs[0], w13, w2, inputs[1], inputs[2]),
        )


@requires_sm90
def test_public_ep1_pool_overflow_traps_in_subprocess() -> None:
    code = r"""
import torch
from flashinfer.moe_ep import (
    BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer, MoEEpTensors,
    MoEWeightPack, Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
)
H, I, E, K, T = 256, 256, 4, 2, 64
w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda") * H ** -0.5
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * I ** -0.5
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=1, rank=0),
    fleet_params=FleetParams(
        num_experts=E, max_tokens_per_rank=T, token_hidden_size=H,
    ),
    weights=MoEWeightPack(w13=w13, w2=w2),
    backend=MegaConfig(
        megakernel=Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(
            intermediate_size=I, top_k=K, capacity_factor=0.25,
        ),
        quantize_input=True,
        preprocess_weights=True,
    ),
)
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda").expand(T, K).contiguous()
weights = torch.full((T, K), 0.5, dtype=torch.float32, device="cuda")
layer(MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights))
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=_subprocess_env_with_free_port(),
    )
    _assert_device_trap(result, "sm90_push: dedup pool overflow")


@requires_sm90
def test_internal_silu_rejects_device_row_count_above_capacity() -> None:
    code = r"""
import torch
from flashinfer.moe_ep import (
    BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer, MoEWeightPack,
    Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
)
H, I, E, K, T = 256, 256, 4, 2, 8
w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda")
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda")
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=1, rank=0),
    fleet_params=FleetParams(
        num_experts=E, max_tokens_per_rank=T, token_hidden_size=H,
    ),
    weights=MoEWeightPack(w13=w13, w2=w2),
    backend=MegaConfig(
        megakernel=Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(intermediate_size=I, top_k=K),
        quantize_input=True,
        preprocess_weights=True,
    ),
)
workspace = layer._ensure_workspace()
g = torch.empty(4, I, dtype=torch.bfloat16, device="cuda")
h = torch.empty(4, 2 * I, dtype=torch.bfloat16, device="cuda")
m_dev = torch.tensor([5], dtype=torch.int32, device="cuda")
workspace.pipe.module.sm90_silu_mul_gated(g, h, m_dev, 4)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=_subprocess_env_with_free_port(),
    )
    _assert_device_trap(result, "sm90_push: SiLU row count 5 exceeds capacity 4")


@requires_sm90
def test_transformed_weights_bypass_preprocessing() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        make_sm90_push_bf16_weights,
    )

    device = torch.device("cuda", 0)
    local_w13, local_w2 = _make_weights(LOCAL_EXPERTS, 229, device)
    transformed = make_sm90_push_bf16_weights(local_w13, local_w2)
    layer, _, _ = _build_layer(
        1,
        0,
        device,
        transformed_weights=transformed,
        seed=229,
    )
    hidden_states, ids, weights = _make_inputs(79, LOCAL_EXPERTS, TOP_K, 233, device)

    output = _forward(layer, hidden_states, ids, weights)
    torch.cuda.synchronize()

    _assert_matches_reference(
        output,
        _reference_bf16_staged(hidden_states, local_w13, local_w2, ids, weights),
    )


@requires_sm90
def test_transformed_weight_validation_rejects_noncontiguous_bundle() -> None:
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.weights import (
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        Sm90PushBf16Weights,
    )

    device = torch.device("cuda", 0)
    storage = torch.empty(
        LOCAL_EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        2,
        dtype=torch.bfloat16,
        device=device,
    )
    w13 = storage[..., 0]
    w2 = torch.empty(
        LOCAL_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    transformed = object.__new__(Sm90PushBf16Weights)
    object.__setattr__(transformed, "w13", w13)
    object.__setattr__(transformed, "w2", w2)

    with pytest.raises(MoEEpConfigError, match="contiguous"):
        validate_transformed_mega_weights(
            transformed,
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            num_local_experts=LOCAL_EXPERTS,
        )


@requires_sm90
def test_transformed_weight_validation_requires_fused_wmma_alignment() -> None:
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.weights import (
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        Sm90PushBf16Weights,
    )

    device = torch.device("cuda", 0)
    w13_elements = LOCAL_EXPERTS * 2 * INTERMEDIATE * HIDDEN
    w13_storage = torch.empty(w13_elements + 8, dtype=torch.bfloat16, device=device)
    w13 = w13_storage[8:].view(LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    w2 = torch.empty(
        LOCAL_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        dtype=torch.bfloat16,
        device=device,
    )
    transformed = Sm90PushBf16Weights(w13=w13, w2=w2)
    assert w13.is_contiguous()
    assert w13.data_ptr() % 32 == 16

    validate_transformed_mega_weights(
        transformed,
        intermediate_size=INTERMEDIATE,
        hidden_size=HIDDEN,
        num_local_experts=LOCAL_EXPERTS,
        fuse_fc1_epilogue=False,
    )
    with pytest.raises(MoEEpConfigError, match="32-byte aligned"):
        validate_transformed_mega_weights(
            transformed,
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            num_local_experts=LOCAL_EXPERTS,
            fuse_fc1_epilogue=True,
        )


@requires_sm90
def test_transformed_weight_validation_requires_cutlass_alignment() -> None:
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.weights import (
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        Sm90PushBf16Weights,
    )

    device = torch.device("cuda", 0)
    w13 = torch.empty(
        LOCAL_EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        dtype=torch.bfloat16,
        device=device,
    )
    w2_elements = LOCAL_EXPERTS * HIDDEN * INTERMEDIATE
    w2_storage = torch.empty(w2_elements + 1, dtype=torch.bfloat16, device=device)
    w2 = w2_storage[1:].view(LOCAL_EXPERTS, HIDDEN, INTERMEDIATE)
    transformed = Sm90PushBf16Weights(w13=w13, w2=w2)
    assert w2.is_contiguous()
    assert w2.data_ptr() % 16 == 2

    with pytest.raises(MoEEpConfigError, match="w2 must be 16-byte aligned"):
        validate_transformed_mega_weights(
            transformed,
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            num_local_experts=LOCAL_EXPERTS,
        )


@requires_sm90
@pytest.mark.parametrize(
    "fuse_fc1_epilogue",
    [False, pytest.param(True, marks=archived_engine)],
)
@pytest.mark.parametrize("grouped_combine", [False, True])
def test_public_ep1_validation_output_identity_and_graph_replay(
    fuse_fc1_epilogue: bool,
    grouped_combine: bool,
) -> None:
    from flashinfer.moe_ep import MoEEpConfigError

    device = torch.device("cuda", 0)
    layer, _, _ = _build_layer(
        1,
        0,
        device,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
        grouped_combine=grouped_combine,
    )
    first = _make_inputs(89, LOCAL_EXPERTS, TOP_K, 239, device)
    second = _make_inputs(89, LOCAL_EXPERTS, TOP_K, 241, device)

    with pytest.raises(MoEEpConfigError, match="bf16"):
        _forward(layer, first[0].half(), first[1], first[2])
    with pytest.raises(MoEEpConfigError, match="int32"):
        _forward(layer, first[0], first[1].long(), first[2])
    with pytest.raises(MoEEpConfigError, match="float32"):
        _forward(layer, first[0], first[1], first[2].half())

    first_output = _forward(layer, *first)
    torch.cuda.synchronize()
    first_snapshot = first_output.clone()
    second_output = _forward(layer, *second)
    torch.cuda.synchronize()
    assert first_output.data_ptr() != second_output.data_ptr()
    assert torch.equal(first_output, first_snapshot)
    assert not torch.equal(first_output, second_output)
    eager = [first_snapshot, second_output.clone()]

    static_hidden = first[0].clone()
    static_ids = first[1].clone()
    static_weights = first[2].clone()
    side_stream = torch.cuda.Stream()
    for _ in range(2):
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(layer, static_hidden, static_ids, static_weights)
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _forward(layer, static_hidden, static_ids, static_weights)
    for inputs, expected in zip((first, second), eager, strict=True):
        static_hidden.copy_(inputs[0])
        static_ids.copy_(inputs[1])
        static_weights.copy_(inputs[2])
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_output, expected, rtol=5e-3, atol=1e-2)


@requires_sm90
@pytest.mark.parametrize("wave_schedule", ["serial2", "pipe2"])
def test_public_ep1_two_wave_rejects_graph_capture(wave_schedule: str) -> None:
    device = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, device, wave_schedule=wave_schedule)
    inputs = _make_inputs(65, LOCAL_EXPERTS, TOP_K, 247, device)
    _forward(layer, *inputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="two-wave scheduling"),
        torch.cuda.graph(graph),
    ):
        _forward(layer, *inputs)


@requires_sm90
def test_public_ep1_destroy_releases_workspace_allocations() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.synchronize(device)
    gc.collect()

    layer, _, _ = _build_layer(1, 0, device)
    after_weights = torch.cuda.memory_allocated(device)
    layer._ensure_workspace()
    torch.cuda.synchronize(device)
    with_workspace = torch.cuda.memory_allocated(device)

    layer.destroy()
    layer.destroy()
    gc.collect()
    torch.cuda.synchronize(device)
    after_destroy = torch.cuda.memory_allocated(device)

    assert layer._workspace is None
    assert with_workspace > after_weights
    assert with_workspace - after_destroy >= 1 << 20
    assert after_destroy <= after_weights + (1 << 20)


def _dist_setup() -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        except (ValueError, RuntimeError):
            dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", str(rank))))
    return rank, world_size


@requires_dist
@pytest.mark.parametrize(
    (
        "dedup_dispatch",
        "route_mode",
        "wave_schedule",
        "grouped_combine",
        "fuse_fc1_epilogue",
    ),
    [
        (False, "random", "mono", False, False),
        (True, "all_remote", "mono", True, False),
        (True, "random", "serial2", False, False),
        (True, "all_remote", "pipe2", True, False),
        pytest.param(True, "random", "mono", False, True, marks=archived_engine),
        pytest.param(True, "all_remote", "mono", True, True, marks=archived_engine),
        pytest.param(True, "random", "pipe2", False, True, marks=archived_engine),
    ],
)
def test_public_multirank_forward_configs(
    dedup_dispatch: bool,
    route_mode: str,
    wave_schedule: str,
    grouped_combine: bool,
    fuse_fc1_epilogue: bool,
) -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", torch.cuda.current_device())
    layer, w13, w2 = _build_layer(
        world_size,
        rank,
        device,
        dedup_dispatch=dedup_dispatch,
        wave_schedule=wave_schedule,
        grouped_combine=grouped_combine,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    hidden_states, ids, weights = _make_inputs(
        91,
        LOCAL_EXPERTS * world_size,
        TOP_K,
        251 + rank,
        device,
        mode=route_mode,
        rank=rank,
    )

    output = _forward(layer, hidden_states, ids, weights)
    torch.cuda.synchronize()
    reference = _reference_bf16_staged(hidden_states, w13, w2, ids, weights)
    _assert_matches_reference(output, reference)
    dist.barrier()


@requires_dist
def test_public_multirank_uneven_empty_and_recovery() -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", torch.cuda.current_device())
    layer, w13, w2 = _build_layer(world_size, rank, device)
    num_tokens = 0 if rank == 1 else max(TOKEN_CAPACITY - 17 * rank, 1)
    inputs = _make_inputs(
        num_tokens,
        LOCAL_EXPERTS * world_size,
        TOP_K,
        263 + rank,
        device,
        rank=rank,
    )
    output = _forward(layer, *inputs)
    torch.cuda.synchronize()
    _assert_matches_reference(
        output,
        _reference_bf16_staged(inputs[0], w13, w2, inputs[1], inputs[2]),
    )

    recovery = _make_inputs(
        97,
        LOCAL_EXPERTS * world_size,
        TOP_K,
        277 + rank,
        device,
        rank=rank,
    )
    recovered = _forward(layer, *recovery)
    torch.cuda.synchronize()
    _assert_matches_reference(
        recovered,
        _reference_bf16_staged(recovery[0], w13, w2, recovery[1], recovery[2]),
    )
    dist.barrier()


@requires_dist
def test_public_multirank_soak() -> None:
    import torch.distributed as dist

    rounds = int(os.environ.get("SM90_PUSH_BF16_MULTIRANK_SOAK_ROUNDS", "60"))
    assert rounds >= 1
    rank, world_size = _dist_setup()
    device = torch.device("cuda", torch.cuda.current_device())
    layer, w13, w2 = _build_layer(world_size, rank, device)
    token_choices = (0, 1, 8, 63, 97)
    for round_index in range(rounds):
        num_tokens = token_choices[(round_index + rank * 3) % len(token_choices)]
        route_mode = "all_remote" if round_index % 2 else "random"
        inputs = _make_inputs(
            num_tokens,
            LOCAL_EXPERTS * world_size,
            TOP_K,
            5000 + round_index * world_size + rank,
            device,
            mode=route_mode,
            rank=rank,
        )
        output = _forward(layer, *inputs)
        torch.cuda.synchronize()
        _assert_matches_reference(
            output,
            _reference_bf16_staged(inputs[0], w13, w2, inputs[1], inputs[2]),
        )
    dist.barrier()
