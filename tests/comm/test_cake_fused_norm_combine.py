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

Tests for the SM100 / SM103 eight-peer Cake fused norm-combine export.

The CPU tests check the token dispatch, the persistent grid rule, the
generated programs' launch records, the target scope, the workspace layout
helper and the argument validation of the public API.  The distributed test
runs the public API on eight SM100 or eight SM103 GPUs against an independent
PyTorch reference of the same math, over every dispatch band and both
directions of every band switch on one workspace."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import socket

import pytest
import torch

import flashinfer.comm as comm
from flashinfer.jit import cake_fused_norm_combine as loader

# ``flashinfer.comm`` re-exports the launch function under the module's own name, so the
# attribute path resolves to the function; bind the module object explicitly.
api = importlib.import_module("flashinfer.comm.cake_fused_norm_combine")

# Every dispatch band, both sides of every band edge, and the band interiors.
TOKENS_UNDER_TEST = (
    1,
    2,
    3,
    4,
    8,
    37,
    38,
    64,
    256,
    479,
    480,
    592,
    593,
    1024,
    1772,
    1773,
    2048,
)
ATOL = 1e-2
RTOL = 1e-2
WORKER_TIMEOUT_SECONDS = 20 * 60


@pytest.mark.parametrize(
    "tokens,sm_count,expected",
    [
        (1, 148, api.VARIANT_PARALLEL_SINGLE),
        (2, 148, api.VARIANT_PARALLEL),
        (api.SPLIT_MIN_TOKENS - 1, 148, api.VARIANT_PARALLEL),
        (api.SPLIT_MIN_TOKENS, 148, api.VARIANT_PARALLEL_SPLIT),
        (api.STREAM_MIN_TOKENS - 1, 148, api.VARIANT_PARALLEL_SPLIT),
        (api.STREAM_MIN_TOKENS, 148, api.VARIANT_OWNER_STREAM),
        (api.PIPELINED_MIN_TOKENS - 1, 148, api.VARIANT_OWNER_STREAM),
        (api.PIPELINED_MIN_TOKENS, 148, api.VARIANT_OWNER_PIPELINED),
        # 148 SMs: four CTAs per SM hold 592 tokens in one step; the
        # three-CTA grid of 443 CTAs covers 1772 tokens in four steps.
        (592, 148, api.VARIANT_OWNER_PIPELINED),
        (593, 148, api.VARIANT_OWNER_PIPELINED_EARLY),
        (1772, 148, api.VARIANT_OWNER_PIPELINED_EARLY),
        (1773, 148, api.VARIANT_OWNER_PIPELINED),
        (2048, 148, api.VARIANT_OWNER_PIPELINED),
        # 152 SMs: 608 tokens in one step, a 455-CTA grid for four steps.
        (608, 152, api.VARIANT_OWNER_PIPELINED),
        (609, 152, api.VARIANT_OWNER_PIPELINED_EARLY),
        (1820, 152, api.VARIANT_OWNER_PIPELINED_EARLY),
        (1821, 152, api.VARIANT_OWNER_PIPELINED),
    ],
)
def test_token_dispatch(tokens: int, sm_count: int, expected: str) -> None:
    assert 1 < api.SPLIT_MIN_TOKENS < api.STREAM_MIN_TOKENS < api.PIPELINED_MIN_TOKENS
    assert api.select_variant(tokens, sm_count) == expected
    assert expected in api.VARIANTS


@pytest.mark.parametrize("tokens,sm_count", [(0, 148), (-1, 148), (True, 148), (8, 0)])
def test_token_dispatch_rejects_non_positive_counts(tokens, sm_count) -> None:
    with pytest.raises(ValueError):
        api.select_variant(tokens, sm_count)


@pytest.mark.parametrize(
    "tokens,capacity,expected",
    [
        (1, 592, 1),
        (592, 592, 592),
        (593, 592, 591),
        (2048, 592, 591),
        (2048, 148, 147),
        (2048, 149, 149),
        (2048, 2, 1),
    ],
)
def test_persistent_grid_is_every_token_or_the_largest_odd_co_resident_grid(
    tokens: int, capacity: int, expected: int
) -> None:
    assert api.persistent_grid(tokens, capacity) == expected


def test_grid_rules() -> None:
    assert api.launch_grid_x(api.VARIANT_PARALLEL_SINGLE, 1, 148) == 1
    assert api.launch_grid_x(api.VARIANT_PARALLEL, 2, 148) == 2
    assert api.launch_grid_x(api.VARIANT_PARALLEL_SPLIT, 8, 148) == 64
    assert api.launch_grid_x(api.VARIANT_OWNER_STREAM, 64, 148) == 64
    assert api.launch_grid_x(api.VARIANT_OWNER_STREAM, 256, 148) == 147
    assert api.launch_grid_x(api.VARIANT_OWNER_PIPELINED, 512, 148) == 512
    assert api.launch_grid_x(api.VARIANT_OWNER_PIPELINED, 2048, 148) == 591
    assert api.launch_grid_x(api.VARIANT_OWNER_PIPELINED_EARLY, 593, 148) == 443
    assert api.launch_grid_x(api.VARIANT_OWNER_PIPELINED_EARLY, 1772, 148) == 443
    with pytest.raises(ValueError):
        api.launch_grid_x("unknown", 8, 148)
    with pytest.raises(ValueError):
        api.persistent_grid(0, 148)


def test_generated_programs_match_the_host_plan() -> None:
    programs = tuple(getattr(loader, name) for name in api.VARIANTS)
    assert programs == loader.PROGRAMS
    host_args = (
        "x",
        "residual",
        "weight",
        "norm_out",
        "residual_out",
        "collective_out",
        "workspace",
        "rank",
        "tokens",
        "epsilon",
        "grid_x",
        "grid_y",
        "grid_z",
    )
    for name in api.VARIANTS:
        program = getattr(loader, name)
        assert program.targets == ("sm_100a", "sm_103a")
        assert program.args == host_args
        assert program.use_pdl
        assert program.cluster == (1, 1, 1)
        # Only the persistent programs synchronize grid-wide.
        assert program.cooperative == (name in api.CTAS_PER_SM)
        assert program.target_for((10, 0)) == "sm_100a"
        assert program.target_for((10, 3)) == "sm_103a"
        for capability in ((9, 0), (10, 7), (12, 0)):
            assert program.target_for(capability) is None


def test_target_scope(monkeypatch) -> None:
    facts = {0: ((10, 0), 148), 1: ((10, 3), 148), 2: ((9, 0), 132)}
    monkeypatch.setattr(api, "_device_facts", lambda index: facts[index])
    assert api.cake_fused_norm_combine_target(torch.device("cuda", 0)) == "sm_100a"
    assert api.cake_fused_norm_combine_target(torch.device("cuda", 1)) == "sm_103a"
    with pytest.raises(ValueError, match="10.0"):
        api.cake_fused_norm_combine_target(torch.device("cuda", 2))


def test_workspace_layout_matches_the_kernel_contract() -> None:
    sizes = api.cake_fused_norm_combine_workspace_bytes(8, 2048, 2560)
    assert sizes["data_bytes"] == 4 * 2048 * 2560
    assert sizes["flags_bytes"] == 4 * 2048 * 8
    assert sizes["rotation_bytes"] == 2 * 8 * 2048 * 2560
    assert sizes["lamport_bytes"] == 3 * sizes["rotation_bytes"]
    assert sizes["control_bytes"] == 20
    assert sizes["table_entries"] == 25
    with pytest.raises(ValueError):
        api.cake_fused_norm_combine_workspace_bytes(4, 2048, 2560)
    with pytest.raises(ValueError):
        api.cake_fused_norm_combine_workspace_bytes(8, 0, 2560)
    with pytest.raises(ValueError):
        api.cake_fused_norm_combine_workspace_bytes(8, 2048, 4096)


def test_public_api_rejects_host_tensors_before_touching_cuda() -> None:
    tokens = 4
    x = torch.zeros(tokens, 2, 2560, dtype=torch.bfloat16)
    weight = torch.ones(2, 2560, dtype=torch.bfloat16)
    collective = torch.empty(tokens, 2560, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        comm.cake_fused_norm_combine(
            x,
            x.clone(),
            weight,
            norm_out=torch.empty_like(x),
            residual_out=torch.empty_like(x),
            collective_out=collective,
            workspace=None,
            epsilon=1e-6,
        )
    with pytest.raises(ValueError, match="backend"):
        comm.cake_fused_norm_combine(
            x,
            x.clone(),
            weight,
            norm_out=torch.empty_like(x),
            residual_out=torch.empty_like(x),
            collective_out=collective,
            workspace=None,
            epsilon=1e-6,
            backend="trtllm",
        )
    with pytest.raises(TypeError):
        comm.cake_fused_norm_combine_destroy_workspace(object())


def test_comm_package_exports_the_public_entry_points() -> None:
    assert comm.cake_fused_norm_combine is api.cake_fused_norm_combine
    assert (
        comm.cake_fused_norm_combine_create_workspace
        is api.cake_fused_norm_combine_create_workspace
    )
    assert (
        comm.cake_fused_norm_combine_destroy_workspace
        is api.cake_fused_norm_combine_destroy_workspace
    )
    assert comm.CakeFusedNormCombineWorkspace is api.CakeFusedNormCombineWorkspace


def _get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
            return int(sock.getsockname()[1])


def _source_ordered_square_sum(values: torch.Tensor) -> torch.Tensor:
    """FP32 sum of squares over the hidden axis in the kernel's accumulation order.

    Sixteen sequential FP32 multiply-adds per vector thread, an XOR warp tree at
    offsets 16/8/4/2/1, then the same tree over the lane-zero warp sums with the
    inactive lanes zero. A plain ``mean`` groups the additions differently; that
    can move one BF16 rounding boundary in a normalized track, and the eight-peer
    BF16 sum then exceeds the tolerance for a few elements at large T.
    """

    tokens, tracks, hidden = values.shape
    vector_threads = hidden // 16
    warp_count = (vector_threads + 31) // 32
    vectors = values.reshape(tokens, tracks, vector_threads, 16)
    partial = torch.zeros(
        (tokens, tracks, vector_threads), dtype=torch.float32, device=values.device
    )
    for element in range(16):
        value = vectors[..., element]
        partial = torch.addcmul(partial, value, value)
    padded = torch.zeros(
        (tokens, tracks, warp_count * 32), dtype=torch.float32, device=values.device
    )
    padded[..., :vector_threads] = partial
    lanes = torch.arange(32, device=values.device)
    warp_values = padded.reshape(tokens, tracks, warp_count, 32)
    for offset in (16, 8, 4, 2, 1):
        warp_values = warp_values + warp_values.index_select(-1, lanes ^ offset)
    block_values = torch.zeros(
        (tokens, tracks, 32), dtype=torch.float32, device=values.device
    )
    block_values[..., :warp_count] = warp_values[..., 0]
    for offset in (16, 8, 4, 2, 1):
        block_values = block_values + block_values.index_select(-1, lanes ^ offset)
    return block_values[..., :1]


def _local_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual add, per-track RMSNorm and the two-track mean in FP32 with BF16 boundaries.

    Independent eager PyTorch with the kernel's FP32 operation order: rounded
    BF16 residual, source-ordered sum of squares divided by an FP32 hidden-size
    tensor, FP32 epsilon, ``torch.rsqrt``, two FP32 multiplies, BF16 rounding.
    """

    residual_out = (x.float() + residual.float()).to(torch.bfloat16)
    propagated = residual_out.float()
    divisor = torch.tensor(
        float(propagated.shape[-1]), dtype=torch.float32, device=propagated.device
    )
    epsilon_fp32 = torch.tensor(epsilon, dtype=torch.float32, device=propagated.device)
    inverse_rms = torch.rsqrt(
        _source_ordered_square_sum(propagated) / divisor + epsilon_fp32
    )
    normalized = ((propagated * inverse_rms) * weight.float().unsqueeze(0)).to(
        torch.bfloat16
    )
    inverse_tracks = torch.tensor(
        1.0 / normalized.shape[1], dtype=torch.float32, device=propagated.device
    )
    mean = torch.zeros_like(normalized[:, 0, :], dtype=torch.float32)
    for track in range(normalized.shape[1]):
        mean = mean + normalized[:, track, :].float() * inverse_tracks
    contribution = mean.to(torch.bfloat16)
    # One-shot publication canonicalizes negative zero before the reduction.
    contribution = torch.where(
        (contribution == 0) & torch.signbit(contribution),
        torch.zeros_like(contribution),
        contribution,
    )
    return normalized, residual_out, contribution


def _assert_close_all_ranks(
    actual: torch.Tensor, expected: torch.Tensor, *, label: str
) -> None:
    import torch.distributed as dist

    actual_f32 = actual.float()
    expected_f32 = expected.float()
    close = torch.isclose(actual_f32, expected_f32, atol=ATOL, rtol=RTOL).all()
    failure = (~close).to(torch.int32)
    max_abs = torch.nan_to_num(
        (actual_f32 - expected_f32).abs(), nan=float("inf")
    ).max()
    dist.all_reduce(failure, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    if failure.item():
        raise AssertionError(
            f"{label} failed the distributed close check: max_abs={max_abs.item():.6g}"
        )


def _worker(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    workspace = None
    try:
        workspace = comm.cake_fused_norm_combine_create_workspace(
            rank=rank,
            world_size=world_size,
            max_tokens=max(TOKENS_UNDER_TEST),
            group=dist.group.WORLD,
            device=device,
        )
        epsilon = 1e-6
        cases = {}
        for tokens in TOKENS_UNDER_TEST:
            generator = torch.Generator(device="cpu").manual_seed(
                17 + 104729 * rank + tokens
            )
            x = torch.randn(
                tokens, 2, 2560, generator=generator, dtype=torch.bfloat16
            ).to(device)
            residual = torch.randn(
                tokens, 2, 2560, generator=generator, dtype=torch.bfloat16
            ).to(device)
            weight = torch.randn(2, 2560, generator=generator, dtype=torch.bfloat16).to(
                device
            )
            normalized, residual_ref, contribution = _local_reference(
                x, residual, weight, epsilon
            )
            gathered = [torch.empty_like(contribution) for _ in range(world_size)]
            dist.all_gather(gathered, contribution)
            reduced = gathered[0].clone()
            for peer in range(1, world_size):
                reduced = (reduced.float() + gathered[peer].float()).to(torch.bfloat16)
            cases[tokens] = (x, residual, weight, normalized, residual_ref, reduced)
        # Two launches per token count exercise the payload rotation; the
        # descending pass switches every band in the other direction, so each
        # program clears what another program left in the shared workspace.
        order = [tokens for tokens in TOKENS_UNDER_TEST for _launch in range(2)]
        order += list(reversed(TOKENS_UNDER_TEST))
        for tokens in order:
            x, residual, weight, normalized, residual_ref, reduced = cases[tokens]
            norm_out = torch.zeros_like(x)
            residual_out = torch.zeros_like(x)
            collective_out = torch.zeros(
                tokens, 2560, dtype=torch.bfloat16, device=device
            )
            comm.cake_fused_norm_combine(
                x,
                residual,
                weight,
                norm_out=norm_out,
                residual_out=residual_out,
                collective_out=collective_out,
                workspace=workspace,
                epsilon=epsilon,
            )
            torch.cuda.synchronize(device)
            variant = api.select_variant(tokens, sm_count)
            _assert_close_all_ranks(
                norm_out, normalized, label=f"norm_out T={tokens} {variant}"
            )
            _assert_close_all_ranks(
                residual_out, residual_ref, label=f"residual_out T={tokens} {variant}"
            )
            _assert_close_all_ranks(
                collective_out, reduced, label=f"collective_out T={tokens} {variant}"
            )
    finally:
        if workspace is not None:
            comm.cake_fused_norm_combine_destroy_workspace(workspace)
        dist.destroy_process_group()


def _exported_eight_gpu_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        return False
    capabilities = {
        tuple(torch.cuda.get_device_capability(index)) for index in range(8)
    }
    return len(capabilities) == 1 and capabilities.issubset(api.SUPPORTED_CAPABILITIES)


@pytest.mark.skipif(
    not _exported_eight_gpu_node(),
    reason="requires eight SM100 or eight SM103 GPUs on one node",
)
def test_eight_peer_fused_norm_combine_matches_the_reference() -> None:
    world_size = 8
    port = _get_open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, world_size, port), daemon=False)
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=WORKER_TIMEOUT_SECONDS)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.kill()
    assert not alive, "distributed fused norm-combine workers timed out"
    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]
