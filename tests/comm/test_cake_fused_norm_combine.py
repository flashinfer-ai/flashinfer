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

Tests for the SM100 eight-peer Cake fused norm-combine export.

The CPU tests check the verified module inventory, the token dispatch, the
workspace layout helper and the argument validation of the public API.  The
distributed test runs the public API on eight SM100 GPUs against an
independent PyTorch reference of the same math."""

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

TOKENS_UNDER_TEST = (1, 8, 64, 256, 1024, 2048)
ATOL = 1e-2
RTOL = 1e-2
WORKER_TIMEOUT_SECONDS = 20 * 60


def test_module_inventory_is_verified_source_only() -> None:
    assert loader.MODULES
    assert set(loader.ROUTES) == {loader.VARIANT_ONE_SHOT, loader.VARIANT_OWNER_REDUCE}
    assert set(loader.ROUTES.values()) == set(loader.MODULES)
    for name, record in loader.MODULES.items():
        assert name.startswith("cake_") and record["cache_name"].startswith("cake_")
        assert record["kernel_symbol"].startswith("kernel_cake_")
        assert record["ffi_entry"] == "run"
        paths = loader.verified_sources(name)
        assert len(paths) == 2 and all(path.suffix == ".cu" for path in paths)
        kinds = {kind for kind, _key in record["arg_plan"]}
        assert kinds == {"buffer", "parameter", "grid"}
        assert {key for kind, key in record["arg_plan"] if kind == "buffer"} == {
            "x",
            "residual",
            "weight",
            "norm_out",
            "residual_out",
            "collective_out",
            "workspace",
        }
        assert {key for kind, key in record["arg_plan"] if kind == "parameter"} == {
            "rank",
            "tokens",
            "epsilon",
        }
        assert record["arg_plan"][-3:] == [
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ]
        launch = record["launch"]
        assert launch["use_pdl"] is True
        assert launch["cooperative"] is False
        assert tuple(launch["cluster"]) == (1, 1, 1)
    one_shot = loader.MODULES[loader.ROUTES[loader.VARIANT_ONE_SHOT]]["launch"]
    owner = loader.MODULES[loader.ROUTES[loader.VARIANT_OWNER_REDUCE]]["launch"]
    # The one-shot half runs both tracks concurrently (two 160-thread halves);
    # the owner reduce keeps the source's 160-thread row.
    assert tuple(one_shot["block"]) == (320, 1, 1)
    assert tuple(owner["block"]) == (160, 1, 1)


def test_jit_spec_math_flags_follow_the_source_build() -> None:
    # The source arm compiles the same translation units without fast-math unless
    # the recorded compile flags say otherwise; the JIT build must not add it.
    for name, record in loader.MODULES.items():
        flags = loader.spec(name).extra_cuda_cflags or []
        fast = any(flag in ("-use_fast_math", "--use_fast_math") for flag in flags)
        assert fast == ("--use_fast_math" in record["compile_flags"]), name


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (1, loader.VARIANT_ONE_SHOT),
        (8, loader.VARIANT_ONE_SHOT),
        (64, loader.VARIANT_ONE_SHOT),
        (255, loader.VARIANT_ONE_SHOT),
        (256, loader.VARIANT_OWNER_REDUCE),
        (1024, loader.VARIANT_OWNER_REDUCE),
        (2048, loader.VARIANT_OWNER_REDUCE),
    ],
)
def test_token_dispatch(tokens: int, expected: str) -> None:
    assert loader.LARGE_MIN_TOKENS == 256
    assert loader.select_variant(tokens) == expected
    assert loader.route_module_name(tokens) == loader.ROUTES[expected]
    assert api.select_variant(tokens) == expected


@pytest.mark.parametrize("tokens", [0, -1, True])
def test_token_dispatch_rejects_non_positive_counts(tokens) -> None:
    with pytest.raises(ValueError):
        loader.select_variant(tokens)


def test_route_scope_is_eight_sm100_peers_with_hidden_2560() -> None:
    assert loader.route_applies(
        world_size=8, device_capability=(10, 0), hidden_dim=2560
    )
    assert not loader.route_applies(
        world_size=4, device_capability=(10, 0), hidden_dim=2560
    )
    assert not loader.route_applies(
        world_size=8, device_capability=(10, 3), hidden_dim=2560
    )
    assert not loader.route_applies(
        world_size=8, device_capability=(10, 0), hidden_dim=4096
    )


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
            norm_out = torch.zeros_like(x)
            residual_out = torch.zeros_like(x)
            collective_out = torch.zeros(
                tokens, 2560, dtype=torch.bfloat16, device=device
            )
            normalized, residual_ref, contribution = _local_reference(
                x, residual, weight, epsilon
            )
            gathered = [torch.empty_like(contribution) for _ in range(world_size)]
            dist.all_gather(gathered, contribution)
            reduced = gathered[0].clone()
            for peer in range(1, world_size):
                reduced = (reduced.float() + gathered[peer].float()).to(torch.bfloat16)
            # Two launches on the same workspace exercise the payload rotation.
            for _launch in range(2):
                norm_out.zero_()
                residual_out.zero_()
                collective_out.zero_()
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
                _assert_close_all_ranks(
                    norm_out, normalized, label=f"norm_out T={tokens}"
                )
                _assert_close_all_ranks(
                    residual_out, residual_ref, label=f"residual_out T={tokens}"
                )
                _assert_close_all_ranks(
                    collective_out, reduced, label=f"collective_out T={tokens}"
                )
    finally:
        if workspace is not None:
            comm.cake_fused_norm_combine_destroy_workspace(workspace)
        dist.destroy_process_group()


def _sm100_eight_gpu_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        return False
    return all(torch.cuda.get_device_capability(index) == (10, 0) for index in range(8))


@pytest.mark.skipif(
    not _sm100_eight_gpu_node(), reason="requires eight SM100 GPUs on one node"
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
