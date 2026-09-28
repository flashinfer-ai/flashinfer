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

Kimi-K3 TP12 fused LatentMoE tail (Cake backend).

The CPU tests check the host runtime (partition, route selection, generated
module inventory) on any machine.  The GPU test needs the operator's real
platform, a twelve-rank multi-node NVLink domain (three GB200 / GB300 NVL72
compute trays); launch it with torchrun over the three nodes::

    torchrun --nnodes 3 --nproc-per-node 4 --node-rank <n> --master-addr <host> \
        -m pytest tests/experimental/test_cake_kimi_k3_tp12_tail.py -k twelve_ranks

Single-process runs skip it.
"""

import os

import pytest
import torch

from flashinfer.experimental.kimi_k3_tp12_tail import cake_backend as cb
from flashinfer.experimental.kimi_k3_tp12_tail import cake_jit

ATOL = RTOL = 1e-2
SEED = 620
# One-shot rows, two-shot grouped rows and one pinned row (M > 256).
GPU_ROWS = (1, 4, 8, 16, 32, 128, 256, 300)
GPU_MAX_TOKENS = 512


def test_partition_covers_hidden_in_128_column_blocks():
    assert len(cb.PARTITION) == cb.WORLD_SIZE == 12
    assert sum(cb.PARTITION) == cb.HIDDEN == 7168
    assert all(width % 128 == 0 for width in cb.PARTITION)
    assert cb.COL_BEGIN[0] == 0 and cb.COL_BEGIN[-1] == cb.HIDDEN
    assert cb.col_begin(cb.PARTITION) == cb.COL_BEGIN
    with pytest.raises(ValueError):
        cb.col_begin((640,) * 12)


def test_route_selection_by_token_count():
    # K2-stream regime: Cake SIMT slice GEMM (fp32) + fp32-add K3, module by the rank's column width
    assert cb.K2_STREAM_MAX_TOKENS == 4
    for M in (1, 2, 3, 4):
        for rank in range(cb.WORLD_SIZE):
            assert cb.route_kernel_keys(M, rank) == (
                f"k1_oneshot:r{rank}",
                f"k2_stream:n{cb.PARTITION[rank]}",
                "k3_f32:grouped",
            )
    assert cb.k2_form_for(4) == "stream" and cb.k2_form_for(5) == "cublas"
    assert cb.k2_kernel_key(4, 0) == "k2_stream:n640"
    assert cb.k2_kernel_key(4, 11) == "k2_stream:n512"
    assert cb.k2_kernel_key(5, 0) is None
    for M in (5, 8, 16):
        for rank in range(cb.WORLD_SIZE):
            assert cb.route_kernel_keys(M, rank) == (
                f"k1_oneshot:r{rank}",
                "k3:grouped",
            )
    for M in (17, 32, 128, 255):
        assert cb.route_kernel_keys(M, 3) == ("k1_twoshot:grouped", "k3:grouped")
    # persistent K3 pipeline from K3_PERSIST_MIN_TOKENS on (the poll schedule still follows M)
    assert cb.K3_PERSIST_MIN_TOKENS == 256
    assert cb.route_kernel_keys(256, 3) == ("k1_twoshot:grouped", "k3_persist:grouped")
    for M in (257, 512, 4096):
        assert cb.route_kernel_keys(M, 3) == ("k1_twoshot:pinned", "k3_persist:pinned")
    assert cb.k3_form_for(255) == "lamport" and cb.k3_form_for(256) == "persist"
    assert cb.k3_grid(255, 152) == (255, 2, 1)
    assert cb.k3_grid(256, 152) == (152, 2, 1)
    assert cb.k3_grid(100, 152) == (100, 2, 1)
    assert cb.k3_grid(4096, 148) == (148, 2, 1)
    with pytest.raises(ValueError):
        cb.k3_grid(4096, 0)
    assert set(cake_jit.required_kernel_keys()) == {
        *(f"k1_oneshot:r{r}" for r in range(12)),
        "k1_twoshot:grouped",
        "k1_twoshot:pinned",
        "k3:grouped",
        "k3_persist:grouped",
        "k3_persist:pinned",
        "k2_stream:n640",
        "k2_stream:n512",
        "k3_f32:grouped",
    }


def test_workspace_sizing():
    sizes = cb.workspace_buffer_bytes(4096)
    assert sizes["k1_oneshot"] == 16 * 3584 * 12 * 2
    assert sizes["k1_twoshot"] == 2 * 4104 * 3584 * 2
    assert sizes["k3"] == 2 * 4096 * 12 * 640 * 2
    assert cb.workspace_buffer_bytes(1)["k1_oneshot"] == 3584 * 12 * 2
    with pytest.raises(ValueError):
        cb.workspace_buffer_bytes(0)


def test_operand_validation_is_shape_and_dtype_strict():
    good = torch.empty((4, cb.LATENT), dtype=torch.bfloat16)
    cb._check(good, (4, cb.LATENT), "routed_partial")
    with pytest.raises(ValueError):
        cb._check(good.float(), (4, cb.LATENT), "routed_partial")
    with pytest.raises(ValueError):
        cb._check(good[:, ::2], (4, cb.LATENT // 2), "routed_partial")
    with pytest.raises(ValueError):
        cb._check(good, (4, cb.HIDDEN), "shared_partial")
    with pytest.raises(TypeError):
        cb._check(None, (4, cb.LATENT), "routed_partial")


def test_generated_module_inventory():
    if not cake_jit.MODULES:
        pytest.skip(
            "no generated Kimi-K3 TP12 tail program is registered in this checkout"
        )
    required = set(cake_jit.required_kernel_keys())
    for arch, table in cake_jit.KERNELS.items():
        assert arch in cake_jit.ARCHES
        assert set(table) == required
        for key, name in table.items():
            record = cake_jit.MODULES[name]
            assert record["arch"] == arch
            assert record["ffi_entry"] == "run"
            assert len(record["sources"]) == 2
            kinds = {kind for kind, _ in record["arg_plan"]}
            assert kinds <= {"buffer", "raw_pointer", "parameter", "grid"}
            names = {n for _, n in record["arg_plan"]}
            raw = {n for kind, n in record["arg_plan"] if kind == "raw_pointer"}
            assert record["launch"]["use_pdl"] is True
            assert tuple(record["launch"]["block"]) == (cb.THREADS, 1, 1)
            if key.startswith("k1_oneshot:"):
                assert raw == {"mcast_ptr", "local_unicast_ptr"}
                assert {
                    "routed",
                    "y_out",
                    "gamma",
                    "buffer_flags",
                    "num_tokens",
                    "epsilon",
                } <= names
            elif key.startswith("k2_stream:"):
                assert raw == set()
                assert {"y", "w_slice", "out", "num_tokens"} <= names
            elif key.startswith("k1_twoshot:"):
                assert raw == {"mcast_ptr"}
                assert {
                    "routed",
                    "y_out",
                    "gamma",
                    "peer_ptrs",
                    "buffer_flags",
                    "rank",
                } <= names
            else:
                assert raw == {"mcast_ptr"}
                assert {
                    "shared",
                    "gemm_slice",
                    "out",
                    "peer_ptrs",
                    "buffer_flags",
                    "my_col_begin",
                    "my_cols",
                    "gemm_plane_stride",
                    "num_gemm_splits",
                } <= names
    for arch in cake_jit.KERNELS:
        assert cake_jit.route_available(arch, tuple(required))


# ---------------------------------------------------------------------------
# Twelve-rank GPU test (torchrun over three NVL72 compute trays)
# ---------------------------------------------------------------------------


def _twelve_rank_platform() -> str:
    if os.environ.get("WORLD_SIZE") != "12" or "RANK" not in os.environ:
        return "needs a torchrun launch with WORLD_SIZE=12 (three GB200 / GB300 NVL72 trays)"
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability(0) not in cb.SUPPORTED_COMPUTE_CAPABILITIES:
        return "generated kernels target Blackwell SM100 / SM103"
    return ""


def make_inputs(M: int, rank: int, device: torch.device, seed: int = SEED) -> dict:
    """Model-scale inputs: weights identical on every rank, per-rank partials."""
    g = torch.Generator(device=device).manual_seed(seed * 1000 + M)
    norm_w = (1.0 + 0.1 * torch.randn((cb.LATENT,), generator=g, device=device)).to(
        torch.bfloat16
    )
    up_w = (0.01 * torch.randn((cb.HIDDEN, cb.LATENT), generator=g, device=device)).to(
        torch.bfloat16
    )
    gr = torch.Generator(device=device).manual_seed(seed * 1000 + M + 17 * (rank + 1))
    routed = (0.3 * torch.randn((M, cb.LATENT), generator=gr, device=device)).to(
        torch.bfloat16
    )
    shared = (0.1 * torch.randn((M, cb.HIDDEN), generator=gr, device=device)).to(
        torch.bfloat16
    )
    out = torch.empty((M, cb.HIDDEN), dtype=torch.bfloat16, device=device)
    return dict(norm_w=norm_w, up_w=up_w, routed=routed, shared=shared, out=out)


def reference(inp: dict, world: int) -> torch.Tensor:
    """FP32 partial sums, one BF16 rounding of the latent, KimiRMSNorm round points, one final BF16 rounding."""
    import torch.distributed as dist

    rs = [torch.empty_like(inp["routed"]) for _ in range(world)]
    ss = [torch.empty_like(inp["shared"]) for _ in range(world)]
    dist.all_gather(rs, inp["routed"])
    dist.all_gather(ss, inp["shared"])
    routed_sum = torch.stack([t.float() for t in rs]).sum(0).to(torch.bfloat16)
    shared_sum = torch.stack([t.float() for t in ss]).sum(0)
    xf = routed_sum.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cb.RMS_EPS)
    y = inp["norm_w"] * xf.to(torch.bfloat16)
    gemm = y.float() @ inp["up_w"].float().t()
    return (gemm + shared_sum).to(torch.bfloat16)


def _rank_invariant(out: torch.Tensor, world: int) -> bool:
    import torch.distributed as dist

    outs = [torch.empty_like(out) for _ in range(world)]
    dist.all_gather(outs, out)
    return all(torch.equal(outs[0], t) for t in outs[1:])


@pytest.mark.skipif(bool(_twelve_rank_platform()), reason=_twelve_rank_platform() or "")
def test_tail_matches_reference_on_twelve_ranks():
    import torch.distributed as dist

    from flashinfer.kimi_k3_tp12_tail import (
        create_kimi_k3_tp12_tail_workspace,
        prepare_kimi_k3_tp12_tail,
    )

    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == cb.WORLD_SIZE
    local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if not cb.generated_program_available(device):
        pytest.skip("no generated program registered for this architecture")
    workspace = create_kimi_k3_tp12_tail_workspace(rank=rank, max_tokens=GPU_MAX_TOKENS)
    try:
        for M in GPU_ROWS:
            inp = make_inputs(M, rank, device)
            expected = reference(inp, world)
            runner = prepare_kimi_k3_tp12_tail(
                inp["routed"],
                inp["shared"],
                inp["norm_w"],
                inp["up_w"],
                inp["out"],
                workspace=workspace,
            )
            assert runner.kernel_keys == cb.route_kernel_keys(M, rank)
            assert runner.k3.kwargs["grid"] == cb.k3_grid(M, workspace.sm_count)
            assert (runner.k2 is not None) == (M <= cb.K2_STREAM_MAX_TOKENS)
            if runner.k2 is not None:
                assert runner.k2.kwargs["grid"] == (cb.K2_STREAM_GRID_X, 1, 1)
                assert runner.gemm.dtype == torch.float32
            inp["out"].fill_(float("nan"))
            runner()
            torch.cuda.synchronize()
            first = inp["out"].clone()
            assert torch.isfinite(first.float()).all(), f"M={M}: non-finite output"
            torch.testing.assert_close(
                first.float(), expected.float(), atol=ATOL, rtol=RTOL
            )
            assert _rank_invariant(first, world), f"M={M}: output differs between ranks"
            # idempotent second launch (Lamport buffers rotate and clear themselves)
            inp["out"].fill_(float("nan"))
            runner()
            torch.cuda.synchronize()
            assert torch.equal(inp["out"], first), f"M={M}: second launch differs"
            # CUDA-graph replay is bitwise identical to the eager launch
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                runner()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                runner()
            inp["out"].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(inp["out"], first), f"M={M}: graph replay differs"
            del graph
        with pytest.raises(ValueError):
            inp = make_inputs(GPU_MAX_TOKENS + 1, rank, device)
            prepare_kimi_k3_tp12_tail(
                inp["routed"],
                inp["shared"],
                inp["norm_w"],
                inp["up_w"],
                inp["out"],
                workspace=workspace,
            )
        dist.barrier()
    finally:
        torch.cuda.synchronize()
        workspace.destroy()
        if owns_group:
            dist.destroy_process_group()
