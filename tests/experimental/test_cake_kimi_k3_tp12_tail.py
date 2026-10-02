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
program inventory) on any machine.  The single-GPU test builds and loads every
registered program for the device's architecture (SM100 / SM103, no process
group).  The numerical GPU test needs the operator's real platform, a
twelve-rank multi-node NVLink domain (three GB200 / GB300 NVL72 compute
trays); launch it with torchrun over the three nodes::

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
# Fused K23 rows (M <= 8: the four-token module at 1 / 4, the eight-token module at 6 / 8), K3-ESS rows with the
# one-shot / two-shot ESS K1, the plain persistent K3 at 256 and one cp.async.bulk pinned row (M > 256).
GPU_ROWS = (1, 4, 6, 8, 16, 32, 128, 256, 300)
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
    # fused K23 regime: ESS one-shot K1 + fused slice GEMM / tail, module by the rank's column width and by the
    # accumulator capacity ladder (the four-token module for M <= 4, the eight-token module for 5 <= M <= 8)
    assert cb.K23_MAX_TOKENS == 8 and cb.K23_ROWS == 8 and cb.K23_LADDER == (4, 8)
    for M in (1, 2, 3, 4, 5, 6, 7, 8):
        capacity = 4 if M <= 4 else 8
        assert cb.k23_capacity_for(M) == capacity
        for rank in range(cb.WORLD_SIZE):
            # one program per capacity on every rank: the rank is a launch argument, the column width a grid size
            assert cb.route_kernel_keys(M, rank) == (
                "k1_oneshot_ess",
                f"k23:c{capacity}",
            )
    assert cb.up_proj_form_for(8) == "k23" and cb.up_proj_form_for(9) == "cublas"
    assert cb.k3_kernel_key(4, 0) == "k23:c4"
    assert cb.k3_kernel_key(5, 0) == "k23:c8"
    assert cb.k3_kernel_key(8, 11) == "k23:c8"
    with pytest.raises(ValueError):
        cb.k23_capacity_for(9)
    # cuBLAS + K3-ESS regime: the ESS K1 scattered the shared partial, one CTA per token and column half
    for M in (9, 12, 16):
        for rank in range(cb.WORLD_SIZE):
            assert cb.route_kernel_keys(M, rank) == ("k1_oneshot_ess", "k3_ess:grouped")
    for M in (17, 32, 128, 255):
        assert cb.route_kernel_keys(M, 3) == (
            "k1_twoshot_ess:grouped",
            "k3_ess:grouped",
        )
    # persistent K3 pipeline from K3_PERSIST_MIN_TOKENS on: plain at 256 (grouped), cp.async.bulk pushes above (pinned)
    assert cb.K3_PERSIST_MIN_TOKENS == 256 and cb.BULK_MIN_TOKENS == 257
    assert cb.route_kernel_keys(256, 3) == ("k1_twoshot:grouped", "k3_persist:grouped")
    for M in (257, 512, 4096):
        assert cb.route_kernel_keys(M, 3) == (
            "k1_twoshot:pinned",
            "k3_persist_bulk:pinned",
        )
    for M in (1, 4, 5, 255, 256, 257, 4096):
        assert len(cb.route_kernel_keys(M, 0)) == 2
    assert cb.k3_form_for(8) == "k23" and cb.k3_form_for(9) == "ess"
    assert cb.k3_form_for(255) == "ess" and cb.k3_form_for(256) == "persist"
    assert cb.k3_form_for(257) == "persist_bulk"
    # K23: one CTA per K23_ROWS output columns of the rank (80 for 640, 64 for 512)
    assert cb.k3_grid(1, 152, 0) == (80, 1, 1)
    assert cb.k3_grid(4, 148, 11) == (64, 1, 1)
    assert cb.k3_grid(8, 152, 0) == (80, 1, 1)
    # K3-ESS: one CTA per token and column half; persistent forms: min(M, SM count)
    assert cb.k3_grid(9, 152, 0) == (9, 2, 1)
    assert cb.k3_grid(100, 152, 3) == (100, 2, 1)
    assert cb.k3_grid(255, 152, 3) == (255, 2, 1)
    assert cb.k3_grid(256, 152, 3) == (152, 2, 1)
    assert cb.k3_grid(4096, 148, 3) == (148, 2, 1)
    with pytest.raises(ValueError):
        cb.k3_grid(4096, 0, 3)
    assert len(cake_jit.required_kernel_keys()) == 9
    assert set(cake_jit.required_kernel_keys()) == {
        "k1_oneshot_ess",
        "k1_twoshot_ess:grouped",
        "k1_twoshot:grouped",
        "k1_twoshot:pinned",
        "k23:c4",
        "k23:c8",
        "k3_ess:grouped",
        "k3_persist:grouped",
        "k3_persist_bulk:pinned",
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
    # one program per logical kernel, each built for both architectures, launched with PDL
    assert set(cake_jit.KERNELS) == required
    assert len(set(cake_jit.KERNELS.values())) == len(cake_jit.MODULES)
    for key, name in cake_jit.KERNELS.items():
        record = cake_jit.MODULES[name]
        assert set(record["arches"]) == set(cake_jit.ARCHES), key
        assert record["ffi_entry"] == "run"
        assert len(record["sources"]) == 2
        assert {kind for kind, _ in record["arg_plan"]} <= {
            "buffer",
            "raw_pointer",
            "parameter",
            "grid",
        }
        assert record["launch"]["use_pdl"] is True
        assert tuple(record["launch"]["block"]) == (cb.THREADS, 1, 1)
    for arch in cake_jit.ARCHES:
        assert cake_jit.route_available(arch, tuple(required))
        for key in required:
            assert cake_jit.kernel_module_name(arch, key) == cake_jit.KERNELS[key]


def _single_gpu_platform() -> str:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability(0) not in cb.SUPPORTED_COMPUTE_CAPABILITIES:
        return "generated kernels target Blackwell SM100 / SM103"
    return ""


@pytest.mark.skipif(bool(_single_gpu_platform()), reason=_single_gpu_platform() or "")
def test_registered_programs_compile_on_this_device():
    """Every registered program builds and loads for this device's architecture (one GPU, no process group)."""
    if not cake_jit.MODULES:
        pytest.skip(
            "no generated Kimi-K3 TP12 tail program is registered in this checkout"
        )
    arch = cb.SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(0)]
    names = sorted(
        {
            cake_jit.kernel_module_name(arch, key)
            for key in cake_jit.required_kernel_keys()
        }
    )
    assert set(names) == set(cake_jit.MODULES)
    for name in names:
        module = cake_jit.load_cake_kimi_k3_tp12_tail_module(name)
        assert callable(getattr(module, cake_jit.MODULES[name]["ffi_entry"]))


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
            assert runner.k3.kwargs["grid"] == cb.k3_grid(M, workspace.sm_count, rank)
            assert runner.cublas == (M > cb.K23_MAX_TOKENS)
            assert runner.launch_count == (2 if M <= cb.K23_MAX_TOKENS else 3)
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
