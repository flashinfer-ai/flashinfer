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

import os

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.pcie_ipc_policy import (
    IpcLaunchConfig,
    IpcVariant,
    _is_fused_launchable,
    _is_launchable,
    fused_rms_norm_threads,
)
from tests.comm.test_pcie_ipc_all_reduce import (
    _init_process_group,
    multi_process_parallel,
)

_EPS = 1e-6
# batch % world_size != 0 is deliberately over-represented. Ownership of the
# flat pack index space only splits into whole rows when it divides, and a
# design that normalises on the owner's shard cannot serve the rest -- which is
# most of the decode range. These cases are what pins that this one does.
#
# The large batches are here for the row -> block mapping rather than for the
# reduction: they are the ones where a block normalises several rows, so the
# staging slot it clears is one it wrote several iterations earlier.
_SHAPES = [
    (1, 4096),
    (1, 6144),
    (2, 6144),
    (3, 6144),
    (4, 4096),
    (5, 6144),
    (8, 6144),
    (32, 4096),
    (33, 6144),
    (64, 4096),
]

# Block counts swept per shape. Both sides of `rows == blocks` matter: above it
# a block carries several rows, below it some blocks carry none and must still
# reach every barrier, and the ambiguity bug this protocol is prone to only
# appears at two or more blocks in the first place.
_SWEEP_BLOCKS = (1, 2, 4, 8)

# Calls per configuration. One is not enough: the failure this kernel family
# risks is a race between a block clearing a staging slot and another block
# still shipping it, which needs the epoch double buffer to have turned over at
# least once to show up at all.
_REPEATS = 3


def _sweep_configs(
    world_size: int, hidden: int, max_blocks: int, sm_count=None, numel: int = None
) -> list:
    """Every (transport, blocks) pair the fused dispatch can reach for a shape.

    Explicit rather than whatever :meth:`fused_launch_config` returns: the seed
    picks one transport per payload, so leaving coverage to it would silently
    drop a kernel the moment a threshold moves.
    """
    threads = fused_rms_norm_threads(hidden, 2)
    # transport_blocks 0 means "every block"; 1 is the other extreme, where the
    # collective runs on one block while the rest only normalise. Both sides of
    # that split have to work. 4 is here for the world-8 island-block kernel,
    # whose admission rule requires a multiple of four -- without it every split
    # configuration of that kernel is filtered out and only the undivided grid
    # is ever exercised.
    configs = [
        IpcLaunchConfig(blocks, threads, variant, transport)
        # Every variant, both data planes. A hardcoded list is how the
        # copy-engine transports went untested after the fused dispatch learned
        # to reach them: the admission predicate said yes and nothing ever
        # asked it.
        for variant in IpcVariant
        for blocks in _SWEEP_BLOCKS
        for transport in (0, 1, 2, 4)
    ]
    return [
        config
        for config in configs
        if _is_fused_launchable(
            world_size, config, max_blocks, hidden, 2, sm_count, numel
        )
    ]


def _reference(
    inp: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    group,
) -> tuple:
    """float64 reference for the fused op, built from NCCL's own reduction.

    ``residual_out`` is exact: the inputs are small integers, so the group sum
    lands on a value both reduction orders represent exactly, and it can be
    compared with zero tolerance. ``norm_out`` cannot be -- ``rsqrt`` is not --
    so it is computed in float64 *from the rounded residual*, which is what the
    kernel normalises and what the other fusion backends normalise.
    """
    reduced = inp.clone()
    dist.all_reduce(reduced, group=group)
    residual_out = (reduced + residual_in).to(inp.dtype)

    pre = residual_out.to(torch.float64)
    inv_rms = torch.rsqrt(pre.pow(2).mean(dim=-1, keepdim=True) + eps)
    norm_out = pre * inv_rms * gamma.to(torch.float64)
    return norm_out, residual_out


def _check_fused(
    norm_out, residual_out, ref_norm, ref_residual, group, label: str
) -> None:
    """The three assertions, at deliberately different strengths.

    The residual is exact, the norm is not, and the norm must additionally agree
    *between ranks* -- a rank-local tolerance would pass while the group
    disagreed, and a divergent norm makes every downstream TP GEMM diverge.
    """
    device = norm_out.device
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    # Group-wide: a mis-selected kernel corrupts a subset of ranks, so a
    # rank-local assertion can pass on rank 0 while the collective is wrong
    # elsewhere.
    wrong = torch.tensor(
        [int((residual_out != ref_residual).sum().item())], device=device
    )
    dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
    assert int(wrong.item()) == 0, (
        f"{label}: residual_out has {int(wrong.item())} wrong elements; "
        "the reduction itself is exact for these inputs"
    )

    torch.testing.assert_close(
        norm_out.to(torch.float64),
        ref_norm,
        rtol=0.05,
        atol=0.15,
        msg=lambda m: f"{label}: norm_out\n{m}",
    )

    # Bit-identical across ranks. Every rank normalises from the same rounded
    # payload, so this holds exactly -- unless the row reduction is
    # order-dependent, which is why the kernel may not use atomics.
    gathered = [torch.empty_like(norm_out) for _ in range(world_size)]
    dist.all_gather(gathered, norm_out, group=group)
    for peer, other in enumerate(gathered):
        assert torch.equal(norm_out, other), (
            f"{label}: norm_out differs between rank {rank} and rank {peer}; "
            "the row reduction is not deterministic"
        )


def _fused_worker(world_size: int, rank: int, port: int) -> None:
    """The fused op against a float64 reference, on shapes the tuner will see.

    Every transport and block count the dispatch can reach is swept per shape,
    not just the one the seed picks: the two transports are separate kernels,
    and which one a shape lands on is a threshold that moves.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16
        max_numel = max(b * h for b, h in _SHAPES)
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=max_numel, dtype=dtype
        )
        shared = torch.Generator(device=device).manual_seed(0)
        local = torch.Generator(device=device).manual_seed(1234 + rank)

        for batch, hidden in _SHAPES:
            shape = (batch, hidden)
            # The contribution differs per rank; the residual and the weight do
            # not. Both are replicated in tensor parallelism, and the fused op
            # relies on it -- the residual is added once, by whichever rank owns
            # the pack, so ranks holding different residuals would produce a
            # result stitched together from several of them.
            inp = torch.randint(
                0, 16, shape, dtype=torch.int32, device=device, generator=local
            ).to(dtype)
            residual_in = torch.randint(
                0, 16, shape, dtype=torch.int32, device=device, generator=shared
            ).to(dtype)
            gamma = torch.randint(
                1, 5, (hidden,), dtype=torch.int32, device=device, generator=shared
            ).to(dtype)
            inp_before = inp.clone()

            ref_norm, ref_residual = _reference(inp, residual_in, gamma, _EPS, group)

            if not ws.supports_fused_add_rms_norm(inp):
                continue
            # The seed's own choice first, then every configuration it could
            # have made instead.
            for config in [None] + _sweep_configs(
                world_size, hidden, ws.max_blocks, ws.sm_count, inp.numel()
            ):
                for call in range(_REPEATS):
                    norm_out, residual_out = ws.all_reduce_fused_add_rms_norm(
                        inp,
                        residual_in=residual_in,
                        rms_gamma=gamma,
                        rms_eps=_EPS,
                        config=config,
                    )
                    where = (
                        "seed"
                        if config is None
                        else (
                            f"{config.variant.name} blocks={config.blocks} "
                            f"transport={config.effective_transport_blocks()}"
                        )
                    )
                    _check_fused(
                        norm_out,
                        residual_out,
                        ref_norm,
                        ref_residual,
                        group,
                        f"{shape} {where} call {call}",
                    )

            assert torch.equal(inp, inp_before), f"{shape}: input was mutated"
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_fused_add_rms_norm(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _fused_worker)


def _pack_region(ws) -> tuple:
    """Byte range of this rank's pack scratch region, as (offset, length).

    Recomputed from the workspace's immutables rather than read out of the
    header, and cross-checked against the size the launcher reports, so a
    layout change fails this loudly instead of quietly inspecting the wrong
    bytes.
    """
    from flashinfer.comm.pcie_ipc_ar import get_pcie_ipc_comm_module

    def align128(n: int) -> int:
        return (n + 127) & ~127

    signal_phases = 8  # kSignalPhases
    slots = (
        ws.max_blocks  # epoch slots
        + signal_phases * ws.max_blocks * ws.world_size  # barrier phases
        + ws.max_blocks  # barrier flags
        + 2 * 2  # {epoch, arrival} per scratch region
        + 2  # {arrival, generation} for the grid barrier
    )
    signal_bytes = align128(4 * slots)
    max_payload = align128(ws.max_numel * ws.elem_size)
    scratch_bytes = align128(2 * ws.world_size * max_payload)

    # The copy-engine block. Nothing here reads it -- it holds no sentinel and
    # the fused kernels never touch it -- but it is part of the total, so the
    # cross-check below cannot be made without it. Restated from
    # test_pcie_ipc_workspace_layout.py, which is the authority; the two move
    # together.
    ce_slots = 2 * (ws.world_size - 1) * 4 + 2  # kCePieces = 4
    ce_flag_bytes = align128(ce_slots * 128)  # kCeFlagStride = 128
    ce_counter_bytes = align128(2 * ce_slots * 4)
    flat = 2 * (ws.world_size - 1) * align128(max_payload // ws.world_size)
    island = 7 * align128(max_payload // 4) if ws.world_size == 8 else 0
    ce_bytes = ce_flag_bytes + ce_counter_bytes + max(flat, island)

    # The stream-memop flag block, appended outside compute_workspace_layout()
    # and only where that protocol can run. Asked of the module rather than
    # assumed: this test passed on every device that cannot run it and failed on
    # every device that can, which makes the verdict a property of the machine.
    module = get_pcie_ipc_comm_module()
    memop_bytes = (
        2 * (ws.world_size - 1) * 128
        if ws.world_size in (4, 8) and module.memop_supported()
        else 0
    )

    total = module.workspace_size(
        ws.world_size, ws.max_numel, ws.elem_size, ws.max_blocks
    )
    mirrored = signal_bytes + 2 * scratch_bytes + ce_bytes + memop_bytes
    assert mirrored == total, (
        "the workspace layout this test mirrors no longer matches the one the "
        f"launcher computes ({mirrored} vs {total})"
    )
    # [signal | pack scratch | block scratch | copy engine]: the fused family
    # lives in pack, and the copy-engine block is at the tail precisely so
    # these two offsets do not move when it is present.
    return signal_bytes, scratch_bytes


def _scratch_clean_worker(world_size: int, rank: int, port: int) -> None:
    """Every fused kernel must hand its scratch region back zeroed.

    That is the whole of the sentinel contract: a poll reads +0.0 as "not
    written yet", so a kernel that leaves payload behind does not fail itself,
    it fails whichever kernel runs next on that region -- and that one exits its
    poll immediately with stale data, which is wrong and fast. Checked directly
    here so the failure lands on the kernel that caused it rather than on a
    later call.
    """
    import ctypes

    from flashinfer.comm.cuda_ipc import cudart

    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16
        batch, hidden = 33, 6144
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=batch * hidden, dtype=dtype
        )
        shared = torch.Generator(device=device).manual_seed(0)
        shape = (batch, hidden)
        inp = torch.randint(0, 16, shape, dtype=torch.int32, device=device).to(dtype)
        residual_in = torch.randint(
            0, 16, shape, dtype=torch.int32, device=device, generator=shared
        ).to(dtype)
        gamma = torch.randint(
            1, 5, (hidden,), dtype=torch.int32, device=device, generator=shared
        ).to(dtype)
        if not ws.supports_fused_add_rms_norm(inp):
            return

        offset, length = _pack_region(ws)
        mirror = torch.empty(length, dtype=torch.uint8, device=device)
        base = ws._ipc_ptrs[rank]

        for config in _sweep_configs(
            world_size, hidden, ws.max_blocks, ws.sm_count, inp.numel()
        ):
            ws.all_reduce_fused_add_rms_norm(
                inp,
                residual_in=residual_in,
                rms_gamma=gamma,
                rms_eps=_EPS,
                config=config,
            )
            torch.cuda.synchronize(device)
            cudart.cudaMemcpy(
                ctypes.c_void_p(mirror.data_ptr()),
                ctypes.c_void_p(base + offset),
                length,
            )
            dirty = int(mirror.count_nonzero().item())
            verdict = torch.tensor([dirty], device=device)
            dist.all_reduce(verdict, op=dist.ReduceOp.MAX, group=group)
            assert int(verdict.item()) == 0, (
                f"{config.variant.name} blocks={config.blocks} left "
                f"{dirty} non-zero bytes in the pack region; the next sentinel "
                "kernel on it would poll stale payload instead of waiting"
            )
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_fused_norm_leaves_scratch_clean(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _scratch_clean_worker)


def _interleave_worker(world_size: int, rank: int, port: int) -> None:
    """Plain all-reduce and the fused op, alternating on one workspace.

    The two are separate protocol families in separate scratch regions, and the
    families do not mix: a barrier kernel leaves its payload in place, so a
    sentinel poll landing on those leftovers takes the stale value instead of
    waiting. At world size 8 the block region belongs to the barrier kernels,
    which is what puts the whole fused family in the pack region.

    Every (plain variant, fused variant) pair the shape admits is swept, in both
    orders, because the hazard is per (variant, region) pair rather than per
    call. The fused side needs an explicit configuration: at eight ranks the
    seed reaches only the one-shot and the ring, so leaving it to choose would
    never exercise the island-block kernel -- the one that mirrors the plain
    kStaged this test is most likely to collide with.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16
        batch, hidden = 8, 6144
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=batch * hidden, dtype=dtype
        )
        shared = torch.Generator(device=device).manual_seed(0)
        local = torch.Generator(device=device).manual_seed(1234 + rank)
        shape = (batch, hidden)
        inp = torch.randint(
            0, 16, shape, dtype=torch.int32, device=device, generator=local
        ).to(dtype)
        residual_in = torch.randint(
            0, 16, shape, dtype=torch.int32, device=device, generator=shared
        ).to(dtype)
        gamma = torch.randint(
            1, 5, (hidden,), dtype=torch.int32, device=device, generator=shared
        ).to(dtype)

        ref_reduced = inp.clone()
        dist.all_reduce(ref_reduced, group=group)
        ref_norm, ref_residual = _reference(inp, residual_in, gamma, _EPS, group)
        if not ws.supports_fused_add_rms_norm(inp):
            return

        # blocks divisible by four: the TP8 block kernel derives its chunk from
        # blockIdx.x & 3 and the launcher rejects anything else.
        plain = [
            IpcLaunchConfig(4, 256, variant)
            for variant in IpcVariant
            if _is_launchable(
                world_size, IpcLaunchConfig(4, 256, variant), ws.max_blocks
            )
        ]
        assert plain, f"no plain variant is launchable at {world_size} ranks"
        threads = fused_rms_norm_threads(hidden, 2)
        fused = [
            IpcLaunchConfig(4, threads, variant)
            for variant in IpcVariant
            if _is_fused_launchable(
                world_size,
                IpcLaunchConfig(4, threads, variant),
                ws.max_blocks,
                hidden,
                2,
                ws.sm_count,
            )
        ]
        assert fused, f"no fused variant is launchable at {world_size} ranks"

        def _check_plain(config):
            reduced = ws.all_reduce(inp, config=config)
            wrong = torch.tensor(
                [int((reduced != ref_reduced).sum().item())], device=device
            )
            dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
            return int(wrong.item())

        def _check_fused(config, label):
            norm_out, residual_out = ws.all_reduce_fused_add_rms_norm(
                inp,
                residual_in=residual_in,
                rms_gamma=gamma,
                rms_eps=_EPS,
                config=config,
            )
            wrong = torch.tensor(
                [int((residual_out != ref_residual).sum().item())], device=device
            )
            dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
            assert int(wrong.item()) == 0, (
                f"{label}: the fused op produced {int(wrong.item())} wrong "
                "elements; the two families are sharing a scratch region"
            )
            torch.testing.assert_close(
                norm_out.to(torch.float64),
                ref_norm,
                rtol=0.05,
                atol=0.15,
                msg=lambda m, l=label: f"{l}: norm_out\n{m}",
            )

        for plain_config in plain:
            for fused_config in fused:
                pair = (
                    f"plain {plain_config.variant.name} / fused "
                    f"{fused_config.variant.name}"
                )
                # Plain first, then fused.
                assert _check_plain(plain_config) == 0, (
                    f"{plain_config.variant.name}: the plain all-reduce itself "
                    "is wrong before the fused call even runs"
                )
                _check_fused(fused_config, f"after {pair}")
                # And the other order: a region left dirty by the fused call is
                # only visible to a plain call that follows it.
                _check_fused(fused_config, f"before {pair}")
                assert _check_plain(plain_config) == 0, (
                    f"{pair}: the plain all-reduce is wrong after a fused call; "
                    "the fused kernel left its region dirty"
                )
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_fused_norm_interleaved_with_plain(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _interleave_worker)


def _no_row_sync_worker(
    world_size: int, rank: int, port: int, variant: IpcVariant = IpcVariant.STAGED
) -> None:
    """Negative control: the row reduction without its barrier must be wrong.

    Rebuilt through the same compile-time switch the other protocol controls
    use. Without this a passing fused test says little -- the reduction's
    failure mode is a race between warps, which usually does not fire.

    Parametrised by transport because each one is a separate kernel with its own
    copy of the row reduction, and a control that only covers the flat push says
    nothing about the two ring kernels.
    """
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        from flashinfer.jit.comm import gen_pcie_ipc_comm_debug_module

        module = gen_pcie_ipc_comm_debug_module(
            stall_ns=0, stall_island=0, no_row_sync=1
        ).build_and_load()

        hidden, batch = 6144, 2
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=batch * hidden, dtype=torch.bfloat16
        )
        try:
            shared = torch.Generator(device=device).manual_seed(0)
            inp = torch.randint(
                1, 16, (batch, hidden), dtype=torch.int32, device=device
            ).to(torch.bfloat16)
            residual_in = torch.randint(
                1,
                16,
                (batch, hidden),
                dtype=torch.int32,
                device=device,
                generator=shared,
            ).to(torch.bfloat16)
            gamma = torch.ones(hidden, dtype=torch.bfloat16, device=device)
            config = ws.fused_launch_config(inp)
            norm_out = torch.empty_like(inp)
            residual_out = torch.empty_like(inp)
            module.pcie_ipc_all_reduce_fused_add_rmsnorm(
                ws.handle,
                inp,
                residual_in,
                gamma,
                residual_out,
                norm_out,
                hidden,
                1e-6,
                config.blocks,
                config.threads,
                int(variant),
                False,
            )
            torch.cuda.synchronize()

            # The residual is unaffected -- only the denominator is -- so this
            # pins the control on the norm rather than on the collective.
            pre = residual_out.to(torch.float64)
            inv_rms = torch.rsqrt(pre.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            expected = pre * inv_rms
            wrong = torch.tensor(
                [int((norm_out.to(torch.float64) - expected).abs().gt(0.05).sum())],
                device=device,
            )
            dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
            assert int(wrong.item()) > 0, (
                "the build without the row barrier produced a correct norm; the "
                "control has no power and the fused test proves nothing"
            )
        finally:
            ws.destroy()
    finally:
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.skipif(
    os.environ.get("FLASHINFER_TEST_PCIE_IPC_RACE") != "1",
    reason="opt-in: builds an extra JIT module",
)
# One case per fused kernel: the flat push, the TP4 ring, the TP8 island ring
# and the TP8 island block each carry their own copy of the row reduction, so a
# barrier dropped from one is invisible in the others.
@pytest.mark.parametrize(
    "world_size,variant",
    [
        (2, IpcVariant.STAGED),
        (4, IpcVariant.STAGED_RING),
        (8, IpcVariant.STAGED_RING),
        (8, IpcVariant.STAGED),
    ],
)
def test_pcie_ipc_fused_norm_needs_the_row_barrier(
    world_size: int, variant: IpcVariant
) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _no_row_sync_worker, (variant,))


def _fused_graph_capture_worker(world_size: int, rank: int, port: int) -> None:
    """The fused op needs preparing too, and prepare() has to cover it.

    Resolution is per entry point: the two keep separate caches, so a shape
    prepared for the plain all-reduce is still unresolved here. Both halves are
    pinned -- an unprepared shape must fail with an error naming the remedy,
    and prepare() must make the same shape capture, replay and stay correct.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16
        hidden, batch = 4096, 4
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=batch * hidden, dtype=dtype
        )
        shared = torch.Generator(device=device).manual_seed(0)
        local = torch.Generator(device=device).manual_seed(99 + rank)
        inp = torch.randint(
            0, 16, (batch, hidden), dtype=torch.int32, device=device, generator=local
        ).to(dtype)
        residual_in = torch.randint(
            0, 16, (batch, hidden), dtype=torch.int32, device=device, generator=shared
        ).to(dtype)
        gamma = torch.randint(
            1, 5, (hidden,), dtype=torch.int32, device=device, generator=shared
        ).to(dtype)
        if not ws.supports_fused_add_rms_norm(inp):
            return
        ref_norm, ref_residual = _reference(inp, residual_in, gamma, _EPS, group)
        norm_out = torch.empty_like(inp)
        residual_out = torch.empty_like(inp)

        def _call():
            ws.all_reduce_fused_add_rms_norm(
                inp,
                residual_in=residual_in,
                rms_gamma=gamma,
                rms_eps=_EPS,
                residual_out=residual_out,
                norm_out=norm_out,
            )

        # Unprepared: the guard fires before any collective is issued, so every
        # rank raises at the same point and the group stays in step.
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        unprepared = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(unprepared):
                _call()
            raised = ""
        except RuntimeError as exc:
            raised = str(exc)
        assert "prepare()" in raised, (
            "capturing an unresolved fused shape should name prepare() as the "
            f"remedy, got: {raised or 'no error at all'}"
        )

        ws.prepare([(batch, hidden)], dtype=dtype)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        prepared = torch.cuda.CUDAGraph()
        with torch.cuda.graph(prepared):
            _call()
        norm_out.zero_()
        residual_out.zero_()
        prepared.replay()
        torch.cuda.synchronize(device)

        wrong = torch.tensor(
            [int((residual_out != ref_residual).sum().item())], device=device
        )
        dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
        assert int(wrong.item()) == 0, (
            f"the replayed graph produced {int(wrong.item())} wrong residual elements"
        )
        torch.testing.assert_close(
            norm_out.to(torch.float64), ref_norm, rtol=0.05, atol=0.15
        )
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_prepare_makes_a_fused_shape_capturable(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _fused_graph_capture_worker, timeout_s=300)
