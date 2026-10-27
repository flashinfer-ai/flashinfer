"""Multirank CUDA-graph capture/replay for Blackwell MegaMoE.

Multirank completion of ``test_mega_cuda_graph.py``: the mega kernel has cross-rank device-side barriers, so ranks must
replay together — this test warms up collectively, captures each rank's
``layer.forward`` (capture records without executing, so no cross-rank
dependency during capture), then replays in lockstep and compares against
the eager reference bit-exactly (default non-ikr config is deterministic).

Communication probes cover sparse route identities and tail-reset boundaries.
Full NVFP4/MXFP8 kernels also replay changing routes with poisoned combine cells.

Launched via torchrun (any even world size >= 2; runs at world size 2+):
    torchrun --nproc_per_node=2 -m pytest \\
        tests/moe_ep/test_mega_cuda_graph_multirank.py -v \\
        -m "gpu_2 and arch_blackwell"
"""

import os

import pytest
import torch
import torch.distributed as dist

from .test_moe_ep_nvfp4_cutedsl_mega_multirank import (
    _assert_ikr_close,
    _launcher_ranks,
    _mega_problem,
    _require_cuda,
)

_REPLAYS = 4


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("capacity", [65, 129, 1024])
@pytest.mark.parametrize("early", [True, False])
@pytest.mark.parametrize("tile_n,cluster_m", [(128, 4), (64, 4), (64, 2)])
def test_blackwell_genphase_graph(
    sparse_ep_runtime, capacity, early, tile_n, cluster_m
):
    from .test_nvfp4_cutedsl_kernel_vs_reference import _run_blackwell_genphase_graph

    rank, world = sparse_ep_runtime
    _run_blackwell_genphase_graph(rank, world, capacity, early, tile_n, cluster_m)


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "mode,token_back_mode,tuning,in_kernel_fc2_reduce,alpha_source",
    [
        ("w4a4", "epi_warps", "manual", False, "config"),
        ("w4a16", "epi_warps", "manual", False, "config"),
        ("w4a16", None, "auto", False, "config"),
        ("w4a16", "reuse_dispatch_warps", "manual", False, "runtime"),
        ("w4a16", "reuse_dispatch_warps", "manual", True, "runtime"),
    ],
)
def test_nvfp4_mega_two_rank_graph_replay_lockstep(
    mode, token_back_mode, tuning, in_kernel_fc2_reduce, alpha_source
):
    pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 2:
        pytest.skip("needs >=2 ranks")

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        MoEWeightPack,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size)

    assert mode in ("w4a4", "w4a16"), mode
    assert tuning in ("manual", "auto"), tuning
    assert tuning == "manual" or (mode == "w4a16" and token_back_mode is None)
    alphas = {}
    options = {}
    if mode == "w4a16":
        local_experts = problem["num_experts"] // world_size
        alphas = dict(
            fc1_alpha=torch.linspace(0.71013, 1.23017, local_experts, device="cuda"),
            fc2_alpha=torch.linspace(1.17019, 0.83023, local_experts, device="cuda"),
        )
        options = dict(
            enable_in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            knobs="auto"
            if tuning == "auto"
            else {
                "token_back_mode": token_back_mode,
                "in_kernel_fc2_reduce": in_kernel_fc2_reduce,
            },
            **(alphas if alpha_source == "config" else {}),
        )
    configs = {
        "w4a4": Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        "w4a16": Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    }
    mega = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=problem["num_experts"],
            max_tokens_per_rank=problem["max_tokens"],
            token_hidden_size=problem["hidden"],
        ),
        weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
        backend=MegaConfig(
            megakernel=configs[mode](
                intermediate_size=problem["intermediate"],
                top_k=problem["topk"],
                gate_up_clamp=problem["gate_up_clamp"],
                **options,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    assert isinstance(mega, MoEEpMegaLayer)
    graph = None
    try:
        t = MoEEpTensors(
            hidden_states=problem["hidden_states"],
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
            **(alphas if alpha_source == "runtime" else {}),
        )

        # Collective warmup: compile + workspace + one real launch, all ranks.
        mega.warmup(t)
        dist.barrier()

        y_eager = mega.forward(t).clone()
        torch.cuda.synchronize()
        dist.barrier()

        # Capture records without executing — safe per-rank; barrier keeps
        # ranks aligned so no rank starts REPLAYING while a peer captures.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_graph = mega.forward(t)
        dist.barrier()

        # Lockstep replays: the kernel's cross-rank barriers require every
        # rank to replay the same iteration together.
        for _ in range(_REPLAYS):
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
        if in_kernel_fc2_reduce:
            _assert_ikr_close(y_graph, y_eager, topk=problem["topk"])
        else:
            assert torch.equal(y_graph, y_eager), (
                f"rank {rank}: lockstep graph replay diverged from eager"
            )

        if alpha_source == "runtime":
            # Change only scales, then replay before eager can pre-stage them.
            alpha_ptrs = (t.fc1_alpha.data_ptr(), t.fc2_alpha.data_ptr())
            t.fc1_alpha.mul_(0.5)
            t.fc2_alpha.mul_(1.25)
            assert (t.fc1_alpha.data_ptr(), t.fc2_alpha.data_ptr()) == alpha_ptrs
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            y_alpha_replay = y_graph.clone()
            y_alpha_eager = mega.forward(t)
            torch.cuda.synchronize()
            dist.barrier()
            assert not torch.equal(y_alpha_eager, y_eager)
            if in_kernel_fc2_reduce:
                _assert_ikr_close(y_alpha_replay, y_alpha_eager, topk=problem["topk"])
            else:
                assert torch.equal(y_alpha_replay, y_alpha_eager)

        # Replay over mutated inputs (fresh values, same buffers).
        g = torch.Generator(device="cuda").manual_seed(1234 + rank)
        t.hidden_states.copy_(
            torch.randn(
                *t.hidden_states.shape,
                dtype=t.hidden_states.dtype,
                device="cuda",
                generator=g,
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        y_replay = y_graph.clone()

        y_eager2 = mega.forward(t)
        torch.cuda.synchronize()
        dist.barrier()
        if in_kernel_fc2_reduce:
            _assert_ikr_close(y_replay, y_eager2, topk=problem["topk"])
        else:
            assert torch.equal(y_replay, y_eager2), (
                f"rank {rank}: replay-after-mutation diverged from eager"
            )
        if in_kernel_fc2_reduce:
            t.hidden_states.zero_()
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            assert torch.count_nonzero(y_graph) == 0
    finally:
        if graph is not None:
            graph.reset()
        mega.destroy()
        torch.cuda.synchronize()
        dist.barrier()


def _tail_probe(world, sm_count):
    """Exercise the real communication tail with peer writes from other roles."""
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda

    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    args_type = pkg.TokenCommArgs

    class Probe:
        def __init__(self):
            self.comm = pkg.TokenInPullTokenBackPush(
                world_size=world,
                num_topk=4,
                num_experts_per_rank=4,
                num_total_experts=4 * world,
                hidden=128,
                fc1_token_dtype=cutlass.BFloat16,
                sf_uint32_per_token=0,
                token_padding_block=64,
                sf_padding_block=128,
                cluster_tile_tokens=64,
                cluster_shape_mn=(2, 1),
                dispatch_warp_start=8,
                num_other_warps=8,
            )

        @cute.jit
        def __call__(
            self,
            shared: cute.Tensor,
            signals: cute.Tensor,
            local: cute.Tensor,
            phase: cute.Tensor,
            control: cute.Tensor,
            mapper: pkg.SymBufferHost,
            stream: cuda.CUstream,
        ):
            args = args_type(
                input_token_buffer=None,
                input_sf_buffer=None,
                topk_idx=None,
                input_topk_weights_buffer=None,
                expert_send_count=None,
                expert_recv_count=None,
                expert_recv_count_sum=None,
                src_token_topk_idx=None,
                fc1_input_token_buffer=None,
                fc1_input_sf_buffer=None,
                fc1_input_topk_weights_buffer=None,
                fc1_ready_counter=None,
                token_src_metadata=None,
                combine_output=None,
                nvlink_barrier_signal=signals,
                nvlink_barrier_counter=phase,
                grid_sync_counter=cute.make_tensor(
                    local.iterator, cute.make_layout((2,))
                ),
                local_zero_prefix=local,
                shared_zero_prefix=shared,
                peer_rank_ptr_mapper=mapper.make_device_obj(),
                world_size=world,
                local_rank=mapper.rank_idx,
                num_total_experts=4 * world,
                num_experts_per_rank=4,
                num_topk=4,
                hidden_bytes=256,
                sf_uint32_per_token=0,
                token_padding_block=64,
                sf_padding_block=128,
                sm_count=sm_count,
            )
            self.kernel(args, control).launch(
                grid=(sm_count, 1, 1),
                block=(384, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(self, args: args_type, control: cute.Tensor):
            tid = cute.arch.thread_idx()[0]
            cta = cute.arch.block_idx()[0]
            warp, lane = tid // 32, tid % 32
            # Leave both grid-counter slots live, as in the real layout.
            index = cta * 384 + tid + 2
            while index < cute.size(args.local_zero_prefix):
                args.local_zero_prefix[index] = cutlass.Int32(7)
                index += sm_count * 384
            if tid == 0:
                if control[0] != 0:
                    address = args.peer_rank_ptr_mapper.map(
                        args.shared_zero_prefix.iterator.toint(),
                        (args.local_rank + 1) % world,
                        cutlass.Int64((cta % cute.size(args.shared_zero_prefix)) * 4),
                    )
                    pointer = cute.make_ptr(
                        cutlass.Int32, address, cute.AddressSpace.gmem
                    )
                    cute.arch.atomic_add(
                        pointer, cutlass.Int32(1), sem="relaxed", scope="sys"
                    )
            cute.arch.sync_threads()
            if warp >= 8:
                # The real dispatch has one peer barrier before the tail.
                self.comm.nvlink_barrier(
                    args.nvlink_barrier_signal,
                    args.nvlink_barrier_counter,
                    args.grid_sync_counter,
                    args.peer_rank_ptr_mapper,
                    cta,
                    warp - 8,
                    lane,
                    num_sms=sm_count,
                    prologue_grid_sync=True,
                    epilogue_grid_sync=True,
                )
            self.comm.kernel_tail(args, warp_idx=warp, lane_idx=lane, tidx=tid)

    return Probe()


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "count,ncu", [(16, False), (4096, False), (4097, False), (16, True)]
)
def test_tail_reset_multirank_graph(count, ncu, monkeypatch):
    """Clear small/large prefixes and preserve the phase over empty/work replays."""
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda

    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4):
        pytest.skip("requires torchrun with 2 or 4 ranks")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    monkeypatch.setenv("MEGA_USE_NCU", "1" if ncu else "0")
    sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
    probe = _tail_probe(world, sm_count)
    single_cta = count <= 4096 and not ncu
    assert probe.comm.use_single_cta_tail_reset(count) == single_cta
    signal_offset = (count + 3) // 4 * 4
    heap = symm.empty(
        (signal_offset + 4,), device=torch.device("cuda", rank), dtype=torch.int32
    )
    handle = symm.rendezvous(heap, dist.group.WORLD)
    heap.zero_()
    shared = heap[:count]
    signals = heap[signal_offset : signal_offset + 2]
    local = torch.zeros(256, device="cuda", dtype=torch.int32)
    phase = torch.zeros(1, device="cuda", dtype=torch.int32)
    control = torch.ones(1, device="cuda", dtype=torch.int32)
    mapper = pkg.SymBufferHost(
        base_addr=heap.data_ptr(),
        offsets=tuple(p - heap.data_ptr() for p in handle.buffer_ptrs),
        rank_idx=rank,
        num_max_ranks=world,
    )
    kwargs = dict(
        shared=from_dlpack(shared),
        signals=from_dlpack(signals),
        local=from_dlpack(local),
        phase=from_dlpack(phase),
        control=from_dlpack(control),
        mapper=mapper,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    executable = cute.compile(probe, **kwargs)
    torch.cuda.synchronize()
    dist.barrier()
    executable(**kwargs)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            kwargs["stream"] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            executable(**kwargs)
        launches = 1
        for active in (1, 0, 1):
            control.fill_(active)
            for replay in range(65):
                graph.replay()
                launches += 1
                if replay in (0, 64):
                    torch.cuda.synchronize()
                    dist.barrier()
                    expected_grid_phase = (
                        -(1 << 31) if single_cta and launches % 2 else 0
                    )
                    failures = torch.stack(
                        [
                            torch.count_nonzero(shared) != 0,
                            torch.count_nonzero(local[1:]) != 0,
                            local[0] != expected_grid_phase,
                            phase[0] != (4 if ncu else 3) * launches,
                        ]
                    ).to(torch.int32)
                    dist.all_reduce(failures, op=dist.ReduceOp.MAX)
                    assert not failures.any().item(), failures.tolist()
                    dist.barrier()
    finally:
        graph.reset()
        torch.cuda.synchronize()
        dist.barrier()


def _sparse_dispatch_probe(world, sm_count, capacity, topk, sparse, local_experts):
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda

    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    experts = local_experts * world
    shared_prefix = 2 * (experts + local_experts)
    metadata_elements = local_experts * world * capacity * topk
    args_type = pkg.TokenCommArgs

    class Probe:
        def __init__(self):
            self.comm = pkg.TokenInPullTokenBackPush(
                world_size=world,
                num_topk=topk,
                num_experts_per_rank=local_experts,
                num_total_experts=experts,
                hidden=128,
                fc1_token_dtype=cutlass.BFloat16,
                sf_uint32_per_token=0,
                token_padding_block=64,
                sf_padding_block=128,
                cluster_tile_tokens=64,
                cluster_shape_mn=(1, 1),
                dispatch_warp_start=0,
                num_other_warps=0,
            )

        @cute.jit
        def __call__(
            self,
            heap: cute.Tensor,
            local: cute.Tensor,
            ids: cute.Tensor,
            snapshot: cute.Tensor,
            mapper: pkg.SymBufferHost,
            stream: cuda.CUstream,
        ):
            def view(pointer, dtype, shape):
                strides = []
                stride = 1
                for extent in reversed(shape):
                    strides.insert(0, stride)
                    stride *= extent
                return cute.make_tensor(
                    cute.recast_ptr(pointer, dtype=dtype),
                    cute.make_layout(shape, stride=tuple(strides)),
                )

            args = args_type(
                input_token_buffer=None,
                input_sf_buffer=None,
                topk_idx=ids,
                input_topk_weights_buffer=None,
                expert_send_count=view(local.iterator, cutlass.Int64, (experts,)),
                expert_recv_count=view(
                    heap.iterator, cutlass.Int64, (world, local_experts)
                ),
                expert_recv_count_sum=view(
                    heap.iterator + 2 * experts, cutlass.Int64, (local_experts,)
                ),
                src_token_topk_idx=view(
                    heap.iterator + shared_prefix,
                    cutlass.Int32,
                    (local_experts, world, capacity * topk),
                ),
                fc1_input_token_buffer=None,
                fc1_input_sf_buffer=None,
                fc1_input_topk_weights_buffer=None,
                fc1_ready_counter=None,
                token_src_metadata=None,
                combine_output=None,
                nvlink_barrier_signal=view(
                    heap.iterator + shared_prefix + metadata_elements,
                    cutlass.Int32,
                    (2,),
                ),
                nvlink_barrier_counter=view(
                    local.iterator + 2 * experts + 4, cutlass.Int32, (1,)
                ),
                grid_sync_counter=view(
                    local.iterator + 2 * experts, cutlass.Int32, (2,)
                ),
                local_zero_prefix=view(
                    local.iterator, cutlass.Int32, (2 * experts + 4,)
                ),
                shared_zero_prefix=view(heap.iterator, cutlass.Int32, (shared_prefix,)),
                dispatch_route_extent=(
                    view(local.iterator + 2 * experts + 2, cutlass.Int32, (2,))
                    if cutlass.const_expr(sparse)
                    else None
                ),
                peer_rank_ptr_mapper=mapper.make_device_obj(),
                world_size=world,
                local_rank=mapper.rank_idx,
                num_total_experts=experts,
                num_experts_per_rank=local_experts,
                num_topk=topk,
                hidden_bytes=256,
                sf_uint32_per_token=0,
                token_padding_block=64,
                sf_padding_block=128,
                sm_count=sm_count,
            )
            self.kernel(args, snapshot).launch(
                grid=(sm_count, 1, 1), block=(128, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(self, args: args_type, snapshot: cute.Tensor):
            storage = cutlass.utils.SmemAllocator().allocate(
                self.comm.extra_smem_storage_class()
            )
            tid, cta = cute.arch.thread_idx()[0], cute.arch.block_idx()[0]
            warp, lane = tid // 32, tid % 32
            self.comm.dispatch_prep(
                storage,
                args.topk_idx,
                args.expert_send_count,
                args.src_token_topk_idx,
                args.peer_rank_ptr_mapper,
                cta,
                warp,
                lane,
                local_rank=args.local_rank,
                num_tokens=capacity,
                num_sms=sm_count,
                route_extent=args.dispatch_route_extent,
            )
            self.comm.dispatch_barrier(
                args.expert_send_count,
                args.expert_recv_count,
                args.expert_recv_count_sum,
                args.nvlink_barrier_signal,
                args.grid_sync_counter,
                args.peer_rank_ptr_mapper,
                cta,
                warp,
                lane,
                local_rank=args.local_rank,
                num_sms=sm_count,
                nvlink_barrier_counter=args.nvlink_barrier_counter,
            )
            index = cta * 128 + tid
            if index < experts:
                snapshot[index] = args.expert_send_count[index]
                snapshot[experts + index] = args.expert_recv_count[
                    index // local_experts, index % local_experts
                ]
            if index < local_experts:
                snapshot[2 * experts + index] = args.expert_recv_count_sum[index]
            self.comm.kernel_tail(args, warp_idx=warp, lane_idx=lane, tidx=tid)

    return Probe()


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("capacity,topk,local_experts", [(129, 4, 4), (1024, 10, 128)])
@pytest.mark.parametrize("sparse", [False, True])
def test_sparse_dispatch_graph(capacity, topk, local_experts, sparse, monkeypatch):
    """Keep holes and last-capacity routes; every expert retains full-grid completion."""
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm

    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4):
        pytest.skip("requires torchrun with 2 or 4 ranks")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    monkeypatch.setenv("MEGA_USE_NCU", "0")
    experts = local_experts * world
    sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
    prefix = 2 * (experts + local_experts)
    heap = symm.empty(
        (prefix + experts * capacity * topk + 4,),
        dtype=torch.int32,
        device=torch.device("cuda", rank),
    )
    handle = symm.rendezvous(heap, dist.group.WORLD)
    heap.zero_()
    local = torch.zeros(2 * experts + 8, device="cuda", dtype=torch.int32)
    ids = torch.zeros((capacity, topk), device="cuda", dtype=torch.int64)
    base = (
        torch.arange(capacity * topk, device="cuda").reshape_as(ids) + rank
    ) % experts
    snapshot = torch.empty(
        2 * experts + local_experts, device="cuda", dtype=torch.int64
    )
    metadata = heap[prefix : prefix + experts * capacity * topk].reshape(
        local_experts, world, -1
    )
    mapper = pkg.SymBufferHost(
        base_addr=heap.data_ptr(),
        offsets=tuple(p - heap.data_ptr() for p in handle.buffer_ptrs),
        rank_idx=rank,
        num_max_ranks=world,
    )
    probe = _sparse_dispatch_probe(
        world, sm_count, capacity, topk, sparse, local_experts
    )
    kwargs = dict(
        heap=from_dlpack(heap),
        local=from_dlpack(local),
        ids=from_dlpack(ids),
        snapshot=from_dlpack(snapshot),
        mapper=mapper,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    executable = cute.compile(probe, **kwargs)
    torch.cuda.synchronize()
    dist.barrier()
    executable(**kwargs)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            kwargs["stream"] = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            executable(**kwargs)
        for state in (
            "dense",
            "one",
            "empty",
            "last",
            "holes",
            "asymmetric",
            "restore",
        ):
            ids.copy_(base)
            if state in ("one", "empty", "last"):
                ids.fill_(-1)
            if state == "one":
                ids[0] = base[0]
            if state == "last":
                ids[-1, -1] = base[-1, -1]
            if state == "holes":
                ids.reshape(-1)[::3] = -1
            if state == "asymmetric":
                ids[(0, 1, 129, capacity)[rank] :] = -1
            gathered = [torch.empty_like(ids) for _ in range(world)]
            dist.all_gather(gathered, ids)
            reference = torch.stack(gathered).cpu()
            torch.cuda.synchronize()
            dist.barrier()
            for replay in range(17):
                graph.replay()
                if replay in (0, 16):
                    torch.cuda.synchronize()
                    dist.barrier()
                    counts = snapshot.cpu()
                    errors = 0
                    for expert in range(experts):
                        packed = int(counts[expert])
                        errors += (packed >> 32) != sm_count
                        errors += (packed & 0xFFFFFFFF) != int(
                            (reference[rank] == expert).sum()
                        )
                    for expert in range(local_experts):
                        global_expert = rank * local_experts + expert
                        total = 0
                        for source in range(world):
                            expected = torch.where(
                                reference[source].reshape(-1) == global_expert
                            )[0]
                            count = int(
                                counts[experts + source * local_experts + expert]
                            )
                            errors += count != expected.numel()
                            actual = (
                                metadata[expert, source, :count]
                                .cpu()
                                .to(torch.int64)
                                .sort()
                                .values
                            )
                            errors += not torch.equal(actual, expected)
                            total += expected.numel()
                        packed = int(counts[2 * experts + expert])
                        errors += (packed >> 32) != sm_count * world
                        errors += (packed & 0xFFFFFFFF) != total
                    errors += int(
                        torch.count_nonzero(
                            local[2 * experts + 2 : 2 * experts + 4]
                        ).item()
                        != 0
                    )
                    verdict = torch.tensor(errors, device="cuda", dtype=torch.int32)
                    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
                    assert verdict.item() == 0, (state, replay, errors, verdict.item())
                    dist.barrier()
    finally:
        graph.reset()
        torch.cuda.synchronize()
        dist.barrier()


@pytest.fixture(scope="module")
def sparse_ep_runtime():
    from flashinfer.moe_ep import (
        BootstrapConfig,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world not in (2, 4) or os.environ.get("MEGA_NO_DIST") == "1":
        pytest.skip("requires torchrun with 2 or 4 ranks and NVSHMEM")
    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires native Blackwell SM100/SM103")
    kernel = create_mega_kernel(
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=256, top_k=4)
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    try:
        yield rank, world
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "dtype,early", [("nvfp4", False), ("nvfp4", True), ("mxfp8", True)]
)
@pytest.mark.parametrize("capacity", [129, 1024])
def test_sparse_to_dense_graph(sparse_ep_runtime, dtype, early, capacity):
    """Check every capacity row after first/65th replay of each routing state.

    Dense per-route cells are the numerical control; independent dense Torch
    oracles live in the existing NVFP4/MXFP8 multirank suites.
    """
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as pkg

    rank, world = sparse_ep_runtime
    create = (
        pkg.create_dummy_nvfp4_inputs
        if dtype == "nvfp4"
        else pkg.create_dummy_mxfp8_inputs
    )
    make_thunk = (
        pkg.nvfp4_mega_launch_thunk if dtype == "nvfp4" else pkg.mxfp8_mega_launch_thunk
    )
    torch.manual_seed(6447 + rank)
    options = dict(apply_routing_weights_before_fc2=early) if dtype == "nvfp4" else {}
    _, w1, w2, buffer = create(
        rank,
        world,
        16,
        capacity,
        capacity,
        4,
        512,
        256,
        gate_up_clamp=10.0,
        seed=6447,
        **options,
    )
    graph = torch.cuda.CUDAGraph()
    try:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            thunk = make_thunk(w1, w2, buffer)
            thunk()
        ids = buffer.topk_idx.clone()
        scores = buffer.topk_weights.clone()
        torch.cuda.synchronize()
        dist.barrier()
        mega = buffer._frontend._mega
        kernel = mega.kernel
        assert kernel.token_comm.use_single_cta_tail_reset(kernel.shared_zero_i32_count)
        region = kernel._shared_region_by_name["combine_quant"]
        start = kernel._shared_offsets["combine_quant"]
        combine = mega.shared_workspace.narrow(0, start, region.nbytes).view(
            torch.bfloat16
        )
        combine = combine.reshape(capacity, 4, 512)
        dense_terms = combine.float().clone()
        finite = torch.isfinite(dense_terms).all().to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        assert finite.item() == 1
        with torch.cuda.graph(graph, stream=capture_stream):
            thunk()
        base = torch.arange(capacity * 4, device="cuda").reshape(capacity, 4)
        for state in ("dense", "empty", "last", "holes", "asymmetric", "restore"):
            valid = torch.ones_like(ids, dtype=torch.bool)
            if state in ("empty", "last"):
                valid.zero_()
            if state == "last":
                valid[-1, -1] = True
            if state == "holes":
                valid = base % 3 != 0
            if state == "asymmetric":
                valid[(0, 1, 129, capacity)[rank] :] = False
            buffer.topk_idx.copy_(ids).masked_fill_(~valid, -1)
            buffer.topk_weights.copy_(scores).masked_fill_(~valid, float("nan"))
            expected = torch.zeros((capacity, 512), device="cuda", dtype=torch.float32)
            for k in range(4):
                term = torch.where(valid[:, k, None], dense_terms[:, k], 0)
                if not early:
                    term *= torch.where(valid[:, k, None], scores[:, k, None], 0)
                expected += term
            combine.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
            for replay in range(65):
                graph.replay()
                if replay in (0, 64):
                    torch.cuda.synchronize()
                    dist.barrier()
                    actual = buffer.output_activation.float()
                    error = (actual - expected).norm() / expected.norm().clamp_min(
                        1e-12
                    )
                    failures = torch.stack(
                        [
                            ~torch.isfinite(actual).all(),
                            torch.count_nonzero(actual[~valid.any(dim=1)]) != 0,
                            ~torch.isfinite(error) | (error >= 0.02),
                        ]
                    ).to(torch.int32)
                    dist.all_reduce(failures, op=dist.ReduceOp.MAX)
                    assert not failures.any().item(), (
                        state,
                        replay,
                        error.item(),
                        failures.tolist(),
                    )
                    dist.barrier()
    finally:
        graph.reset()
        torch.cuda.synchronize()
        dist.barrier()
        buffer.destroy()
