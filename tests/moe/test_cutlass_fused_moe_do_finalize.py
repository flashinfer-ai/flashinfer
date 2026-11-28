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

Tests for ``cutlass_fused_moe(..., do_finalize=False)``: the unreduced GEMM2 rows
and route-to-row map, alone, under expert parallelism, under CUDA graph replay,
and as the input of ``trtllm_moe_finalize_allreduce_fusion``.
"""

import multiprocessing as mp
import socket

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
import flashinfer.fused_moe as fused_moe
from flashinfer.fused_moe.core import ActivationType, get_cutlass_fused_moe_module
from flashinfer.utils import get_compute_capability

# trtllm_moe_finalize_allreduce_fusion, the consumer of this output, runs on SM90 and SM10x.
pytestmark = [
    pytest.mark.skipif(
        get_compute_capability(torch.device("cuda"))[0] not in (9, 10),
        reason="do_finalize=False is tested on SM90 and SM10x",
    ),
    # The two-GPU case needs every GPU of the node.
    pytest.mark.node_exclusive,
]

DTYPES = [torch.bfloat16, torch.float16]


def _make_moe(num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen, dtype=torch.float32)

    # Unit-variance GEMM outputs keep bf16 rounding errors comparable across shapes.
    x = randn(num_tokens, hidden_size).to(dtype)
    w31 = (randn(num_experts, 2 * inter_size, hidden_size) / hidden_size**0.5).to(dtype)
    w2 = (randn(num_experts, hidden_size, inter_size) / inter_size**0.5).to(dtype)
    probs = torch.softmax(randn(num_tokens, num_experts), dim=-1)
    weights, experts = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x, w31, w2, experts.to(torch.int32), weights.float()


def _finalize(gemm2_output, expert_weights, permuted_idx):
    """Reference top-k reduction of the do_finalize=False output, in FP32."""
    rows = permuted_idx.long()
    gathered = gemm2_output[rows.clamp(min=0)].float()
    weights = (expert_weights.float() * (rows >= 0)).unsqueeze(-1)
    return (gathered * weights).sum(dim=1)


def _run(x, w31, w2, experts, weights, dtype, **kwargs):
    return fused_moe.cutlass_fused_moe(
        x, experts, weights, w31, w2, dtype, quant_scales=None, **kwargs
    )


def _finalized(x, w31, w2, experts, weights, dtype, **kwargs):
    # The same unfused finalize path that do_finalize=False hands over.
    result = _run(
        x, w31, w2, experts, weights, dtype, use_fused_finalize=False, **kwargs
    )
    return result[0] if isinstance(result, (list, tuple)) else result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 300])
@pytest.mark.parametrize("hidden_size,inter_size", [(128, 128), (2048, 1024)])
@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (8, 6), (16, 1)])
def test_do_finalize_false_matches_finalize(
    dtype, num_tokens, hidden_size, inter_size, num_experts, top_k
):
    x, w31, w2, experts, weights = _make_moe(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed=0
    )
    ref = _finalized(x, w31, w2, experts, weights, dtype)

    gemm2_output, expert_weights, permuted_idx = _run(
        x, w31, w2, experts, weights, dtype, do_finalize=False
    )

    assert gemm2_output.shape == (num_tokens * top_k, hidden_size)
    assert gemm2_output.dtype == dtype
    assert expert_weights is weights
    assert permuted_idx.shape == (num_tokens, top_k)
    assert permuted_idx.dtype == torch.int32
    # Without EP every route is local, so the map is a permutation of the rows.
    assert torch.equal(
        permuted_idx.flatten().sort().values,
        torch.arange(num_tokens * top_k, dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        _finalize(gemm2_output, expert_weights, permuted_idx).to(dtype),
        ref,
        rtol=1e-2,
        atol=1e-2,
    )


def _kernel_names(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return {
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    }


def test_do_finalize_false_skips_finalize_kernel():
    dtype = torch.bfloat16
    x, w31, w2, experts, weights = _make_moe(16, 1024, 512, 8, 4, dtype, seed=10)
    # Warm up (JIT, autotune) outside the profiled region.
    _finalized(x, w31, w2, experts, weights, dtype)
    _run(x, w31, w2, experts, weights, dtype, do_finalize=False)

    finalized = _kernel_names(lambda: _finalized(x, w31, w2, experts, weights, dtype))
    unfinalized = _kernel_names(
        lambda: _run(x, w31, w2, experts, weights, dtype, do_finalize=False)
    )
    # The finalized run proves that the name match works on this build.
    assert any("finalizeMoeRouting" in n for n in finalized), sorted(finalized)
    assert not any("finalizeMoeRouting" in n for n in unfinalized), sorted(unfinalized)
    assert any("buildUnfinalizedPermutedIdx" in n for n in unfinalized), sorted(
        unfinalized
    )


def test_do_finalize_false_rows_match_torch_reference():
    """Each route's GEMM2 row matches an FP32 torch evaluation of its expert."""
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size, num_experts, top_k = 16, 256, 128, 8, 2
    x, w31, w2, experts, weights = _make_moe(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed=11
    )
    gemm2_output, _, permuted_idx = _run(
        x, w31, w2, experts, weights, dtype, do_finalize=False
    )
    tokens = torch.arange(num_tokens, device="cuda").repeat_interleave(top_k)
    expert_ids = experts.flatten().long()
    # w31 holds [up; gate] per expert; SwiGLU is silu(gate) * up.
    up, gate = w31.float()[expert_ids].chunk(2, dim=1)
    xr = x.float()[tokens].unsqueeze(-1)
    act = torch.nn.functional.silu((gate @ xr).squeeze(-1)) * (up @ xr).squeeze(-1)
    ref = (w2.float()[expert_ids] @ act.unsqueeze(-1)).squeeze(-1)
    rows = gemm2_output[permuted_idx.flatten().long()].float()
    torch.testing.assert_close(rows, ref, rtol=3e-2, atol=3e-2)
    assert ((rows - ref).norm() / ref.norm()).item() < 1e-2


# 16 tokens use the fused expert-map prologue and 300 the blocked fallback, which leaves
# stale map entries for routes to other ranks.
@pytest.mark.parametrize("num_tokens", [16, 300])
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_do_finalize_false_expert_parallel(num_tokens, ep_rank):
    dtype = torch.bfloat16
    hidden_size, inter_size, num_experts, top_k, ep_size = 1024, 512, 8, 4, 2
    x, w31, w2, experts, weights = _make_moe(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed=1
    )
    experts_per_rank = num_experts // ep_size
    start = ep_rank * experts_per_rank
    local = slice(start, start + experts_per_rank)
    ep = dict(ep_size=ep_size, ep_rank=ep_rank)

    workspace = torch.empty(
        fused_moe.cutlass_fused_moe_workspace_size(
            num_tokens,
            hidden_size,
            inter_size,
            num_experts,
            top_k,
            x_dtype=dtype,
            weight_dtype=dtype,
            output_dtype=dtype,
            use_fused_finalize=False,
            device=x.device,
            **ep,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    # Fill the workspace with in-range map entries for every route first: a route to
    # another rank must still come back as -1, not as a row of this earlier call.
    all_local = (
        start
        + torch.arange(num_tokens * top_k, dtype=torch.int32, device="cuda").view(
            num_tokens, top_k
        )
        % experts_per_rank
    )
    _run(
        x,
        w31[local],
        w2[local],
        all_local,
        weights,
        dtype,
        workspace_buffer=workspace,
        do_finalize=False,
        **ep,
    )

    ref = _finalized(x, w31[local], w2[local], experts, weights, dtype, **ep)
    gemm2_output, expert_weights, permuted_idx = _run(
        x,
        w31[local],
        w2[local],
        experts,
        weights,
        dtype,
        workspace_buffer=workspace,
        do_finalize=False,
        **ep,
    )

    is_local = (experts >= start) & (experts < start + experts_per_rank)
    assert is_local.any() and not is_local.all()
    assert (permuted_idx[~is_local] == -1).all()
    local_rows = permuted_idx[is_local]
    assert local_rows.unique().numel() == local_rows.numel()
    assert ((local_rows >= 0) & (local_rows < num_tokens * top_k)).all()
    torch.testing.assert_close(
        _finalize(gemm2_output, expert_weights, permuted_idx).to(dtype),
        ref,
        rtol=1e-2,
        atol=1e-2,
    )


def test_do_finalize_false_cuda_graph():
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size, num_experts, top_k = 32, 1024, 512, 8, 4
    x, w31, w2, experts, weights = _make_moe(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed=2
    )
    _run(x, w31, w2, experts, weights, dtype, do_finalize=False)  # autotune eagerly

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm2_output, expert_weights, permuted_idx = _run(
            x, w31, w2, experts, weights, dtype, do_finalize=False
        )

    # New inputs and routing in the captured buffers, then replay.
    x2, _, _, experts2, weights2 = _make_moe(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed=3
    )
    x.copy_(x2)
    experts.copy_(experts2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _finalize(gemm2_output, expert_weights, permuted_idx).to(dtype),
        _finalized(x, w31, w2, experts, weights, dtype),
        rtol=1e-2,
        atol=1e-2,
    )


def test_do_finalize_false_every_gemm2_tactic():
    """Each GEMM2 tactic (TMA or not, swap_ab or not) writes the caller's buffer densely."""
    dtype = torch.bfloat16
    num_tokens, hidden_size, inter_size, num_experts, top_k = 48, 1024, 512, 8, 4
    x, w31, w2, experts, weights = _make_moe(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed=8
    )
    major, minor = get_compute_capability(x.device)
    runner = get_cutlass_fused_moe_module(f"{major * 10 + minor}").MoERunner(
        x_dtype=dtype,
        weight_dtype=dtype,
        output_dtype=dtype,
        top_k=top_k,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        enable_pdl=True,
        activation_type=ActivationType.Swiglu,
        use_packed_weights=False,
        use_fused_finalize=False,
        use_wfp4afp8_humming=False,
    )
    tactics = list(
        runner.fused_moe_runner.get_valid_tactics_for_shape(2, hidden_size, inter_size)
    )
    assert tactics
    tested = 0
    for tactic in tactics:
        profile_ids = [-1, tactic]
        try:
            ref = _finalized(
                x, w31, w2, experts, weights, dtype, profile_ids=profile_ids
            )
            torch.cuda.synchronize()
        except RuntimeError:
            continue  # tactic not runnable for this shape; not specific to do_finalize
        gemm2_output, expert_weights, permuted_idx = _run(
            x,
            w31,
            w2,
            experts,
            weights,
            dtype,
            profile_ids=profile_ids,
            do_finalize=False,
        )
        torch.testing.assert_close(
            _finalize(gemm2_output, expert_weights, permuted_idx).to(dtype),
            ref,
            rtol=1e-2,
            atol=1e-2,
            msg=lambda m, tactic=tactic: f"GEMM2 tactic {tactic}: {m}",
        )
        tested += 1
    assert tested > 0


def test_do_finalize_false_fp8_per_tensor():
    """GEMM2 with FP8 weights and activations still writes rows in the output dtype."""
    dtype, fp8 = torch.bfloat16, torch.float8_e4m3fn
    num_tokens, hidden_size, inter_size, num_experts, top_k = 32, 1024, 512, 8, 4
    x, w31, w2, experts, weights = _make_moe(
        num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed=9
    )
    fp8_max = torch.finfo(fp8).max

    def quant(t, dims):
        scale = t.float().abs().amax(dim=dims, keepdim=True) / fp8_max
        return (t.float() / scale).to(fp8), scale.flatten()

    x_q, x_scale = quant(x, (0, 1))
    w31_q, w31_scale = quant(w31, (1, 2))
    w2_q, w2_scale = quant(w2, (1, 2))
    # [fc1 dequant per expert, fc2 activation quant, fc2 dequant per expert, fc1 input dequant]
    quant_scales = [
        (w31_scale * x_scale).float(),
        torch.tensor(1.0, device="cuda"),
        w2_scale.float(),
        x_scale.reshape(()).float(),
    ]

    def run(**kwargs):
        return fused_moe.cutlass_fused_moe(
            x_q,
            experts,
            weights,
            w31_q,
            w2_q,
            dtype,
            quant_scales=quant_scales,
            use_fused_finalize=False,
            **kwargs,
        )

    ref = run()
    ref = ref[0] if isinstance(ref, (list, tuple)) else ref
    gemm2_output, expert_weights, permuted_idx = run(do_finalize=False)
    assert gemm2_output.dtype == dtype
    torch.testing.assert_close(
        _finalize(gemm2_output, expert_weights, permuted_idx).to(dtype),
        ref,
        rtol=1e-2,
        atol=1e-2,
    )


def _assert_close_rounded_chain(actual, ref):
    # The fused kernel rounds to the activation dtype after every step (each weighted
    # route, the scaling, the shared add and the all-reduce payload), so elementwise
    # errors reach a few bf16 ulps. Bound them like the upstream pattern-7 test does, and
    # bound the overall relative error tightly so that a wrong row or a missed route fails.
    actual = actual.float()
    torch.testing.assert_close(actual, ref, rtol=0.2, atol=0.2)
    rel = ((actual - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"relative error {rel:.3e}"


def _rms_norm(x, weight, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def _finalize_allreduce_worker(world_size, rank, port, dtype):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    num_tokens, hidden_size, inter_size, num_experts, top_k = 16, 2048, 512, 8, 4
    eps, routed_scaling_factor = 1e-5, 2.5
    ipc_handles, workspace = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
        rank, world_size, 128, hidden_size, group=group
    )
    try:
        # Tokens, routing, residual and norm are the same on every rank; each rank owns
        # its own expert weights, as a TP shard of the intermediate dimension would.
        x, _, _, experts, weights = _make_moe(
            num_tokens, hidden_size, inter_size, num_experts, top_k, dtype, seed=6
        )
        _, w31, w2, _, _ = _make_moe(
            num_tokens,
            hidden_size,
            inter_size,
            num_experts,
            top_k,
            dtype,
            seed=100 + rank,
        )
        gen = torch.Generator(device="cuda").manual_seed(7)
        residual = torch.randn(
            num_tokens, hidden_size, device="cuda", generator=gen
        ).to(dtype)
        norm_weight = torch.randn(hidden_size, device="cuda", generator=gen).to(dtype)
        shared = (torch.randn(num_tokens, hidden_size, device="cuda") / 5).to(dtype)

        gemm2_output, expert_weights, permuted_idx = _run(
            x, w31, w2, experts, weights, dtype, do_finalize=False
        )
        norm_out = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        comm.trtllm_moe_finalize_allreduce_fusion(
            allreduce_in=gemm2_output,
            residual_in=residual,
            norm_weight=norm_weight,
            expanded_idx_to_permuted_idx=permuted_idx,
            workspace_ptrs=workspace,
            launch_with_pdl=True,
            world_rank=rank,
            world_size=world_size,
            eps=eps,
            shared_expert_output=shared,
            expert_scale_factor=expert_weights.to(dtype),
            norm_out=norm_out,
            residual_out=residual_out,
            quant_out=None,
            scale_out=None,
            routed_scaling_factor=routed_scaling_factor,
        )

        # Reference: the finalized CUTLASS output, then scale, shared add, all-reduce,
        # residual and RMSNorm in FP32.
        moe_out = _finalized(x, w31, w2, experts, weights, dtype).float()
        partial = moe_out * routed_scaling_factor + shared.float()
        dist.all_reduce(partial, group=group)
        ref_residual = partial + residual.float()
        torch.cuda.synchronize()
        _assert_close_rounded_chain(residual_out, ref_residual)
        _assert_close_rounded_chain(norm_out, _rms_norm(ref_residual, norm_weight, eps))
    finally:
        dist.barrier(group=group)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
            ipc_handles, group=group
        )
        dist.destroy_process_group(group=group)


def _open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("dtype", DTYPES)
def test_do_finalize_false_feeds_moe_finalize_allreduce(dtype):
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    ctx = mp.get_context("spawn")
    port = _open_port()
    procs = [
        ctx.Process(
            target=_finalize_allreduce_worker, args=(world_size, rank, port, dtype)
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
