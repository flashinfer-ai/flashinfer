"""
Copyright (c) 2025 by FlashInfer team.

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

"""
Real multi-GPU test for the peer-scatter combine fused into the Blackwell
CuTe-DSL MoE GEMM2 finalize epilogue.

Launch with four ranks on one node::

    python -m torch.distributed.run --nproc_per_node=4 -m pytest -q \
        tests/moe/test_cute_dsl_moe_peer_scatter_multigpu.py

(`python -m torch.distributed.run`, not the bare `torchrun`: in a container
where flashinfer lives in a venv, the `torchrun` on PATH belongs to the base
interpreter and its workers would not see the venv.)

Setup mirrors what the vLLM wiring plan does. Every rank holds the same
all-gathered token batch (that is what `AgRsAll2AllManager.dispatch()`'s
`all_gatherv` produces), but owns a distinct slice of the experts, so each
`(token, k_slot)` route is computed by exactly one rank in the world. GEMM2
then writes that route straight into the combine buffer of the rank that owns
the token, over NVLink, through `SymmetricBuffer.peer_addresses`.

Two things are checked, and they are the two claims the design rests on:

1. Every destination slot has exactly one writer, so the buffer needs no
   zero-initialisation. The buffer is pre-filled with NaN and must contain
   none afterwards.
2. Reducing the peer-written buffer locally gives the same answer as the
   baseline this is meant to replace: run the identical GEMM2 without peer
   writes and all-reduce the per-rank partial results, which is what
   `AgRsAll2AllManager.combine()`'s `reduce_scatterv` does.

The weights are deliberately identical on every rank while the global expert
ranges differ. That makes the "global model" inconsistent, which does not
matter here: both arms of the comparison use the same per-rank weights, so the
test isolates the combine mechanism, which is what is under test.
"""

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.fused_moe.cute_dsl.fused_moe import cute_dsl_fused_moe
from flashinfer.fused_moe.cute_dsl.moe_utils import (
    allocate_peer_combine_buffer,
    build_peer_scatter_destination_map,
    peer_scatter_local_reduce,
)
from flashinfer.utils import is_sm100a_supported

from .utils import create_moe_tensors

WORLD_SIZE = 4
NUM_TOKENS = 256
TOP_K = 4
NUM_EXPERTS = 256
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
SEED = 42


@pytest.fixture(scope="module")
def distributed_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != WORLD_SIZE:
        pytest.skip(f"Run this test with {WORLD_SIZE} distributed ranks")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not is_sm100a_supported(device):
        pytest.skip("Peer scatter targets the Blackwell CuTe-DSL MoE GEMM2 kernel")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    try:
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()


def _expert_parallel_case(rank: int, device: torch.device):
    """Build this rank's slice of an expert-parallel MoE step.

    Same seed on every rank, so the token batch and the routing are identical
    everywhere, exactly as they are after an all-gather. Only
    `local_expert_offset` differs, which is what makes each route land on one
    rank.
    """
    num_local_experts = NUM_EXPERTS // WORLD_SIZE
    tensors = create_moe_tensors(
        num_tokens=NUM_TOKENS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_experts=NUM_EXPERTS,
        num_local_experts=num_local_experts,
        top_k=TOP_K,
        device=str(device),
        seed=SEED,
    )
    call = dict(
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        w1_weight=tensors["w1_weight"],
        w1_weight_sf=tensors["w1_weight_sf"],
        w1_alpha=tensors["w1_alpha"],
        w2_weight=tensors["w2_weight"],
        w2_weight_sf=tensors["w2_weight_sf"],
        w2_alpha=tensors["w2_alpha"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_local_experts=num_local_experts,
        local_expert_offset=rank * num_local_experts,
        use_fused_finalize=False,
        quant_mode="w4a4",
        x=tensors["x"],
        x_sf=tensors["x_sf"],
        fc2_input_scale=tensors["fc2_input_scale"],
        per_token_scale=tensors["x_per_token_scale"],
    )
    return tensors, call


@pytest.mark.parametrize("release_fence", [False, True])
def test_peer_scatter_four_ranks_matches_reduce_scatter_baseline(
    distributed_group, release_fence
):
    group = distributed_group
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens_per_rank = NUM_TOKENS // WORLD_SIZE

    tensors, call = _expert_parallel_case(rank, device)

    # --- Baseline arm: unmodified GEMM2, then a cross-rank sum -------------
    # Each rank reduces only its own routes (moe_sort masks the rest with -1),
    # so summing across ranks reconstructs the full output. That is what
    # AgRsAll2AllManager.combine()'s reduce_scatterv does today.
    local_partial = cute_dsl_fused_moe(**call)
    assert local_partial.shape == (NUM_TOKENS, HIDDEN_SIZE)
    baseline = local_partial.float().clone()
    dist.all_reduce(baseline, op=dist.ReduceOp.SUM, group=group)
    my_baseline = baseline[rank * tokens_per_rank : (rank + 1) * tokens_per_rank]

    # --- Peer arm: GEMM2 writes into the owning rank's combine buffer ------
    combine = allocate_peer_combine_buffer(
        (tokens_per_rank * TOP_K, HIDDEN_SIZE), torch.bfloat16, device, group
    )
    assert combine.peer_addresses is not None
    assert combine.peer_addresses.shape == (WORLD_SIZE,)
    if rank == 0:
        print(f"[peer table via {combine.source}]", flush=True)

    # Poison every slot. The design claims each (token, k_slot) has exactly one
    # writer in the whole world, so no slot may survive as NaN and the buffer
    # needs no zero-init. Anything unwritten shows up here.
    combine.tensor.fill_(float("nan"))

    token_dst_rank, token_dst_local_idx = build_peer_scatter_destination_map(
        [tokens_per_rank] * WORLD_SIZE, device
    )
    dist.barrier(group)

    returned = cute_dsl_fused_moe(
        **call,
        peer_addresses=combine.peer_addresses,
        token_dst_rank=token_dst_rank,
        token_dst_local_idx=token_dst_local_idx,
        combine_buffer=combine.tensor,
        # With the fence, GEMM2 drains its bulk copies and issues a
        # system-scope release before exiting, so the writes are guaranteed
        # visible rather than relying on kernel completion alone.
        peer_release_fence=release_fence,
    )
    assert returned.data_ptr() == combine.tensor.data_ptr()

    # Completion signal: every rank's peer writes must have landed before any
    # rank reads its own buffer.
    torch.cuda.synchronize(device)
    dist.barrier(group)

    unwritten = int(torch.isnan(combine.tensor).any(dim=1).sum().item())
    assert unwritten == 0, (
        f"rank {rank}: {unwritten} of {tokens_per_rank * TOP_K} combine slots "
        "were never written; the one-writer-per-slot invariant does not hold"
    )

    # Establish that this really was a cross-GPU write and not every rank
    # filling its own buffer. The writer of slot (t, k) is the rank owning the
    # expert routed to, which the routing tensor names outright.
    my_experts = tensors["token_selected_experts"][
        rank * tokens_per_rank : (rank + 1) * tokens_per_rank
    ]
    writers = torch.bincount(
        (my_experts.long() // (NUM_EXPERTS // WORLD_SIZE)).flatten(),
        minlength=WORLD_SIZE,
    )
    assert int(writers.sum().item()) == tokens_per_rank * TOP_K
    remote = int(writers.sum().item() - writers[rank].item())
    assert (writers > 0).all(), (
        f"rank {rank}: not every peer wrote into this buffer ({writers.tolist()}), "
        "so this run did not exercise cross-GPU scatter"
    )
    print(
        f"[rank {rank}] writers per rank={writers.tolist()} remote_slots={remote}",
        flush=True,
    )

    my_scales = tensors["token_final_scales"][
        rank * tokens_per_rank : (rank + 1) * tokens_per_rank
    ]
    fused = peer_scatter_local_reduce(combine.tensor, my_scales, tokens_per_rank, TOP_K)
    assert fused.shape == (tokens_per_rank, HIDDEN_SIZE)

    # The two arms differ only in summation order: the baseline sums bf16
    # partials across ranks through NCCL, the peer arm accumulates the top_k
    # slots in fp32 inside moe_unpermute and rounds once. So this is close, not
    # bitwise.
    diff = (fused.float() - my_baseline).abs()
    scale = my_baseline.abs().max().clamp_min(1e-6)
    max_abs = float(diff.max().item())
    max_rel = float((diff.max() / scale).item())
    print(
        f"[rank {rank}] slots={tokens_per_rank * TOP_K} unwritten=0 "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
        flush=True,
    )
    torch.testing.assert_close(fused.float(), my_baseline, rtol=2e-2, atol=2e-2)

    dist.barrier(group)


def test_peer_scatter_four_ranks_rejects_short_combine_buffer(distributed_group):
    """The opt-in bounds check must catch a combine buffer that is too small."""
    group = distributed_group
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens_per_rank = NUM_TOKENS // WORLD_SIZE

    _, call = _expert_parallel_case(rank, device)
    # One row short of what token_dst_local_idx will ask for.
    combine = allocate_peer_combine_buffer(
        (tokens_per_rank * TOP_K - 1, HIDDEN_SIZE), torch.bfloat16, device, group
    )
    token_dst_rank, token_dst_local_idx = build_peer_scatter_destination_map(
        [tokens_per_rank] * WORLD_SIZE, device
    )

    os.environ["FLASHINFER_VALIDATE_INPUTS"] = "1"
    try:
        with pytest.raises(ValueError, match="combine-buffer row"):
            cute_dsl_fused_moe(
                **call,
                peer_addresses=combine.peer_addresses,
                token_dst_rank=token_dst_rank,
                token_dst_local_idx=token_dst_local_idx,
                combine_buffer=combine.tensor,
            )
    finally:
        os.environ.pop("FLASHINFER_VALIDATE_INPUTS", None)
    dist.barrier(group)
