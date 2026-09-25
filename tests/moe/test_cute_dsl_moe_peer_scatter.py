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
Tests for the peer-scatter (cross-GPU fused combine) path of the Blackwell
CuTe-DSL MoE GEMM2 finalize kernel.

The kernel change replaces the epilogue's destination address: instead of
storing each ``(token, k_slot)`` row into the local ``out`` buffer, it stores
into the combine buffer of the rank that owns the token, selected from a
peer-pointer table. Every route has exactly one writer across the world, so
the store stays the plain non-accumulating bulk copy and no zero-init is
needed.

The multi-rank behaviour is exercised on a single GPU by handing the kernel a
peer table of ordinary local allocations: the kernel only ever loads a base
address out of that table and stores to it, so several "ranks" can be
simulated with several buffers on one device. That covers the rank selection,
the (rank, row) metadata packing, and the address arithmetic without needing a
distributed job.
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported
from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    PEER_DST_RANK_LIMIT,
    PEER_DST_ROW_BITS,
    PEER_DST_ROW_LIMIT,
    PEER_DST_ROW_MASK,
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
)
from flashinfer.fused_moe.cute_dsl.fused_moe import cute_dsl_fused_moe
from flashinfer.fused_moe.cute_dsl.moe_utils import (
    build_peer_scatter_destination_map,
    peer_scatter_local_reduce,
)

from .utils import check_accuracy, create_moe_tensors


def _is_sm100_family() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in (
        (10, 0),
        (10, 3),
        (10, 7),
    )


cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)
sm100_required = pytest.mark.skipif(
    not _is_sm100_family(),
    reason="Peer scatter targets the Blackwell CuTe-DSL MoE GEMM2 kernel",
)
requires_dsl_arch = pytest.mark.skipif(
    torch.cuda.is_available()
    and not is_cute_dsl_arch_supported(
        *torch.cuda.get_device_capability(0), native_only=True
    ),
    reason="installed CuTe DSL does not support this GPU architecture",
)
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)
# moe_unpermute's JIT module is generated only for SM90 and SM100
# (gen_moe_utils_module passes supported_major_versions=[9, 10]), so the local
# reduce cannot even be built on older cards.
moe_utils_arch_required = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0)[0] not in (9, 10),
    reason="moe_unpermute is compiled only for SM90 and SM100",
)


# ---------------------------------------------------------------------------
# Metadata packing. Host-only: no GPU needed.
# ---------------------------------------------------------------------------


def test_packing_constants_are_consistent():
    assert PEER_DST_ROW_LIMIT == 1 << PEER_DST_ROW_BITS
    assert PEER_DST_ROW_MASK == PEER_DST_ROW_LIMIT - 1
    assert PEER_DST_RANK_LIMIT == 1 << (31 - PEER_DST_ROW_BITS)


def test_packing_is_lossless_and_fits_int32():
    """The packed (rank, row) pair must round-trip and stay a positive int32.

    The epilogue reuses the single int32 ``sMetaTokenIdx`` slot for both
    fields, so a signed overflow here would silently corrupt destinations.
    """
    worst = (PEER_DST_RANK_LIMIT - 1) * PEER_DST_ROW_LIMIT + (PEER_DST_ROW_LIMIT - 1)
    assert worst == 2**31 - 1

    for rank, row in (
        (0, 0),
        (0, PEER_DST_ROW_LIMIT - 1),
        (PEER_DST_RANK_LIMIT - 1, 0),
        (PEER_DST_RANK_LIMIT - 1, PEER_DST_ROW_LIMIT - 1),
        (3, 12345),
    ):
        packed = rank * PEER_DST_ROW_LIMIT + row
        assert packed >= 0
        assert packed // PEER_DST_ROW_LIMIT == rank
        assert packed & PEER_DST_ROW_MASK == row


def test_kernel_rejects_peer_scatter_with_fused_finalize():
    with pytest.raises(ValueError, match="use_fused_finalize=False"):
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
            sf_vec_size=16,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            use_fused_finalize=True,
            use_peer_scatter=True,
        )


def test_default_path_passes_no_peer_kwargs():
    """The non-peer path must trace and launch with exactly its old arguments.

    Peer scatter must not perturb the default kernel: if any peer keyword
    leaked into ``cute.compile`` when the feature is off, every existing
    caller would trace a different wrapper signature.
    """
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        _peer_wrapper_kwargs,
    )

    assert _peer_wrapper_kwargs(False, "p", "r", "l", world_size=4, peer_rows=8) == {}
    # Compile site: pointers plus the two compile-time scalars.
    assert _peer_wrapper_kwargs(True, "p", "r", "l", world_size=4, peer_rows=8) == {
        "peer_addresses_ptr": "p",
        "token_dst_rank_ptr": "r",
        "token_dst_local_idx_ptr": "l",
        "world_size": 4,
        "peer_rows": 8,
    }
    # Launch site: pointers only, the scalars are baked in.
    assert _peer_wrapper_kwargs(True, "p", "r", "l") == {
        "peer_addresses_ptr": "p",
        "token_dst_rank_ptr": "r",
        "token_dst_local_idx_ptr": "l",
    }


# ---------------------------------------------------------------------------
# Destination map. Host-only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes,expect_rank,expect_local",
    [
        ([4], [0, 0, 0, 0], [0, 1, 2, 3]),
        ([2, 3], [0, 0, 1, 1, 1], [0, 1, 0, 1, 2]),
        ([3, 0, 2], [0, 0, 0, 2, 2], [0, 1, 2, 0, 1]),
        ([0, 0], [], []),
    ],
)
def test_build_peer_scatter_destination_map(sizes, expect_rank, expect_local):
    """The map is a pure function of ``sizes``: all-gather is rank-contiguous."""
    rank, local = build_peer_scatter_destination_map(sizes, "cpu")
    assert rank.dtype == torch.int32
    assert local.dtype == torch.int32
    assert rank.shape == (sum(sizes),)
    assert rank.tolist() == expect_rank
    assert local.tolist() == expect_local


def test_build_peer_scatter_destination_map_rejects_bad_sizes():
    with pytest.raises(ValueError, match="non-negative"):
        build_peer_scatter_destination_map([1, -1], "cpu")
    with pytest.raises(ValueError, match="at least one rank"):
        build_peer_scatter_destination_map([], "cpu")


@cuda_required
def test_build_peer_scatter_destination_map_on_device():
    rank, local = build_peer_scatter_destination_map([2, 2], "cuda")
    assert rank.device.type == "cuda"
    assert local.device.type == "cuda"
    assert rank.tolist() == [0, 0, 1, 1]
    assert local.tolist() == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Local reduce. Needs CUDA but not Blackwell: moe_unpermute is a plain kernel.
# ---------------------------------------------------------------------------


@moe_utils_arch_required
@pytest.mark.parametrize("num_local_tokens", [1, 7, 64])
@pytest.mark.parametrize("top_k", [2, 4])
def test_peer_scatter_local_reduce_matches_reference(num_local_tokens, top_k):
    """The local half of p2p+local_reduce is a plain weighted sum over top_k."""
    torch.manual_seed(0)
    hidden_size = 128
    combine_buffer = torch.randn(
        (num_local_tokens * top_k, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    topk_scales = torch.rand(
        (num_local_tokens, top_k), dtype=torch.float32, device="cuda"
    )

    result = peer_scatter_local_reduce(
        combine_buffer, topk_scales, num_local_tokens, top_k
    )

    reference = (
        combine_buffer.float().view(num_local_tokens, top_k, hidden_size)
        * topk_scales.unsqueeze(-1)
    ).sum(dim=1)

    assert result.shape == (num_local_tokens, hidden_size)
    torch.testing.assert_close(
        result.float(), reference.to(result.dtype).float(), rtol=2e-2, atol=2e-2
    )


@cuda_required
def test_peer_scatter_local_reduce_validates_shapes():
    combine_buffer = torch.zeros((8, 16), dtype=torch.bfloat16, device="cuda")
    scales = torch.ones((4, 2), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="at least"):
        peer_scatter_local_reduce(combine_buffer, scales, num_local_tokens=8, top_k=2)
    with pytest.raises(ValueError, match="topk_scales must have shape"):
        peer_scatter_local_reduce(
            combine_buffer,
            torch.ones((3, 2), dtype=torch.float32, device="cuda"),
            num_local_tokens=4,
            top_k=2,
        )


# ---------------------------------------------------------------------------
# End to end through the Blackwell kernel.
# ---------------------------------------------------------------------------


def _moe_case(num_tokens: int, top_k: int, num_experts: int):
    hidden_size, intermediate_size = 256, 512
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
    )
    api_inputs = {
        "x": tensors["x"],
        "x_sf": tensors["x_sf"],
        "fc2_input_scale": tensors["fc2_input_scale"],
        "per_token_scale": tensors["x_per_token_scale"],
    }
    common = dict(
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        w1_weight=tensors["w1_weight"],
        w1_weight_sf=tensors["w1_weight_sf"],
        w1_alpha=tensors["w1_alpha"],
        w2_weight=tensors["w2_weight"],
        w2_weight_sf=tensors["w2_weight_sf"],
        w2_alpha=tensors["w2_alpha"],
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_experts,
        use_fused_finalize=False,
        quant_mode="w4a4",
        **api_inputs,
    )
    return tensors, common, hidden_size


@cute_dsl_available
@sm100_required
@requires_dsl_arch
@pytest.mark.parametrize("num_tokens", [128, 256])
@pytest.mark.parametrize("top_k", [2, 8])
def test_peer_scatter_single_rank_matches_local(num_tokens, top_k):
    """A world of one must reproduce the ordinary deterministic path exactly.

    With ``world_size == 1`` the destination row the meta warp computes,
    ``dst_local_idx * top_k + topk_idx``, is the same ``token_idx * top_k +
    topk_idx`` the deterministic path already writes, and the peer table's only
    entry is this rank's own buffer. So the peer path must agree with the
    unmodified path to the bit.
    """
    num_experts = 256
    tensors, common, hidden_size = _moe_case(num_tokens, top_k, num_experts)

    reference = cute_dsl_fused_moe(**common)

    combine_buffer = torch.zeros(
        (num_tokens * top_k, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    peer_addresses = torch.tensor(
        [combine_buffer.data_ptr()], dtype=torch.int64, device="cuda"
    )
    token_dst_rank, token_dst_local_idx = build_peer_scatter_destination_map(
        [num_tokens], "cuda"
    )

    returned = cute_dsl_fused_moe(
        **common,
        peer_addresses=peer_addresses,
        token_dst_rank=token_dst_rank,
        token_dst_local_idx=token_dst_local_idx,
        combine_buffer=combine_buffer,
    )
    # GEMM2 hands back the raw per-slot buffer; the reduction is the caller's
    # job, after the completion barrier.
    assert returned.data_ptr() == combine_buffer.data_ptr()

    reduced = peer_scatter_local_reduce(
        combine_buffer, tensors["token_final_scales"], num_tokens, top_k
    )
    assert reduced.shape == (num_tokens, hidden_size)
    assert not torch.isnan(reduced).any()
    torch.testing.assert_close(reduced, reference, rtol=0, atol=0)


@cute_dsl_available
@sm100_required
@requires_dsl_arch
@pytest.mark.parametrize("world_size", [2, 4])
def test_peer_scatter_simulated_multi_rank_matches_local(world_size):
    """Simulate several owning ranks with several buffers on one GPU.

    The kernel only loads a base address out of the peer table and stores to
    it, so distinct local allocations stand in for distinct peers. This is what
    actually exercises rank selection and the packed-metadata unpack; the
    single-rank test above cannot, because rank 0 packs to zero.
    """
    num_tokens, top_k, num_experts = 256, 4, 256
    assert num_tokens % world_size == 0
    tokens_per_rank = num_tokens // world_size

    tensors, common, hidden_size = _moe_case(num_tokens, top_k, num_experts)
    reference = cute_dsl_fused_moe(**common)

    buffers = [
        torch.zeros(
            (tokens_per_rank * top_k, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(world_size)
    ]
    peer_addresses = torch.tensor(
        [buf.data_ptr() for buf in buffers], dtype=torch.int64, device="cuda"
    )
    token_dst_rank, token_dst_local_idx = build_peer_scatter_destination_map(
        [tokens_per_rank] * world_size, "cuda"
    )

    cute_dsl_fused_moe(
        **common,
        peer_addresses=peer_addresses,
        token_dst_rank=token_dst_rank,
        token_dst_local_idx=token_dst_local_idx,
        # `out` is this process' own buffer; the stores for every rank still go
        # through the peer table, including rank 0's entry.
        combine_buffer=buffers[0],
    )

    scales = tensors["token_final_scales"]
    reduced = torch.cat(
        [
            peer_scatter_local_reduce(
                buffers[r],
                scales[r * tokens_per_rank : (r + 1) * tokens_per_rank],
                tokens_per_rank,
                top_k,
            )
            for r in range(world_size)
        ],
        dim=0,
    )

    assert reduced.shape == (num_tokens, hidden_size)
    torch.testing.assert_close(reduced, reference, rtol=0, atol=0)


@cute_dsl_available
@sm100_required
@requires_dsl_arch
def test_peer_scatter_accuracy_against_moe_reference():
    """Sanity-check the fused path against the numerical MoE reference."""
    from .utils import compute_reference_moe_fp4

    num_tokens, top_k, num_experts = 128, 2, 256
    hidden_size, intermediate_size = 256, 512
    tensors, common, _ = _moe_case(num_tokens, top_k, num_experts)

    combine_buffer = torch.zeros(
        (num_tokens * top_k, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    peer_addresses = torch.tensor(
        [combine_buffer.data_ptr()], dtype=torch.int64, device="cuda"
    )
    token_dst_rank, token_dst_local_idx = build_peer_scatter_destination_map(
        [num_tokens], "cuda"
    )
    cute_dsl_fused_moe(
        **common,
        peer_addresses=peer_addresses,
        token_dst_rank=token_dst_rank,
        token_dst_local_idx=token_dst_local_idx,
        combine_buffer=combine_buffer,
    )
    result = peer_scatter_local_reduce(
        combine_buffer, tensors["token_final_scales"], num_tokens, top_k
    )

    ref_output = compute_reference_moe_fp4(
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_states=tensors["x_ref"].float(),
        gemm1_weights=tensors["w1_weight_bf16"].float(),
        gemm2_weights=tensors["w2_weight_bf16"].float(),
        gemm1_alpha=tensors["w1_alpha"],
        gemm2_alpha=tensors["w2_alpha"],
        fc2_input_scale=tensors["fc2_input_scale"],
        use_per_token_activation=tensors["x_per_token_scale"] is not None,
    )
    passed, percent_within, atol = check_accuracy(result, ref_output)
    assert passed, (
        f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
    )


# ---------------------------------------------------------------------------
# Input validation on the dispatcher.
# ---------------------------------------------------------------------------


@cute_dsl_available
@sm100_required
@requires_dsl_arch
def test_peer_scatter_rejects_fused_finalize_end_to_end():
    num_tokens, top_k, num_experts = 128, 2, 256
    tensors, common, hidden_size = _moe_case(num_tokens, top_k, num_experts)
    common["use_fused_finalize"] = True

    combine_buffer = torch.zeros(
        (num_tokens * top_k, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    token_dst_rank, token_dst_local_idx = build_peer_scatter_destination_map(
        [num_tokens], "cuda"
    )
    with pytest.raises(ValueError, match="use_fused_finalize=False"):
        cute_dsl_fused_moe(
            **common,
            peer_addresses=torch.tensor(
                [combine_buffer.data_ptr()], dtype=torch.int64, device="cuda"
            ),
            token_dst_rank=token_dst_rank,
            token_dst_local_idx=token_dst_local_idx,
            combine_buffer=combine_buffer,
        )


@cute_dsl_available
@sm100_required
@requires_dsl_arch
def test_peer_scatter_requires_combine_buffer():
    num_tokens, top_k, num_experts = 128, 2, 256
    _, common, hidden_size = _moe_case(num_tokens, top_k, num_experts)
    token_dst_rank, token_dst_local_idx = build_peer_scatter_destination_map(
        [num_tokens], "cuda"
    )
    scratch = torch.zeros(
        (num_tokens * top_k, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(ValueError, match="combine_buffer"):
        cute_dsl_fused_moe(
            **common,
            peer_addresses=torch.tensor(
                [scratch.data_ptr()], dtype=torch.int64, device="cuda"
            ),
            token_dst_rank=token_dst_rank,
            token_dst_local_idx=token_dst_local_idx,
            combine_buffer=None,
        )
