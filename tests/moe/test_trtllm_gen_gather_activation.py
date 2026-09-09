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

Standalone tests for the trtllm-gen MoE activation gather
(``trtllm_gen_moe_gather_activation``).

Notes on test construction:
- The op is a pure copy, so every comparison against the torch reference is
  bitwise (``atol=rtol=0``).
- The synthetic permutation maps the ``num_tokens * top_k`` expanded slots into
  a larger padded row space, exactly like the routing padding does. Unmapped
  rows are filled with NaN, so any out-of-contract read (a padding row, or a
  row an inactive slot used to point at) poisons the output and fails the
  comparison rather than passing silently.
- Widths are swept across and around the kernel's 128-bit vector width, so both
  the vectorized and the element-wise path are exercised.
"""

import zlib

import pytest
import torch

from flashinfer import shuffle_matrix_a
from flashinfer.fused_moe import (
    WeightLayout,
    convert_to_block_layout,
    trtllm_bf16_routed_moe,
    trtllm_gen_moe_gather_activation,
)
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import device_support_pdl, get_compute_capability

from .trtllm_gen_fused_moe_utils import pack_topk_for_routed_moe


@pytest.fixture(autouse=True)
def require_supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = get_compute_capability(torch.device("cuda"))
    sm = major * 10 + minor
    if not trtllm_gen_moe_gather_activation.is_compute_capability_supported(sm):
        pytest.skip(f"trtllm-gen activation gather is not supported on SM{sm}")


def stable_seed(*parts):
    """Deterministic across processes (unlike hash(), which salts strings)."""
    return zlib.crc32(repr(parts).encode()) % (2**31)


def reference_gather(activation_output, expanded_idx_to_permuted_idx, top_k):
    """The eager gather this op replaces (``bgmv_moe_gemm2_lora_delta``), reshaped
    to the op's ``[num_tokens, top_k, intermediate_size]`` output."""
    intermediate_size = activation_output.shape[1]
    perm = expanded_idx_to_permuted_idx.reshape(-1).to(torch.int64)
    num_slots = perm.numel()
    valid = perm >= 0
    out = torch.zeros(
        num_slots,
        intermediate_size,
        dtype=activation_output.dtype,
        device=activation_output.device,
    )
    out[valid] = activation_output[perm[valid]]
    return out.reshape(num_slots // top_k, top_k, intermediate_size)


def make_case(
    num_tokens,
    top_k,
    intermediate_size,
    *,
    extra_rows=64,
    inactive_frac=0.0,
    dtype=torch.bfloat16,
    seed=0,
):
    """Synthetic permuted activation with NaN poison outside the contract.

    Returns ``(activation_output, expanded_idx_to_permuted_idx, inactive)``.
    """
    device = torch.device("cuda")
    torch.manual_seed(seed)
    num_slots = num_tokens * top_k
    num_padded = num_slots + extra_rows

    # Map each expanded slot to a distinct padded row; NaN-fill the rows no slot
    # points at so that reading the routing padding is caught.
    row_of_slot = torch.randperm(num_padded, device=device)[:num_slots]
    activation_output = torch.full(
        (num_padded, intermediate_size), float("nan"), dtype=dtype, device=device
    )
    activation_output[row_of_slot] = torch.randn(
        num_slots, intermediate_size, dtype=dtype, device=device
    )

    expanded_idx_to_permuted_idx = row_of_slot.to(torch.int32).reshape(
        num_tokens, top_k
    )
    if inactive_frac > 0:
        inactive = torch.rand(num_tokens, top_k, device=device) < inactive_frac
        expanded_idx_to_permuted_idx = torch.where(
            inactive, -1, expanded_idx_to_permuted_idx
        )
        # NaN-poison the rows the now-inactive slots pointed at: the kernel must
        # not read them.
        activation_output[row_of_slot.reshape(num_tokens, top_k)[inactive]] = float(
            "nan"
        )
    else:
        inactive = torch.zeros(num_tokens, top_k, dtype=torch.bool, device=device)
    return activation_output, expanded_idx_to_permuted_idx, inactive


def assert_gather_matches(activation_output, expanded_idx_to_permuted_idx, top_k):
    out = trtllm_gen_moe_gather_activation(
        activation_output, expanded_idx_to_permuted_idx, top_k
    )
    ref = reference_gather(activation_output, expanded_idx_to_permuted_idx, top_k)
    assert out.shape == ref.shape
    assert out.dtype == activation_output.dtype
    assert not out.isnan().any(), "output poisoned by an out-of-contract read"
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
    return out


@pytest.mark.parametrize("num_tokens", [1, 2, 17, 1024, 8192])
@pytest.mark.parametrize("top_k", [1, 2, 4, 6, 8])
def test_gather_matches_reference(num_tokens, top_k):
    activation_output, e2p, _ = make_case(
        num_tokens,
        top_k,
        512,
        inactive_frac=0.3,
        seed=stable_seed(num_tokens, top_k),
    )
    assert_gather_matches(activation_output, e2p, top_k)


@pytest.mark.parametrize("intermediate_size", [128, 512, 1536, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gather_widths(intermediate_size, dtype):
    activation_output, e2p, _ = make_case(
        129,
        4,
        intermediate_size,
        inactive_frac=0.3,
        dtype=dtype,
        seed=stable_seed(intermediate_size, str(dtype)),
    )
    assert_gather_matches(activation_output, e2p, 4)


@pytest.mark.parametrize("intermediate_size", [1, 7, 100, 132, 1023])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gather_widths_not_a_multiple_of_the_vector(intermediate_size, dtype):
    """Widths the 128-bit path cannot handle fall back to element-wise copies."""
    activation_output, e2p, _ = make_case(
        33,
        4,
        intermediate_size,
        inactive_frac=0.3,
        dtype=dtype,
        seed=stable_seed(intermediate_size, str(dtype), "unaligned"),
    )
    assert_gather_matches(activation_output, e2p, 4)


@pytest.mark.parametrize("inactive_frac", [0.0, 0.5, 1.0])
# 512 elements take the 128-bit path, 1023 the element-wise one; each has its own
# zero constant, and only a bitwise check tells +0.0 from -0.0.
@pytest.mark.parametrize("intermediate_size", [512, 1023])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_inactive_slots_are_bitwise_zero(inactive_frac, intermediate_size, dtype):
    """Slots with a negative index must produce exactly zero, and only those."""
    num_tokens, top_k = 65, 4
    activation_output, e2p, inactive = make_case(
        num_tokens,
        top_k,
        intermediate_size,
        inactive_frac=inactive_frac,
        dtype=dtype,
        seed=stable_seed("inactive", inactive_frac, intermediate_size, str(dtype)),
    )
    if inactive_frac == 1.0:
        # `torch.rand < 1.0` is almost surely all-True, but not by construction.
        e2p = torch.full_like(e2p, -1)
        inactive = torch.ones_like(inactive)

    out = assert_gather_matches(activation_output, e2p, top_k)

    assert (e2p < 0).equal(inactive)
    # Bitwise zero, not just numerically zero (-0.0 would compare equal to 0.0).
    bits = out.view(torch.int16)
    assert (bits[inactive] == 0).all()
    if not inactive.all():
        # Active rows carry the source rows, which are non-zero with probability 1.
        assert (bits[~inactive] != 0).float().mean() > 0.99


def test_padded_source_rows_are_never_read():
    """The source has strictly more rows than the permutation references; the
    trailing rows hold NaN and must not appear in the output."""
    num_tokens, top_k, intermediate_size = 128, 4, 512
    device = torch.device("cuda")
    torch.manual_seed(stable_seed("padded"))
    num_slots = num_tokens * top_k
    tail_rows = 257

    # Every slot maps into the first num_slots rows, so the trailing rows are
    # unreachable padding.
    row_of_slot = torch.randperm(num_slots, device=device)
    activation_output = torch.full(
        (num_slots + tail_rows, intermediate_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    activation_output[:num_slots] = torch.randn(
        num_slots, intermediate_size, dtype=torch.bfloat16, device=device
    )
    e2p = row_of_slot.to(torch.int32).reshape(num_tokens, top_k)
    inactive = torch.rand(num_tokens, top_k, device=device) < 0.25
    e2p = torch.where(inactive, -1, e2p)
    activation_output[row_of_slot.reshape(num_tokens, top_k)[inactive]] = float("nan")

    assert activation_output.shape[0] > int(e2p.max()) + 1
    assert activation_output[num_slots:].isnan().all(), (
        "the poison rows were not planted"
    )
    assert_gather_matches(activation_output, e2p, top_k)


def test_gather_into_preallocated_out():
    num_tokens, top_k, intermediate_size = 33, 4, 512
    activation_output, e2p, _ = make_case(
        num_tokens,
        top_k,
        intermediate_size,
        inactive_frac=0.3,
        seed=stable_seed("out"),
    )
    out = torch.full(
        (num_tokens, top_k, intermediate_size),
        float("inf"),
        dtype=activation_output.dtype,
        device=activation_output.device,
    )
    returned = trtllm_gen_moe_gather_activation(activation_output, e2p, top_k, out=out)
    assert returned.data_ptr() == out.data_ptr()
    ref = reference_gather(activation_output, e2p, top_k)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


def test_gather_rejects_bad_inputs():
    # ValueError, not Exception: a JIT build failure or a device-side assert also
    # satisfies Exception and would let these cases pass for the wrong reason.
    num_tokens, top_k, intermediate_size = 8, 4, 512
    activation_output, e2p, _ = make_case(
        num_tokens, top_k, intermediate_size, seed=stable_seed("reject")
    )
    device = activation_output.device

    with pytest.raises(ValueError, match="must be 2D"):
        trtllm_gen_moe_gather_activation(activation_output.unsqueeze(0), e2p, top_k)
    with pytest.raises(ValueError, match="int32"):
        trtllm_gen_moe_gather_activation(activation_output, e2p.to(torch.int64), top_k)
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        trtllm_gen_moe_gather_activation(activation_output, e2p, 0)
    with pytest.raises(ValueError, match="multiple of top_k"):
        trtllm_gen_moe_gather_activation(activation_output, e2p, 3)
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        trtllm_gen_moe_gather_activation(activation_output.float(), e2p, top_k)
    if torch.cuda.device_count() > 1:
        with pytest.raises(ValueError, match="same device"):
            trtllm_gen_moe_gather_activation(activation_output, e2p.to("cuda:1"), top_k)
    with pytest.raises(ValueError, match="shape"):
        trtllm_gen_moe_gather_activation(
            activation_output,
            e2p,
            top_k,
            out=torch.empty(
                num_tokens,
                top_k,
                intermediate_size + 8,
                dtype=activation_output.dtype,
                device=device,
            ),
        )
    with pytest.raises(ValueError, match="dtype"):
        trtllm_gen_moe_gather_activation(
            activation_output,
            e2p,
            top_k,
            out=torch.empty(
                num_tokens,
                top_k,
                intermediate_size,
                dtype=torch.float32,
                device=device,
            ),
        )
    with pytest.raises(ValueError, match="device"):
        trtllm_gen_moe_gather_activation(
            activation_output,
            e2p,
            top_k,
            out=torch.empty(
                num_tokens,
                top_k,
                intermediate_size,
                dtype=activation_output.dtype,
                device="cpu",
            ),
        )
    # A strided out= (a view into a wider workspace) must be caught in Python, not
    # by the binding's CHECK_INPUT.
    with pytest.raises(ValueError, match="contiguous"):
        wide = torch.empty(
            num_tokens,
            top_k,
            intermediate_size + 8,
            dtype=activation_output.dtype,
            device=device,
        )
        trtllm_gen_moe_gather_activation(
            activation_output, e2p, top_k, out=wide[:, :, :intermediate_size]
        )


@pytest.mark.parametrize("enable_pdl", [True, False])
def test_gather_pdl_toggle(enable_pdl):
    activation_output, e2p, _ = make_case(
        129, 4, 512, inactive_frac=0.3, seed=stable_seed("pdl", enable_pdl)
    )
    out = trtllm_gen_moe_gather_activation(
        activation_output, e2p, 4, enable_pdl=enable_pdl
    )
    torch.testing.assert_close(
        out, reference_gather(activation_output, e2p, 4), atol=0.0, rtol=0.0
    )


def test_gather_row_spans_multiple_blocks():
    """A row wider than one block's worth of copy units splits over grid.x."""
    activation_output, e2p, _ = make_case(
        16, 2, 8192, inactive_frac=0.25, extra_rows=8, seed=stable_seed("wide")
    )
    assert_gather_matches(activation_output, e2p, 2)


def test_gather_more_tokens_than_grid_z():
    """More tokens than the gridDim.z cap (8192) drives the token grid-stride loop."""
    activation_output, e2p, _ = make_case(
        8193, 1, 128, inactive_frac=0.25, extra_rows=8, seed=stable_seed("gridz")
    )
    assert_gather_matches(activation_output, e2p, 1)


@pytest.mark.parametrize("top_k", [1, 4])
def test_gather_zero_tokens_is_a_noop(top_k):
    """An idle expert-parallel rank calls this with an empty batch, and a zero grid
    dimension is a CUDA error."""
    device = torch.device("cuda")
    activation_output = torch.randn(64, 512, dtype=torch.bfloat16, device=device)
    e2p = torch.empty(0, dtype=torch.int32, device=device)
    out = trtllm_gen_moe_gather_activation(activation_output, e2p, top_k)
    assert out.shape == (0, top_k, 512)
    assert out.dtype == activation_output.dtype
    # A bad launch config surfaces here, not at the call.
    torch.cuda.synchronize()


@pytest.mark.parametrize("num_tokens", [8, 300])
@pytest.mark.parametrize("ep_shard", [False, True])
def test_gather_matches_fused_moe_activation_output(num_tokens, ep_shard):
    """E2E: gather the activation output the fused MoE actually returns.

    ``trtllm_bf16_routed_moe`` with a ``gemm1_lora_delta`` returns the permuted
    post-activation FC1 output together with its permutation map; the op must
    reorder that real buffer exactly like the torch gather does. The sharded case
    routes half the slots outside the local experts, so the map carries the
    routing kernels' own -1 rather than one this file planted.
    """
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm_bf16_routed_moe is only supported on SM100/SM103")
    device = torch.device("cuda")
    torch.manual_seed(stable_seed("e2e", num_tokens, ep_shard))
    enable_pdl = device_support_pdl(device)
    hidden_size = 1024
    intermediate_size = 1024
    num_experts = 16
    top_k = 4
    local_num_experts = num_experts // 2 if ep_shard else num_experts

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    gemm1_weights = torch.randn(
        local_num_experts, 2 * intermediate_size, hidden_size, device=device
    ).to(torch.bfloat16)
    gemm2_weights = torch.randn(
        local_num_experts, hidden_size, intermediate_size, device=device
    ).to(torch.bfloat16)
    block_k = 128
    gemm1_weights = torch.stack(
        [
            convert_to_block_layout(
                shuffle_matrix_a(gemm1_weights[i].view(torch.uint8), 64), block_k
            )
            for i in range(local_num_experts)
        ]
    ).view(torch.bfloat16)
    gemm2_weights = torch.stack(
        [
            convert_to_block_layout(
                shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), 64), block_k
            )
            for i in range(local_num_experts)
        ]
    ).view(torch.bfloat16)

    # Distinct experts per token spanning the GLOBAL expert range, and a LoRA delta
    # so the activation output is returned at all.
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device), dim=-1
    ).to(torch.bfloat16)
    gemm1_lora_delta = (
        torch.randn(
            num_tokens,
            top_k,
            2 * intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01
    )

    _, e2p, activation_output = trtllm_bf16_routed_moe(
        topk_ids=pack_topk_for_routed_moe(topk_ids, topk_weights),
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
        enable_pdl=enable_pdl,
        gemm1_lora_delta=gemm1_lora_delta,
    )

    assert activation_output.shape[1] == intermediate_size
    assert e2p.numel() == num_tokens * top_k
    # Without these the comparison below is satisfied by an all-zero return, and by
    # a source that carries no routing padding for the gather to skip.
    assert torch.count_nonzero(activation_output), (
        "the launcher returned an empty activation output"
    )
    assert activation_output.shape[0] > int(e2p.max()) + 1, (
        "expected a routing-padded row count strictly above max(perm) + 1"
    )

    inactive = e2p.reshape(num_tokens, top_k) < 0
    if ep_shard:
        assert inactive.any(), "the expert-parallel shard produced no remote slots"
        assert not inactive.all(), "the expert-parallel shard produced no local slots"
    else:
        assert not inactive.any()

    out = trtllm_gen_moe_gather_activation(
        activation_output, e2p, top_k, enable_pdl=enable_pdl
    )
    ref = reference_gather(activation_output, e2p, top_k)
    assert out.shape == (num_tokens, top_k, intermediate_size)
    assert out.dtype == activation_output.dtype
    assert torch.count_nonzero(out)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
    # Rows for the routing kernels' own sentinel must be bitwise zero.
    assert (out.view(torch.int16)[inactive] == 0).all()
