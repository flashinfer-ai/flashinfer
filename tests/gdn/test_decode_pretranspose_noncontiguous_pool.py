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

from __future__ import annotations

import random

import pytest
import torch

from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm90_or_later() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


@pytest.mark.parametrize("page_gap", [2, 3])
def test_decode_pretranspose_pool_noncontiguous_state(page_gap: int) -> None:
    _skip_if_not_sm90_or_later()

    seed = 20260309 + page_gap
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    B, T, H, HV, K, V = 8, 1, 16, 32, 128, 128
    pool_size = B * 3
    device = torch.device("cuda")
    qkv_dtype = torch.bfloat16

    with device:
        q = torch.randn(B, T, H, K, dtype=qkv_dtype)
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, dtype=qkv_dtype), p=2.0, dim=-1
        )
        v = torch.randn(B, T, HV, V, dtype=qkv_dtype)

        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        a = torch.randn(B, T, HV, dtype=qkv_dtype) * 0.1
        b = torch.randn(B, T, HV, dtype=qkv_dtype)

        # Build a non-contiguous [pool, HV, V, K] view with page stride on dim-0.
        pool_storage = torch.randn(pool_size, page_gap, HV, V, K, dtype=torch.float32)
        pool_source = pool_storage[:, page_gap - 1]
        assert not pool_source.is_contiguous()

        indices = (torch.arange(B, dtype=torch.int32, device=device) * 2) % pool_size

    # Pool path under test: initial_state is a non-contiguous view.
    pool_under_test_storage = pool_storage.clone()
    pool_under_test = pool_under_test_storage[:, page_gap - 1]
    out_pool, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
        initial_state=pool_under_test,
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    # Gather + direct-state reference path.
    gathered_state = pool_source[indices].clone()
    out_direct, updated_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=gathered_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
    )
    torch.cuda.synchronize()

    atol = 5e-3
    rtol = 5e-3
    torch.testing.assert_close(out_pool, out_direct, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        pool_under_test[indices], updated_state, atol=atol, rtol=rtol
    )

    untouched = torch.ones(pool_size, dtype=torch.bool, device=device)
    untouched[indices] = False
    torch.testing.assert_close(
        pool_under_test[untouched], pool_source[untouched], atol=0.0, rtol=0.0
    )


# ============================================================================
# Regression test: int32 element-offset overflow when
# ``pool_idx * initial_state.stride(0) >= 2**31`` corrupted memory and crashed
# the kernel with ``CUDA error: an illegal memory access was encountered``.
#
# Original reproducer (B=4, fp32 SSM state, slot stride = 540672 elements,
# the layout vLLM hands to FlashInfer for Qwen3.5-class GDN models):
#
#   | pool_idx | element offset (= idx * 540672) | result (pre-fix) |
#   |----------|---------------------------------|------------------|
#   | 3970     | 2_146_467_840  (<  2**31)       | OK               |
#   | 3972     | 2_147_549_184  (>= 2**31)       | CRASH            |
#   | 3973     | 2_148_089_856                   | CRASH            |
#   | 8191     | 4_429_645_152                   | CRASH            |
#
# The threshold is exactly ``ceil(2**31 / 540672) = 3972``.  The fix widens
# the pool indices to Int64 inside the CuTe-DSL kernel before they multiply
# stride[0], so this test exercises a single high pool slot that would have
# wrapped under the original Int32 arithmetic.  Memory footprint is dominated
# by the pool itself (~8.6 GB for fp32 with the padded slot stride), so we
# skip when free VRAM is insufficient.
# ============================================================================


# Padded slot stride used by vLLM for Qwen3.5-class models: 32 HV rows of
# state plus one extra HV-row's worth of padding for an adjacent conv state
# packed into the same paged-KV slot.  ``540672 = (32 + 1) * 128 * 128``.
_PADDED_SLOT_STRIDE = 540672
_HV_WITH_PAD = _PADDED_SLOT_STRIDE // (128 * 128)  # 33

# First pool index whose element offset in the padded layout exceeds INT32_MAX.
_OVERFLOW_POOL_IDX = (2**31 + _PADDED_SLOT_STRIDE - 1) // _PADDED_SLOT_STRIDE  # 3973


def _skip_if_low_vram(required_bytes: int) -> None:
    # Release any tensors cached by PyTorch's allocator from previous tests so
    # ``mem_get_info`` reflects the true driver-level free memory, not the
    # pool-reserved memory.  Without this, large multi-test runs can spuriously
    # skip later tests when earlier tests left big allocations in the cache.
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < int(required_bytes * 1.2):
        pytest.skip(
            f"Requires ~{required_bytes / 1024**3:.1f}GB free VRAM, "
            f"only {free / 1024**3:.1f}GB available"
        )


@pytest.mark.parametrize("explicit_pool_idx", [_OVERFLOW_POOL_IDX, 8191])
def test_decode_pretranspose_pool_int64_offset(explicit_pool_idx: int) -> None:
    """Pool indices whose element offset wraps int32 must not crash and must
    produce the same output as the gather + direct-state reference path.

    Reproduces the int32 element-offset overflow originally observed when
    integrating ``gated_delta_rule_decode_pretranspose`` into vLLM's GDN
    decode path.  Uses B=1 with a single high pool index to keep the memory
    footprint as small as possible (~8.6 GB for the padded fp32 pool).
    """
    _skip_if_not_sm90_or_later()

    B, T, H, HV, K, V = 1, 1, 16, 32, 128, 128
    pool_size = explicit_pool_idx + 1

    # Padded fp32 pool storage (matches vLLM's paged-KV slot stride).
    required_bytes = pool_size * _HV_WITH_PAD * V * K * 4  # fp32
    _skip_if_low_vram(required_bytes)

    seed = 20260505 + explicit_pool_idx
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda")
    qkv_dtype = torch.bfloat16

    with device:
        # Allocate the padded pool, then take a [pool, HV, V, K] view that
        # shares the same storage so ``stride(0) == _PADDED_SLOT_STRIDE`` —
        # exactly the layout vLLM hands to FlashInfer.  The view must NOT be
        # cloned (clone would compact stride(0) back down to HV*V*K and hide
        # the bug), so we always re-slice the storage we want to use.
        pool_storage = torch.zeros(pool_size, _HV_WITH_PAD, V, K, dtype=torch.float32)
        # Only fill the slot we'll actually read so the rest of the pool is
        # left as zeros — keeps host-side initialization cheap and avoids a
        # multi-GB ``randn`` call.
        pool_storage[explicit_pool_idx, :HV, :, :].copy_(
            torch.randn(HV, V, K, dtype=torch.float32) * 0.1
        )

        q = torch.randn(B, T, H, K, dtype=qkv_dtype) * 0.05
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, dtype=qkv_dtype), p=2.0, dim=-1
        )
        v = torch.randn(B, T, HV, V, dtype=qkv_dtype) * 0.05

        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        a = torch.randn(B, T, HV, dtype=qkv_dtype) * 0.1
        b = torch.randn(B, T, HV, dtype=qkv_dtype)

        indices = torch.full((B,), explicit_pool_idx, dtype=torch.int32, device=device)

    # Snapshot the initial slot for the reference path; ``pool_under_test``
    # will be mutated in place by the kernel.
    initial_slot = pool_storage[explicit_pool_idx, :HV, :, :].clone()

    # Take the strided [pool, HV, V, K] view of the underlying storage that we
    # actually pass to the kernel.  This view's stride(0) = _PADDED_SLOT_STRIDE
    # is what triggers the int32 element-offset overflow in the kernel.
    pool_under_test = pool_storage[:, :HV, :, :]
    assert pool_under_test.shape == (pool_size, HV, V, K)
    assert pool_under_test.stride() == (_PADDED_SLOT_STRIDE, V * K, K, 1), (
        pool_under_test.stride()
    )

    # Pool path under test — would crash without the Int64 widening fix.
    out_pool, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
        initial_state=pool_under_test,
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    # Direct-state reference path: gather the single slot into a contiguous
    # [B, HV, V, K] tensor and run the non-pool kernel.
    gathered_state = initial_slot.unsqueeze(0).clone()
    out_direct, updated_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=gathered_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
    )
    torch.cuda.synchronize()

    atol = 5e-3
    rtol = 5e-3
    torch.testing.assert_close(out_pool, out_direct, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        pool_under_test[explicit_pool_idx],
        updated_state[0],
        atol=atol,
        rtol=rtol,
    )


# ============================================================================
# Regression test: int32 element-offset overflow in the BF16 state fast path.
# The pool+indices API at K=V=128 bfloat16 dispatches into one of two CuTe-DSL
# kernels in ``flashinfer/gdn_kernels/gdn_decode_bf16_state.py``:
#
#   * ``gdn_decode_bf16state_mtp_ilp4_kernel`` (small-batch fallback,
#     reached when ``B * HV <= 128`` — what this test exercises with B=1)
#   * ``gdn_wide_vec_kernel``                  (production fast path, reached
#     when ``B * HV >= 512``)
#
# Both kernels share the same overflow site: the API reshapes the pool to
# ``[pool_size * HV, V, K]`` internally, so after the reshape:
#   * ``stride[0] = V * K = 16_384`` BF16 elements (32 KB / row)
#   * the kernel computes ``flat_state_idx = cache_idx * HV + i_hv``
#     and indexes ``h0_source[(flat_state_idx, ...)]`` via ``cute.local_tile``
#
# Per-slot element-offset arithmetic ``(cache_idx * HV + i_hv) * V * K``
# crosses INT32_MAX at:
#   cache_idx * HV * V * K >= 2**31
#   cache_idx >= ceil(2**31 / (HV * V * K)) = ceil(2**31 / 524_288) = 4096
#                                              (HV=32, V=K=128)
#
# Empirical bisect (B=1, contiguous bf16 pool):
#
#   | cache_idx | offset (= idx * 524288) | result (pre-fix) |
#   |-----------|-------------------------|------------------|
#   | 4095      | 2_147_287_040  (< 2**31) | OK              |
#   | 4096      | 2_147_483_648  (= 2**31) | CRASH           |
#
# The fix widens ``cache_idx`` (and ``write_cache_idx``) to Int64 inside both
# BF16 kernels, so the propagated ``flat_state_idx`` / ``flat_write_state_idx``
# stay Int64 and the ``cute.local_tile`` offset multiplications cannot wrap.
# Memory footprint of the contiguous bf16 pool is ~4.3 GB at the threshold.
# ============================================================================


# Tight (no-padding) slot stride for the bf16 pool: ``HV * V * K = 524288``
# bf16 elements = 1 MB / slot.  The bf16 fast path only accepts contiguous
# pools, so ``stride[0]`` is fixed at this value (no padded variant exists).
_BF16_TIGHT_SLOT_STRIDE = 32 * 128 * 128

# First pool index whose element offset exceeds INT32_MAX in the bf16 layout.
_BF16_OVERFLOW_POOL_IDX = (
    2**31 + _BF16_TIGHT_SLOT_STRIDE - 1
) // _BF16_TIGHT_SLOT_STRIDE
assert _BF16_OVERFLOW_POOL_IDX == 4096


@pytest.mark.parametrize(
    "explicit_pool_idx", [_BF16_OVERFLOW_POOL_IDX, _BF16_OVERFLOW_POOL_IDX + 100]
)
def test_decode_pretranspose_pool_int64_offset_bf16(explicit_pool_idx: int) -> None:
    """BF16 state pool indices whose element offset wraps int32 must not crash
    and must produce the same output as the gather + direct-state reference
    path that runs the non-pool BF16 kernel.

    Reproduces the same int32 element-offset overflow as the fp32 pretranspose
    path, but for the BF16 fast path.  At B=1, HV=32 the pool+indices dispatch
    routes through ``gdn_decode_bf16state_mtp_ilp4_kernel`` (the small-batch
    fallback in ``flashinfer/gdn_kernels/gdn_decode_bf16_state.py``); the
    ``gdn_wide_vec_kernel`` production path shares the identical overflow
    site (``cache_idx * HV * V * K`` in Int32) and the identical Int64 fix.
    Uses B=1 with a single high pool index to keep the memory footprint as
    small as possible (~4.3 GB for the contiguous bf16 pool at the threshold).
    """
    _skip_if_not_sm90_or_later()

    B, T, H, HV, K, V = 1, 1, 16, 32, 128, 128
    pool_size = explicit_pool_idx + 1

    # Contiguous bf16 pool storage (the only layout the bf16 path accepts —
    # the API's internal ``reshape(pool_size * HV, V, K)`` requires
    # ``stride[0] == HV * V * K``).
    required_bytes = pool_size * HV * V * K * 2  # bf16
    _skip_if_low_vram(required_bytes)

    seed = 20260505 + explicit_pool_idx
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device("cuda")
    qkv_dtype = torch.bfloat16

    with device:
        # Allocate a contiguous bf16 pool.  Only fill the slot we'll actually
        # read so the rest of the pool stays at zero — keeps host-side init
        # cheap and avoids a multi-GB ``randn`` call.
        pool_storage = torch.zeros(pool_size, HV, V, K, dtype=torch.bfloat16)
        assert pool_storage.is_contiguous()
        assert pool_storage.stride() == (_BF16_TIGHT_SLOT_STRIDE, V * K, K, 1), (
            pool_storage.stride()
        )
        pool_storage[explicit_pool_idx].copy_(
            torch.randn(HV, V, K, dtype=qkv_dtype) * 0.1
        )

        q = torch.randn(B, T, H, K, dtype=qkv_dtype) * 0.05
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, dtype=qkv_dtype), p=2.0, dim=-1
        )
        v = torch.randn(B, T, HV, V, dtype=qkv_dtype) * 0.05

        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        a = torch.randn(B, T, HV, dtype=qkv_dtype) * 0.1
        b = torch.randn(B, T, HV, dtype=qkv_dtype)

        indices = torch.full((B,), explicit_pool_idx, dtype=torch.int32, device=device)

    # Snapshot the initial slot for the reference path; ``pool_storage`` will
    # be mutated in place by the kernel.
    initial_slot = pool_storage[explicit_pool_idx].clone()

    # Pool path under test — would crash without the Int64 widening fix in
    # the BF16 kernels.  This routes through
    # ``_gated_delta_rule_bf16_state_mtp`` (pool+indices forces the MTP
    # backend even at T=1), and at B=1 HV=32 the dispatcher selects
    # ``gdn_decode_bf16state_mtp_ilp4_kernel`` — see the module-level
    # comment above for the shared-overflow / shared-fix discussion.
    out_pool, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
        initial_state=pool_storage,
        initial_state_indices=indices,
    )
    torch.cuda.synchronize()

    # Direct-state reference path: feed the snapshotted slot as the per-batch
    # ``state`` (no pool).  This routes through the non-pool T=1 BF16 path
    # and never touches the high-index pool slot.
    gathered_state = initial_slot.unsqueeze(0).clone()
    out_direct, updated_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=gathered_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=1.0,
        use_qk_l2norm=True,
    )
    torch.cuda.synchronize()

    atol = 5e-3
    rtol = 5e-3
    torch.testing.assert_close(out_pool, out_direct, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        pool_storage[explicit_pool_idx],
        updated_state[0],
        atol=atol,
        rtol=rtol,
    )
