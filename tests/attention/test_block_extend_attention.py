"""Tests for the native Block Extend prefill option."""

import inspect
import math

import pytest
import torch

from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache,
)


DTYPE_TOLERANCES = {
    torch.float16: 1e-3,
    torch.bfloat16: 1e-2,
}


def tolerance_for(dtype: torch.dtype) -> float:
    return DTYPE_TOLERANCES[dtype]


def get_available_backends(device: torch.device) -> list[str]:
    from flashinfer.utils import is_sm90a_supported

    return ["fa2", "fa3"] if is_sm90a_supported(device) else ["fa2"]


def test_block_extend_mask_mode_is_not_public_plan_api():
    """Block Extend is enabled through wrapper configuration, not a raw mode."""
    for wrapper_cls in (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    ):
        assert "mask_mode" not in inspect.signature(wrapper_cls.plan).parameters


def block_extend_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    *,
    q_offset: int = 0,
    kv_offset: int = 0,
    sm_scale: float,
    backend: str = "fa2",
) -> torch.Tensor:
    """Reference the block mask with the existing custom-mask path.

    The FA3 backend now supports ``custom_mask`` for the single-prefill path
    (SM90+), so an FA3 reference can be produced for cross-checking by
    passing ``backend="fa3"``.
    """
    q_pos = torch.arange(q.shape[0], device=q.device) + q_offset
    kv_pos = torch.arange(k.shape[0], device=k.device) + kv_offset
    mask = ((q_pos[:, None] // block_size) >= (kv_pos[None, :] // block_size)).to(
        torch.uint8
    )
    return single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask, sm_scale=sm_scale, backend=backend
    )


def test_block_extend_batch_generator_routing(monkeypatch):
    """Block Extend uses the dedicated generator directly."""
    import flashinfer.prefill as prefill_module

    class DummySpec:
        def build_and_load(self):
            return "block-extend-module"

    def dedicated_generator(*args, **kwargs):
        return DummySpec()

    monkeypatch.setattr(
        prefill_module,
        "gen_customize_block_extend_batch_prefill_module",
        dedicated_generator,
    )

    module = prefill_module.get_block_extend_batch_prefill_module(
        backend="fa2",
        uri="test_block_extend_batch_generator_routing",
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        dtype_o=torch.float16,
        idtype=torch.int32,
        head_dim_qk=128,
        head_dim_vo=128,
        additional_tensor_names=[],
        additional_tensor_dtypes=[],
        additional_scalar_names=[],
        additional_scalar_dtypes=[],
        variant_name="TestBlockExtendAttention",
        variant_decl="",
    )

    assert module == "block-extend-module"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_block_extend_single_prefill_matches_reference(dtype: torch.dtype):
    device = torch.device("cuda:0")
    tol = tolerance_for(dtype)

    # Cover all supported block sizes, GQA/MQA, both supported head dimensions,
    # incremental offsets, non-aligned boundaries, and a long KV sequence.
    configs = [
        (16, 64, 128, 0, 0, 32, 8, 128),
        (32, 64, 128, 0, 0, 32, 8, 128),
        (64, 64, 128, 0, 0, 32, 8, 128),
        (32, 64, 192, 64, 0, 32, 8, 128),
        (32, 64, 128, 0, 0, 32, 4, 128),
        (32, 64, 128, 0, 0, 32, 1, 128),
        (32, 64, 128, 0, 0, 32, 8, 64),
        (32, 128, 2048, 0, 0, 32, 8, 128),
        (64, 33, 97, 17, 0, 32, 8, 128),
        # Offset the KV sequence as well: only a prefix of it is visible.
        (32, 128, 128, 0, 256, 32, 8, 128),
        # Non-power-of-2 block sizes.
        (24, 64, 128, 0, 0, 32, 8, 128),
        (48, 64, 192, 48, 0, 32, 8, 128),
        (72, 128, 256, 0, 0, 32, 8, 128),
        (100, 50, 200, 10, 30, 8, 4, 128),
    ]
    for (
        block_size,
        qo_len,
        kv_len,
        q_offset,
        kv_offset,
        num_heads,
        num_kv_heads,
        head_dim,
    ) in configs:
        sm_scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        for backend in get_available_backends(device):
            ref = block_extend_reference(
                q,
                k,
                v,
                block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                sm_scale=sm_scale,
                backend=backend,
            )
            out = single_prefill_with_kv_cache(
                q,
                k,
                v,
                sm_scale=sm_scale,
                block_extend=True,
                block_size=block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                backend=backend,
            )
            diff = (out.float() - ref.float()).abs().max().item()
            assert diff < tol, (
                backend,
                dtype,
                block_size,
                qo_len,
                kv_len,
                q_offset,
                kv_offset,
                num_heads,
                num_kv_heads,
                head_dim,
                diff,
                tol,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_block_extend_ragged_wrapper_matches_reference(dtype: torch.dtype):
    device = torch.device("cuda:0")
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    block_size = 32
    sm_scale = 1.0 / math.sqrt(head_dim)
    tol = tolerance_for(dtype)

    # Keep an all-invisible request first so FA3 must safely advance to a
    # following normal request in the same launch. The rest cover a prefix,
    # a non-aligned boundary, and a normal prompt in one heterogeneous batch.
    requests = [
        (128, 128, 0, 256),
        (64, 192, 64, 0),
        (33, 97, 17, 0),
        (64, 128, 0, 0),
    ]
    qo_ends = torch.tensor([r[0] for r in requests]).cumsum(0).tolist()
    kv_ends = torch.tensor([r[1] for r in requests]).cumsum(0).tolist()
    qo_indptr = torch.tensor([0, *qo_ends], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, *kv_ends], dtype=torch.int32, device=device)
    q_offsets = torch.tensor([r[2] for r in requests], dtype=torch.int32, device=device)
    kv_offsets = torch.tensor(
        [r[3] for r in requests], dtype=torch.int32, device=device
    )
    q = torch.randn(int(qo_indptr[-1]), num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(
        int(kv_indptr[-1]), num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v = torch.randn(
        int(kv_indptr[-1]), num_kv_heads, head_dim, dtype=dtype, device=device
    )

    for backend in get_available_backends(device):
        refs = []
        for i, (_, _, q_offset, kv_offset) in enumerate(requests):
            refs.append(
                block_extend_reference(
                    q[qo_indptr[i] : qo_indptr[i + 1]],
                    k[kv_indptr[i] : kv_indptr[i + 1]],
                    v[kv_indptr[i] : kv_indptr[i + 1]],
                    block_size,
                    q_offset=q_offset,
                    kv_offset=kv_offset,
                    sm_scale=sm_scale,
                    backend=backend,
                )
            )
        ref = torch.cat(refs)

        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend=backend,
            block_extend=True,
            block_size=block_size,
        )
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_data_type=dtype,
            sm_scale=sm_scale,
            q_offsets=q_offsets,
            kv_offsets=kv_offsets,
        )
        out = wrapper.run(q, k, v)
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < tol, (backend, dtype, diff, tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_block_extend_paged_wrapper_matches_reference(dtype: torch.dtype):
    device = torch.device("cuda:0")
    tol = tolerance_for(dtype)
    num_heads, page_size = 32, 16
    # Include all block sizes, an incremental prompt, MQA, and an incomplete
    # final page. The K/V reference is extracted from paged_kv itself so the
    # page layout and reference always describe identical inputs.
    configs = [
        (16, 64, 128, 0, 0, 8, 128),
        (32, 64, 128, 0, 0, 8, 128),
        (64, 64, 128, 0, 0, 8, 128),
        (32, 64, 192, 64, 0, 8, 128),
        (32, 64, 128, 0, 0, 1, 128),
        (64, 33, 97, 17, 0, 8, 128),
        (32, 64, 128, 64, 64, 8, 128),
    ]
    for (
        block_size,
        qo_len,
        kv_len,
        q_offset,
        kv_offset,
        num_kv_heads,
        head_dim,
    ) in configs:
        sm_scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
        num_pages = (kv_len + page_size - 1) // page_size
        paged_kv = torch.randn(
            num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        k = paged_kv[:, 0].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        v = paged_kv[:, 1].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        last_page_len = kv_len - (num_pages - 1) * page_size
        paged_kv_last_page_len = torch.tensor(
            [last_page_len], dtype=torch.int32, device=device
        )
        q_offsets = torch.tensor([q_offset], dtype=torch.int32, device=device)
        kv_offsets = torch.tensor([kv_offset], dtype=torch.int32, device=device)

        for backend in get_available_backends(device):
            ref = block_extend_reference(
                q,
                k,
                v,
                block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                sm_scale=sm_scale,
                backend=backend,
            )
            workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                kv_layout="NHD",
                backend=backend,
                block_extend=True,
                block_size=block_size,
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets,
                kv_offsets=kv_offsets,
            )
            out = wrapper.run(q, paged_kv)
            diff = (out.float() - ref.float()).abs().max().item()
            assert diff < tol, (
                backend,
                dtype,
                block_size,
                qo_len,
                kv_len,
                q_offset,
                kv_offset,
                num_kv_heads,
                head_dim,
                diff,
                tol,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_block_extend_paged_zero_visible_then_normal(dtype: torch.dtype):
    """Paged FA3 must advance past zero-visible data and preserve page bases."""
    device = torch.device("cuda:0")
    block_size = 32
    page_size = 16
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    qo_len = 64
    kv_lens = [64, 97]
    batch_size = 2
    pages_per_request = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    num_pages = sum(pages_per_request)
    sm_scale = 1.0 / math.sqrt(head_dim)
    tol = tolerance_for(dtype)

    q = torch.randn(
        batch_size * qo_len, num_heads, head_dim, dtype=dtype, device=device
    )
    paged_kv = torch.randn(
        num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    k = paged_kv[:, 0].reshape(-1, num_kv_heads, head_dim)
    v = paged_kv[:, 1].reshape(-1, num_kv_heads, head_dim)

    # Request 0 sees no KV (kv_offset=256); request 1 is normal and has a
    # different number of pages. This catches accidental compaction of the
    # page indptr, which would shift request 1's physical page-table base.
    q_offset_values = [0, 0]
    kv_offset_values = [256, 0]
    q_offsets = torch.tensor(q_offset_values, dtype=torch.int32, device=device)
    kv_offsets = torch.tensor(kv_offset_values, dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, qo_len, 2 * qo_len], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor(
        [0, pages_per_request[0], num_pages], dtype=torch.int32, device=device
    )
    paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor(
        [page_size, kv_lens[1] - (pages_per_request[1] - 1) * page_size],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    seq_lens_before_plan = seq_lens.clone()

    for backend in get_available_backends(device):
        refs = []
        kv_starts = [0, pages_per_request[0] * page_size]
        for i in range(batch_size):
            start_q, end_q = i * qo_len, (i + 1) * qo_len
            start_kv = kv_starts[i]
            end_kv = start_kv + kv_lens[i]
            refs.append(
                block_extend_reference(
                    q[start_q:end_q],
                    k[start_kv:end_kv],
                    v[start_kv:end_kv],
                    block_size,
                    q_offset=q_offset_values[i],
                    kv_offset=kv_offset_values[i],
                    sm_scale=sm_scale,
                    backend=backend,
                )
            )
        ref = torch.cat(refs)

        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend=backend,
            block_extend=True,
            block_size=block_size,
        )
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            q_data_type=dtype,
            sm_scale=sm_scale,
            q_offsets=q_offsets,
            kv_offsets=kv_offsets,
            seq_lens=seq_lens,
        )
        assert torch.equal(seq_lens, seq_lens_before_plan)
        out = wrapper.run(q, paged_kv)
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < tol, (backend, dtype, diff, tol)
        del wrapper
        torch.cuda.empty_cache()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_block_extend_non_power_of_two_single(dtype: torch.dtype):
    """Non-power-of-2 block sizes should produce correct results for single prefill.

    Since we removed the power-of-2 restriction, the kernel must handle arbitrary
    positive block sizes correctly.  This tests several non-power-of-2 values
    across both FA2 and FA3 backends.
    """
    device = torch.device("cuda:0")
    sm_scale = 1.0 / math.sqrt(128)
    tol = tolerance_for(dtype)
    num_heads, num_kv_heads, head_dim = 32, 8, 128

    # (block_size, qo_len, kv_len, q_offset, kv_offset)
    configs = [
        (24, 128, 128, 0, 0),
        (48, 200, 300, 50, 0),
        (72, 256, 256, 0, 100),
        (100, 50, 200, 10, 30),
        (127, 127, 127, 0, 0),
        (191, 200, 512, 30, 60),
    ]
    for block_size, qo_len, kv_len, q_offset, kv_offset in configs:
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        for backend in get_available_backends(device):
            ref = block_extend_reference(
                q,
                k,
                v,
                block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                sm_scale=sm_scale,
                backend=backend,
            )
            out = single_prefill_with_kv_cache(
                q,
                k,
                v,
                sm_scale=sm_scale,
                block_extend=True,
                block_size=block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                backend=backend,
            )
            diff = (out.float() - ref.float()).abs().max().item()
            assert diff < tol, (
                f"backend={backend} block_size={block_size} "
                f"qo_len={qo_len} kv_len={kv_len} "
                f"q_offset={q_offset} kv_offset={kv_offset} "
                f"diff={diff:.6f} tol={tol:.6f}"
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_block_extend_non_power_of_two_batch(dtype: torch.dtype):
    """Non-power-of-2 block sizes should produce correct results for batch prefill."""
    device = torch.device("cuda:0")
    sm_scale = 1.0 / math.sqrt(128)
    tol = tolerance_for(dtype)
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    page_size = 16

    configs = [(24, 128, 128, 0, 0), (48, 200, 300, 50, 0), (100, 50, 200, 10, 30)]
    for block_size, qo_len, kv_len, q_offset, kv_offset in configs:
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)

        # --- Ragged path ---
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        for backend in get_available_backends(device):
            ref = block_extend_reference(
                q,
                k,
                v,
                block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                sm_scale=sm_scale,
                backend=backend,
            )
            workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
            q_offsets = torch.tensor([q_offset], dtype=torch.int32, device=device)
            kv_offsets = torch.tensor([kv_offset], dtype=torch.int32, device=device)

            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                kv_layout="NHD",
                backend=backend,
                block_extend=True,
                block_size=block_size,
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets,
                kv_offsets=kv_offsets,
            )
            out = wrapper.run(q, k, v)
            diff = (out.float() - ref.float()).abs().max().item()
            assert diff < tol, (
                f"ragged backend={backend} block_size={block_size} diff={diff:.6f}"
            )

        # --- Paged path ---
        num_pages = (kv_len + page_size - 1) // page_size
        paged_kv = torch.randn(
            num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        # Copy K/V into the paged buffer.
        paged_kv[: kv_len // page_size, 0, :, :, :] = k[
            : page_size * (kv_len // page_size)
        ].view(kv_len // page_size, page_size, num_kv_heads, head_dim)
        paged_kv[: kv_len // page_size, 1, :, :, :] = v[
            : page_size * (kv_len // page_size)
        ].view(kv_len // page_size, page_size, num_kv_heads, head_dim)
        if kv_len % page_size:
            paged_kv[kv_len // page_size, 0, : kv_len % page_size, :, :] = k[
                page_size * (kv_len // page_size) :
            ]
            paged_kv[kv_len // page_size, 1, : kv_len % page_size, :, :] = v[
                page_size * (kv_len // page_size) :
            ]

        for backend in get_available_backends(device):
            ref = block_extend_reference(
                q,
                k,
                v,
                block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                sm_scale=sm_scale,
                backend=backend,
            )
            workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            paged_kv_indptr = torch.tensor(
                [0, num_pages], dtype=torch.int32, device=device
            )
            paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
            last_page_len_val = kv_len - (num_pages - 1) * page_size
            paged_kv_last_page_len = torch.tensor(
                [last_page_len_val], dtype=torch.int32, device=device
            )
            q_offsets_t = torch.tensor([q_offset], dtype=torch.int32, device=device)
            kv_offsets_t = torch.tensor([kv_offset], dtype=torch.int32, device=device)

            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                kv_layout="NHD",
                backend=backend,
                block_extend=True,
                block_size=block_size,
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                q_data_type=dtype,
                sm_scale=sm_scale,
                q_offsets=q_offsets_t,
                kv_offsets=kv_offsets_t,
            )
            out = wrapper.run(q, paged_kv)
            diff = (out.float() - ref.float()).abs().max().item()
            assert diff < tol, (
                f"paged backend={backend} block_size={block_size} diff={diff:.6f}"
            )
