"""Tests for the native block-diffusion prefill option."""

import math

import torch

from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache,
)


def get_available_backends(device: torch.device) -> list[str]:
    from flashinfer.utils import is_sm90a_supported

    return ["fa2", "fa3"] if is_sm90a_supported(device) else ["fa2"]


def block_diffusion_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    *,
    q_offset: int = 0,
    kv_offset: int = 0,
    sm_scale: float,
) -> torch.Tensor:
    """Reference the block mask with the existing custom-mask path."""
    q_pos = torch.arange(q.shape[0], device=q.device) + q_offset
    kv_pos = torch.arange(k.shape[0], device=k.device) + kv_offset
    mask = ((q_pos[:, None] // block_size) >= (kv_pos[None, :] // block_size)).to(
        torch.uint8
    )
    return single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask, sm_scale=sm_scale, backend="fa2"
    )


def test_block_diffusion_batch_generator_routing(monkeypatch):
    """Only a fixed block-expanding request takes the dedicated generator."""
    import flashinfer.prefill as prefill_module
    from flashinfer.utils import MaskMode

    class DummySpec:
        def build_and_load(self):
            return "block-diffusion-module"

    def dedicated_generator(*args, **kwargs):
        return DummySpec()

    def shared_generator(*args, **kwargs):
        raise AssertionError("block-expanding mask must not use the shared generator")

    monkeypatch.setattr(
        prefill_module,
        "gen_customize_block_extend_batch_prefill_module",
        dedicated_generator,
    )
    monkeypatch.setattr(
        prefill_module, "gen_customize_batch_prefill_module", shared_generator
    )

    module = prefill_module.get_customize_batch_prefill_module(
        backend="fa2",
        uri="test_block_diffusion_batch_generator_routing",
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
        variant_name="TestBlockDiffusionAttention",
        variant_decl="",
        mask_modes=[MaskMode.BLOCK_EXPANDING.value],
    )

    assert module == "block-diffusion-module"


def test_block_diffusion_single_prefill_matches_reference():
    device = torch.device("cuda:0")
    dtype = torch.float16
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    block_size = 32
    sm_scale = 1.0 / math.sqrt(head_dim)
    tol = 1e-2

    # Includes regular, incremental, partially-visible, and fully-invisible KV.
    configs = [(64, 128, 0, 0), (64, 128, 64, 64), (128, 128, 0, 96), (128, 128, 0, 256)]
    for qo_len, kv_len, q_offset, kv_offset in configs:
        q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        ref = block_diffusion_reference(
            q, k, v, block_size, q_offset=q_offset, kv_offset=kv_offset, sm_scale=sm_scale
        )
        for backend in get_available_backends(device):
            out = single_prefill_with_kv_cache(
                q,
                k,
                v,
                sm_scale=sm_scale,
                block_diffusion=True,
                block_size=block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                backend=backend,
            )
            diff = (out.float() - ref.float()).abs().max().item()
            assert diff < tol, (backend, qo_len, kv_len, q_offset, kv_offset, diff)


def test_block_diffusion_ragged_wrapper_matches_reference():
    device = torch.device("cuda:0")
    dtype = torch.float16
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    block_size = 32
    sm_scale = 1.0 / math.sqrt(head_dim)
    tol = 1e-2

    # Keep an all-invisible request first so FA3 must safely advance to a
    # following normal request in the same launch.
    requests = [(128, 128, 0, 256), (64, 128, 64, 64), (32, 64, 32, 0)]
    qo_ends = torch.tensor([r[0] for r in requests]).cumsum(0).tolist()
    kv_ends = torch.tensor([r[1] for r in requests]).cumsum(0).tolist()
    qo_indptr = torch.tensor([0, *qo_ends], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, *kv_ends], dtype=torch.int32, device=device)
    q_offsets = torch.tensor([r[2] for r in requests], dtype=torch.int32, device=device)
    kv_offsets = torch.tensor([r[3] for r in requests], dtype=torch.int32, device=device)
    q = torch.randn(int(qo_indptr[-1]), num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(int(kv_indptr[-1]), num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(int(kv_indptr[-1]), num_kv_heads, head_dim, dtype=dtype, device=device)

    refs = []
    for i, (_, _, q_offset, kv_offset) in enumerate(requests):
        refs.append(
            block_diffusion_reference(
                q[qo_indptr[i] : qo_indptr[i + 1]],
                k[kv_indptr[i] : kv_indptr[i + 1]],
                v[kv_indptr[i] : kv_indptr[i + 1]],
                block_size,
                q_offset=q_offset,
                kv_offset=kv_offset,
                sm_scale=sm_scale,
            )
        )
    ref = torch.cat(refs)

    for backend in get_available_backends(device):
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend=backend,
            block_diffusion=True,
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
        assert diff < tol, (backend, diff)


def test_block_diffusion_paged_wrapper_matches_reference():
    device = torch.device("cuda:0")
    dtype = torch.float16
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    block_size, page_size, qo_len, kv_len = 32, 16, 64, 128
    q_offset, kv_offset = 64, 64
    sm_scale = 1.0 / math.sqrt(head_dim)
    tol = 1e-2

    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    ref = block_diffusion_reference(
        q, k, v, block_size, q_offset=q_offset, kv_offset=kv_offset, sm_scale=sm_scale
    )
    num_pages = kv_len // page_size
    paged_kv = torch.stack(
        (
            k.reshape(num_pages, page_size, num_kv_heads, head_dim),
            v.reshape(num_pages, page_size, num_kv_heads, head_dim),
        ),
        dim=1,
    )
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)
    q_offsets = torch.tensor([q_offset], dtype=torch.int32, device=device)
    kv_offsets = torch.tensor([kv_offset], dtype=torch.int32, device=device)

    for backend in get_available_backends(device):
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend=backend,
            block_diffusion=True,
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
        assert diff < tol, (backend, diff)
