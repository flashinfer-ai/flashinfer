"""
Copyright (c) 2024 by FlashInfer team.

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

import math

import numpy as np
import pytest
import torch

try:
    import scipy as sp

    HAVE_SCIPY = True
except ImportError:
    sp = None
    HAVE_SCIPY = False

from tests.test_helpers.jit_utils import (
    gen_decode_attention_modules,
    gen_prefill_attention_modules,
)

import flashinfer
import flashinfer.sparse as sparse_module
from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.utils import has_flashinfer_jit_cache, is_sm100a_supported


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


def bsr_attention_ref(
    q,
    k,
    v,
    indptr,
    indices,
    mask_data,
):
    """Dense reference for block-sparse attention, built from the BSR mask."""
    M = q.shape[0]
    N = k.shape[0]
    if HAVE_SCIPY:
        bsr = sp.sparse.bsr_matrix(
            (mask_data.cpu().numpy(), indices.cpu().numpy(), indptr.cpu().numpy()),
            shape=(M, N),
        )
        dense_mask = torch.tensor(bsr.toarray(), dtype=bool, device=q.device)
    else:
        dense_mask = _bsr_to_dense_torch(indptr, indices, mask_data, M, N).to(q.device)
    o = flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, custom_mask=dense_mask)
    return o


def set_seed(seed: int = 42):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _bsr_to_dense_torch(
    indptr: "torch.Tensor",
    indices: "torch.Tensor",
    mask_data: "torch.Tensor",
    M: int,
    N: int,
) -> "torch.Tensor":
    """Convert BSR format to dense boolean mask without scipy."""
    R = mask_data.shape[1]
    C = mask_data.shape[2]
    device = mask_data.device
    dense = torch.zeros(M, N, dtype=torch.bool, device=device)
    n_block_rows = indptr.numel() - 1
    for br in range(n_block_rows):
        for ki in range(indptr[br].item(), indptr[br + 1].item()):
            bc = indices[ki].item()
            dense[br * R : (br + 1) * R, bc * C : (bc + 1) * C] = mask_data[ki]
    return dense


# Shared with test_block_sparse_cutile.py so both matrices use the same oracle.
def _run_block_sparse_attention_case(
    backend, R, C, M, N, num_qo_heads, num_kv_heads, head_dim, mask_inside_block
):
    """Run one block-sparse backend case against the dense reference."""
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    if M % R != 0 or N % C != 0:
        pytest.skip("BSR test dimensions require M % R == 0 and N % C == 0")

    if backend == "vsa_blackwell":
        if not is_sm100a_supported(torch.device(0)):
            pytest.skip("vsa_blackwell requires sm100a (Blackwell GPU)")
        if R != 128 or C != 128:
            pytest.skip("vsa_blackwell requires R == C == 128")
        if M % 128 != 0 or N % 128 != 0:
            pytest.skip("vsa_blackwell requires M and N divisible by 128")
        if head_dim not in (64, 96, 128):
            pytest.skip("vsa_blackwell requires head_dim in {64, 96, 128}")
        if mask_inside_block:
            pytest.skip(
                "vsa_blackwell does not support per-element block masks (mask_inside_block=True)"
            )

    if backend == "cutile":
        if not is_cuda_tile_available():
            pytest.skip("cuda-tile / tileiras compiler not available")
        # cuTile block-sparse maps each block-row onto a paged prefill batch with
        # page_size == C; it expresses sparsity at block granularity only.
        if mask_inside_block:
            pytest.skip(
                "cuTile block-sparse does not support per-element intra-block masks."
            )
        if C < 16:
            # The BSR column-block size C maps to the paged-KV page_size; the
            # prefill autotune (_get_prefill_autotune_configs) only yields configs
            # with BLOCK_N <= page_size, and the smallest BLOCK_N is 16, so C < 16
            # leaves an empty search space.
            pytest.skip("cuTile block-sparse requires C >= 16 (min prefill BLOCK_N).")

    set_seed(33)
    rng = np.random.default_rng(33)

    MB = M // R
    NB = N // C
    if HAVE_SCIPY:
        S = sp.sparse.random(MB, NB, density=0.25, random_state=rng).tocsr()
        indptr = torch.from_numpy(S.indptr).to(0)
        indices = torch.from_numpy(S.indices).to(0)
        nnz = S.nnz
    else:
        # Generate random sparse CSR pattern without scipy
        sp_mask = torch.rand(MB, NB) < 0.25
        indptr_list = [0]
        indices_list = []
        for br in range(MB):
            cols = sp_mask[br].nonzero(as_tuple=True)[0].tolist()
            indices_list.extend(cols)
            indptr_list.append(len(indices_list))
        indptr = torch.tensor(indptr_list, dtype=torch.int32, device=0)
        indices = torch.tensor(indices_list, dtype=torch.int32, device=0)
        nnz = len(indices_list)
    if mask_inside_block:
        data_mask = (torch.rand((nnz, R, C)) > 0.5).to(0)
    else:
        data_mask = torch.full((nnz, R, C), True, dtype=bool, device=0)
    q = torch.randn((M, num_qo_heads, head_dim), dtype=torch.float16, device=0)
    k = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device=0)
    v = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device=0)

    o_ref = bsr_attention_ref(q, k, v, indptr, indices, data_mask)
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=0)
    sparse_attention_wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(
        workspace_buffer, backend=backend
    )

    sparse_attention_wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        mask=data_mask if mask_inside_block else None,
    )

    o = sparse_attention_wrapper.run(q, k, v)
    torch.testing.assert_close(o_ref, o, atol=1e-2, rtol=1e-3)

    # test with pre-allocated output
    o_buffer = torch.empty_like(o)
    sparse_attention_wrapper.run(q, k, v, out=o_buffer)
    torch.testing.assert_close(o_ref, o_buffer, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("backend", ["auto", "vsa_blackwell"])
@pytest.mark.parametrize("R", [1, 4, 16, 128])
@pytest.mark.parametrize("C", [1, 4, 16, 128])
@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("mask_inside_block", [True, False])
def test_block_sparse_attention(
    backend, R, C, M, N, num_qo_heads, num_kv_heads, head_dim, mask_inside_block
):
    """Block-sparse attention must match the dense reference for each backend."""
    _run_block_sparse_attention_case(
        backend,
        R,
        C,
        M,
        N,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        mask_inside_block,
    )


def _ref_attention(
    q: torch.Tensor,  # [gqa_group_size, qo_len, head_dim]
    k: torch.Tensor,  # [1, kv_len, head_dim]
    v: torch.Tensor,  # [1, kv_len, head_dim]
    block_mask_map: torch.Tensor,  # [MB, NB]
    block_row_sz: torch.Tensor,  # [MB]
    block_col_sz: torch.Tensor,  # [NB]
) -> torch.Tensor:
    # convert block mask map to element mask
    def _block_mask_to_element_mask(
        block_mask_map: torch.Tensor,  # [MB, NB] – bool
        block_row_sz: torch.Tensor,  # [MB]     – int (rows per block-row)
        block_col_sz: torch.Tensor,  # [NB]     – int (cols per block-col)
    ) -> torch.Tensor:
        block_row_sz = block_row_sz.to(block_mask_map.device, dtype=torch.long)
        block_col_sz = block_col_sz.to(block_mask_map.device, dtype=torch.long)
        expanded_rows = torch.repeat_interleave(block_mask_map, block_row_sz, dim=0)
        element_mask = torch.repeat_interleave(expanded_rows, block_col_sz, dim=1)

        return element_mask

    dense_mask = _block_mask_to_element_mask(
        block_mask_map, block_row_sz, block_col_sz
    ).to(dtype=torch.bool, device=q.device)

    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()
    v = v.transpose(0, 1).contiguous()
    o = flashinfer.prefill.single_prefill_with_kv_cache(
        q, k, v, custom_mask=dense_mask
    )  # [qo_len, gqa_group_size, head_dim]
    o = o.transpose(0, 1).contiguous()

    return o


@pytest.mark.parametrize("num_qo_heads", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [256, 4096, 8192])
@pytest.mark.parametrize("num_blocks_row", [10, 20])
@pytest.mark.parametrize("num_blocks_col", [50, 100])
@pytest.mark.parametrize("block_density", [0.2, 0.7, 0.9])
def test_variable_block_sparse_attention_wrapper(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    num_blocks_row: int,
    num_blocks_col: int,
    block_density: float,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    if seq_len // num_blocks_row < 1:
        pytest.skip("seq_len must be greater than num_blocks_row")
    if seq_len // num_blocks_col < 1:
        pytest.skip("seq_len must be greater than num_blocks_col")

    set_seed(330)

    def random_partition_batch(
        seq_len: int,
        num_blocks: int,
        bsz: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        assert seq_len >= num_blocks
        sizes = torch.empty((bsz, num_blocks), dtype=dtype, device=device)
        for i in range(bsz):
            cut_pts = torch.randperm(seq_len - 1, device=device)[: num_blocks - 1] + 1
            cut_pts, _ = torch.sort(cut_pts)
            row_sizes = torch.diff(
                torch.cat(
                    (
                        torch.tensor([0], device=device),
                        cut_pts,
                        torch.tensor([seq_len], device=device),
                    )
                )
            )
            sizes[i] = row_sizes

        assert sizes.min() >= 1
        assert sizes.max() <= seq_len
        assert torch.all(sizes.sum(dim=-1) == seq_len)

        return sizes.to(device=device)

    def _test_variable_block_sparse_attention(
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        # qkv: HND
        qo_len = block_row_sz.sum(dim=1)[0].item()
        kv_len = block_col_sz.sum(dim=1)[0].item()
        assert torch.all(block_col_sz.sum(dim=1) == block_col_sz.sum(dim=1)[0])
        assert torch.all(block_row_sz.sum(dim=1) == block_row_sz.sum(dim=1)[0])

        q = torch.randn(num_qo_heads, qo_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(num_kv_heads, kv_len, head_dim, device=device, dtype=dtype)

        float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=device)
        wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
            float_workspace_buffer, backend="auto"
        )

        wrapper.plan(
            block_mask_map=block_mask_map,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=dtype,
        )

        o: torch.Tensor = wrapper.run(q, k, v)  # [num_qo_heads, qo_len, head_dim]
        o = o.reshape(num_kv_heads, -1, *o.shape[-2:])
        q = q.reshape(num_kv_heads, -1, *q.shape[-2:])
        for kv_head_idx in range(num_kv_heads):
            o_ref = _ref_attention(
                q[kv_head_idx],
                k[kv_head_idx : kv_head_idx + 1, :, :],
                v[kv_head_idx : kv_head_idx + 1, :, :],
                block_mask_map[kv_head_idx],
                block_row_sz[kv_head_idx],
                block_col_sz[kv_head_idx],
            )
            torch.testing.assert_close(o[kv_head_idx], o_ref, atol=1e-2, rtol=1e-2)

    block_row_sz = random_partition_batch(
        seq_len, num_blocks_row, num_kv_heads, device="cuda:0"
    )
    block_col_sz = random_partition_batch(
        seq_len, num_blocks_col, num_kv_heads, device="cuda:0"
    )
    block_mask_map = (
        torch.rand(num_kv_heads, num_blocks_row, num_blocks_col) > block_density
    ).to(device="cuda:0")

    _test_variable_block_sparse_attention(
        num_qo_heads,
        num_kv_heads,
        head_dim,
        block_mask_map,
        block_row_sz,
        block_col_sz,
    )


def _paged_reference(q, k_cache, v_cache, route, page_size, layout, num_kv_heads):
    """Gather the routed KV entries and run dense attention on them."""
    pages = k_cache.shape[0]
    if layout == "HND":
        flat_k = k_cache.permute(0, 2, 1, 3).reshape(
            pages * page_size, num_kv_heads, -1
        )
        flat_v = v_cache.permute(0, 2, 1, 3).reshape(
            pages * page_size, num_kv_heads, -1
        )
    else:
        flat_k = k_cache.reshape(pages * page_size, num_kv_heads, -1)
        flat_v = v_cache.reshape(pages * page_size, num_kv_heads, -1)
    rows, width = route.shape
    heads = q.shape[1]
    group = heads // num_kv_heads
    k = flat_k[route.reshape(-1)].reshape(rows, width, num_kv_heads, -1)
    v = flat_v[route.reshape(-1)].reshape(rows, width, num_kv_heads, -1)
    k = k.repeat_interleave(group, dim=2).float()
    v = v.repeat_interleave(group, dim=2).float()
    logits = torch.einsum("rhd,rwhd->rhw", q.float(), k) / math.sqrt(q.shape[-1])
    return torch.einsum("rhw,rwhd->rhd", torch.softmax(logits, dim=-1), v).to(q.dtype)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("page_size", [1, 16])
def test_block_sparse_paged_route(num_kv_heads, layout, page_size):
    """A route of physical slots over a cache that still stores whole pages.

    The wrapper must divide each index back into (page, entry); reading it as
    a page id would address a different token, which the reference catches.
    """
    torch.manual_seed(42)
    num_qo_heads, head_dim, pages, rows, width = 8, 128, 12, 6, 9
    entries = pages * page_size
    device = "cuda:0"
    shape = (
        (pages, page_size, num_kv_heads, head_dim)
        if layout == "NHD"
        else (pages, num_kv_heads, page_size, head_dim)
    )
    k_cache = torch.randn(*shape, dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(rows, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    # scatter the route so consecutive entries land in different pages, and
    # repeat one page so several entries share it
    route = torch.randint(0, entries, (rows, width), dtype=torch.int32, device=device)

    indptr = torch.arange(
        0, (rows + 1) * width, width, dtype=torch.int32, device=device
    )
    wrapper = flashinfer.BlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout=layout,
    )
    wrapper.plan(
        indptr,
        route.reshape(-1).contiguous(),
        rows,
        entries,
        1,
        1,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        mask=torch.ones(rows * width, 1, 1, dtype=torch.bool, device=device),
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        o_data_type=torch.float16,
        kv_cache_page_size=page_size,
    )
    out = wrapper.run(q, k_cache, v_cache)
    ref = _paged_reference(q, k_cache, v_cache, route, page_size, layout, num_kv_heads)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_block_sparse_paged_route_rejects_bad_geometry():
    """Every bound the kernel cannot check for itself has to raise here."""
    device = "cuda:0"
    num_qo_heads, num_kv_heads, head_dim, pages, page_size = 4, 2, 128, 8, 16
    entries = pages * page_size
    rows, width = 4, 5
    k_cache = torch.randn(
        pages, num_kv_heads, page_size, head_dim, dtype=torch.float16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(rows, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    route = torch.randint(0, entries, (rows, width), dtype=torch.int32, device=device)
    indptr = torch.arange(
        0, (rows + 1) * width, width, dtype=torch.int32, device=device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    def make_plan(**overrides):
        wrapper = flashinfer.BlockSparseAttentionWrapper(
            workspace, kv_layout=overrides.pop("kv_layout", "HND")
        )
        kwargs = dict(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=torch.ones(rows * width, 1, 1, dtype=torch.bool, device=device),
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            o_data_type=torch.float16,
            kv_cache_page_size=page_size,
        )
        kwargs.update(overrides)
        indices = kwargs.pop("indices", route.reshape(-1).contiguous())
        wrapper.plan(indptr, indices, rows, entries, 1, 1, **kwargs)
        return wrapper

    # a negative route element is loaded as a huge uint32
    negative = route.clone()
    negative[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        make_plan(indices=negative.reshape(-1).contiguous())

    # one past the last entry is still out of bounds
    past_end = route.clone()
    past_end[0, 0] = entries
    with pytest.raises(ValueError, match="out of bound"):
        make_plan(indices=past_end.reshape(-1).contiguous())

    # HND describes a paged cache; without one there is no page axis
    with pytest.raises(ValueError, match="HND"):
        make_plan(kv_cache_page_size=None)

    # a route of slots needs a logical block of one entry
    with pytest.raises(ValueError, match="C must be 1"):
        flashinfer.BlockSparseAttentionWrapper(workspace, kv_layout="HND").plan(
            indptr,
            route.reshape(-1).contiguous(),
            rows,
            entries,
            1,
            2,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            o_data_type=torch.float16,
            kv_cache_page_size=page_size,
        )

    # a value cache the route can read past
    wrapper = make_plan()
    with pytest.raises(ValueError, match="KV entries"):
        wrapper.run(q, k_cache, v_cache[:-1])

    # both caches are large enough on their own, but the route addresses one
    # cache and the kernel takes its geometry from the other
    wide_k = torch.cat([k_cache, k_cache], dim=0)
    with pytest.raises(ValueError, match="pages"):
        wrapper.run(q, wide_k, v_cache)

    # the page size the tensor carries has to be the planned one
    with pytest.raises(ValueError, match="entries per page"):
        wrapper.run(q, k_cache[:, :, :-1], v_cache[:, :, :-1])

    # a cache too small for the planned route would be read past the end
    small = flashinfer.BlockSparseAttentionWrapper(workspace, kv_layout="HND")
    small.plan(
        indptr,
        route.reshape(-1).contiguous(),
        rows,
        entries,
        1,
        1,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        mask=torch.ones(rows * width, 1, 1, dtype=torch.bool, device=device),
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        o_data_type=torch.float16,
        kv_cache_page_size=page_size,
    )
    with pytest.raises(ValueError, match="KV entries"):
        small.run(q, k_cache[:-1], v_cache[:-1])


if __name__ == "__main__":
    # This test verifies the INT32_T overflow issue.
    for seq_len in [16 * 1024, 32 * 1024, 40 * 1024, 48 * 1024, 64 * 1024]:
        test_block_sparse_attention(128, 128, seq_len, seq_len, 1, 1, 128, False)


DEV = "cuda:0"

requires_cuda_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(DEV)[0] < 8,
    reason="the FA2 large-head path starts at sm_80",
)


@requires_cuda_sm80
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_paged_route_reads_a_wide_quantized_head(head_dim, kv_dtype):
    """A one-byte cache of any supported width reads on this architecture.

    The FA2 large-head path rebuilds a quantized cache from raw bytes before
    the dots, which does not depend on the architecture the bytes were written
    for. This pins that down for the widths the read paths claim.
    """
    num_qo_heads, num_kv_heads, page_size = 8, 1, 64
    rows, width, pages = 4, 64, 8
    g = torch.Generator(device=DEV).manual_seed(head_dim)

    keys = (
        torch.randn(
            pages,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device=DEV,
            generator=g,
        )
        * 0.3
    )
    values = torch.randn_like(keys) * 0.3
    scale = 0.5
    if kv_dtype == torch.float8_e4m3fn:
        keys = (keys.float() / scale).to(kv_dtype)
        values = (values.float() / scale).to(kv_dtype)
        run_kwargs = {"k_scale": scale, "v_scale": scale}
        # The reference reads the same bytes, dequantized ahead of time.
        ref_keys = (keys.float() * scale).to(torch.bfloat16)
        ref_values = (values.float() * scale).to(torch.bfloat16)
    else:
        run_kwargs = {}
        ref_keys, ref_values = keys, values

    query = torch.randn(
        rows, num_qo_heads, head_dim, dtype=torch.bfloat16, device=DEV, generator=g
    )
    route = torch.randint(
        0, pages * page_size, (rows, width), dtype=torch.int32, device=DEV, generator=g
    )
    indptr = torch.arange(0, (rows + 1) * width, width, dtype=torch.int32, device=DEV)
    mask = torch.ones(rows * width, 1, 1, dtype=torch.bool, device=DEV)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)

    def run(k, v, dtype, **kwargs):
        wrapper = flashinfer.BlockSparseAttentionWrapper(workspace, kv_layout="HND")
        wrapper.plan(
            indptr,
            route.reshape(-1).contiguous(),
            rows,
            pages * page_size,
            1,
            1,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=mask,
            q_data_type=torch.bfloat16,
            kv_data_type=dtype,
            o_data_type=torch.bfloat16,
            kv_cache_page_size=page_size,
        )
        return wrapper.run(query, k, v, **kwargs)

    out = run(keys, values, kv_dtype, **run_kwargs)
    expected = run(ref_keys, ref_values, torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@requires_cuda_sm80
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_paged_fp8_cache_sizes_its_default_scales_by_the_head_count(layout):
    """A higher-precision query over an FP8 paged cache, with no explicit
    per-head scales.

    The defaults were sized from ``k.shape[1]``. For a raw paged cache that
    axis is the page size under NHD and the head count only by coincidence, so
    a page size below the head count handed the kernel a scale tensor shorter
    than the head it indexes by. The shape here is page_size 1 against four KV
    heads, the smallest that axis gets.

    What it checks is the tensor the wrapper builds, not the attention output:
    the FA2 module this runs on takes the scales in its signature and does not
    apply them, so a short one is passed but never dereferenced and nothing
    downstream moves. Reading them is the FA3 path, which needs SM90.
    """
    torch.manual_seed(7)
    num_qo_heads, num_kv_heads, head_dim = 8, 4, 128
    page_size, pages, rows, width = 1, 32, 4, 8
    entries = pages * page_size
    device = "cuda:0"
    shape = (
        (pages, page_size, num_kv_heads, head_dim)
        if layout == "NHD"
        else (pages, num_kv_heads, page_size, head_dim)
    )
    k_cache = torch.randn(*shape, dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    v_cache = torch.randn(*shape, dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    q = torch.randn(rows, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device)
    route = torch.randint(0, entries, (rows, width), dtype=torch.int32, device=device)
    indptr = torch.arange(
        0, (rows + 1) * width, width, dtype=torch.int32, device=device
    )

    wrapper = flashinfer.BlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        kv_layout=layout,
    )
    wrapper.plan(
        indptr,
        route.reshape(-1).contiguous(),
        rows,
        entries,
        1,
        1,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        mask=torch.ones(rows * width, 1, 1, dtype=torch.bool, device=device),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        kv_cache_page_size=page_size,
    )

    seen = []
    inner = wrapper._cached_module.paged_run

    def spy(*args, **kwargs):
        seen.extend(
            a
            for a in args
            if isinstance(a, torch.Tensor) and a.dtype == torch.float32 and a.ndim == 1
        )
        return inner(*args, **kwargs)

    wrapper._cached_module.paged_run = spy
    try:
        wrapper.run(q, k_cache, v_cache)
    finally:
        wrapper._cached_module.paged_run = inner

    # The module takes other float vectors too, so the check is that the two
    # KV scales are there rather than that every vector is one of them. Sized
    # from k.shape[1] they would be page_size long under NHD -- one element for
    # four heads. HND puts the head count on that axis, so it passes either way
    # and is here to show the coincidence rather than to catch anything.
    widths = [t.numel() for t in seen]
    assert widths.count(num_kv_heads) >= 2, widths


@requires_cuda_sm80
def test_a_second_plan_still_takes_a_page_size_after_auto_resolved(monkeypatch):
    """plan() resolves "auto" and keeps the answer on the wrapper. Gating the
    page size on that resolved value refused a second plan for a backend the
    caller never named -- so a wrapper that first planned a flat route could not
    then plan a paged one."""
    device = "cuda:0"
    rows, width, heads, head_dim = 4, 8, 4, 128
    wrapper = flashinfer.BlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        backend="auto",
    )
    indptr = torch.arange(
        0, (rows + 1) * width, width, dtype=torch.int32, device=device
    )
    indices = torch.zeros(rows * width, dtype=torch.int32, device=device)
    common = dict(
        num_qo_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        mask=torch.ones(rows * width, 1, 1, dtype=torch.bool, device=device),
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        o_data_type=torch.float16,
    )
    # First a flat route, which lets plan() resolve and store a backend.
    wrapper.plan(indptr, indices, rows, 64, 1, 1, **common)
    # What it resolves to depends on the device: sm_80 gives fa2, which the
    # gate accepts either way, so the case that matters is written out rather
    # than waited for. This is what plan() leaves behind on sm_90.
    wrapper._backend = "fa3"
    assert wrapper._requested_backend == "auto"
    # Then a paged one on the same wrapper. Gating on the resolved backend
    # refuses this for an fa3 the caller never named.
    wrapper.plan(indptr, indices, rows, 64, 1, 1, kv_cache_page_size=1, **common)
    assert wrapper._backend == "fa2"
    # And back to a flat one. The paged plan left fa2 behind, which is not
    # "auto", so a plan that does not reset first never resolves again -- it
    # keeps every later flat route on whatever the paged one settled on.
    #
    # Comparing against what the resolver would return does not show that: on
    # this device it returns fa2 as well, so a stale fa2 and a fresh one look
    # the same. The resolver is intercepted instead, and the question becomes
    # whether it was consulted at all.
    calls = []
    real = sparse_module.determine_attention_backend

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(sparse_module, "determine_attention_backend", spy)
    wrapper.plan(indptr, indices, rows, 64, 1, 1, **common)
    assert calls, "the flat plan reused the paged plan's backend"
    assert wrapper._requested_backend == "auto"


@requires_cuda_sm80
def test_a_caller_supplied_fp8_out_holds_the_scaled_result():
    """`out=` is the caller's buffer, and folding `v_scale` into an 8-bit output
    has to go through float32. Doing that by rebinding the local would return a
    scaled tensor and leave the caller's holding the unscaled one.

    An 8-bit output only exists on the cuda-core decode path -- the tensor-core
    prefill generators refuse it -- so the plan here is what selects that path:
    one head each side, a narrow route and no custom mask.
    """
    torch.manual_seed(3)
    rows, width, entries, head_dim = 4, 8, 64, 128
    k = torch.randn(entries, 1, head_dim, dtype=torch.float16, device=DEV)
    v = torch.randn_like(k)
    q = torch.randn(rows, 1, head_dim, dtype=torch.float16, device=DEV)
    indptr = torch.arange(0, (rows + 1) * width, width, dtype=torch.int32, device=DEV)
    indices = torch.randint(0, entries, (rows * width,), dtype=torch.int32, device=DEV)

    def plan():
        wrapper = flashinfer.BlockSparseAttentionWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
        )
        wrapper.plan(
            indptr,
            indices,
            rows,
            entries,
            1,
            1,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=head_dim,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            o_data_type=torch.float8_e4m3fn,
        )
        assert not wrapper._use_tensor_cores, "this case has to take the decode path"
        return wrapper

    scale = 2.0
    buffer = torch.empty(rows, 1, head_dim, dtype=torch.float8_e4m3fn, device=DEV)
    returned = plan().run(q, k, v, out=buffer, v_scale=scale)
    unscaled = plan().run(q, k, v)
    torch.cuda.synchronize()

    # The caller's buffer is what came back, and it holds the scaled values --
    # rebinding the local would leave it holding the unscaled ones.
    assert returned.data_ptr() == buffer.data_ptr()
    want = (unscaled.to(torch.float32) * scale).to(torch.float8_e4m3fn)
    torch.testing.assert_close(buffer.float(), want.float(), rtol=0, atol=0)
    assert not torch.equal(buffer.float(), unscaled.float())
