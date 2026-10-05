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

from types import SimpleNamespace

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
from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.utils import (
    get_compute_capability,
    has_flashinfer_jit_cache,
    is_sm90a_supported,
    is_sm100a_supported,
)


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    if torch.cuda.is_available() and get_compute_capability(torch.device(0)) == (10, 7):
        # SM107 does not run the vsa_blackwell backend. Let any supported
        # backend compile lazily instead of building unrelated modules here.
        yield
        return
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
        if get_compute_capability(torch.device(0)) == (10, 7):
            pytest.skip("vsa_blackwell supports SM100 and SM103, not SM107")
        if not is_sm100a_supported(torch.device(0)):
            pytest.skip("vsa_blackwell requires sm100a (Blackwell GPU)")
        if torch.cuda.get_device_capability(0) == (10, 7):
            pytest.skip("vsa_blackwell supports SM100/SM103, not SM107")
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


def _fp8_fa3(M):
    """FP8 Q/K/V over a diagonal BSR layout with C=1, below the KV-head count."""
    w = flashinfer.BlockSparseAttentionWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda:0"), backend="fa3"
    )
    rows = torch.arange(M + 1, dtype=torch.int32, device="cuda:0")
    fp8 = torch.float8_e4m3fn
    w.plan(rows, rows[:-1], M, M, 1, 1, 8, 4, 128, q_data_type=fp8, kv_data_type=fp8)
    q, k, v = (torch.randn(M, h, 128, device="cuda:0").to(fp8) for h in (8, 4, 4))
    return w, q, k, v


@pytest.mark.parametrize("paged", [False, True])
def test_fp8_default_scales_are_sized_by_the_kv_head_count(monkeypatch, paged):
    seen = []  # paged_run's scale_k and scale_v
    rec = SimpleNamespace(
        plan=lambda *a: [], paged_run=lambda *a, **k: seen.extend(a[25:27])
    )
    monkeypatch.setattr(
        flashinfer.sparse, "get_batch_prefill_module", lambda *a, **k: rec
    )
    if paged:  # a bf16 query over an FP8 paged cache whose page size is 1
        c = _paged_case("fp8", page=1, hkv=4)
        _plan(c).run(c.q, *c.kv)
    else:
        w, q, k, v = _fp8_fa3(16)
        w.run(q, k, v)
    assert [s.numel() for s in seen] == [4, 4]


def test_fp8_default_scales_match_explicit_unit_scales():
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("FP8 block-sparse runs on FA3")
    w, q, k, v = _fp8_fa3(64)
    q_s, k_s, v_s = (torch.ones(h, device="cuda:0") for h in (8, 4, 4))
    explicit = w.run(q, k, v, scale_q=q_s, scale_k=k_s, scale_v=v_s)
    torch.testing.assert_close(w.run(q, k, v), explicit, rtol=0, atol=0)


def _u8(n, device="cuda", **kw):
    return torch.empty(n, dtype=torch.uint8, device=device, **kw)


def _bsa(**kw):
    return flashinfer.BlockSparseAttentionWrapper(_u8(128 << 20), **kw)


def _decode_nvfp4(data, sf, scale):
    """NVFP4 bytes to float in torch: two e2m1 values per byte, low nibble first."""
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=data.device)
    nib = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    sign = 1 - 2 * (nib >> 3)
    return e2m1[nib & 7] * sign * sf.float().repeat_interleave(16, -1) * scale


def _paged_case(fmt="bf16", layout="NHD", page=16, hkv=1, d=128, h=8, seed=0):
    """q, a 64-page K/V cache in ``fmt`` with its float copy, and a route of random
    flat slots; 35-wide rows keep a packed mask off byte boundaries."""
    g = torch.Generator("cuda").manual_seed(seed)
    rows, width, entries = 16, 35, 64 * page
    shape = (2, 64, page, hkv) if layout == "NHD" else (2, 64, hkv, page)
    kv = torch.randn(*shape, d, device="cuda", generator=g)
    s, run_kw = torch.tensor([1.5, 0.75], device="cuda").view(2, 1, 1, 1, 1), {}
    if fmt == "bf16":
        kv = kv.bfloat16()
        ref = kv.float()
    elif fmt == "fp8":
        kv = (kv / s).to(torch.float8_e4m3fn)
        ref, run_kw = kv.float() * s, dict(k_scale=1.5, v_scale=0.75)
    else:  # nvfp4, from FlashInfer's own writer, decoded in torch
        src = torch.randn(2, entries, hkv, d, device="cuda", generator=g).bfloat16()
        kv = torch.zeros(*shape, d // 2, dtype=torch.uint8, device="cuda")
        sf = torch.zeros(*shape, d // 16, dtype=torch.float8_e4m3fn, device="cuda")
        slots = torch.arange(entries, dtype=torch.int32, device="cuda")
        flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
            *src, slots, tuple(kv), tuple(sf), 1.5, 0.75, kv_layout=layout
        )
        ref = _decode_nvfp4(kv, sf, s)
        run_kw = dict(kv_cache_sf=tuple(sf), k_scale=1.5, v_scale=0.75)
    q = torch.randn(rows, h, d, device="cuda", generator=g).bfloat16()
    indptr = torch.arange(0, rows * width + 1, width, dtype=torch.int32, device="cuda")
    route = torch.randint(0, entries, (rows * width,), device="cuda", generator=g).int()
    dims = dict(num_qo_heads=h, num_kv_heads=hkv, head_dim=d, kv_cache_page_size=page)
    dims.update(q_data_type=q.dtype, kv_data_type=kv.dtype, o_data_type=q.dtype)
    kv, ref = kv.unbind(), ref.unbind()
    return SimpleNamespace(**locals())


def _plan(c, w=None, C=1, **kw):
    w = w or _bsa(kv_layout=c.layout)
    w.plan(c.indptr, c.route, c.rows, c.entries, 1, C, **{**c.dims, **kw})
    return w


def _size(c, w=None, **kw):
    return (w or _bsa()).workspace_size(c.indptr, c.rows, 1, 1, **{**c.dims, **kw})[1]


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "fmt,page,hkv,d",
    [("bf16", p, h, 128) for p in (1, 16) for h in (1, 2, 4)]
    + [("fp8", 16, 1, 256), ("fp8", 16, 1, 512), ("nvfp4", 16, 2, 128)],
)
def test_block_sparse_paged_route(fmt, layout, page, hkv, d):
    """Each index is a flat slot into a cache that still stores whole pages."""
    c = _paged_case(fmt, layout, page, hkv, d)
    out = torch.empty_like(c.q)
    _plan(c).run(c.q, *c.kv, out=out, **c.run_kw)
    k, v = (t.transpose(1, 2) if layout == "HND" else t for t in c.ref)
    k, v = (t.reshape(-1, hkv, d)[c.route.view(16, -1)] for t in (k, v))
    k, v = (t.repeat_interleave(8 // hkv, dim=2) for t in (k, v))
    p = torch.einsum("rhd,rwhd->rhw", c.q.float(), k).mul(d**-0.5).softmax(-1)
    ref = torch.einsum("rhw,rwhd->rhd", p, v)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "plan_kw,cut,match",
    [
        ({"first": -1}, None, "non-negative"),
        ({"first": 1024}, None, "out of bound"),
        ({"kv_cache_page_size": None}, None, "HND"),
        ({"C": 2}, None, "C must be 1"),
        ({}, lambda k, v: (k, v[:-1]), "KV entries"),
        ({}, lambda k, v: (k[:-1], v[:-1]), "KV entries"),
        ({}, lambda k, v: (torch.cat([k, k]), v), "pages"),
        ({}, lambda k, v: (k[:, :, :-1], v[:, :, :-1]), "entries per page"),
    ],
)
def test_block_sparse_paged_route_rejects_bad_geometry(plan_kw, cut, match):
    c, plan_kw = _paged_case(layout="HND", hkv=2), dict(plan_kw)
    c.route[0] = plan_kw.pop("first", c.route[0])
    with pytest.raises(ValueError, match=match):
        _plan(c, **plan_kw).run(c.q, *(cut(*c.kv) if cut else c.kv))


def test_a_second_plan_resolves_the_requested_backend_again(monkeypatch):
    c = _paged_case()
    w = _plan(c, kv_cache_page_size=None)
    w._backend = "fa3"  # what a flat plan leaves behind on SM90
    assert _plan(c, w)._backend == "fa2"
    seen, target = [], "flashinfer.sparse.determine_attention_backend"
    monkeypatch.setattr(target, lambda *a, **k: seen.append(a) or "fa2")
    _plan(c, w, kv_cache_page_size=None)
    assert seen and w._requested_backend == "auto"


def test_a_caller_fp8_out_holds_the_scaled_result():
    """Folding v_scale into an 8-bit output has to write the caller's buffer."""
    c, fp8 = _paged_case(h=1), torch.float8_e4m3fn
    w = _plan(c, kv_cache_page_size=None, o_data_type=fp8)
    assert not w._use_tensor_cores  # 8-bit outputs exist on the decode path only
    flat, out = [t.reshape(-1, 1, 128) for t in c.kv], torch.empty_like(c.q, dtype=fp8)
    assert w.run(c.q, *flat, out=out, v_scale=2.0).data_ptr() == out.data_ptr()
    assert torch.equal(out.float(), (w.run(c.q, *flat).float() * 2).to(fp8).float())


def test_sizing_plans_nothing_and_the_default_buffers_are_the_wrappers_own():
    c, w = _paged_case(), _bsa()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    assert _size(c, w) > 0 and torch.cuda.max_memory_allocated() == before
    assert w._backend == "auto" and not hasattr(w, "_plan_info")
    assert w._int_workspace_buffer.numel() == 8 << 20
    assert w._pin_memory_int_workspace_buffer.is_pinned()


@pytest.mark.parametrize("page", [None, 16])  # cuda-core decode, FA2 prefill
def test_the_sized_int_workspace_is_exactly_enough(page):
    c = _paged_case(h=1)
    n = _size(c, kv_cache_page_size=page)
    w = _plan(c, _bsa(int_workspace_buffer=_u8(n)), kv_cache_page_size=page)
    assert w._use_tensor_cores == bool(page)
    with pytest.raises(RuntimeError, match="Buffer overflow"):
        _plan(c, _bsa(int_workspace_buffer=_u8(n - 1)), kv_cache_page_size=page)


def test_live_plans_on_exact_slices_of_one_arena_run_and_replay():
    cs = [_paged_case(seed=s) for s in (1, 2)]
    n = _size(cs[0])
    step = -(-n // 16) * 16
    arena, kw = (
        _u8(2 * step),
        dict(pin_memory_int_workspace_buffer=_u8(n, "cpu", pin_memory=True)),
    )
    ws = [
        _plan(c, _bsa(int_workspace_buffer=arena[i * step :][:n], **kw))
        for i, c in enumerate(cs)
    ]
    for w, c in zip(ws, cs, strict=True):  # both planned before either runs
        assert torch.equal(w.run(c.q, *c.kv), _plan(c).run(c.q, *c.kv))
    c, w = cs[0], ws[0]
    out = torch.empty_like(c.q)
    want = w.run(c.q, *c.kv, out=out).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(c.q, *c.kv, out=out)
    out.zero_()
    graph.replay()
    assert torch.equal(out, want)


@pytest.mark.parametrize(
    "make,match",
    [
        (lambda n: (_u8(n, "cpu"), None), "on cuda"),
        (lambda n: (_u8(n).view(torch.int32), None), "on cuda"),
        (lambda n: (_u8(n).view(1, n), None), "on cuda"),
        (lambda n: (_u8(2 * n)[::2], None), "contiguous"),
        (lambda n: (_u8(n + 16)[1:], None), "aligned"),
        (lambda n: (_u8(n), _u8(n)), "on cpu"),
        (lambda n: (_u8(n), _u8(2 * n, "cpu", pin_memory=True)[::2]), "contiguous"),
        (lambda n: (_u8(n), _u8(n, "cpu")), "pinned"),
        (lambda n: (_u8(n), _u8(n - 1, "cpu", pin_memory=True)), "pinned"),
    ],
)
def test_an_unusable_int_workspace_is_refused_before_planning(make, match):
    int_ws, staging = make(4096)
    with pytest.raises(ValueError, match=match):
        _bsa(int_workspace_buffer=int_ws, pin_memory_int_workspace_buffer=staging)


def test_a_caller_packed_mask_matches_the_one_plan_packs():
    c = _paged_case()
    mask = torch.rand(16, 35, device="cuda") > 0.3
    packed = torch.cat([flashinfer.packbits(m, bitorder="little") for m in mask])
    got = _plan(c, packed_mask=packed).run(c.q, *c.kv)
    assert torch.equal(got, _plan(c, mask=mask.view(-1, 1, 1)).run(c.q, *c.kv))


def test_sizing_and_a_caller_int_workspace_inside_a_default_device_context():
    c = _paged_case()
    with torch.device("cuda"):
        inside = _size(c, use_custom_mask=True)
        w = _bsa(int_workspace_buffer=_u8(4096))
    assert inside == _size(c, use_custom_mask=True)
    assert w._pin_memory_int_workspace_buffer.is_pinned()


if __name__ == "__main__":
    # This test verifies the INT32_T overflow issue.
    for seq_len in [16 * 1024, 32 * 1024, 40 * 1024, 48 * 1024, 64 * 1024]:
        test_block_sparse_attention(128, 128, seq_len, seq_len, 1, 1, 128, False)
