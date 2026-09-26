# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cuTile MLA paged decode backend of BatchMLAPagedAttentionWrapper."""

import ctypes
import gc
import importlib
import math
from types import SimpleNamespace
import warnings

import pytest
import torch

import flashinfer
from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.utils import get_compute_capability


@pytest.fixture
def _require_blackwell():
    """Skip unless this is a cuTile MLA-validated Blackwell target."""
    if not torch.cuda.is_available():
        pytest.skip("cuTile MLA decode requires CUDA")
    pytest.importorskip("cuda.tile.compilation")
    if not is_cuda_tile_available():
        pytest.skip("cuTile compiler toolchain is unavailable")
    capability = get_compute_capability(torch.device("cuda"))
    if capability not in {(10, 0), (10, 3), (12, 0), (12, 1)}:
        pytest.skip("cuTile MLA decode requires SM100, SM103, SM120, or SM121")


def _torch_mla_decode_ref(
    q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table, page_size, sm_scale
):
    """Naive per-request paged MLA decode reference (fp32 math).

    scores = (q_nope . ckv + q_pe . kpe) * sm_scale over valid kv positions;
    out = softmax(scores) @ ckv  (V shares the compressed latent, head_dim_vo=512).
    """
    batch_size, num_heads, head_dim_ckv = q_nope.shape
    out = torch.empty(
        batch_size, num_heads, head_dim_ckv, dtype=torch.float32, device=q_nope.device
    )
    for b in range(batch_size):
        seq_len = int(kv_lens[b].item())
        n_pages = math.ceil(seq_len / page_size)
        pages = page_table[b, :n_pages]
        # gather [seq_len, dim] from the paged cache
        ckv = ckv_cache[pages].reshape(-1, head_dim_ckv)[:seq_len].float()
        kpe = kpe_cache[pages].reshape(-1, kpe_cache.shape[-1])[:seq_len].float()
        qn = q_nope[b].float()  # [H, 512]
        qp = q_pe[b].float()  # [H, 64]
        # [H, seq_len]
        scores = (qn @ ckv.t() + qp @ kpe.t()) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = probs @ ckv  # [H, 512]
    return out


def _run_mla_decode_case(
    *,
    batch_size,
    max_seq_len,
    page_size,
    num_heads,
    dtype=torch.bfloat16,
    packed=False,
    legacy_api=False,
):
    device = torch.device("cuda")
    torch.manual_seed(42 + page_size + num_heads)
    head_dim_ckv = 512
    head_dim_kpe = 64
    total_page_num = 512
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    q_nope = torch.randn(
        batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(batch_size, num_heads, head_dim_kpe, dtype=dtype, device=device)
    ckv_cache = torch.randn(
        total_page_num, page_size, head_dim_ckv, dtype=dtype, device=device
    )
    kpe_cache = torch.randn(
        total_page_num, page_size, head_dim_kpe, dtype=dtype, device=device
    )
    # random but valid seq lengths (at least 1 token)
    kv_lens = torch.randint(
        1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )
    kv_lens[0] = max_seq_len  # ensure the long case is covered
    if batch_size > 1:
        kv_lens[1] = max_seq_len - 1  # guarantee a non-page-aligned tail
    pages_per_batch = math.ceil(max_seq_len / page_size)
    page_table = torch.randperm(total_page_num, dtype=torch.int32, device=device)[
        : batch_size * pages_per_batch
    ].reshape(batch_size, pages_per_batch)
    identity_pages = torch.arange(
        batch_size * pages_per_batch, dtype=torch.int32, device=device
    ).reshape_as(page_table)
    assert not torch.equal(page_table, identity_pages)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutile")

    if legacy_api:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            wrapper.plan(
                torch.arange(batch_size + 1, dtype=torch.int32, device=device),
                torch.arange(batch_size + 1, dtype=torch.int32, device=device)
                * pages_per_batch,
                page_table.reshape(-1),
                kv_lens,
                num_heads,
                head_dim_ckv,
                head_dim_kpe,
                page_size,
                False,
                sm_scale,
                dtype,
                dtype,
            )
            out = wrapper.run(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                kv_len=kv_lens,
                page_table=page_table,
            )
    else:
        metadata = flashinfer.mla.MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            block_tables=page_table,
            seq_lens=kv_lens,
        )
        layout = "packed" if packed else "split"
        wrapper.plan(
            metadata=metadata,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            page_size=page_size,
            causal=False,
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
            query_layout=layout,
            kv_cache_layout=layout,
        )
        query = torch.cat((q_nope, q_pe), dim=-1) if packed else (q_nope, q_pe)
        kv_cache = (
            torch.cat((ckv_cache, kpe_cache), dim=-1)
            if packed
            else (ckv_cache, kpe_cache)
        )
        out = wrapper.run(query=query, kv_cache=kv_cache)

    ref = _torch_mla_decode_ref(
        q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table, page_size, sm_scale
    )

    assert out.shape == (batch_size, num_heads, head_dim_ckv)
    assert not out.isnan().any()
    torch.testing.assert_close(out.float(), ref, rtol=2e-1, atol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [256, 1024])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.usefixtures("_require_blackwell")
def test_mla_decode_cutile_vs_torch(batch_size, max_seq_len, page_size, num_heads):
    """cuTile paged MLA decode must match the torch reference across the shape sweep."""
    _run_mla_decode_case(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
        num_heads=num_heads,
    )


@pytest.mark.parametrize(
    ("dtype", "num_heads"),
    [
        (torch.float16, 32),
        (torch.bfloat16, 8),
        (torch.bfloat16, 48),
        (torch.bfloat16, 64),
        (torch.bfloat16, 96),
        (torch.bfloat16, 128),
    ],
)
@pytest.mark.usefixtures("_require_blackwell")
def test_mla_decode_cutile_supported_dtype_and_head_contract(dtype, num_heads):
    """Persistent coverage for the restored dtype and head-count support."""
    _run_mla_decode_case(
        batch_size=1,
        max_seq_len=127,
        page_size=64,
        num_heads=num_heads,
        dtype=dtype,
    )


@pytest.mark.usefixtures("_require_blackwell")
def test_mla_decode_cutile_packed_inputs():
    """Packed canonical inputs must lower to the unchanged split kernel."""
    _run_mla_decode_case(
        batch_size=2,
        max_seq_len=255,
        page_size=16,
        num_heads=32,
        packed=True,
    )


@pytest.mark.usefixtures("_require_blackwell")
def test_mla_decode_cutile_legacy_flat_api():
    """The original flat-plan and positional-run compatibility path still works."""
    _run_mla_decode_case(
        batch_size=2,
        max_seq_len=127,
        page_size=64,
        num_heads=32,
        legacy_api=True,
    )


@pytest.mark.parametrize(
    "pages_per_batch",
    [2, 8],
    ids=["no_split", "split_kv"],
)
@pytest.mark.usefixtures("_require_blackwell")
def test_mla_decode_cutile_zero_length_rows(pages_per_batch):
    """Empty KV rows must be zero for auto- and caller-allocated outputs."""
    device = torch.device("cuda")
    torch.manual_seed(73 + pages_per_batch)
    dtype = torch.bfloat16
    batch_size, num_heads, page_size = 2, 16, 64
    head_dim_ckv, head_dim_kpe = 512, 64
    max_seq_len = pages_per_batch * page_size
    total_page_num = batch_size * pages_per_batch
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    q_nope = torch.randn(
        batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(batch_size, num_heads, head_dim_kpe, dtype=dtype, device=device)
    ckv_cache = torch.randn(
        total_page_num, page_size, head_dim_ckv, dtype=dtype, device=device
    )
    kpe_cache = torch.randn(
        total_page_num, page_size, head_dim_kpe, dtype=dtype, device=device
    )
    kv_lens = torch.tensor([0, max_seq_len - 1], dtype=torch.int32, device=device)
    page_table = torch.arange(total_page_num, dtype=torch.int32, device=device).reshape(
        batch_size, pages_per_batch
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutile")
    wrapper.plan(
        metadata=flashinfer.mla.MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            block_tables=page_table,
            seq_lens=kv_lens,
        ),
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        query_layout="split",
        kv_cache_layout="split",
    )

    query = (q_nope, q_pe)
    kv_cache = (ckv_cache, kpe_cache)
    auto_out = wrapper.run(query=query, kv_cache=kv_cache)
    caller_out = torch.full_like(auto_out, 7.0)
    actual = wrapper.run(query=query, kv_cache=kv_cache, out=caller_out)

    assert actual is caller_out
    expected_empty = torch.zeros_like(auto_out[0])
    torch.testing.assert_close(auto_out[0], expected_empty, rtol=0, atol=0)
    torch.testing.assert_close(caller_out[0], expected_empty, rtol=0, atol=0)
    ref = _torch_mla_decode_ref(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_lens,
        page_table,
        page_size,
        sm_scale,
    )
    torch.testing.assert_close(auto_out.float(), ref, rtol=2e-1, atol=1e-2)
    torch.testing.assert_close(caller_out.float(), ref, rtol=2e-1, atol=1e-2)


@pytest.mark.usefixtures("_require_blackwell")
def test_mla_decode_cutile_preallocated_out():
    """Passing a preallocated out tensor must match the auto-allocated path."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    dtype = torch.bfloat16
    batch_size, num_heads, page_size, max_seq_len = 2, 32, 64, 512
    head_dim_ckv, head_dim_kpe = 512, 64
    total_page_num = 256
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    q_nope = torch.randn(
        batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(batch_size, num_heads, head_dim_kpe, dtype=dtype, device=device)
    ckv_cache = torch.randn(
        total_page_num, page_size, head_dim_ckv, dtype=dtype, device=device
    )
    kpe_cache = torch.randn(
        total_page_num, page_size, head_dim_kpe, dtype=dtype, device=device
    )
    kv_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    pages_per_batch = math.ceil(max_seq_len / page_size)
    page_table = torch.randint(
        0,
        total_page_num,
        (batch_size, pages_per_batch),
        dtype=torch.int32,
        device=device,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutile")
    metadata = flashinfer.mla.MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        block_tables=page_table,
        seq_lens=kv_lens,
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        query_layout="split",
        kv_cache_layout="split",
    )

    o_auto = wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv_cache, kpe_cache))
    o_pre = torch.empty_like(o_auto)
    actual = wrapper.run(
        query=(q_nope, q_pe),
        kv_cache=(ckv_cache, kpe_cache),
        out=o_pre,
    )
    assert actual is o_pre
    torch.testing.assert_close(o_auto, o_pre)


@pytest.mark.usefixtures("_require_blackwell")
def test_mla_decode_cutile_cuda_graph_replays_mutated_plan_metadata():
    """Captured runs must read updated values through stable metadata pointers."""
    device = torch.device("cuda")
    torch.manual_seed(11)
    dtype = torch.bfloat16
    batch_size, num_heads, page_size = 2, 16, 64
    head_dim_ckv, head_dim_kpe = 512, 64
    total_page_num = 8
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    q_nope = torch.randn(
        batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(batch_size, num_heads, head_dim_kpe, dtype=dtype, device=device)
    ckv_cache = torch.randn(
        total_page_num, page_size, head_dim_ckv, dtype=dtype, device=device
    )
    kpe_cache = torch.randn(
        total_page_num, page_size, head_dim_kpe, dtype=dtype, device=device
    )
    kv_lens = torch.tensor([96, 64], dtype=torch.int32, device=device)
    page_table = torch.tensor([[3, 1], [6, 2]], dtype=torch.int32, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    out = torch.empty(batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace,
        use_cuda_graph=True,
        backend="cutile",
    )
    wrapper.plan(
        metadata=flashinfer.mla.MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            ),
            block_tables=page_table,
            seq_lens=kv_lens,
        ),
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
        query_layout="split",
        kv_cache_layout="split",
    )

    # Compile and warm the exact launch before capture.
    assert (
        wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv_cache, kpe_cache), out=out)
        is out
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = wrapper.run(
            query=(q_nope, q_pe), kv_cache=(ckv_cache, kpe_cache), out=out
        )
    assert captured_out is out
    graph.replay()
    torch.cuda.synchronize()
    initial = out.clone()

    updated_kv_lens = torch.tensor([0, 95], dtype=torch.int32, device=device)
    updated_page_table = torch.tensor(
        [[0, 5], [4, 7]], dtype=torch.int32, device=device
    )
    kv_lens.copy_(updated_kv_lens)
    page_table.copy_(updated_page_table)
    graph.replay()
    torch.cuda.synchronize()

    ref = _torch_mla_decode_ref(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        updated_kv_lens,
        updated_page_table,
        page_size,
        sm_scale,
    )
    assert not torch.equal(initial, out)
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]), rtol=0, atol=0)
    torch.testing.assert_close(out.float(), ref, rtol=2e-1, atol=1e-2)


# Prepared kernel ABI, launch limits, and execution lifetime.


@pytest.mark.parametrize("fail_load", [False, True])
def test_cutile_library_budget_preserves_cached_kernels(monkeypatch, fail_load):
    from flashinfer.mla._batch_mla._backends import _cutile_prepared as prepared
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )

    driver = pytest.importorskip("cuda.bindings.driver")
    compilation = pytest.importorskip("cuda.tile.compilation")
    # Isolate ownership; fake libraries must never enter the real process cache.
    monkeypatch.setattr(prepared, "_LOADED_LIBRARIES", {})
    monkeypatch.setattr(prepared, "_MAX_LOADED_LIBRARIES", 2)
    loaded, unloaded, exported = [], [], []

    def load(*args):
        library = len(loaded) + 1
        loaded.append(library)
        return (0, library)

    monkeypatch.setattr(driver, "cuLibraryLoadData", load)
    monkeypatch.setattr(driver, "cuLibraryGetKernel", lambda lib, _: (0, lib * 10))
    monkeypatch.setattr(driver, "cuKernelGetFunction", lambda fn: (0, fn))
    monkeypatch.setattr(driver, "cuLibraryUnload", lambda lib: unloaded.append(lib))
    monkeypatch.setattr(compilation, "KernelSignature", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        compilation, "export_kernel", lambda *args, **kwargs: exported.append(True)
    )
    kernel = object()

    def prepare(signature):
        with prepared._COMPILE_LOCK:
            return prepared._load_kernel(kernel, (signature,), "decode", "sm_100a", 0)

    if fail_load:
        with monkeypatch.context() as failed:
            failed.setattr(driver, "cuKernelGetFunction", lambda fn: (1, None))
            with pytest.raises(RuntimeError, match="CUDA driver call failed"):
                prepare(0)
        assert unloaded == [1]
        assert not prepared._LOADED_LIBRARIES

    first, second = prepare(0), prepare(1)
    with pytest.raises(_BackendPlanUnsupportedError, match="library.*limit"):
        prepare(2)
    assert prepare(0) == first
    assert prepare(1) == second
    assert len(prepared._LOADED_LIBRARIES) == 2
    assert len(exported) == len(loaded) == 2 + int(fail_load)
    assert unloaded == ([1] if fail_load else [])


def _argument_array(shape, strides):
    return SimpleNamespace(shape=shape, stride=lambda: strides, data_ptr=lambda: 4096)


def test_cutile_v1_constraints_do_not_require_v2_api():
    from flashinfer.mla._batch_mla._backends._cutile_prepared import _array

    ct = SimpleNamespace(
        int32="int32", int64="int64", float16="float16", bfloat16="bfloat16"
    )
    seen = {}

    def constructor(dtype, ndim, **kwargs):
        seen.update(kwargs)
        return (dtype, ndim)

    compilation = SimpleNamespace(
        ArrayConstraint=constructor,
        CallingConvention=SimpleNamespace(cutile_python_v1=lambda: None),
    )
    assert _array(compilation, ct, ct.bfloat16, (None, 128, 512)) == ("bfloat16", 3)
    assert "shape_constant" not in seen
    assert seen["index_dtype"] == "int64"
    assert seen["base_addr_divisible_by"] == 2
    assert seen["stride_constant"] == (None, None, 1)


@pytest.mark.parametrize(
    "batch,heads,block_h,splits",
    [
        (1, 65535, 1, 1),
        (1, 65536, 16, 1),
        (1, 65535, 1, 2),
        (1, 1, 1, 65535),
        (2**31 - 1, 1, 1, 1),
    ],
)
def test_cutile_launch_grid_exact_boundaries(batch, heads, block_h, splits):
    from flashinfer.mla._batch_mla._backends._cutile_prepared import (
        _validate_launch_grids,
    )

    _validate_launch_grids(batch, heads, block_h, splits)


@pytest.mark.parametrize(
    "batch,heads,block_h,splits,message",
    [
        (1, 65537, 1, 1, "decode grid Y"),
        (1, 65536, 16, 2, "reduction grid Y"),
        (1, 1, 1, 65536, "decode grid Z"),
        (2**31, 1, 1, 1, "decode grid X"),
    ],
)
def test_cutile_launch_grid_overflow_is_typed(batch, heads, block_h, splits, message):
    from flashinfer.mla._batch_mla._backends._capabilities import (
        _BackendPlanUnsupportedError,
    )
    from flashinfer.mla._batch_mla._backends._cutile_prepared import (
        _validate_launch_grids,
    )

    with pytest.raises(_BackendPlanUnsupportedError, match=message):
        _validate_launch_grids(batch, heads, block_h, splits)


def test_cutile_int64_abi_preserves_large_known_and_dynamic_strides():
    from flashinfer.mla._batch_mla._backends._cutile_prepared import (
        _configuration,
        _parameters,
        _validate_launch_grids,
    )

    heads = 4194304
    config = _configuration(16, heads, 128, 128, 148, (10, 0))
    _validate_launch_grids(16, heads, config[0], config[2])
    query = _argument_array((16, heads, 512), (heads * 576, 576, 1))
    pool = _argument_array((2**31 + 17, 128, 512), (128 * 576, 576, 1))
    values, types = _parameters((query, pool), (2**31 + 1,))
    assert values == (
        4096,
        16,
        heads,
        512,
        2415919104,
        576,
        1,
        4096,
        2**31 + 17,
        128,
        512,
        73728,
        576,
        1,
        2**31 + 1,
    )
    assert types == ((ctypes.c_void_p,) + (ctypes.c_int64,) * 6) * 2 + (ctypes.c_int64,)
    # Exercise the actual ctypes conversion rather than merely compare metadata.
    assert [
        kind(value).value for value, kind in zip(values, types, strict=True)
    ] == list(values)


@pytest.mark.parametrize(
    "shape,strides,scalars",
    [
        ((2**63,), (1,), ()),
        ((2,), (2**63,), ()),
        ((2,), (-1,), ()),
        ((2,), (1,), (2**63,)),
    ],
)
def test_cutile_int64_abi_rejects_unrepresentable_arguments(shape, strides, scalars):
    from flashinfer.mla._batch_mla._backends._cutile_prepared import _parameters

    with pytest.raises(ValueError, match="int64"):
        _parameters((_argument_array(shape, strides),), scalars)


def _forbidden(*args, **kwargs):
    raise AssertionError(
        "prepared cuTile run attempted compilation, tuning or allocation"
    )


@pytest.fixture
def cutile_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("prepared cuTile acceptance requires SM100")
    pytest.importorskip("cuda.tile.compilation")
    from flashinfer.cutile.cutile_common import is_cuda_tile_available

    if not is_cuda_tile_available():
        pytest.skip("cuTile compiler toolchain is unavailable")
    prior = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior


@pytest.fixture
def cutile_runtime_case(cutile_sm100):
    """Zero queries and constant values give an exact output without an oracle."""
    from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata

    def make(batch=1, heads=16, page=128, length=96):
        width = math.ceil(length / page)
        metadata = MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
            block_tables=torch.arange(
                batch * width, device="cuda", dtype=torch.int32
            ).view(batch, width),
            seq_lens=torch.full((batch,), length, device="cuda", dtype=torch.int32),
            max_q_len=1,
        )
        wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(1024, device="cuda", dtype=torch.uint8), backend="cutile"
        )
        wrapper.plan(
            metadata=metadata,
            num_heads=heads,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=page,
            causal=True,
            sm_scale=1 / math.sqrt(576),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            query_layout="split",
            kv_cache_layout="split",
        )
        return wrapper, metadata

    return make


@pytest.mark.parametrize(
    "batch,heads,page,length", [(1, 16, 128, 96), (16, 64, 64, 384)]
)
def test_cutile_prepared_run_only_launches(
    cutile_runtime_case, monkeypatch, batch, heads, page, length
):
    import cuda.tile as ct
    from cuda.tile import compilation

    wrapper, metadata = cutile_runtime_case(batch, heads, page, length)
    prepared = wrapper._planned_backend._decode_mla_kv_paged_cutile
    assert (prepared.partial is not None) == (batch == 16)
    scratch_pointer = (
        prepared.partial.data_ptr() if prepared.partial is not None else None
    )
    native = importlib.import_module(
        "flashinfer.attention.kernels.cutile.fmha_decode_bsr_cutile"
    )
    compile_module = importlib.import_module("cuda.tile._compile")
    monkeypatch.setattr(ct, "launch", _forbidden)
    monkeypatch.setattr(compilation, "export_kernel", _forbidden)
    monkeypatch.setattr(compile_module, "compile_tile", _forbidden)
    monkeypatch.setattr(native, "exhaustive_search", _forbidden)
    monkeypatch.setattr(native, "decode_mla_kv_paged_cutile", _forbidden)
    for extra_pools in (0, 17):
        for adjacent in (False, True):
            pools = metadata.block_tables.numel() + extra_pools
            if adjacent:
                q = torch.zeros(batch, heads, 576, device="cuda", dtype=torch.bfloat16)
                kv = torch.full(
                    (pools, page, 576), 2.0, device="cuda", dtype=torch.bfloat16
                )
                query = (q[..., :512], q[..., 512:])
                cache = (kv[..., :512], kv[..., 512:])
            else:
                query = tuple(
                    torch.zeros(batch, heads, d, device="cuda", dtype=torch.bfloat16)
                    for d in (512, 64)
                )
                cache = tuple(
                    torch.full(
                        (pools, page, d), 2.0, device="cuda", dtype=torch.bfloat16
                    )
                    for d in (512, 64)
                )
            out = torch.empty(
                batch * heads * 512 + 1, device="cuda", dtype=torch.bfloat16
            )[1:].view(batch, heads, 512)

            def run():
                with monkeypatch.context() as guard:
                    for name in ("empty", "zeros", "full", "zeros_like", "empty_like"):
                        guard.setattr(torch, name, _forbidden)
                    assert wrapper.run(query=query, kv_cache=cache, out=out) is out

            for current_length in (length, 0, 13, length):
                metadata.seq_lens.fill_(current_length)
                out.fill_(math.nan)
                run()
                torch.testing.assert_close(
                    out, torch.full_like(out, 2.0 if current_length else 0.0)
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            out.fill_(math.nan)
            graph.replay()
            torch.testing.assert_close(out, torch.full_like(out, 2.0))
    assert (
        prepared.partial.data_ptr() if prepared.partial is not None else None
    ) == scratch_pointer


def test_cutile_binary_lifetime_survives_preparer_collection(
    cutile_runtime_case, monkeypatch
):
    from flashinfer.mla._batch_mla._backends import _cutile_prepared
    import cuda.tile.compilation as compilation

    first, _ = cutile_runtime_case()
    binary = first._planned_backend._decode_mla_kv_paged_cutile.decode
    del first
    gc.collect()
    monkeypatch.setattr(compilation, "export_kernel", _forbidden)
    second, _ = cutile_runtime_case()
    assert second._planned_backend._decode_mla_kv_paged_cutile.decode == binary
    query = tuple(
        torch.zeros(1, 16, d, device="cuda", dtype=torch.bfloat16) for d in (512, 64)
    )
    cache = tuple(
        torch.full((1, 128, d), 2.0, device="cuda", dtype=torch.bfloat16)
        for d in (512, 64)
    )
    out = torch.empty(16 * 512 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
        1, 16, 512
    )
    assert second.run(query=query, kv_cache=cache, out=out) is out
    torch.testing.assert_close(out, torch.full_like(out, 2.0))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        second.run(query=query, kv_cache=cache, out=out)
    del second
    gc.collect()
    # A refused new specialization must not evict code held by this graph.
    monkeypatch.setattr(
        _cutile_prepared,
        "_MAX_LOADED_LIBRARIES",
        len(_cutile_prepared._LOADED_LIBRARIES),
    )
    with (
        pytest.raises(
            _cutile_prepared._BackendPlanUnsupportedError, match="library.*limit"
        ),
        _cutile_prepared._COMPILE_LOCK,
    ):
        _cutile_prepared._load_kernel(object(), (), "new", "sm_100a", 0)
    out.fill_(math.nan)
    graph.replay()
    torch.testing.assert_close(out, torch.full_like(out, 2.0))


if __name__ == "__main__":
    test_mla_decode_cutile_vs_torch(4, 1024, 64, 32)
    test_mla_decode_cutile_preallocated_out()
