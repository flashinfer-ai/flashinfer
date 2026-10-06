from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from b12x.preparation import PreparedCall, PreparationSession
from b12x.attention._shared.mla.compressed_api import (
    _validate_compressed_cache_layout,
)
from b12x.attention._shared.mla.kernel import (
    _cache_block_stride_bytes as _decode_cache_block_stride_bytes,
)
from b12x.attention._shared.mla.prefill import (
    _cache_block_stride_bytes as _prefill_cache_block_stride_bytes,
)
from b12x.attention._shared.mla.prefill_mg import (
    _cache_block_stride_bytes as _prefill_mg_cache_block_stride_bytes,
)
from b12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN,
    COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
    COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
    compressed_sparse_mla_page_nbytes,
    compressed_sparse_mla_reference,
    pack_compressed_sparse_mla_kv_cache_reference,
    pack_deepseek_v41_cache_reference,
    unpack_deepseek_v41_cache_reference,
)
from b12x.attention import compressed_sparse_mla
from b12x.attention.compressed_sparse_mla._scratch import B12XCompressedSparseMLAScratchCaps
from b12x.attention._shared.mla.api import clear_mla_caches

from ..conftest import require_b12x as require_sm120


_COMPRESSED_HEAD_DIM = 512
_SHARED_CORE_HEAD_DIM = 576
_SHARED_CORE_V_HEAD_DIM = 512
_LOCAL_Q_HEADS = 32
_SM_SCALE = 1.0 / math.sqrt(_COMPRESSED_HEAD_DIM)


@pytest.mark.parametrize("page_size", [16, 64, 256])
def test_compressed_sparse_mla_layout_accepts_contiguous_and_padded_pages(
    page_size: int,
) -> None:
    payload_nbytes = page_size * COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
    padded_nbytes = compressed_sparse_mla_page_nbytes(page_size)

    _validate_compressed_cache_layout(
        torch.empty((2, payload_nbytes), dtype=torch.uint8),
        page_size=page_size,
        name="cache",
    )
    _validate_compressed_cache_layout(
        torch.empty((2, padded_nbytes), dtype=torch.uint8),
        page_size=page_size,
        name="cache",
    )


def test_compressed_sparse_mla_layout_rejects_short_page() -> None:
    payload_nbytes = (
        COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE * COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
    )
    with pytest.raises(ValueError, match="contiguous payload"):
        _validate_compressed_cache_layout(
            torch.empty((2, payload_nbytes - 1), dtype=torch.uint8),
            page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
            name="cache",
        )


@pytest.mark.parametrize(("cache_kind", "record_bytes"), [("swa", 528), ("indexed", 288)])
def test_deepseek_v41_reference_page_records_are_distinct(
    cache_kind: str, record_bytes: int
) -> None:
    values = torch.tensor(
        [[0.0, -0.0, 0.125, -0.125] + [0.0] * 508],
        dtype=torch.bfloat16,
    )
    cache = pack_deepseek_v41_cache_reference(
        values, page_size=16, cache_kind=cache_kind
    )
    assert cache.shape == (1, 16 * record_bytes)
    unpacked = unpack_deepseek_v41_cache_reference(
        cache, page_size=16, cache_kind=cache_kind
    )
    assert unpacked.shape == (16, 512)
    assert torch.isfinite(unpacked).all()

@pytest.mark.parametrize(
    "stride_fn,kwargs",
    [
        (_decode_cache_block_stride_bytes, {"model_type": 0}),
        (_prefill_cache_block_stride_bytes, {"model_type": 0}),
        (_prefill_mg_cache_block_stride_bytes, {"is_glm": False}),
    ],
)
@pytest.mark.parametrize("padded", [False, True])
def test_compressed_sparse_mla_dispatch_uses_physical_page_stride(
    stride_fn: Callable[..., int],
    kwargs: dict[str, object],
    padded: bool,
) -> None:
    payload_nbytes = (
        COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE * COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
    )
    physical_nbytes = (
        compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE)
        if padded
        else payload_nbytes
    )
    storage = torch.empty(2 * physical_nbytes, dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, payload_nbytes),
        stride=(physical_nbytes, 1),
    )

    assert (
        stride_fn(cache, page_size=COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE, **kwargs)
        == physical_nbytes
    )


def _make_split_merge_tensors(
    *,
    rows: int,
    heads: int,
    chunks: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    tmp_storage = torch.randn(
        rows * heads * chunks * _COMPRESSED_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=gen,
    )
    tmp_output = tmp_storage.as_strided(
        (rows, heads, chunks, _COMPRESSED_HEAD_DIM),
        (
            heads * _COMPRESSED_HEAD_DIM,
            _COMPRESSED_HEAD_DIM,
            rows * heads * _COMPRESSED_HEAD_DIM,
            1,
        ),
    )
    tmp_lse = torch.randn(
        (rows, heads, chunks),
        dtype=torch.float32,
        device=device,
        generator=gen,
    )
    output = torch.empty(
        (rows, heads, _COMPRESSED_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    num_chunks_ptr = torch.tensor([chunks], dtype=torch.int32, device=device)
    attn_sink = torch.zeros((heads,), dtype=torch.float32, device=device)
    return tmp_output, tmp_lse, num_chunks_ptr, attn_sink, output

def _page_size_from_cache(cache: torch.Tensor) -> int:
    """Recover the V4 page geometry from the supplied physical page metadata."""
    page_nbytes = int(cache.shape[1])
    matches = [
        page_size
        for page_size in (1, 2, 4, 16, 64, 256)
        if page_nbytes in (
            page_size * COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN,
            compressed_sparse_mla_page_nbytes(page_size),
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cache page width {page_nbytes} does not identify one DSV4 page size"
        )
    return matches[0]



def _prepare_compressed_binding(
    *,
    device: torch.device | str,
    mode: str,
    max_q_rows: int,
    max_kv_rows: int,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    attn_sink: torch.Tensor | None,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_lengths: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    use_cuda_graph: bool = False,
):
    """Prepare the exact native carrier that the test binding will execute."""
    device = torch.device(device)
    swa_width = int(swa_indices.shape[1])
    indexed_width = 0 if indexed_indices is None else int(indexed_indices.shape[1])
    swa_page_size = _page_size_from_cache(swa_k_cache)
    indexed_page_size = (
        0
        if indexed_k_cache is None
        else _page_size_from_cache(indexed_k_cache)
    )
    caps = B12XCompressedSparseMLAScratchCaps(
        device=device,
        num_q_heads=int(q.shape[-2]),
        max_q_rows=max_q_rows,
        max_width=swa_width + indexed_width,
        max_batch=max_q_rows,
        max_kv_rows=max_kv_rows,
        max_page_table_width=swa_width + indexed_width,
        mode=mode,
        swa_width=swa_width,
        indexed_width=indexed_width,
        swa_page_size=swa_page_size,
        indexed_page_size=(
            swa_page_size if indexed_k_cache is None else indexed_page_size
        ),
        use_cuda_graph=use_cuda_graph,
    )
    declaration = compressed_sparse_mla.plan(
        caps,
        invocation=compressed_sparse_mla.invocation_from_tensors(
            q=q,
            swa_k_cache=swa_k_cache,
            indexed_k_cache=indexed_k_cache,
            attn_sink=attn_sink,
            out=out,
        ),
    )
    owned = {}

    def prepare_call(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind_for_preparation(
            scratch=scratch,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
        )
        owned["scratch"] = scratch
        return PreparedCall(
            run=lambda: state.run(
                binding,
                swa_k_cache=swa_k_cache,
                indexed_k_cache=indexed_k_cache,
                attn_sink=attn_sink,
                sm_scale=_SM_SCALE,
                out=out,
            )
        )

    result = PreparationSession(device=device, autotune=False).prepare((
        declaration.request(
            name="compressed-sparse-mla-test",
            prepare_call=prepare_call,
        ),
    ))
    plan = declaration
    return (
        result,
        plan,
        compressed_sparse_mla.bind(
            plan,
            scratch=owned["scratch"],
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
        ),
    )


def _make_cache(
    *,
    tokens: int,
    page_size: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k_nope = (
        torch.randn((tokens, 448), generator=gen, dtype=torch.float32, device=device)
        * 0.05
    )
    k_rope = (
        torch.randn((tokens, 64), generator=gen, dtype=torch.float32, device=device)
        * 0.05
    )
    return pack_compressed_sparse_mla_kv_cache_reference(
        k_nope,
        k_rope.to(dtype=torch.bfloat16),
        page_size=page_size,
    )


def _make_q(*, rows: int, seed: int, device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = (
        torch.randn(
            (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
            generator=gen,
            dtype=torch.float32,
            device=device,
        )
        * 0.04
    )
    return q.to(dtype=torch.bfloat16)


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize(
    "mode,large_pool",
    [("decode", False), ("decode", True), ("extend", False), ("extend", True)],
)
@torch.inference_mode()
def test_compressed_sparse_mla_ignores_nan_in_unused_page(
    heads: int, mode: str, large_pool: bool
) -> None:
    """Masked candidates must not contribute unused-page NaNs to P.V."""
    device = require_sm120()
    rows = 1 if mode == "decode" else 17
    page_size = 64
    stride = 1_002_240
    live_page = (1 << 31) // stride + 1 if large_pool else 1
    rope = torch.zeros((8, 64), dtype=torch.bfloat16, device=device)
    rope[:, 0] = 4
    nope = torch.ones((8, 448), device=device)
    compact = pack_compressed_sparse_mla_kv_cache_reference(
        -nope, -rope, page_size=page_size
    )
    indexed_compact = pack_compressed_sparse_mla_kv_cache_reference(
        nope, rope, page_size=page_size
    )
    page_bytes = compact.shape[1]
    storage = torch.empty((live_page + 1) * stride, dtype=torch.uint8, device=device)
    swa_cache = storage.as_strided((live_page + 1, page_bytes), (stride, 1))
    indexed_cache = storage.as_strided(
        (live_page + 1, page_bytes), (stride, 1), storage_offset=page_bytes
    )
    for cache, source in ((swa_cache, compact), (indexed_cache, indexed_compact)):
        cache[0].zero_()
        cache[live_page].copy_(source[0])
    q = torch.zeros((rows, heads, 512), dtype=torch.bfloat16, device=device)
    q[:, :, 448] = 16
    swa_indices = torch.full((rows, 128), -1, dtype=torch.int32, device=device)
    indexed_indices = torch.full((rows, 512), -1, dtype=torch.int32, device=device)
    swa_indices[:, :8] = live_page * page_size + torch.arange(8, device=device)
    indexed_indices[:, :3] = live_page * page_size + torch.arange(3, device=device)
    swa_lengths = torch.arange(rows, device=device, dtype=torch.int32) % 8 + 1
    indexed_lengths = torch.full((rows,), 3, device=device, dtype=torch.int32)
    attn_sink = torch.zeros(heads, dtype=torch.float32, device=device)
    prepared, plan, binding = _prepare_compressed_binding(
        device=device,
        mode=mode,
        max_q_rows=rows,
        max_kv_rows=rows * 640,
        q=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        attn_sink=attn_sink,
        use_cuda_graph=True,
    )

    def run():
        return compressed_sparse_mla.run(
            plan=plan,
            binding=binding,
            swa_k_cache=swa_cache,
            swa_page_size=page_size,
            indexed_k_cache=indexed_cache,
            indexed_page_size=page_size,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
        )

    baseline = run().clone()
    expected = compressed_sparse_mla_reference(
        q,
        compact,
        swa_indices - live_page * page_size,
        swa_lengths,
        swa_page_size=page_size,
        extra_k_cache=indexed_compact,
        extra_indices=indexed_indices - live_page * page_size,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=page_size,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    torch.testing.assert_close(
        baseline.float(), expected.float(), atol=0.002, rtol=0.01
    )
    assert torch.count_nonzero(baseline).item() > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for cache in (swa_cache, indexed_cache):
        cache[0].fill_(255)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, baseline, atol=0, rtol=0)


@torch.inference_mode()
def test_compressed_sparse_mla_shared_core_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    q = _make_q(rows=1, seed=21, device=device)
    swa_cache_bytes = _make_cache(
        tokens=32,
        page_size=COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
        seed=22,
        device=device,
    )
    indexed_cache_bytes = _make_cache(
        tokens=32,
        page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
        seed=23,
        device=device,
    )
    swa_cache = swa_cache_bytes.view(torch.float8_e4m3fn)
    indexed_cache = indexed_cache_bytes.view(torch.float8_e4m3fn)
    swa_indices = torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0)
    indexed_indices = torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0)
    swa_lengths = torch.tensor([11], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([7], dtype=torch.int32, device=device)
    attn_sink = torch.nn.Parameter(
        torch.linspace(-0.1, 0.1, _LOCAL_Q_HEADS, dtype=torch.float32, device=device)
    )
    prepared, plan, binding = _prepare_compressed_binding(
        device=device,
        mode="decode",
        max_q_rows=8,
        max_kv_rows=8 * (swa_indices.shape[1] + indexed_indices.shape[1]),
        q=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_k_cache=indexed_cache,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        attn_sink=attn_sink,
        use_cuda_graph=True,
    )
    captured_out: torch.Tensor | None = None


    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_sparse_mla.run(
            plan=plan,
            binding=binding,
            swa_k_cache=swa_cache,
            indexed_k_cache=indexed_cache,
            indexed_page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
        )
        return captured_out

    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_out is not None

    expected = compressed_sparse_mla_reference(
        q,
        swa_cache_bytes,
        swa_indices,
        swa_lengths,
        extra_k_cache=indexed_cache_bytes,
        extra_indices=indexed_indices,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995

    # Replay the same captured graph with shorter live sections. The launch grid,
    # workspace, and tensor addresses stay fixed; one of the two capacity-planned
    # chunks is now wholly empty and must contribute a neutral LSE without running
    # its gather/MMA pipeline.
    swa_lengths.fill_(1)
    indexed_lengths.zero_()
    graph.replay()
    torch.cuda.synchronize(device)

    expected_short = compressed_sparse_mla_reference(
        q,
        swa_cache_bytes,
        swa_indices,
        swa_lengths,
        extra_k_cache=indexed_cache_bytes,
        extra_indices=indexed_indices,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    max_abs_short = (captured_out.float() - expected_short.float()).abs().max().item()
    cos_short = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected_short.float().reshape(-1), dim=0
    )
    assert max_abs_short <= 0.10
    assert cos_short.item() >= 0.9995


@torch.inference_mode()
def test_compressed_sparse_mla_out_param_writes_directly_and_matches() -> None:
    device = require_sm120()
    clear_mla_caches()

    rows = 8
    q = _make_q(rows=rows, seed=311, device=device)
    swa_cache = _make_cache(
        tokens=32,
        page_size=COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
        seed=312,
        device=device,
    )
    attn_sink = torch.linspace(
        -0.2, 0.15, _LOCAL_Q_HEADS, dtype=torch.float32, device=device
    )

    def _make_swa(width: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.full((rows, width), -1, dtype=torch.int32, device=device)
        lengths = torch.empty((rows,), dtype=torch.int32, device=device)
        for row in range(rows):
            length = min(width, row + 1)
            indices[row, :length] = torch.arange(
                row, row - length, -1, dtype=torch.int32, device=device
            )
            lengths[row] = length
        return indices, lengths

    # The MG prefill kernel requires the FP8 topk widths (512/1024/2048);
    # decode has no such floor.  Each output ownership mode is a distinct
    # prepared ABI, so both plans retain the actual route they invoke.
    for mode, width in (("decode", 8), ("extend", 512)):
        swa_indices, swa_lengths = _make_swa(width)
        _, baseline_plan, baseline_binding = _prepare_compressed_binding(
            device=device,
            mode=mode,
            max_q_rows=rows,
            max_kv_rows=rows * width,
            q=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            attn_sink=attn_sink,
        )
        baseline = compressed_sparse_mla.run(
            plan=baseline_plan,
            binding=baseline_binding,
            swa_k_cache=swa_cache,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
        ).clone()

        # NaN canary: every output position must be written by the kernel.
        out = torch.full(
            (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        _, output_plan, output_binding = _prepare_compressed_binding(
            device=device,
            mode=mode,
            max_q_rows=rows,
            max_kv_rows=rows * width,
            q=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            attn_sink=attn_sink,
            out=out,
        )
        returned = compressed_sparse_mla.run(
            plan=output_plan,
            binding=output_binding,
            swa_k_cache=swa_cache,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
            out=out,
        )
        assert returned.data_ptr() == out.data_ptr(), mode
        assert not torch.isnan(out.float()).any(), mode
        assert torch.equal(out, baseline), mode

    swa_indices, swa_lengths = _make_swa(512)
    valid_out = torch.empty(
        (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    _, plan, binding = _prepare_compressed_binding(
        device=device,
        mode="extend",
        max_q_rows=rows,
        max_kv_rows=rows * 512,
        q=q,
        swa_k_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        attn_sink=attn_sink,
        out=valid_out,
    )
    bad_shape = torch.empty(
        (rows + 1, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    with pytest.raises(ValueError, match="out must have shape"):
        compressed_sparse_mla.run(
            plan=plan, binding=binding, swa_k_cache=swa_cache,
            attn_sink=attn_sink, sm_scale=_SM_SCALE, out=bad_shape,
        )
    bad_dtype = torch.empty(
        (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
        dtype=torch.float16,
        device=device,
    )
    with pytest.raises(TypeError, match="out must be bfloat16"):
        compressed_sparse_mla.run(
            plan=plan, binding=binding, swa_k_cache=swa_cache,
            attn_sink=attn_sink, sm_scale=_SM_SCALE, out=bad_dtype,
        )
    non_contiguous = torch.empty(
        (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM * 2),
        dtype=torch.bfloat16,
        device=device,
    )[..., ::2]
    with pytest.raises(ValueError, match="out must be contiguous"):
        compressed_sparse_mla.run(
            plan=plan, binding=binding, swa_k_cache=swa_cache,
            attn_sink=attn_sink, sm_scale=_SM_SCALE, out=non_contiguous,
        )
