from __future__ import annotations

import math

import pytest
import torch

import b12x.attention._shared.mla.compressed_api as compressed_api_impl
import b12x.attention._shared.mla.merge as mla_split_impl
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.attention._shared.workspace import (
    B12XAttentionArena,
    B12XAttentionArenaCaps,
)
from b12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
    COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
    COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
    COMPRESSED_SPARSE_MLA_SWA_TOKENS,
    compressed_sparse_mla_page_nbytes,
    compressed_sparse_mla_reference,
    gather_compressed_sparse_mla_kv_cache_reference,
    pack_compressed_sparse_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.api import clear_mla_caches
from b12x.attention._shared.mla.compressed_api import (
    compressed_sparse_mla_decode_forward,
)
from b12x.attention._shared.mla.compressed_config import (
    compressed_sparse_mla_split_chunks_for_contract,
    compressed_sparse_mla_split_config_for_contract,
)
from b12x.attention._shared.mla.kernel import _dsv4_h16_auto
from b12x.attention.compressed_sparse_mla._scratch import (
    B12XCompressedSparseMLAScratchCaps,
)
from b12x.attention.compressed_sparse_mla import (
    plan as plan_compressed_sparse_mla_scratch,
)
from b12x._lib.compiler import clear_compile_cache, compile_cache_info

from b12x.testing.reference.helpers import require_b12x


_COMPRESSED_HEAD_DIM = 512
_SHARED_CORE_HEAD_DIM = 576
_SHARED_CORE_V_HEAD_DIM = 512
_LOCAL_Q_HEADS = 32
_SM_SCALE = 1.0 / math.sqrt(_COMPRESSED_HEAD_DIM)


def test_spark_h16_policy_keeps_swa_h8_and_promotes_c4() -> None:
    assert not _dsv4_h16_auto(
        rows=8,
        heads=32,
        num_chunks=2,
        h8_num_splits=1,
        sm_count=48,
    )
    assert _dsv4_h16_auto(
        rows=8,
        heads=32,
        num_chunks=10,
        h8_num_splits=3,
        sm_count=48,
    )
    assert not _dsv4_h16_auto(
        rows=8,
        heads=32,
        num_chunks=10,
        h8_num_splits=3,
        sm_count=188,
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


@torch.inference_mode()
def test_split_sink_merge_live_rows_do_not_resolve_new_kernel() -> None:
    device = require_b12x()
    clear_compile_cache()
    mla_split_impl.clear_sparse_mla_merge_kernel_cache()

    warm_args = _make_split_merge_tensors(
        rows=128,
        heads=2,
        chunks=2,
        device=device,
        seed=6120,
    )
    mla_split_impl.run_sparse_mla_split_decode_merge(
        tmp_output=warm_args[0],
        tmp_lse=warm_args[1],
        num_chunks_ptr=warm_args[2],
        attn_sink=warm_args[3],
        output=warm_args[4],
    )
    torch.cuda.synchronize(device)
    warm_misses = compile_cache_info()["compile_misses"]

    live_args = _make_split_merge_tensors(
        rows=113,
        heads=2,
        chunks=1,
        device=device,
        seed=6121,
    )
    with kernel_resolution_guard(
        "split sink merge live rows and chunks should reuse padded capture"
    ):
        mla_split_impl.run_sparse_mla_split_decode_merge(
            tmp_output=live_args[0],
            tmp_lse=live_args[1],
            num_chunks_ptr=live_args[2],
            attn_sink=live_args[3],
            output=live_args[4],
        )
        torch.cuda.synchronize(device)

    assert compile_cache_info()["compile_misses"] == warm_misses
    assert torch.isfinite(live_args[4].float()).all()


def _make_compressed_binding(
    *,
    device: torch.device | str,
    rows: int,
    topk: int,
    max_kv_rows: int,
    q: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_indices: torch.Tensor | None = None,
    indexed_lengths: torch.Tensor | None = None,
    indexed_page_table: torch.Tensor | None = None,
    use_cuda_graph: bool = False,
    head_dim: int = _COMPRESSED_HEAD_DIM,
    v_head_dim: int = _COMPRESSED_HEAD_DIM,
    max_chunks_per_row: int = 64,
    max_page_table_width: int | None = None,
    decode_row_capacity: int | None = None,
    mode: str = "decode",
):
    plan = plan_compressed_sparse_mla_scratch(
        B12XCompressedSparseMLAScratchCaps(
            device=device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=_LOCAL_Q_HEADS,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            max_width=topk,
            max_page_table_width=max_page_table_width,
            max_q_rows=rows,
            max_batch=rows,
            max_kv_rows=max_kv_rows,
            max_chunks_per_row=max_chunks_per_row,
            mode=mode,
            decode_row_capacity=decode_row_capacity,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = plan.bind(
        scratch=scratch,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
    )
    binding.scratch.use_cuda_graph = bool(use_cuda_graph)
    return binding


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


def test_compressed_sparse_mla_page_byte_widths_match_padded_layout() -> None:
    assert COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE == 256
    assert COMPRESSED_SPARSE_MLA_SWA_TOKENS == 128
    assert (
        COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE == COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE // 4
    )
    assert (
        COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE
        == COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE // 128
    )
    assert (
        compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE)
        == 149760
    )
    assert (
        compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE) == 37440
    )
    assert (
        compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE) == 1728
    )


def test_compressed_sparse_mla_mtp_graph_rows_keep_decode_split_contract() -> None:
    graph_rows_cfg = compressed_sparse_mla_split_config_for_contract(
        rows=192,
        width=384,
        max_chunks=6,
    )
    larger_prefill_cfg = compressed_sparse_mla_split_config_for_contract(
        rows=257,
        width=384,
        max_chunks=6,
    )

    assert graph_rows_cfg.chunk_size == 64
    assert graph_rows_cfg.num_chunks == 6
    assert larger_prefill_cfg.chunk_size == 1024
    assert larger_prefill_cfg.num_chunks == 1


@pytest.mark.parametrize(
    ("decode_row_capacity", "last_decode_row"),
    [
        pytest.param(144, 256, id="legacy-floor"),
        pytest.param(384, 384, id="mns64-k5"),
        pytest.param(512, 512, id="mns64-k7"),
        pytest.param(768, 768, id="mns128-k5"),
    ],
)
def test_compressed_sparse_mla_declared_decode_row_capacity_extends_contract(
    decode_row_capacity: int,
    last_decode_row: int,
) -> None:
    decode_cfg = compressed_sparse_mla_split_config_for_contract(
        rows=last_decode_row,
        width=384,
        max_chunks=6,
        decode_row_capacity=decode_row_capacity,
    )
    next_cfg = compressed_sparse_mla_split_config_for_contract(
        rows=last_decode_row + 1,
        width=384,
        max_chunks=6,
        decode_row_capacity=decode_row_capacity,
    )

    assert decode_cfg.chunk_size == 64
    assert decode_cfg.num_chunks == 6
    assert next_cfg.chunk_size == 1024
    assert next_cfg.num_chunks == 1


def test_compressed_sparse_mla_declared_decode_row_capacity_rejects_nonpositive() -> (
    None
):
    with pytest.raises(ValueError, match="decode_row_capacity must be positive"):
        compressed_sparse_mla_split_config_for_contract(
            rows=1,
            width=384,
            max_chunks=6,
            decode_row_capacity=0,
        )


@pytest.mark.parametrize("rows", [1, 24, 65, 144, 192, 256])
@pytest.mark.parametrize("width", [128, 384, 2176, 4608])
@pytest.mark.parametrize("decode_row_capacity", [144, 384, 768])
def test_declared_capacity_preserves_legacy_plans_through_256_rows(
    rows: int,
    width: int,
    decode_row_capacity: int,
) -> None:
    legacy = compressed_sparse_mla_split_config_for_contract(
        rows=rows,
        width=width,
    )
    declared = compressed_sparse_mla_split_config_for_contract(
        rows=rows,
        width=width,
        decode_row_capacity=decode_row_capacity,
    )

    assert declared == legacy


@pytest.mark.parametrize("rows", [384, 512, 768])
@torch.inference_mode()
def test_declared_decode_capacity_matches_reference_under_graph_replay(
    rows: int,
) -> None:
    device = require_b12x()
    clear_mla_caches()
    width = 384
    live_width = 32
    q = _make_q(rows=rows, seed=6130 + rows, device=device)
    swa_cache_bytes = _make_cache(
        tokens=64,
        page_size=COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
        seed=6131 + rows,
        device=device,
    )
    swa_cache = swa_cache_bytes.view(torch.float8_e4m3fn)
    base_indices = torch.arange(live_width, dtype=torch.int32, device=device)
    swa_indices = torch.full(
        (rows, width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    swa_indices[:, :live_width] = base_indices
    swa_lengths = torch.full(
        (rows,),
        live_width,
        dtype=torch.int32,
        device=device,
    )
    attn_sink = torch.linspace(
        -0.1,
        0.1,
        _LOCAL_Q_HEADS,
        dtype=torch.float32,
        device=device,
    )
    split_cap = math.ceil(width / 64)
    binding = _make_compressed_binding(
        device=device,
        rows=rows,
        topk=width,
        max_kv_rows=rows * width,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        use_cuda_graph=True,
        max_chunks_per_row=split_cap,
        decode_row_capacity=rows,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
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
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1),
        expected.float().reshape(-1),
        dim=0,
    )
    assert max_abs <= 0.10
    assert cosine.item() >= 0.9995


def test_compressed_sparse_mla_arena_scratch_uses_contract_q_chunks() -> None:
    device = require_b12x()
    selected_widths = (128, 640, 2880)
    graph_q_rows = 16
    compressed_prefill_q = 8192
    max_chunks_per_row = max(
        compressed_sparse_mla_split_chunks_for_contract(rows=graph_q_rows, width=width)
        for width in selected_widths
    )
    mla_max_q_chunks = max(
        max(graph_q_rows, compressed_prefill_q)
        * compressed_sparse_mla_split_chunks_for_contract(
            rows=max(graph_q_rows, compressed_prefill_q),
            width=width,
        )
        for width in selected_widths
    )
    decode_q_chunks = graph_q_rows * max_chunks_per_row
    mla_max_q_chunks = max(mla_max_q_chunks, decode_q_chunks)

    base_caps = dict(
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=_LOCAL_Q_HEADS,
        indexer_num_q_heads=64,
        head_dim=_COMPRESSED_HEAD_DIM,
        max_v_head_dim=_COMPRESSED_HEAD_DIM,
        topk=max(selected_widths),
        indexer_topk=512,
        max_page_table_width=4160,
        extend_max_total_q=compressed_prefill_q,
        extend_max_batch=compressed_prefill_q,
        extend_max_kv_rows=0,
        paged_max_q_rows=4096,
        paged_max_batch=4096,
        mla_max_total_q=max(graph_q_rows, compressed_prefill_q),
        page_size=64,
        max_chunks_per_row=max_chunks_per_row,
        reserve_extend_indexer_logits=False,
        reserve_paged_indexer_logits=True,
        reserve_compressed_sparse_mla_staging=True,
        reserve_mhc=True,
        mhc_max_tokens=4096,
        mhc_hidden_size=4096,
        paged_indexer_logits_q_rows=graph_q_rows,
        paged_indexer_logits_k_rows=4160 * 64,
        paged_indexer_tile_logits_k_rows=32768,
    )
    capped = B12XAttentionArena.required_nbytes(
        B12XAttentionArenaCaps(**base_caps, mla_max_q_chunks=mla_max_q_chunks)
    )
    layout = B12XAttentionArena._layout(
        B12XAttentionArenaCaps(**base_caps, mla_max_q_chunks=mla_max_q_chunks)
    )
    legacy_ragged = B12XAttentionArena.required_nbytes(
        B12XAttentionArenaCaps(
            **{
                **base_caps,
                "extend_max_kv_rows": compressed_prefill_q * max(selected_widths),
            },
            mla_max_q_chunks=mla_max_q_chunks,
        )
    )

    assert max_chunks_per_row == 240
    assert layout.ragged_kv_nbytes <= 1024
    assert layout.output_buffer_nbytes == 0
    assert layout.compressed_q_stage_nbytes > 0
    assert layout.compressed_index_stage_nbytes > 0
    assert legacy_ragged > capped * 3
    assert capped < int(2.25 * (1 << 30))


def test_compressed_sparse_mla_reference_pack_gathers_across_padded_pages() -> None:
    device = require_b12x()
    gen = torch.Generator(device=device)
    gen.manual_seed(31)

    for page_size in (
        COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
        COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
    ):
        tokens = page_size * 2 + 1
        k_nope = (
            torch.randn(
                (tokens, 448), generator=gen, dtype=torch.float32, device=device
            )
            * 0.05
        )
        k_rope = (
            torch.randn((tokens, 64), generator=gen, dtype=torch.float32, device=device)
            * 0.05
        ).to(torch.bfloat16)
        cache = pack_compressed_sparse_mla_kv_cache_reference(
            k_nope, k_rope, page_size=page_size
        )
        indices = torch.tensor(
            [0, page_size - 1, page_size, page_size + 1, tokens - 1],
            dtype=torch.int32,
            device=device,
        )

        gathered, _ = gather_compressed_sparse_mla_kv_cache_reference(
            cache, indices, page_size=page_size
        )
        expected_rope = k_rope[indices.to(torch.long)].float()
        assert torch.count_nonzero(gathered[2:]).item() > 0
        torch.testing.assert_close(gathered[:, 448:], expected_rope, atol=0, rtol=0)
        torch.testing.assert_close(
            gathered[:, :448], k_nope[indices.to(torch.long)], atol=0.01, rtol=0.12
        )


@torch.inference_mode()
def test_compressed_sparse_mla_fixed_workspace_split_plan_uses_contract_not_live_shape(
    monkeypatch,
) -> None:
    device = require_b12x()
    clear_mla_caches()
    del monkeypatch

    contract_rows = 128
    contract_width = 2304
    live_rows = 1
    live_width = 512
    page_table_width = 4097
    max_chunks_per_row = compressed_sparse_mla_split_chunks_for_contract(
        rows=live_rows,
        width=contract_width,
    )
    q = _make_q(rows=live_rows, seed=121, device=device)
    swa_cache = torch.empty(
        (1, compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    swa_indices = torch.zeros((live_rows, live_width), dtype=torch.int32, device=device)
    swa_lengths = torch.zeros((live_rows,), dtype=torch.int32, device=device)
    indexed_cache = torch.empty(
        (1, compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    indexed_indices = torch.full(
        (live_rows, live_width), -1, dtype=torch.int32, device=device
    )
    indexed_lengths = torch.zeros((live_rows,), dtype=torch.int32, device=device)
    indexed_page_table = torch.full(
        (live_rows, page_table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    binding = _make_compressed_binding(
        device=device,
        rows=contract_rows,
        topk=contract_width,
        max_kv_rows=1,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
        use_cuda_graph=True,
        max_chunks_per_row=max_chunks_per_row,
        max_page_table_width=page_table_width,
    )

    with (
        torch.inference_mode(),
        torch.no_grad(),
        pytest.raises(ValueError, match="mapped indexed_page_table"),
    ):
        compressed_api_impl.compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache.view(torch.float8_e4m3fn),
            binding=binding,
            indexed_k_cache=indexed_cache,
            indexed_page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
            sm_scale=_SM_SCALE,
        )


@torch.inference_mode()
def test_compressed_sparse_mla_shared_core_replays_under_cuda_graph() -> None:
    device = require_b12x()
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
    binding = _make_compressed_binding(
        device=device,
        rows=8,
        topk=swa_indices.shape[1] + indexed_indices.shape[1],
        max_kv_rows=8 * (swa_indices.shape[1] + indexed_indices.shape[1]),
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
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
def test_compressed_sparse_mla_c128_pv_row_swizzle_replays_under_cuda_graph() -> None:
    device = require_b12x()
    clear_mla_caches()

    width = 32
    q = torch.zeros(
        (1, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    k_nope = torch.zeros((width, 448), dtype=torch.bfloat16, device=device)
    k_nope[20, 0] = 1
    k_rope = torch.zeros((width, 64), dtype=torch.bfloat16, device=device)
    swa_cache = torch.empty(
        (0, compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    indexed_cache = pack_compressed_sparse_mla_kv_cache_reference(
        k_nope,
        k_rope,
        page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
    )
    swa_indices = torch.empty((1, 0), dtype=torch.int32, device=device)
    indexed_indices = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
    swa_lengths = torch.zeros((1,), dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([width], dtype=torch.int32, device=device)
    binding = _make_compressed_binding(
        device=device,
        rows=1,
        topk=width,
        max_kv_rows=width,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            indexed_k_cache=indexed_cache,
            indexed_page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
            sm_scale=1.0,
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

    expected = torch.zeros_like(captured_out.float())
    expected[:, :, 0] = 1.0 / width
    max_abs = (captured_out.float() - expected).abs().max().item()
    assert max_abs <= 1e-4


@torch.inference_mode()
def test_compressed_sparse_mla_swa_page_size_256_replays_under_cuda_graph() -> None:
    device = require_b12x()
    clear_mla_caches()

    swa_page_size = 256
    q = _make_q(rows=1, seed=91, device=device)
    swa_cache = _make_cache(tokens=300, page_size=swa_page_size, seed=92, device=device)
    swa_indices = torch.tensor(
        [[126, 127, 128, 129, 130, 255, 256, 257]],
        dtype=torch.int32,
        device=device,
    )
    swa_lengths = torch.tensor([8], dtype=torch.int32, device=device)
    attn_sink = torch.linspace(
        -0.08, 0.12, _LOCAL_Q_HEADS, dtype=torch.float32, device=device
    )
    binding = _make_compressed_binding(
        device=device,
        rows=q.shape[0],
        topk=swa_indices.shape[1],
        max_kv_rows=q.shape[0] * swa_indices.shape[1],
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            swa_page_size=swa_page_size,
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
        swa_cache,
        swa_indices,
        swa_lengths,
        swa_page_size=swa_page_size,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_sparse_mla_prefill_swa_only_replays_under_cuda_graph() -> None:
    device = require_b12x()
    clear_mla_caches()

    rows = 8
    width = 8
    q = _make_q(rows=rows, seed=81, device=device)
    swa_cache = _make_cache(
        tokens=32,
        page_size=COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
        seed=82,
        device=device,
    )
    swa_indices = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    swa_lengths = torch.empty((rows,), dtype=torch.int32, device=device)
    for row in range(rows):
        length = min(width, row + 1)
        swa_indices[row, :length] = torch.arange(
            row, row - length, -1, dtype=torch.int32, device=device
        )
        swa_lengths[row] = length
    attn_sink = torch.linspace(
        -0.2, 0.15, _LOCAL_Q_HEADS, dtype=torch.float32, device=device
    )
    binding = _make_compressed_binding(
        device=device,
        rows=rows,
        topk=width,
        max_kv_rows=rows * width,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
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
        swa_cache,
        swa_indices,
        swa_lengths,
        attn_sink=attn_sink,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_sparse_mla_clamp_to_one_negative_extra_replays_under_cuda_graph() -> (
    None
):
    device = require_b12x()
    clear_mla_caches()

    q = _make_q(rows=1, seed=71, device=device)
    swa_cache = _make_cache(
        tokens=32,
        page_size=COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
        seed=72,
        device=device,
    )
    indexed_cache = _make_cache(
        tokens=4, page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE, seed=73, device=device
    )
    swa_indices = torch.arange(8, dtype=torch.int32, device=device).unsqueeze(0)
    indexed_indices = torch.full((1, 4), -1, dtype=torch.int32, device=device)
    swa_lengths = torch.tensor([6], dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([1], dtype=torch.int32, device=device)
    binding = _make_compressed_binding(
        device=device,
        rows=q.shape[0],
        topk=swa_indices.shape[1] + indexed_indices.shape[1],
        max_kv_rows=q.shape[0] * (swa_indices.shape[1] + indexed_indices.shape[1]),
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        use_cuda_graph=True,
    )

    captured_out: torch.Tensor | None = None

    def run() -> torch.Tensor:
        nonlocal captured_out
        captured_out = compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            indexed_k_cache=indexed_cache,
            indexed_page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
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
        swa_cache,
        swa_indices,
        swa_lengths,
        extra_k_cache=indexed_cache,
        extra_indices=indexed_indices,
        extra_topk_lengths=indexed_lengths,
        extra_page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
        sm_scale=_SM_SCALE,
    )
    max_abs = (captured_out.float() - expected.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        captured_out.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert max_abs <= 0.10
    assert cos.item() >= 0.9995


@torch.inference_mode()
def test_compressed_sparse_mla_mapped_page_table_is_rejected() -> None:
    device = require_b12x()
    clear_mla_caches()

    q = _make_q(rows=1, seed=181, device=device)
    swa_cache = torch.empty(
        (0, compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    indexed_cache = _make_cache(
        tokens=64,
        page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
        seed=182,
        device=device,
    )
    swa_indices = torch.empty((1, 0), dtype=torch.int32, device=device)
    swa_lengths = torch.zeros((1,), dtype=torch.int32, device=device)
    indexed_lengths = torch.tensor([8], dtype=torch.int32, device=device)
    binding = _make_compressed_binding(
        device=device,
        rows=1,
        topk=8,
        max_kv_rows=8,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=torch.arange(8, dtype=torch.int32, device=device).unsqueeze(0),
        indexed_lengths=indexed_lengths,
        indexed_page_table=torch.arange(2, dtype=torch.int32, device=device).unsqueeze(
            0
        ),
        use_cuda_graph=True,
    )

    with pytest.raises(ValueError, match="mapped indexed_page_table"):
        compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            indexed_k_cache=indexed_cache,
            indexed_page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
            sm_scale=_SM_SCALE,
        )


@torch.inference_mode()
def test_compressed_sparse_mla_rejects_row_shared_mapped_page_table() -> None:
    device = require_b12x()
    clear_mla_caches()

    rows = 3
    q = _make_q(rows=rows, seed=186, device=device)
    swa_cache = torch.empty(
        (0, compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )
    indexed_cache = _make_cache(
        tokens=128,
        page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
        seed=187,
        device=device,
    )
    swa_indices = torch.empty((rows, 0), dtype=torch.int32, device=device)
    swa_lengths = torch.zeros((rows,), dtype=torch.int32, device=device)
    indexed_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
        ],
        dtype=torch.int32,
        device=device,
    )
    indexed_lengths = torch.full(
        (rows,), indexed_indices.shape[1], dtype=torch.int32, device=device
    )
    indexed_page_table = (
        torch.arange(2, dtype=torch.int32, device=device).unsqueeze(0).expand(rows, -1)
    )
    binding = _make_compressed_binding(
        device=device,
        rows=rows,
        topk=indexed_indices.shape[1],
        max_kv_rows=rows * indexed_indices.shape[1],
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_table=indexed_page_table,
        use_cuda_graph=True,
    )

    with pytest.raises(ValueError, match="mapped indexed_page_table"):
        compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            indexed_k_cache=indexed_cache,
            indexed_page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
            sm_scale=_SM_SCALE,
        )


@torch.inference_mode()
def test_compressed_sparse_mla_rejects_live_mapped_page_table() -> None:
    device = require_b12x()
    clear_mla_caches()

    indexed_cache = _make_cache(
        tokens=64,
        page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
        seed=192,
        device=device,
    )
    swa_cache = torch.empty(
        (0, compressed_sparse_mla_page_nbytes(COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE)),
        dtype=torch.uint8,
        device=device,
    )

    q3 = _make_q(rows=3, seed=193, device=device)
    swa_indices3 = torch.empty((3, 0), dtype=torch.int32, device=device)
    swa_lengths3 = torch.zeros((3,), dtype=torch.int32, device=device)
    indexed_indices3 = torch.arange(8, dtype=torch.int32, device=device).repeat(3, 1)
    indexed_lengths3 = torch.full((3,), 8, dtype=torch.int32, device=device)
    page_table3 = torch.arange(2, dtype=torch.int32, device=device).repeat(3, 1)
    binding3 = _make_compressed_binding(
        device=device,
        rows=3,
        topk=8,
        max_kv_rows=8,
        q=q3,
        swa_indices=swa_indices3,
        swa_lengths=swa_lengths3,
        indexed_indices=indexed_indices3,
        indexed_lengths=indexed_lengths3,
        indexed_page_table=page_table3,
        use_cuda_graph=False,
    )
    with pytest.raises(ValueError, match="mapped indexed_page_table"):
        compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding3,
            indexed_k_cache=indexed_cache,
            indexed_page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
            sm_scale=_SM_SCALE,
        )


@torch.inference_mode()
def test_compressed_sparse_mla_out_param_writes_directly_and_matches() -> None:
    device = require_b12x()
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
    # decode has no such floor.
    for mode, width in (("decode", 8), ("extend", 512)):
        swa_indices, swa_lengths = _make_swa(width)
        binding = _make_compressed_binding(
            device=device,
            rows=rows,
            topk=width,
            max_kv_rows=rows * width,
            q=q,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            mode=mode,
        )
        baseline = compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
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
        returned = compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
            out=out,
        )
        assert returned.data_ptr() == out.data_ptr(), mode
        assert not torch.isnan(out.float()).any(), mode
        assert torch.equal(out, baseline), mode

    swa_indices, swa_lengths = _make_swa(512)
    binding = _make_compressed_binding(
        device=device,
        rows=rows,
        topk=512,
        max_kv_rows=rows * 512,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        mode="extend",
    )
    bad_shape = torch.empty(
        (rows + 1, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    with pytest.raises(ValueError, match="out must have shape"):
        compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
            out=bad_shape,
        )
    bad_dtype = torch.empty(
        (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
        dtype=torch.float16,
        device=device,
    )
    with pytest.raises(TypeError, match="out must be bfloat16"):
        compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
            out=bad_dtype,
        )
    non_contiguous = torch.empty(
        (rows, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM * 2),
        dtype=torch.bfloat16,
        device=device,
    )[..., ::2]
    with pytest.raises(ValueError, match="out must be contiguous"):
        compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
            out=non_contiguous,
        )


@torch.inference_mode()
def test_compressed_sparse_mla_prefill_is_run_to_run_deterministic() -> None:
    """Guards the s4_finalize_row_sum_mg2 scratch-reuse barrier: the epilogue's
    persistent-max reads must complete before the row-sum reduction overwrites
    the scratch. Without it, outputs wobble run-to-run (worst for short
    topk_lengths) and depend on unrelated memory traffic."""
    device = require_b12x()
    clear_mla_caches()

    rows, width = 8, 512
    q = _make_q(rows=rows, seed=411, device=device)
    swa_cache = _make_cache(
        tokens=64,
        page_size=COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
        seed=412,
        device=device,
    )
    swa_indices = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    swa_lengths = torch.empty((rows,), dtype=torch.int32, device=device)
    for row in range(rows):
        length = min(width, row + 1)
        swa_indices[row, :length] = (
            torch.arange(row, row - length, -1, dtype=torch.int32, device=device) % 64
        )
        swa_lengths[row] = length
    attn_sink = torch.linspace(
        -0.2, 0.15, _LOCAL_Q_HEADS, dtype=torch.float32, device=device
    )
    binding = _make_compressed_binding(
        device=device,
        rows=rows,
        topk=width,
        max_kv_rows=rows * width,
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        mode="extend",
    )

    def call() -> torch.Tensor:
        return compressed_sparse_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
        ).clone()

    base = call()
    # Dirty L2/DRAM between runs; a stale/racing read would surface as a
    # run-to-run difference.
    garbage = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=device)
    for _ in range(10):
        garbage.uniform_(-1e30, 1e30)
        assert torch.equal(call(), base)
