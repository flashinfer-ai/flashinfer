from __future__ import annotations

import math

import pytest
import torch

from flashinfer.experimental.sm12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_MLA_C128_PAGE_SIZE,
    COMPRESSED_MLA_DSV4_PAGE_SIZE,
    compressed_sparse_mla_reference,
    pack_compressed_mla_kv_cache_reference,
)
from flashinfer.experimental.sm12x.attention import compressed_mla

SM12XCompressedMLAScratchCaps = compressed_mla.Caps
clear_mla_caches = compressed_mla.clear_caches
compressed_mla_decode_forward = compressed_mla.run
plan_compressed_mla_scratch = compressed_mla.plan

from ..conftest import require_sm12x as require_sm120


_COMPRESSED_HEAD_DIM = 512
_SHARED_CORE_HEAD_DIM = 576
_SHARED_CORE_V_HEAD_DIM = 512
_LOCAL_Q_HEADS = 32
_SM_SCALE = 1.0 / math.sqrt(_COMPRESSED_HEAD_DIM)


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
):
    plan = plan_compressed_mla_scratch(
        SM12XCompressedMLAScratchCaps(
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
    return pack_compressed_mla_kv_cache_reference(
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


@torch.inference_mode()
def test_compressed_mla_shared_core_replays_under_cuda_graph() -> None:
    device = require_sm120()
    clear_mla_caches()

    q = _make_q(rows=1, seed=21, device=device)
    swa_cache_bytes = _make_cache(
        tokens=32, page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE, seed=22, device=device
    )
    indexed_cache_bytes = _make_cache(
        tokens=32, page_size=COMPRESSED_MLA_C128_PAGE_SIZE, seed=23, device=device
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
        captured_out = compressed_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            indexed_k_cache=indexed_cache,
            indexed_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
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
        extra_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
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
        extra_page_size=COMPRESSED_MLA_C128_PAGE_SIZE,
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
def test_compressed_mla_out_param_writes_directly_and_matches() -> None:
    device = require_sm120()
    clear_mla_caches()

    rows = 8
    q = _make_q(rows=rows, seed=311, device=device)
    swa_cache = _make_cache(
        tokens=32, page_size=COMPRESSED_MLA_DSV4_PAGE_SIZE, seed=312, device=device
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
        )
        binding.scratch.mode = mode
        baseline = compressed_mla_decode_forward(
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
        returned = compressed_mla_decode_forward(
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
    )
    binding.scratch.mode = "extend"
    bad_shape = torch.empty(
        (rows + 1, _LOCAL_Q_HEADS, _COMPRESSED_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    with pytest.raises(ValueError, match="out must have shape"):
        compressed_mla_decode_forward(
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
        compressed_mla_decode_forward(
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
        compressed_mla_decode_forward(
            swa_k_cache=swa_cache,
            binding=binding,
            attn_sink=attn_sink,
            sm_scale=_SM_SCALE,
            out=non_contiguous,
        )
