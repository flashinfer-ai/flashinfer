"""Correctness coverage for paged prefill top-k on crowded score buckets.

``DSATiledTopkKernel``, reached through ``index_topk_fp8`` with the
``packed_contiguous`` route, stores a radix threshold bucket in bounded shared
memory. A bucket wider than that capacity must use the buffer-free exact MSD
radix rescan rather than dropping candidates.

Cases (verified by score, not index, so a miss is genuine and not a tie-break):

* ``seq_len`` sweep: 32768, 32832, and 65536 cover long-context threshold
  buckets on both sides of a page boundary.
* all-equal scores: one coarse+fine bucket holds every token, so the fallback's tie-fill
  branch must fill the whole top-k from a single pivot key.
* logical output (``output_physical_slots=False``): exercises the two-level fold, whose
  level-1 extent-split pass and level-2 ``run_row_topk`` share the same radix class.
* carry fold (``supertile_k`` < context): multi-chunk streaming fold, so the overflow
  fallback runs in the ``is_first=False`` carry path through the virtual value loader.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention.dsa_indexer._impl import clear_indexer_caches
from b12x.attention.dsa_indexer.paged import (
    index_topk_fp8,
    pack_paged_index_k_cache_reference,
    prepare_paged_indexer_metadata,
)
from tests.experimental.b12x.attention.test_paged_indexer_integration import (
    _bind_paged_indexer,
    _make_real_page_table,
    _paged_index_logits,
)

_PAGE = 64
_ROWS = 16
_NUM_HEADS = 32
_TOPK = 2048
_PAGE_START = 3


def _build_scene(
    device: torch.device,
    seq_len: int,
    scores: str,
    *,
    topk: int = _TOPK,
) -> dict:
    """Build a paged prefill scene plus its fp32 reference top-k threshold.

    ``scores="monotonic"`` gives each token a strictly increasing score so the true
    top-k is unambiguous; ``scores="equal"`` gives every token the same score, which
    lands the whole row in one radix bucket (degenerate tie-fill).
    """
    width_blocks = (seq_len + _PAGE - 1) // _PAGE
    total_pages = _PAGE_START + width_blocks + 8

    q_fp8 = torch.full(
        (_ROWS, _NUM_HEADS, 128), 0.5, dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    weights = torch.ones((_ROWS, _NUM_HEADS), dtype=torch.float32, device=device)
    if scores == "monotonic":
        token_scores = torch.linspace(
            0.25, 1.25, seq_len, dtype=torch.float32, device=device
        )
    elif scores == "equal":
        token_scores = torch.full((seq_len,), 0.5, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"unknown scores mode {scores!r}")
    raw_k = torch.zeros((total_pages * _PAGE, 128), dtype=torch.float32, device=device)
    raw_k[_PAGE_START * _PAGE : _PAGE_START * _PAGE + seq_len] = token_scores.unsqueeze(
        1
    ).expand(-1, 128)
    index_k_cache = pack_paged_index_k_cache_reference(raw_k)

    real_page_table = _make_real_page_table(
        page_starts=[_PAGE_START] * _ROWS,
        seqlens=[seq_len] * _ROWS,
        width_blocks=width_blocks,
        device=device,
    )
    shared_page_table = _make_real_page_table(
        page_starts=[_PAGE_START],
        seqlens=[seq_len],
        width_blocks=width_blocks,
        device=device,
    )
    seqlens = torch.full((_ROWS,), seq_len, dtype=torch.int32, device=device)

    ref_logits = _paged_index_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens=seqlens,
    )
    kth_score = torch.topk(ref_logits, topk, dim=1).values[:, -1]
    return {
        "seq_len": seq_len,
        "page_start": _PAGE_START,
        "topk": topk,
        "width_blocks": width_blocks,
        "q_fp8": q_fp8,
        "weights": weights,
        "index_k_cache": index_k_cache,
        "shared_page_table": shared_page_table,
        "seqlens": seqlens,
        "ref_logits": ref_logits,
        "kth_score": kth_score,
    }


def _run_indexer(
    monkeypatch,
    scene: dict,
    *,
    supertile_k: int,
    output_physical_slots: bool,
    route: str = "packed_contiguous",
    graph_replay: bool = False,
) -> torch.Tensor:
    monkeypatch.setenv("B12X_PAGED_INDEX_SUPERTILE_K", str(supertile_k))
    seqlens = scene["seqlens"]
    topk = int(scene["topk"])
    shared_page_table = scene["shared_page_table"]
    binding = _bind_paged_indexer(
        device=scene["q_fp8"].device,
        num_heads=_NUM_HEADS,
        rows=_ROWS,
        width_blocks=scene["width_blocks"],
        topk=topk,
        real_page_table=shared_page_table.expand(_ROWS, -1),
        seqlens=seqlens,
        supertile_k=supertile_k,
        shared_page_table=True,
        route=route,
        output_physical_slots=output_physical_slots,
    )
    prepare_paged_indexer_metadata(
        real_page_table=shared_page_table.expand(_ROWS, -1),
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=_NUM_HEADS,
        build_schedule=False,
        shared_page_table=True,
    )
    selected = torch.empty(
        (_ROWS, topk), dtype=torch.int32, device=scene["q_fp8"].device
    )
    clear_indexer_caches()
    def run():
        return index_topk_fp8(
            q_fp8=scene["q_fp8"],
            weights=scene["weights"].unsqueeze(-1),
            index_k_cache=scene["index_k_cache"],
            topk=topk,
            expected_num_q_heads=_NUM_HEADS,
            binding=binding,
            out_indices=selected,
            supertile_k=supertile_k,
        )

    run()
    device = scene["q_fp8"].device
    torch.cuda.synchronize(device)
    if graph_replay:
        from b12x._lib.runtime_control import kernel_resolution_guard

        with kernel_resolution_guard('paged top-k high-page-ID replay'):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            pointers = (scene["index_k_cache"].data_ptr(), selected.data_ptr())
            for _ in range(2):
                scene["weights"].neg_()
                scene["ref_logits"].neg_()
                scene["kth_score"] = torch.topk(
                    scene["ref_logits"], topk, dim=1,
                ).values[:, -1]
                selected.fill_(-2)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize(device)
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                    assert before[key] == after[key]
                assert pointers == (scene["index_k_cache"].data_ptr(), selected.data_ptr())
                _assert_selects_true_topk(
                    scene, selected, output_physical_slots=output_physical_slots,
                )
    return selected


def _assert_selects_true_topk(
    scene: dict, selected: torch.Tensor, *, output_physical_slots: bool
) -> None:
    seq_len = scene["seq_len"]
    topk = int(scene["topk"])
    if output_physical_slots:
        # The scene maps consecutive physical pages to logical token positions.
        raw_logical = selected.long() - int(scene["page_start"]) * _PAGE
    else:
        # Logical output is already request-relative.
        raw_logical = selected.long()
    # Every selected slot must map to a real in-range token, and a top-k list must not
    # repeat a token. These hold regardless of score, so they -- unlike the score check
    # below -- catch a tie-fill that emits duplicate or out-of-range indices.
    assert bool(((raw_logical >= 0) & (raw_logical < seq_len)).all()), (
        f"selected index outside [0, {seq_len}) at seq_len={seq_len}"
    )
    for r in range(selected.shape[0]):
        row = raw_logical[r].tolist()
        assert len(set(row)) == len(row), (
            f"row {r} selected duplicate indices at seq_len={seq_len}"
        )
    logical = raw_logical.clamp(0, seq_len - 1)
    selected_score = torch.gather(scene["ref_logits"], 1, logical)
    n_below = int((selected_score < (scene["kth_score"][:, None] - 1e-4)).sum().item())
    assert n_below == 0, (
        f"{n_below}/{_ROWS * topk} selected tokens score below the true top-{topk} "
        f"threshold at seq_len={seq_len}"
    )


@pytest.mark.parametrize("seq_len", [32768, 32832, 65536])
def test_paged_prefill_topk_selects_true_topk_at_long_context(
    monkeypatch, seq_len: int
) -> None:
    device = torch.device("cuda")
    scene = _build_scene(device, seq_len, "monotonic")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=32768, output_physical_slots=True
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=True)


def test_paged_prefill_topk_all_equal_scores_tie_fill(monkeypatch) -> None:
    """One bucket holds every token: the fallback must tie-fill the whole top-k."""
    device = torch.device("cuda")
    scene = _build_scene(device, 32768, "equal")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=32768, output_physical_slots=True
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=True)


def test_paged_prefill_topk_512_all_equal_scores_tie_fill(monkeypatch) -> None:
    """The top-k-512 buffer policy preserves exact tie-fill semantics."""
    device = torch.device("cuda")
    scene = _build_scene(device, 8192, "equal", topk=512)
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=8192, output_physical_slots=True
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=True)


def test_paged_prefill_topk_logical_output_two_level_fold(monkeypatch) -> None:
    """output_physical_slots=False routes through the two-level (extent-split + row) fold."""
    device = torch.device("cuda")
    scene = _build_scene(device, 32768, "monotonic")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=32768, output_physical_slots=False
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=False)


def test_paged_prefill_topk_logical_output_adaptive_carry(monkeypatch) -> None:
    """A zero candidate budget selects the exact carry path for logical output."""
    monkeypatch.setenv("B12X_INDEXER_TWO_LEVEL_FOLD", "auto")
    monkeypatch.setenv("B12X_INDEXER_TWO_LEVEL_FOLD_MAX_MIB", "0")
    device = torch.device("cuda")
    scene = _build_scene(device, 32768, "monotonic")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=8192, output_physical_slots=False
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=False)


def test_paged_prefill_topk_carry_fold_overflow(monkeypatch) -> None:
    """supertile_k < context forces the multi-chunk carry fold (is_first=False path).

    All-equal scores put each 8192-token local chunk in one bucket. The first chunk
    exactly fills the 8192-slot buffer; the next fold adds the carried 2048 winners and
    therefore overflows, exercising the virtual (carry-aware) exact fallback.
    """
    device = torch.device("cuda")
    scene = _build_scene(device, 32768, "equal")
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=8192, output_physical_slots=True
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=True)


def test_paged_prefill_topk_width_does_not_recompile(monkeypatch) -> None:
    """The page-table width must not be part of the tiled top-k compile cache key.

    output_page_table is indexed at runtime as output_page_table[row, page_col]
    (page_col = gidx // output_page_size), so neither shape[1] (the live KV-cache
    page-table width, which grows with sequence/capacity) nor stride(0) (the same
    width for a contiguous page table) is a compile-time input. Pinning either
    into the key re-links a fresh cubin on every capacity bump, per TP rank, on
    the first serving run.

    Two consecutive prefill runs at distinct page-table widths must compile the
    tiled top-k kernel exactly once: the second width reuses the first's cubin.
    """
    from b12x._lib.compiler import clear_compile_cache, compile_cache_info

    device = torch.device("cuda")
    scene_narrow = _build_scene(device, 16384, "monotonic")
    scene_wide = _build_scene(device, 32768, "monotonic")
    width_narrow = scene_narrow["width_blocks"]
    width_wide = scene_wide["width_blocks"]
    assert width_narrow != width_wide, (
        f"test requires distinct page-table widths, got {width_narrow} and {width_wide}"
    )

    # Force every cache key to actually compile (skip the on-disk cache) but keep
    # the in-memory cache on so a repeated identical key is a hit, not a recompile.
    monkeypatch.setenv("B12X_COMPILE_DISK_CACHE", "0")
    monkeypatch.setenv("B12X_COMPILE_MEMORY_CACHE", "1")
    clear_compile_cache()

    selected_narrow = _run_indexer(
        monkeypatch, scene_narrow, supertile_k=32768, output_physical_slots=True
    )
    misses_after_narrow = compile_cache_info()["compile_misses"]
    assert misses_after_narrow >= 1, (
        "run A compiled no kernels; the test is not exercising the topk compile path"
    )

    selected_wide = _run_indexer(
        monkeypatch, scene_wide, supertile_k=32768, output_physical_slots=True
    )
    misses_after_wide = compile_cache_info()["compile_misses"]

    assert misses_after_wide == misses_after_narrow, (
        f"page-table width leaked into the compile cache key: width "
        f"{width_narrow} -> {width_wide} triggered "
        f"{misses_after_wide - misses_after_narrow} extra compile(s); "
        f"misses {misses_after_narrow} -> {misses_after_wide}"
    )
    _assert_selects_true_topk(scene_narrow, selected_narrow, output_physical_slots=True)
    _assert_selects_true_topk(scene_wide, selected_wide, output_physical_slots=True)


@pytest.mark.parametrize("output_physical_slots", (False, True))
def test_paged_topk_512_high_page_ids_graph_replay(
    monkeypatch, output_physical_slots: bool,
) -> None:
    """Read live packed cache pages past the signed 32-bit byte-offset limit."""
    device = torch.device("cuda")
    scene = _build_scene(device, 8192, "monotonic", topk=512)
    packed = scene["index_k_cache"]
    page_stride = packed.stride(0) * packed.element_size()
    page_start = (1 << 31) // page_stride + 1
    live_pages = scene["width_blocks"]
    pool = torch.empty(
        (page_start + live_pages, packed.shape[1]), dtype=packed.dtype, device=device,
    )
    pool[page_start:].copy_(packed[_PAGE_START : _PAGE_START + live_pages])
    scene["index_k_cache"] = pool
    scene["shared_page_table"].add_(page_start - _PAGE_START)
    scene["page_start"] = page_start
    assert page_start * page_stride > (1 << 31)
    assert (page_start + live_pages) * _PAGE < torch.iinfo(torch.int32).max
    # Relocating identical packed pages preserves the small scene's score oracle.
    selected = _run_indexer(
        monkeypatch, scene, supertile_k=4096,
        output_physical_slots=output_physical_slots, route="paged_tiled",
        graph_replay=True,
    )
    _assert_selects_true_topk(scene, selected, output_physical_slots=output_physical_slots)
