"""Tests for the SM120 CUTLASS SVDQuant backend (``backend="cutlass-sm120"``).

Split out of ``test_nvfp4_svdquant_gemm.py``: that file is upstream's and covers
the ``cutlass``, ``cute-dsl`` and ``cute-dsl-unfused`` backends. What is here is
what only this backend has -- the runtime tactic table, the row-class and
geometry equivalences, the LoRA shared-memory swizzle, and the CUDA-graph
capture guards over them.

Not carried over from the merge: seven of upstream's shared tests had SM120
variants in the source repo, gated on ``compute_capability == (12, 0)`` rather
than on a backend name. That gate no longer decides anything now that this
backend is reached by name, so those variants would have exercised
``"cute-dsl"`` instead. Rewriting them to pass ``backend="cutlass-sm120"`` is
open work; until then this backend's rank and reference coverage comes from
``test_svdquant_sm120_linear_routing.py`` and its siblings.
"""

import pytest
import torch
from flashinfer import autotune, mm_nvfp4_svdquant
from flashinfer.gemm.gemm_svdquant import (
    DEFAULT_WORKSPACE_SIZE,
    SVDQUANT_LORA_RANK_GRANULARITY,
)
from flashinfer.gemm.svdquant_sm120_cutlass import (
    _get_nvfp4_svdquant_module_for_device,
    _svdquant_backend_for_capability,
)
from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES
from flashinfer.utils import device_support_pdl, get_compute_capability

# Shared fixtures live in upstream's SVDQuant test module; this file only adds
# the cases specific to the SM120 CUTLASS backend.
from .test_nvfp4_svdquant_gemm import _RANK, _make_gemm_problem, _sqnr_db


def _compute_capability():
    return get_compute_capability(torch.device(device="cuda"))


def _is_sm120():
    return _compute_capability() == (12, 0)


def _skip_unless_svdquant_supported():
    try:
        _svdquant_backend_for_capability(*_compute_capability())
    except ValueError as exc:
        pytest.skip(str(exc))


def _skip_rank_unless_supported(rank: int):
    """SM120 supports LoRA rank 32 only; SM100 supports multiples of 32."""
    if _is_sm120() and rank != SVDQUANT_LORA_RANK_GRANULARITY:
        pytest.skip(f"SM120 SVDQuant supports rank 32 only (requested rank {rank}).")


def _svdquant_module():
    return _get_nvfp4_svdquant_module_for_device(torch.device(device="cuda"))


def test_sm120_fallback_tactic_is_first_feasible_row():
    """The native -1 fallback resolves through feasibility, not a fixed row."""
    _skip_unless_svdquant_supported()
    if not _is_sm120():
        pytest.skip("requires SM120")

    module = _svdquant_module()
    m, n, k = 129, 4096, 3072
    expected = next(
        tactic
        for tactic in range(module.nvfp4_svdquant_gemm_tactic_num())
        if module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)
    )

    assert module.nvfp4_svdquant_gemm_fallback_tactic(m, n, k, _RANK) == expected


def test_mm_nvfp4_svdquant_rejects_noncontiguous_out():
    """A transposed [m, n] view must be rejected, not silently rearranged.

    The validation lives in the SM120 launcher; the SM100 launcher predates it and
    is frozen by the non-regression gate, so this asserts SM120 behavior only.
    """
    _skip_unless_svdquant_supported()
    if not _is_sm120():
        pytest.skip("output-contiguity validation is implemented in the SM120 launcher")
    torch.manual_seed(0)
    m, n, k = 128, 3072, 3072
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    storage = torch.empty(n, m, dtype=torch.bfloat16, device=p["xq"].device)
    out_view = storage.t()  # shape [m, n], non-contiguous strides
    module = _svdquant_module()
    workspace = torch.zeros(
        DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=p["xq"].device
    )
    with pytest.raises(Exception, match="contiguous"):
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out_view,
            workspace,
            0,
            device_support_pdl(p["xq"].device),
        )


@pytest.mark.parametrize("m,n,k", MODEL_SHAPE_CASES)
@pytest.mark.parametrize("use_bias", [True, False])
def test_mm_nvfp4_svdquant_model_shapes(m, n, k, use_bias):
    """Rank-32 correctness across the unified 71 model shapes.

    Covers the 12 Qwen-Image baseline shapes plus the Qwen-Image text-stream,
    Wan2.1 (1.3B/14B), Wan2.2 (A14B/TI2V-5B), and MiniMax-H3 cases from
    docs/model_shape_requirements.md (executable source of truth:
    flashinfer/testing/svdq_model_shapes.py).

    Covers both the autotuned path (tuning-context call plus a cache-replay call)
    and the fixed tactic-0 module path, with the output NaN-poisoned beforehand so
    any unwritten or tail-corrupted region fails the SQNR comparison outright.
    """
    _skip_unless_svdquant_supported()
    torch.manual_seed(3)
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    ref = p["ref_bias"] if use_bias else p["ref"]
    bias = p["bias"] if use_bias else None

    device = p["xq"].device
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    with autotune(True):
        mm_nvfp4_svdquant(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            bias=bias,
            out=out,
        )
    assert _sqnr_db(ref, out.float()) > 40.0

    replay = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=bias,
        out=replay,
    )
    assert _sqnr_db(ref, replay.float()) > 40.0

    module = _svdquant_module()
    out0 = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    workspace = torch.zeros(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    module.nvfp4_svdquant_gemm(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias,
        out0,
        workspace,
        0,
        device_support_pdl(device),
    )
    assert _sqnr_db(ref, out0.float()) > 40.0


def _decode_row(module, tactic):
    packed = int(module.nvfp4_svdquant_gemm_tactic_row(tactic))
    return {
        "kernel_id": packed & 0xFF,
        "splits": (packed >> 8) & 0xFF,
        "raster": (packed >> 16) & 0xFF,
        "swizzle": (packed >> 24) & 0xFF,
        "streamk": bool((packed >> 32) & 1),
    }


def _skip_unless_sm120_table():
    _skip_unless_svdquant_supported()
    if not _is_sm120():
        pytest.skip("the runtime tactic table is an SM120 backend feature")


def test_sm120_runtime_tactic_table_decode():
    """Frozen legacy rows and documented variant blocks decode as designed."""
    _skip_unless_sm120_table()
    module = _svdquant_module()
    assert module.nvfp4_svdquant_gemm_tactic_num() == 83
    for t in range(16):
        row = _decode_row(module, t)
        assert row == {
            "kernel_id": t,
            "splits": 1,
            "raster": 0,
            "swizzle": 1,
            "streamk": t % 4 >= 2,
        }, f"legacy row {t} changed meaning: {row}"
    assert _decode_row(module, 16) == {
        "kernel_id": 2,
        "splits": 2,
        "raster": 0,
        "swizzle": 1,
        "streamk": True,
    }
    assert _decode_row(module, 31)["splits"] == 4
    assert _decode_row(module, 32)["raster"] == 1  # AlongM on kernel 0
    assert _decode_row(module, 63)["swizzle"] == 4
    spikes = [_decode_row(module, t) for t in range(64, 75)]
    assert [s["kernel_id"] for s in spikes] == [
        16,
        17,
        17,
        17,
        18,
        19,
        20,
        21,
        22,
        22,
        22,
    ]
    assert [s["splits"] for s in spikes] == [1, 1, 2, 4, 1, 1, 1, 1, 1, 2, 4]
    # driver rows (70/71) are persistent; the K256-swap Stream-K rows are not
    assert not spikes[6]["streamk"] and not spikes[7]["streamk"]
    assert all(s["streamk"] for s in spikes[8:])
    # the first static-scheduler block uses scheduler-default persistent rows
    statics = [_decode_row(module, t) for t in range(75, 78)]
    assert [s["kernel_id"] for s in statics] == [23, 24, 25]
    assert all(
        s["splits"] == 1 and s["raster"] == 0 and s["swizzle"] == 1 and not s["streamk"]
        for s in statics
    )
    # the fill-geometry row closes the table as a scheduler-default row
    fill = _decode_row(module, 78)
    assert fill == {
        "kernel_id": 26,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }
    assert _decode_row(module, 81) == {
        "kernel_id": 29,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }
    assert _decode_row(module, 82) == {
        "kernel_id": 30,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }
    # its static-scheduler sibling is append-only so row 78 keeps its meaning
    fill_static = _decode_row(module, 79)
    assert fill_static == {
        "kernel_id": 27,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }
    small_m_fill_static = _decode_row(module, 80)
    assert small_m_fill_static == {
        "kernel_id": 28,
        "splits": 1,
        "raster": 0,
        "swizzle": 1,
        "streamk": False,
    }


def test_sm120_splitk_rows_respect_ktile_bound():
    """can_implement enforces splits <= K-tiles; workspace grows with splits."""
    _skip_unless_sm120_table()
    module = _svdquant_module()
    # k = 512: 4 K-tiles at CTA_K=128, 2 at CTA_K=256.
    m, n, k = 256, 3072, 512
    # row 16/17: kernel 2 (128x128x128 Sk) splits 2/4 -> both within 4 K-tiles
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, 17)
    # rows for kernel 6 (128x128x256 Sk): splits 4 > 2 K-tiles -> rejected
    splits4_k256 = next(
        t
        for t in range(16, 32)
        if _decode_row(module, t)
        == {"kernel_id": 6, "splits": 4, "raster": 0, "swizzle": 1, "streamk": True}
    )
    assert not module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, splits4_k256)
    # forced Split-K needs more reduction workspace than the heuristic row
    big = (4096, 3072, 12288)
    ws_default = module.nvfp4_svdquant_gemm_workspace_size(*big, 2)
    ws_split4 = module.nvfp4_svdquant_gemm_workspace_size(*big, 17)
    assert ws_split4 > ws_default, (ws_split4, ws_default)


_NEW_ROW_CLASS_TACTICS = [
    16,
    32,
    35,
    64,
    65,
    68,
    69,
    70,
    71,
    72,
    73,
    75,
    76,
    77,
    78,
    79,
    80,
]


@pytest.mark.parametrize("tactic", _NEW_ROW_CLASS_TACTICS)
@pytest.mark.parametrize("m,n,k", [(256, 3072, 3072), (6889, 3072, 3072)])
def test_sm120_new_row_classes_match_reference(tactic, m, n, k):
    """Every new runtime-row class must reproduce the reference numerics."""
    _skip_unless_sm120_table()
    torch.manual_seed(11)
    module = _svdquant_module()
    if not module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic):
        pytest.skip(f"tactic {tactic} not implementable for {(m, n, k)}")
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    ws_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    module.nvfp4_svdquant_gemm(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        p["bias"],
        out,
        workspace,
        tactic,
        device_support_pdl(device),
    )
    torch.cuda.synchronize()
    sqnr = _sqnr_db(p["ref_bias"], out.float())
    assert sqnr >= 40.0, f"tactic {tactic} on {(m, n, k)}: SQNR {sqnr:.1f} dB"


@pytest.mark.parametrize("m", [64, 129])
def test_sm120_fill_geometry_candidates_are_bitwise_equivalent(m):
    """Scheduler and CTA geometry must not change the target output bits."""
    _skip_unless_sm120_table()
    torch.manual_seed(12)
    n, k = 3072, 3072
    module = _svdquant_module()
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    tactics = (78, 79, 80)
    assert all(
        module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)
        for tactic in tactics
    )
    ws_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        max(
            int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic))
            for tactic in tactics
        ),
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    outputs = []
    for tactic in tactics:
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out,
            workspace,
            tactic,
            device_support_pdl(device),
        )
        torch.cuda.synchronize()
        outputs.append(out)
    assert all(
        torch.equal(outputs[0].view(torch.int16), output.view(torch.int16))
        for output in outputs[1:]
    )


@pytest.mark.parametrize("m", [64, 129])
@pytest.mark.parametrize("lora_epi_overlap", [True, False])
def test_sm120_lora_smem_swizzle_is_bitwise_equivalent(m, lora_epi_overlap):
    """The production swizzle must match its build-wide diagnostic variant.

    Exercise both the target tile count and a partial-M edge with the overlapped
    production consumer and its serial fallback. The overlapped case also pins
    equivalence to the storage-width tail, an independent view of the same
    rank-32 payload; row78/79 equivalence separately anchors the output bits to
    non-swizzled tactics.
    """
    _skip_unless_sm120_table()
    from flashinfer.jit.gemm.svdquant_sm120 import (
        gen_gemm_sm120_module_cutlass_nvfp4_svdquant as _gen,
    )

    torch.manual_seed(18)
    n, k, tactic = 3072, 3072, 80
    production = (
        _svdquant_module()
        if lora_epi_overlap
        else _gen(lora_epi_overlap=False).build_and_load()
    )
    swizzled = _gen(
        lora_epi_overlap=lora_epi_overlap,
        lora_smem_swizzle=True,
    ).build_and_load()
    modules = [production, swizzled]
    if lora_epi_overlap:
        modules.append(_gen(lora_true_rank_tail=False).build_and_load())

    assert all(
        module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)
        for module in modules
    )
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    ws_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        max(
            int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic))
            for module in modules
        ),
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    outputs = []
    for module in modules:
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out,
            workspace,
            tactic,
            device_support_pdl(device),
        )
        torch.cuda.synchronize()
        outputs.append(out.view(torch.int16))

    assert torch.equal(outputs[1], outputs[0]), (
        f"diagnostic swizzle output bits diverged from production (m={m}, "
        f"overlap={lora_epi_overlap})"
    )
    if lora_epi_overlap:
        assert torch.equal(outputs[1], outputs[2]), (
            f"swizzled output bits diverged from the storage-width fallback (m={m})"
        )


@pytest.mark.parametrize("diagnostic_variant", [False, True])
def test_sm120_lora_smem_swizzle_cuda_graph(diagnostic_variant):
    """Production row80 and its diagnostic parity build survive graph replay."""
    _skip_unless_sm120_table()
    from flashinfer.jit.gemm.svdquant_sm120 import (
        gen_gemm_sm120_module_cutlass_nvfp4_svdquant as _gen,
    )

    torch.manual_seed(19)
    m, n, k, tactic = 64, 3072, 3072, 80
    module = (
        _gen(lora_smem_swizzle=True, row80_stages=4).build_and_load()
        if diagnostic_variant
        else _svdquant_module()
    )
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    workspace = torch.zeros(
        max(
            DEFAULT_WORKSPACE_SIZE,
            int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
        ),
        dtype=torch.uint8,
        device=device,
    )
    enable_pdl = device_support_pdl(device)

    def run(out_buf):
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out_buf,
            workspace,
            tactic,
            enable_pdl,
        )

    out_eager = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    run(out_eager)
    torch.cuda.synchronize()

    out_graph = torch.full_like(out_eager, float("nan"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(out_graph)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out_graph)
    for _ in range(2):
        out_graph.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_graph.view(torch.int16), out_eager.view(torch.int16))


@pytest.mark.parametrize("m", [64, 129])
@pytest.mark.parametrize(
    "stages,expected_smem,lora_smem_swizzle",
    [(4, 71680, False), (5, 88064, False), (4, 71680, True)],
)
def test_sm120_row80_stage_override_is_bitwise_equivalent(
    m, stages, expected_smem, lora_smem_swizzle
):
    """Row80 stage parity/alternate builds must preserve production output bits."""
    _skip_unless_sm120_table()
    from flashinfer.jit.gemm.svdquant_sm120 import (
        gen_gemm_sm120_module_cutlass_nvfp4_svdquant as _gen,
    )

    torch.manual_seed(20)
    n, k, tactic = 3072, 3072, 80
    baseline = _svdquant_module()
    staged = _gen(
        row80_stages=stages,
        lora_smem_swizzle=lora_smem_swizzle,
    ).build_and_load()
    assert int(baseline.nvfp4_svdquant_gemm_shared_storage_size(tactic)) == 83968
    assert int(staged.nvfp4_svdquant_gemm_shared_storage_size(tactic)) == expected_smem
    assert int(baseline.nvfp4_svdquant_gemm_shared_storage_size(79)) == 83968
    # The stage override preserves row79 storage; the optional build-wide
    # swizzle changes only its bank mapping, not its allocation size.
    assert int(staged.nvfp4_svdquant_gemm_shared_storage_size(79)) == int(
        baseline.nvfp4_svdquant_gemm_shared_storage_size(79)
    )
    assert staged.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic)

    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    workspace = torch.zeros(
        max(
            DEFAULT_WORKSPACE_SIZE,
            int(staged.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
        ),
        dtype=torch.uint8,
        device=device,
    )
    outputs = []
    for module in (baseline, staged):
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out,
            workspace,
            tactic,
            device_support_pdl(device),
        )
        torch.cuda.synchronize()
        outputs.append(out.view(torch.int16))
    assert torch.equal(outputs[1], outputs[0]), (
        f"row80 stage-{stages} output bits diverged from production "
        f"(m={m}, swizzle={lora_smem_swizzle})"
    )


@pytest.mark.parametrize("tactic", [16, 64, 70, 71, 73, 75, 76, 77, 78, 79, 80])
def test_sm120_cuda_graph_new_rows(tactic):
    """Graph capture/double-replay for a Split-K row, a small-N spike row,
    both 64-row-M driver rows, the K256-swap forced-split row, and the
    static-scheduler rows.

    The workspace is pre-provisioned via the workspace-size query before
    capture; the launcher must never allocate inside the captured region.
    """
    _skip_unless_sm120_table()
    torch.manual_seed(13)
    m, n, k = 256, 3072, 3072
    module = _svdquant_module()
    if not module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic):
        pytest.skip(f"tactic {tactic} not implementable for {(m, n, k)}")
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    enable_pdl = device_support_pdl(device)
    ws_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)

    def run(out_buf):
        module.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out_buf,
            workspace,
            tactic,
            enable_pdl,
        )

    out_eager = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    run(out_eager)
    torch.cuda.synchronize()

    out_graph = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(out_graph)  # capture warmup on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out_graph)
    for _ in range(2):
        out_graph.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_graph, out_eager), (
            f"graph replay diverged from eager for tactic {tactic}"
        )


@pytest.mark.parametrize("mode", ["random", "zeroheavy", "negalpha", "negzero_out"])
@pytest.mark.parametrize("tactic", [68, 75, 79, 80])
def test_sm120_true_rank_tail_bitwise_vs_storage_fallback(mode, tactic):
    """The rank-32 payload slice must match the storage-width fallback in
    exact bit patterns (integer view), including the sign of exact zeros:
    the payload path applies the fallback's additive identity (acc + 0) on
    K256 tiles instead of the skipped zero-MMAs, so -0.0 lanes canonicalize
    identically. The adversarial modes bias inputs toward exact-zero and
    negative-zero accumulator lanes; the observed negative-zero lane count
    is asserted only to be equal between the two builds."""
    _skip_unless_sm120_table()
    from flashinfer.jit.gemm.svdquant_sm120 import (
        gen_gemm_sm120_module_cutlass_nvfp4_svdquant as _gen,
    )

    torch.manual_seed(17)
    m, n, k = (256, 3072, 3072)
    module = _svdquant_module()
    if not module.nvfp4_svdquant_gemm_can_implement(m, n, k, _RANK, tactic):
        pytest.skip(f"tactic {tactic} not implementable for {(m, n, k)}")
    fallback = _gen(lora_true_rank_tail=False).build_and_load()
    p = _make_gemm_problem(m, n, k, rank=_RANK)
    device = p["xq"].device
    if mode == "zeroheavy":
        # zero half the LoRA-down rows and alternate LoRA-up rows so many
        # tail products are exact zeros with mixed signs
        p["d"][m // 2 :].zero_()
        p["d"][: m // 2] *= torch.where(
            torch.rand_like(p["d"][: m // 2]) < 0.5, -1.0, 1.0
        )
        p["l1_scaled"][::2].zero_()
    elif mode == "negalpha":
        # zero activations with a negative dequant scale drive residual
        # accumulator lanes through negative-signed exact zeros
        p["alpha"].fill_(-1.0)
        p["xq"].zero_()
    elif mode == "negzero_out":
        # the strongest zero-sign probe: zero operands with a negative
        # dequant scale and no bias. Measured on SM120: the output is +0.0
        # everywhere in BOTH builds - the epilogue's closing FFMA (zero
        # addend under null bias, exact cancellation otherwise) canonicalizes
        # the sign under round-to-nearest, and the accumulator itself can
        # never hold -0.0 (cleared to +0.0; RN cancellation yields +0.0).
        # The exact -0.0 divergence window is therefore empty at both the
        # accumulator and the output; this case pins that equivalence.
        p["alpha"].fill_(-1.0)
        p["xq"].zero_()
        p["d"].zero_()
        p["bias"] = None
    ws_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    enable_pdl = device_support_pdl(device)
    outs = []
    for mod in (module, fallback):
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        mod.nvfp4_svdquant_gemm(
            p["xq"],
            p["wq"],
            p["x_sf_flat"],
            p["w_sf_flat"],
            p["alpha"],
            p["d"],
            p["l1_scaled"],
            p["bias"],
            out,
            workspace,
            tactic,
            enable_pdl,
        )
        torch.cuda.synchronize()
        outs.append(out)
    iv_default, iv_fallback = outs[0].view(torch.int16), outs[1].view(torch.int16)
    negz_default = int((iv_default == -32768).sum())
    negz_fallback = int((iv_fallback == -32768).sum())
    assert negz_default == negz_fallback, (
        f"negative-zero lane count diverged: {negz_default} vs {negz_fallback}"
    )
    assert torch.equal(iv_default, iv_fallback), (
        f"bit-pattern mismatch vs storage fallback (mode={mode}, tactic={tactic})"
    )


@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize("m,n,k", [(4096, 3072, 3072), (512, 5120, 5120)])
def test_public_entry_serves_rank_64(m, n, k, rank):
    """The rank reaches the module through the public API, not a private getter.

    The collective takes its LoRA rank as a build parameter, so serving rank 64
    means mm_nvfp4_svdquant reading d.shape[1] and getting the module compiled
    for it. A caller should not have to reach for
    get_nvfp4_svdquant_sm120_module(64) by hand.

    Rank 32 rides along as the control: it must keep using the module it always
    did, which is the same object the production build produces.
    """
    _skip_unless_svdquant_supported()
    if not _is_sm120():
        pytest.skip("requires SM120")
    torch.manual_seed(3)
    p = _make_gemm_problem(m, n, k, rank=rank)
    out = mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=None,
        out=None,
        backend="cutlass-sm120",
    )
    assert _sqnr_db(p["ref"], out.float()) > 40.0

    module = _get_nvfp4_svdquant_module_for_device(torch.device("cuda"), rank)
    assert module.nvfp4_svdquant_gemm_lora_rank() == rank
