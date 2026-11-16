"""GPU hardening tests for the CAKE DeepSeek-V4 sparse-MLA host (flashinfer#4671).

Covers padded query rows, separate SWA/compressed tables, length offsets,
all-invalid rows, CUDA graph capture/replay with mutated inputs, and the
no-allocation contract. Reference and generator helpers are imported from the
TRTLLM-GEN DSv4 test module. Skips without an SM100/SM103 GPU.

The CAKE-957 section at the end builds its cases directly in the FlashInfer
API vocabulary: poisoned columns beyond the active length (bit-identical
output), ``-1`` slots inside the compressed active length, non-tile-aligned
lengths, ``seq_lens < 128`` with stale valid SWA slots beyond the causal
window, the sglang call conventions, and the CAKE-939 workspace / counter
contract (fixed 128 MiB buffer, row tiling, capture without priming).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pathlib
from typing import Optional

import pytest
import torch

from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from flashinfer.mla.cake_dsv4 import (
    _route,
    cake_dsv4_workspace_layout,
    cake_dsv4_workspace_requirement,
    cake_dsv4_workspace_reset,
    get_cake_dsv4_workspace_bytes,
)
from flashinfer.utils import get_compute_capability


def _load_reference_module():
    try:
        return importlib.import_module(
            "tests.attention.test_trtllm_gen_sparse_mla_dsv4"
        )
    except ModuleNotFoundError:
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "attention"
            / "test_trtllm_gen_sparse_mla_dsv4.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_cake_dsv4_reference_helpers", path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


ref = _load_reference_module()

HEADS = (32, 64, 128)
DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
Q_LENS = (1, 4)
BATCH = 3
SWA_SEQ_LEN = 512
C4_SEQ_LEN = 1024
SEED_BASE = ref.TEST_SEED_BASE + 10_000
_seed_counter = 0


def _skip_unless_cake_gpu(*, sm107: bool = False) -> None:
    """Skip unless the GPU is SM100 or SM103, or SM107 when ``sm107`` is set
    (the tests whose programs SM107 builds)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CAKE DSv4 hardening tests")
    compute_capability = get_compute_capability(torch.device("cuda"))
    supported = ((10, 0), (10, 3), (10, 7)) if sm107 else ((10, 0), (10, 3))
    if compute_capability not in supported:
        names = "SM100/SM103/SM107" if sm107 else "SM100/SM103"
        pytest.skip(
            f"CAKE DeepSeek V4 sparse MLA requires {names}, "
            f"got SM{compute_capability[0]}{compute_capability[1]}"
        )


def _lowhead_prefill_variant(rows: int) -> str:
    """The variant the host launches for a low-head prefill launch of ``rows``
    query rows on this GPU: on SM107 the V-block sibling up to 27 rows and the
    early-PV sibling up to 64 rows (both ceilings scaled down with the SM count
    below 216 SMs), the route's program above them; elsewhere the route's
    program."""
    import flashinfer.mla.cake_dsv4 as cake

    route = "fp8_lowhead_prefill"
    if cake._target_arch(torch.device("cuda", 0)) != "sm_107a":
        return route
    sm_count = min(torch.cuda.get_device_properties(0).multi_processor_count, 216)
    if rows <= 27 * sm_count // 216:
        return f"{route}_vblock"
    if rows <= 64 * sm_count // 216:
        return f"{route}_overlap"
    return route


def _compressed_topk(h_q: int) -> int:
    # swa128+topk4x profiles: sparse_topk 384 / 640 / 1152 (page size 64).
    return {32: 256, 64: 512, 128: 1024}[h_q]


def _make_case(
    h_q: int,
    dtype: torch.dtype,
    s_q: int,
    *,
    varlen: bool,
    all_invalid: bool = False,
    seed: int | None = None,
    batch: int = BATCH,
):
    global _seed_counter
    if seed is None:
        seed = SEED_BASE + _seed_counter
        _seed_counter += 1
    param = ref.RawTestParamForDecode(
        b=batch,
        h_q=h_q,
        s_q=s_q,
        h_kv=1,
        s_kv=SWA_SEQ_LEN,
        is_varlen=varlen,
        topk=ref.DSV4_SWA_TOPK,
        is_all_indices_invalid=all_invalid,
        extra_s_k=C4_SEQ_LEN,
        extra_topk=_compressed_topk(h_q),
        block_size=ref.SWA_PAGE_SIZE,
        extra_block_size=ref.C4_PAGE_SIZE,
        have_extra_topk_length=True,
        seed=seed,
        dtype=dtype,
        kv_layout="HND",
        sparse_case="swa128+topk4x",
    ).to_test_param()
    return param, ref.generate_testcase_for_decode(param)


class _Inputs:
    """Host tensors for one testcase in both the combined and the separate form."""

    def __init__(self, p, tc):
        assert tc.extra_kv_scope is not None
        assert tc.extra_kv_scope.topk_length is not None
        self.p = p
        self.tc = tc
        self.varlen = p.decode.is_varlen
        self.swa_indices = tc.kv_scope.indices_in_kvcache[tc.valid_q].contiguous()
        self.compressed_indices = tc.extra_kv_scope.indices_in_kvcache[
            tc.valid_q
        ].contiguous()
        self.compressed_lens = ref._topk_length_for_flashinfer(
            tc.extra_kv_scope.topk_length, tc.valid_q
        ).contiguous()
        self.combined_lens = (self.compressed_lens + ref.DSV4_SWA_TOPK).contiguous()
        self.combined_indices = torch.cat(
            (self.swa_indices, self.compressed_indices), dim=-1
        ).contiguous()
        self.swa_kv_cache = tc.kv_scope.get_kvcache_for_flashinfer("HND")
        self.compressed_kv_cache = tc.extra_kv_scope.get_kvcache_for_flashinfer("HND")
        self.seq_lens = tc.kv_scope.cache_seqlens
        if self.varlen:
            self.query = tc.q[tc.valid_q].contiguous()
            self.cum_seq_lens_q = ref._make_cum_seq_lens(tc.q_lens)
            self.max_q_len = p.s_q
        else:
            self.query = tc.q.contiguous()
            self.cum_seq_lens_q = None
            self.max_q_len = None
        self.num_tokens = int(self.swa_indices.shape[0])
        self.sparse_topk = int(self.combined_indices.shape[1])
        self.bmm1_scale = ref._scale_for_flashinfer(p, tc.sm_scale)
        self.bmm2_scale = ref._scale_for_flashinfer(p, 1.0)

    def reference_rows(self) -> torch.Tensor:
        out, _ = ref.ref_sparse_attn_decode(self.p, self.tc)
        return out[self.tc.valid_q].reshape(self.num_tokens, self.p.h_q, self.p.d_v)

    def copy_from(self, other: "_Inputs") -> None:
        """Overwrite every device input in place (same shapes) for graph replay."""
        self.query.copy_(other.query)
        self.combined_indices.copy_(other.combined_indices)
        self.swa_indices.copy_(other.swa_indices)
        self.compressed_indices.copy_(other.compressed_indices)
        self.combined_lens.copy_(other.combined_lens)
        self.compressed_lens.copy_(other.compressed_lens)
        self.swa_kv_cache.copy_(other.swa_kv_cache)
        self.compressed_kv_cache.copy_(other.compressed_kv_cache)
        self.seq_lens.copy_(other.seq_lens)
        if self.tc.attn_sink is not None:
            self.tc.attn_sink.copy_(other.tc.attn_sink)

    def run(
        self,
        *,
        out: torch.Tensor,
        workspace: torch.Tensor,
        separate: bool = False,
        lens_offset: int = 0,
        metadata_rows: int | None = None,
        backend: str = "cake",
        multi_ctas_kv_counter_buffer: torch.Tensor | None = None,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        rows = self.num_tokens if metadata_rows is None else metadata_rows
        kwargs = dict(
            query=self.query,
            swa_kv_cache=self.swa_kv_cache,
            workspace_buffer=workspace,
            compressed_kv_cache=self.compressed_kv_cache,
            seq_lens=self.seq_lens,
            out=out,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            sinks=self.tc.attn_sink,
            kv_layout="HND",
            cum_seq_lens_q=self.cum_seq_lens_q,
            max_q_len=self.max_q_len,
            enable_pdl=enable_pdl,
            backend=backend,
            sparse_topk_lens_offset=lens_offset,
        )
        if multi_ctas_kv_counter_buffer is not None:
            kwargs["multi_ctas_kv_counter_buffer"] = multi_ctas_kv_counter_buffer
        if separate:
            kwargs.update(
                sparse_indices=self.swa_indices[:rows],
                extra_sparse_indices=self.compressed_indices[:rows],
                extra_sparse_topk_lens=(self.compressed_lens[:rows] - lens_offset)
                if lens_offset
                else self.compressed_lens[:rows],
            )
        else:
            kwargs.update(
                sparse_indices=self.combined_indices[:rows],
                sparse_topk_lens=(self.combined_lens[:rows] - lens_offset)
                if lens_offset
                else self.combined_lens[:rows],
            )
        return trtllm_batch_decode_sparse_mla_dsv4(**kwargs)


def _workspace(inputs: _Inputs) -> torch.Tensor:
    num_bytes = get_cake_dsv4_workspace_bytes(
        inputs.num_tokens, inputs.p.h_q, inputs.sparse_topk, inputs.p.dtype
    )
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device="cuda:0")
    cake_dsv4_workspace_reset(workspace)
    return workspace


def _out_like(inputs: _Inputs, fill: float = float("nan")) -> torch.Tensor:
    return torch.full(inputs.query.shape, fill, dtype=torch.bfloat16, device="cuda:0")


def _rows(out: torch.Tensor, inputs: _Inputs) -> torch.Tensor:
    return out.reshape(-1, inputs.p.h_q, inputs.p.d_v)


def _allocation_requests() -> int:
    """Cumulative number of caching-allocator requests on the current device.

    The no-allocation checks compare this monotonic counter rather than
    ``torch.cuda.memory_allocated()``: the live-bytes figure also moves when an
    unrelated tensor is released inside the measured window (a cyclic-garbage
    collection pass can free an earlier case's inputs during the call), which
    is not an allocation by the call under test and made the equality fail with
    *fewer* bytes live after the call (flashinfer-ai/flashinfer#6236).

    Only the native caching allocator maintains the counter; under
    ``PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`` it stays at zero and the
    checks would pass vacuously, so they skip there.
    """
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip(
            "the no-allocation checks read a native caching-allocator statistic"
        )
    return int(torch.cuda.memory_stats()["allocation.all.allocated"])


_CASES = [
    pytest.param(
        h, dtype, s_q, id=f"h{h}-{'bf16' if dtype == torch.bfloat16 else 'fp8'}-q{s_q}"
    )
    for h in HEADS
    for dtype in DTYPES
    for s_q in Q_LENS
]


@pytest.mark.parametrize("h_q,dtype,s_q", _CASES)
@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_padded_query_rows_match_reference(h_q, dtype, s_q, layout):
    """Metadata with fewer rows than the query: prefix computed, tail untouched."""
    _skip_unless_cake_gpu()
    p, tc = _make_case(h_q, dtype, s_q, varlen=layout == "ragged")
    inputs = _Inputs(p, tc)
    rows = (
        inputs.num_tokens - 1 if s_q == 1 else inputs.num_tokens - s_q
    )  # drop a token / a request
    assert rows >= 1
    out = _out_like(inputs)
    inputs.run(out=out, workspace=_workspace(inputs), metadata_rows=rows)
    torch.cuda.synchronize()
    expected = inputs.reference_rows()
    got = _rows(out, inputs)
    _assert_close(got[:rows], expected[:rows], dtype)
    assert torch.isnan(got[rows:]).all(), "padded query rows must not be written"


@pytest.mark.parametrize("h_q,dtype,s_q", _CASES)
def test_separate_tables_match_combined(h_q, dtype, s_q):
    _skip_unless_cake_gpu()
    p, tc = _make_case(h_q, dtype, s_q, varlen=True)
    inputs = _Inputs(p, tc)
    workspace = _workspace(inputs)
    combined = _out_like(inputs)
    separate = _out_like(inputs)
    inputs.run(out=combined, workspace=workspace)
    inputs.run(out=separate, workspace=workspace, separate=True)
    torch.cuda.synchronize()
    _assert_close(_rows(combined, inputs), inputs.reference_rows(), dtype)
    assert torch.equal(separate, combined)


@pytest.mark.parametrize("h_q,dtype,s_q", _CASES)
@pytest.mark.parametrize("separate", [False, True], ids=["combined", "separate"])
def test_lens_offset_matches_pre_added_lens(h_q, dtype, s_q, separate):
    _skip_unless_cake_gpu()
    p, tc = _make_case(h_q, dtype, s_q, varlen=True)
    inputs = _Inputs(p, tc)
    workspace = _workspace(inputs)
    baseline = _out_like(inputs)
    shifted = _out_like(inputs)
    inputs.run(out=baseline, workspace=workspace, separate=separate)
    inputs.run(out=shifted, workspace=workspace, separate=separate, lens_offset=37)
    torch.cuda.synchronize()
    _assert_close(_rows(baseline, inputs), inputs.reference_rows(), dtype)
    assert torch.equal(shifted, baseline)


@pytest.mark.parametrize(
    "h_q,dtype",
    [
        pytest.param(h, d, id=f"h{h}-{'bf16' if d == torch.bfloat16 else 'fp8'}")
        for h in HEADS
        for d in DTYPES
    ],
)
def test_all_invalid_rows(h_q, dtype):
    _skip_unless_cake_gpu()
    p, tc = _make_case(h_q, dtype, 4, varlen=True, all_invalid=True)
    inputs = _Inputs(p, tc)
    assert bool((inputs.combined_indices == -1).all())
    out = _out_like(inputs)
    inputs.run(out=out, workspace=_workspace(inputs))
    torch.cuda.synchronize()
    got = _rows(out, inputs)
    assert not torch.isnan(got).any()
    _assert_close(got, inputs.reference_rows(), dtype)


@pytest.mark.parametrize(
    "h_q,dtype",
    [
        pytest.param(64, torch.bfloat16, id="bf16-h64-prefill"),
        pytest.param(32, torch.float8_e4m3fn, id="fp8-h32-prefill"),
    ],
)
def test_same_workspace_serves_different_kv_caches(h_q, dtype):
    """Two calls with one workspace but different KV caches (consecutive layers).

    The prefill routes of these shapes (``bf16_h64_prefill``,
    ``fp8_h64_prefill_source_persistent_m64_multi_tile``) must read the KV cache
    of each call; nothing bound to the workspace may carry over from the
    previous call.
    """
    _skip_unless_cake_gpu()
    first = _Inputs(*_make_case(h_q, dtype, 257, varlen=True))
    second = _Inputs(*_make_case(h_q, dtype, 257, varlen=True))
    workspace = _workspace(first)
    for inputs in (first, second, first):
        out = _out_like(inputs)
        inputs.run(out=out, workspace=workspace)
        torch.cuda.synchronize()
        _assert_close(_rows(out, inputs), inputs.reference_rows(), dtype)


@pytest.mark.parametrize("h_q,dtype,s_q", _CASES)
def test_cuda_graph_replay_matches_eager(h_q, dtype, s_q):
    """Capture once, mutate every input in place, replay: equals eager; no allocation."""
    _skip_unless_cake_gpu()
    p, tc = _make_case(h_q, dtype, s_q, varlen=True)
    static = _Inputs(p, tc)
    workspace = _workspace(static)
    out = _out_like(static, fill=0.0)

    # Eager warm-up: JIT build, descriptor slab, counters, scale constants.
    static.run(out=out, workspace=workspace)
    static.run(out=out, workspace=workspace)
    torch.cuda.synchronize()
    baseline_requests = _allocation_requests()
    static.run(out=out, workspace=workspace)
    torch.cuda.synchronize()
    assert _allocation_requests() == baseline_requests, "eager call allocated"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static.run(out=out, workspace=workspace)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(_rows(out, static), static.reference_rows(), dtype)

    # Mutate every input in place (same shapes) and replay.
    p2, tc2 = _make_case(h_q, dtype, s_q, varlen=True, seed=p.seed + 5_000)
    mutated = _Inputs(p2, tc2)
    assert mutated.query.shape == static.query.shape
    static.copy_from(mutated)
    torch.cuda.synchronize()
    replay_requests = _allocation_requests()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert _allocation_requests() == replay_requests, "graph replay allocated"

    eager_out = _out_like(static, fill=0.0)
    static.run(out=eager_out, workspace=_workspace(static))
    torch.cuda.synchronize()
    # The reference must be evaluated on the mutated testcase's own tensors.
    _assert_close(_rows(eager_out, static), mutated.reference_rows(), dtype)
    assert torch.equal(out, eager_out)


@pytest.mark.parametrize(
    "h_q,batch,s_q",
    [
        pytest.param(64, BATCH, 4, id="h64-persistent"),
        pytest.param(64, 64, 4, id="h64-m64"),
        pytest.param(128, BATCH, 4, id="h128-persistent"),
        # 64 requests of 6-12 tokens: 576 rows, the twin's range on SM103.
        pytest.param(128, 64, 12, id="h128-persistent-twin"),
        # 2 requests of up to 257 tokens: the twin's range on SM100.
        pytest.param(128, 2, 257, id="h128-persistent-prefill"),
    ],
)
def test_enable_pdl_matches_and_waits_for_the_query_producer(
    monkeypatch, h_q, batch, s_q
):
    """FP8 programs launched as programmatic dependents: bitwise equal to the
    plain launch (the H128 persistent route's PDL-free twin on SM103 from 384
    rows, on SM100 from a 257-token request), and inside a graph they read the
    query the kernel before them writes (the copy that fills it in place)."""
    _skip_unless_cake_gpu()
    import flashinfer.mla.cake_dsv4 as cake

    launched: list = []
    original = cake._prepare_variant

    def spy(variant, **kwargs):
        launched.append(variant)
        return original(variant, **kwargs)

    monkeypatch.setattr(cake, "_prepare_variant", spy)
    dtype = torch.float8_e4m3fn
    p, tc = _make_case(h_q, dtype, s_q, varlen=True, batch=batch)
    static = _Inputs(p, tc)
    workspace = _workspace(static)
    plain = _out_like(static, fill=0.0)
    pdl = _out_like(static, fill=0.0)
    static.run(out=plain, workspace=workspace)
    plain_variants, launched[:] = list(launched), []
    static.run(out=pdl, workspace=workspace, enable_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(plain, pdl)
    (route,) = set(launched)
    capability = tuple(get_compute_capability(torch.device("cuda")))
    rows = static.swa_indices.shape[0]
    twin = route in cake._NO_PDL_TWIN_ROUTES and (
        (capability == (10, 3) and rows >= 384)
        or (capability == (10, 0) and static.max_q_len >= 257)
    )
    assert set(plain_variants) == {f"{route}_no_pdl" if twin else route}

    p2, tc2 = _make_case(h_q, dtype, s_q, varlen=True, seed=p.seed + 7_000, batch=batch)
    mutated = _Inputs(p2, tc2)
    static.copy_from(mutated)
    source = mutated.query.clone()
    static.query.copy_(_Inputs(p, tc).query)
    torch.cuda.synchronize()
    out = _out_like(static, fill=0.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static.query.copy_(source)
        static.run(out=out, workspace=workspace, enable_pdl=True)
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(_rows(out, static), mutated.reference_rows(), dtype)


# --------------------------------------------------------------------------- trtllm-gen host hardening
def _skip_unless_trtllm_gen_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability not in ((10, 0), (10, 3)):
        pytest.skip("TRTLLM-GEN DSv4 sparse MLA requires SM100/SM103")


def _trtllm_workspace() -> torch.Tensor:
    return torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")


@pytest.mark.parametrize(
    "h_q,dtype,s_q",
    [
        pytest.param(64, torch.bfloat16, 4, id="h64-bf16-q4"),
        pytest.param(128, torch.float8_e4m3fn, 1, id="h128-fp8-q1"),
    ],
)
@pytest.mark.parametrize("layout", ["dense", "ragged"])
def test_trtllm_gen_padded_query_rows_match_reference(h_q, dtype, s_q, layout):
    """Default backend: query/out rows beyond the metadata tables are sliced away on the host."""
    _skip_unless_trtllm_gen_gpu()
    p, tc = _make_case(h_q, dtype, s_q, varlen=layout == "ragged")
    inputs = _Inputs(p, tc)
    rows = inputs.num_tokens - 1 if s_q == 1 else inputs.num_tokens - s_q
    assert rows >= 1
    out = _out_like(inputs)
    inputs.run(
        out=out, workspace=_trtllm_workspace(), metadata_rows=rows, backend="trtllm-gen"
    )
    torch.cuda.synchronize()
    expected = inputs.reference_rows()
    got = _rows(out, inputs)
    _assert_close(got[:rows], expected[:rows], dtype)
    assert torch.isnan(got[rows:]).all(), "padded query rows must not be written"


@pytest.mark.parametrize(
    "h_q,dtype,s_q",
    [
        pytest.param(64, torch.bfloat16, 2, id="h64-bf16-q2"),
        pytest.param(128, torch.float8_e4m3fn, 4, id="h128-fp8-q4"),
    ],
)
def test_trtllm_gen_caller_owned_counter_buffer(h_q, dtype, s_q):
    """A caller-owned multi-CTA KV counter buffer is validated, reused and graph-replayable."""
    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    _skip_unless_trtllm_gen_gpu()
    p, tc = _make_case(h_q, dtype, s_q, varlen=True)
    inputs = _Inputs(p, tc)
    sm_count = get_device_sm_count(torch.device("cuda:0"))
    nbytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(BATCH, h_q, sm_count)
    counters = torch.zeros(nbytes, dtype=torch.uint8, device="cuda:0")
    workspace = _trtllm_workspace()
    expected = inputs.reference_rows()
    out = _out_like(inputs)
    for _ in range(3):  # counters must self-reset between launches
        out.fill_(float("nan"))
        inputs.run(
            out=out,
            workspace=workspace,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=counters,
        )
        torch.cuda.synchronize()
        _assert_close(_rows(out, inputs), expected, dtype)
    with pytest.raises(ValueError, match="too small"):
        inputs.run(
            out=out,
            workspace=workspace,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=counters[: nbytes - 8],
        )
    with pytest.raises(ValueError, match="only used by backend='trtllm-gen'"):
        inputs.run(
            out=out,
            workspace=_workspace(inputs),
            backend="cake",
            multi_ctas_kv_counter_buffer=counters,
        )
    # graph capture + replay with mutated inputs
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            inputs.run(
                out=out,
                workspace=workspace,
                backend="trtllm-gen",
                multi_ctas_kv_counter_buffer=counters,
            )
    torch.cuda.synchronize()
    p2, tc2 = _make_case(h_q, dtype, s_q, varlen=True)
    fresh = _Inputs(p2, tc2)
    inputs.copy_from(fresh)
    out.fill_(float("nan"))
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(_rows(out, inputs), fresh.reference_rows(), dtype)


# --------------------------------------------------------------------------- CAKE-957: sparse validity (CAKE-944) + workspace contract (CAKE-939)
#
# Cases are built directly in the FlashInfer API vocabulary (no stock
# generator): causal SWA tables with ``seq_lens < 128``, stale or ``-1`` slots
# beyond the window, ``-1`` slots inside the compressed active length,
# non-tile-aligned lengths, a poison page behind every masked column and the
# sglang call conventions. The reference applies the three validity
# predicates of the metadata ABI; combined column ``c`` of row ``t`` is
# attended iff
#
#     c < clamp(sparse_topk_lens[t], 0, sparse_topk)
#     and index != -1
#     and (c >= 128 or c < visible(t)),
#     visible(t) = clamp(seq_lens[b] - (q_len_b - 1 - q_off), 0, 128)
#
# (pinned against the trtllm-gen kernel on B200 / GB300 for CAKE-957). Every
# masked column points at a valid row of a reserved poison page, so the
# output must be bit-identical whether that page holds random values or
# 400.0 -- a leak of weight w moves the output by ~400 w.

SWA_WINDOW = 128
PAGE_SWA = 256
POISON = 400.0
# Value of the beacon rows behind the boundary columns (the last visible SWA
# slot and the last active compressed column of a row); flipping its sign
# must change every row that attends them.
BEACON = 2.0
WORKSPACE_128MIB = 128 * 1024 * 1024
_REL_TOL = {torch.bfloat16: 2e-2, torch.float8_e4m3fn: 1e-1}
# Elementwise tolerances of the Cake DSv4 contract (atol == rtol): bf16 1e-2, fp8 1e-1.  The
# upstream helper ``ref._assert_close`` keeps a tighter bf16 atol (8e-4) that sits below the
# bf16 P-quantisation noise of sums whose O(1) terms cancel (the beacon rows here), so the
# hardening suite checks elementwise closeness at the contract tolerances and leaves the
# finer per-row validity questions to the relative and poison metrics below.
_ABS_TOL = {torch.bfloat16: 1e-2, torch.float8_e4m3fn: 1e-1}


def _assert_close(
    out: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    """Elementwise ``assert_close`` at the contract tolerance of ``dtype`` (bf16 output)."""
    assert out.shape == expected.shape
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    tol = _ABS_TOL[dtype]
    torch.testing.assert_close(out.float(), expected.float(), rtol=tol, atol=tol)


_case_seed = [SEED_BASE + 50_000]


def _next_seed() -> int:
    _case_seed[0] += 1
    return _case_seed[0]


def _random_rows(
    shape, *, offset: float, dtype: torch.dtype, generator
) -> torch.Tensor:
    rows = torch.randn(shape, dtype=torch.float32, device="cuda:0", generator=generator)
    return rows.mul_(0.05).add_(offset).clamp_(-1.0, 1.0).to(dtype)


@dataclasses.dataclass
class _Case:
    """One ``backend="cake"`` call plus everything the reference needs."""

    dtype: torch.dtype
    num_heads: int
    query: torch.Tensor  # dense [B, Q, H, 512] or ragged [T, H, 512]
    swa_kv_cache: torch.Tensor  # [pages, 1, PAGE_SWA, 512] HND
    compressed_kv_cache: torch.Tensor  # [pages, 1, page, 512] HND (may be the SWA pool)
    sparse_indices: torch.Tensor  # [T, sparse_topk] int32, combined form
    sparse_topk_lens: torch.Tensor  # [T] int32, counting the 128 SWA slots
    seq_lens: torch.Tensor  # [B] int32
    cum_seq_lens_q: Optional[torch.Tensor]
    max_q_len: int
    sinks: Optional[torch.Tensor]
    sm_scale: float
    # (pool, first flat row, rows): the poison page behind every masked column.
    poison_rows: list
    # (pool, first flat row, rows): the beacon rows behind the boundary columns
    # (SWA first, compressed second; empty unless built with beacon=True).
    beacon_rows: list = dataclasses.field(default_factory=list)

    @property
    def rows(self) -> int:
        return int(self.sparse_indices.shape[0])

    @property
    def ragged(self) -> bool:
        return self.cum_seq_lens_q is not None

    @property
    def batch_size(self) -> int:
        return int(self.seq_lens.numel())

    def scale(self, value: float):
        if self.dtype == torch.float8_e4m3fn:
            return torch.tensor([value], dtype=torch.float32, device="cuda:0")
        return value

    def nan_out(self) -> torch.Tensor:
        return torch.full(
            self.query.shape, float("nan"), dtype=torch.bfloat16, device="cuda:0"
        )

    def run(
        self,
        *,
        workspace: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        backend: str = "cake",
        float_scales: bool = False,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        result = trtllm_batch_decode_sparse_mla_dsv4(
            query=self.query,
            swa_kv_cache=self.swa_kv_cache,
            workspace_buffer=workspace,
            sparse_indices=self.sparse_indices,
            compressed_kv_cache=self.compressed_kv_cache,
            sparse_topk_lens=self.sparse_topk_lens,
            seq_lens=self.seq_lens,
            out=out,
            bmm1_scale=self.sm_scale if float_scales else self.scale(self.sm_scale),
            bmm2_scale=1.0 if float_scales else self.scale(1.0),
            sinks=self.sinks,
            kv_layout="HND",
            cum_seq_lens_q=self.cum_seq_lens_q,
            max_q_len=self.max_q_len if self.ragged else None,
            enable_pdl=enable_pdl,
            backend=backend,
        )
        return result.reshape(-1, self.num_heads, 512)

    def requirement(self):
        return cake_dsv4_workspace_requirement(
            dtype=self.dtype,
            num_heads=self.num_heads,
            num_query_tokens=self.rows,
            sparse_topk=int(self.sparse_indices.shape[1]),
            compressed_page_size=int(self.compressed_kv_cache.shape[-2]),
            max_q_len=self.max_q_len,
            batch_size=self.batch_size,
            ragged=self.ragged,
        )

    def fill_poison(self, value: float) -> None:
        for pool, first, count in self.poison_rows:
            pool.reshape(-1, 512)[first : first + count].fill_(value)

    def row_requests(self):
        """Per metadata row: owning request, position in it, its query length (int64)."""
        t = torch.arange(self.rows, device="cuda:0")
        if not self.ragged:
            q = self.max_q_len
            return t // q, t % q, torch.full_like(t, q)
        cum = self.cum_seq_lens_q.long()
        lengths = cum[1:] - cum[:-1]
        batch = torch.repeat_interleave(
            torch.arange(lengths.numel(), device="cuda:0"), lengths
        )
        return batch, t - cum[batch], lengths[batch]

    def visible(self) -> torch.Tensor:
        batch, offset, length = self.row_requests()
        return (self.seq_lens.long()[batch] - (length - 1 - offset)).clamp(
            0, SWA_WINDOW
        )

    def reference(self) -> torch.Tensor:
        """fp32 attention over the attendable columns of every row (three predicates)."""
        q = self.query.reshape(-1, self.num_heads, 512).float()
        swa = self.swa_kv_cache.reshape(-1, 512)
        comp = self.compressed_kv_cache.reshape(-1, 512)
        idx = self.sparse_indices.long()
        rows, topk = idx.shape
        col = torch.arange(topk, device="cuda:0")
        active = self.sparse_topk_lens.long().clamp(0, topk)
        visible = self.visible()
        masked = (
            (idx < 0)
            | (col[None, :] >= active[:, None])
            | ((col[None, :] < SWA_WINDOW) & (col[None, :] >= visible[:, None]))
        )
        out = torch.empty(
            (rows, self.num_heads, 512), dtype=torch.float32, device="cuda:0"
        )
        # Scale of the mass every (row, head) attends: the attention-weighted RMS
        # norm of its attendable value rows, times the sink factor.  The
        # relative metric divides by it (see _relative_rows).
        self._reference_scale = torch.zeros(
            (rows, self.num_heads), dtype=torch.float32, device="cuda:0"
        )
        for start in range(0, rows, 32):
            stop = min(rows, start + 32)
            safe = idx[start:stop].clamp_min(0)
            kv = torch.cat(
                (swa[safe[:, :SWA_WINDOW]].float(), comp[safe[:, SWA_WINDOW:]].float()),
                dim=1,
            )
            scores = torch.einsum("nhd,ncd->nhc", q[start:stop], kv) * self.sm_scale
            scores.masked_fill_(masked[start:stop, None, :], float("-inf"))
            lse = scores.logsumexp(dim=-1)  # -inf for a row without attendable columns
            probs = torch.exp(scores - lse[..., None]).nan_to_num_(nan=0.0)
            o = torch.einsum("nhc,ncd->nhd", probs, kv)
            scale = torch.sqrt(torch.einsum("nhc,nc->nh", probs, kv.pow(2).sum(dim=-1)))
            if self.sinks is not None:
                sink_scale = 1.0 / (1.0 + torch.exp(self.sinks[None, :] - lse))
                sink_scale = torch.where(
                    torch.isfinite(sink_scale), sink_scale, torch.zeros_like(sink_scale)
                )
                o = o * sink_scale[..., None]
                scale = scale * sink_scale
            o[~torch.isfinite(lse)] = 0.0
            scale[~torch.isfinite(lse)] = 0.0
            out[start:stop] = o
            self._reference_scale[start:stop] = scale
        return out.to(torch.bfloat16)


def _build_case(
    *,
    num_heads: int,
    dtype: torch.dtype,
    seq_lens: list,
    q_lens: list,
    layout: str,
    compressed_width: int = 0,
    compressed_page: int = 64,
    swa_tail: str = "minus_one",
    compressed_lens: Optional[list] = None,
    minus_one_inside: bool = False,
    alias_compressed: bool = False,
    tile_padded: bool = False,
    beacon: bool = False,
    newest_first: bool = False,
    compressed_pages: int = 64,
    seed: Optional[int] = None,
) -> _Case:
    """Build one causal DSv4 case.

    ``seq_lens`` / ``q_lens`` are per request (``layout="dense"`` needs equal
    ``q_lens``). SWA slot ``j < visible`` of a row holds the token
    ``token_idx - visible + 1 + j`` (ascending, ending at the query token's
    position, the trtllm-gen table convention; ``newest_first``: slot ``j``
    holds ``token_idx - j``, sglang's convention); slots beyond are ``-1``
    (``swa_tail="minus_one"``, sglang's padding) or stale valid rows of the
    SWA pool's poison page (``"stale"``). ``compressed_width`` compressed
    columns hold random valid rows of a page-``compressed_page`` pool of
    ``compressed_pages`` pages up to the
    row's active count (``compressed_lens``, default: the width); columns at
    or beyond it point at that pool's poison page; ``minus_one_inside`` turns
    every seventh active compressed slot into ``-1``. ``alias_compressed``
    passes the SWA pool as the compressed cache (sglang SWA-only layers);
    ``tile_padded`` makes the tables ``[:T]`` views of 64-row-aligned,
    ``-1``-filled parents (sglang prefill). ``beacon`` points the last visible
    SWA slot and the last active compressed column of every row at a beacon
    row (value ``BEACON``, one reserved page per pool).
    """
    gen = torch.Generator(device="cuda:0")
    gen.manual_seed(seed if seed is not None else _next_seed())
    for s, q in zip(seq_lens, q_lens, strict=True):
        assert s >= q >= 1, (s, q)
    batch_size = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device="cuda:0")
    if layout == "dense":
        assert len(set(q_lens)) == 1, q_lens
        q_len = int(q_lens[0])
        rows = batch_size * q_len
        cum = None
        max_q_len = q_len
        t = torch.arange(rows, device="cuda:0")
        batch, q_off, length = t // q_len, t % q_len, torch.full_like(t, q_len)
    else:
        lengths = torch.tensor(q_lens, dtype=torch.int64, device="cuda:0")
        rows = int(lengths.sum())
        max_q_len = int(max(q_lens))
        cum = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device="cuda:0"),
                lengths.cumsum(0).to(torch.int32),
            )
        )
        batch = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda:0"), lengths
        )
        q_off = torch.arange(rows, device="cuda:0") - cum.long()[batch]
        length = lengths[batch]
    # SWA pool: the pages of every request behind a random page table, plus
    # one poison page at the end.
    pages_per_request = [-(-s // PAGE_SWA) for s in seq_lens]
    num_pages = sum(pages_per_request)
    perm = torch.randperm(num_pages, device="cuda:0", generator=gen)
    page_table = torch.zeros(
        (batch_size, max(pages_per_request)), dtype=torch.int64, device="cuda:0"
    )
    start = 0
    for b, n in enumerate(pages_per_request):
        page_table[b, :n] = perm[start : start + n]
        start += n
    extra_pages = 2 if beacon else 1
    swa_pool = _random_rows(
        (num_pages + extra_pages, 1, PAGE_SWA, 512),
        offset=-0.2,
        dtype=dtype,
        generator=gen,
    )
    swa_poison_first = num_pages * PAGE_SWA
    poison_rows = [(swa_pool, swa_poison_first, PAGE_SWA)]
    beacon_rows = []
    row_ids = torch.arange(rows, device="cuda:0")
    if beacon:
        swa_beacon_first = swa_poison_first + PAGE_SWA
        beacon_rows.append((swa_pool, swa_beacon_first, PAGE_SWA))
    token_idx = seq_lens_t.long()[batch] - length + q_off
    visible = (token_idx + 1).clamp(0, SWA_WINDOW)
    j = torch.arange(SWA_WINDOW, device="cuda:0")
    if newest_first:
        token = (token_idx[:, None] - j[None, :]).clamp_min(0)
    else:
        token = (token_idx[:, None] - visible[:, None] + 1 + j[None, :]).clamp_min(0)
    page = (token // PAGE_SWA).clamp_max(page_table.shape[1] - 1)
    flat = page_table[batch[:, None], page] * PAGE_SWA + token % PAGE_SWA
    in_window = j[None, :] < visible[:, None]
    if swa_tail == "stale":
        tail = swa_poison_first + j[None, :].expand_as(flat)
    elif swa_tail == "minus_one":
        tail = torch.full_like(flat, -1)
    else:
        raise ValueError(swa_tail)
    swa_table = torch.where(in_window, flat, tail)
    if beacon:
        # The last visible slot of every row reads a beacon row.
        swa_table = torch.where(
            j[None, :] == visible[:, None] - 1,
            (swa_beacon_first + row_ids % PAGE_SWA)[:, None].expand_as(flat),
            swa_table,
        )
    swa_table = swa_table.to(torch.int32)
    if compressed_width:
        comp_pages = compressed_pages
        comp_pool = _random_rows(
            (comp_pages + extra_pages, 1, compressed_page, 512),
            offset=0.25,
            dtype=dtype,
            generator=gen,
        )
        comp_rows = comp_pages * compressed_page
        poison_rows.append((comp_pool, comp_rows, compressed_page))
        if beacon:
            comp_beacon_first = comp_rows + compressed_page
            beacon_rows.append((comp_pool, comp_beacon_first, compressed_page))
        active = torch.tensor(
            compressed_lens
            if compressed_lens is not None
            else [compressed_width] * rows,
            dtype=torch.int64,
            device="cuda:0",
        )
        assert active.numel() == rows and int(active.max()) <= compressed_width
        c = torch.arange(compressed_width, device="cuda:0")
        picks = torch.randint(
            0, comp_rows, (rows, compressed_width), device="cuda:0", generator=gen
        )
        tail_c = comp_rows + (c[None, :] % compressed_page).expand_as(picks)
        comp_table = torch.where(c[None, :] < active[:, None], picks, tail_c)
        if minus_one_inside:
            inside = (c[None, :] % 7 == 3) & (c[None, :] < active[:, None])
            comp_table = torch.where(
                inside, torch.full_like(comp_table, -1), comp_table
            )
        if beacon:
            # The last active compressed column of every row reads a beacon row.
            comp_table = torch.where(
                c[None, :] == active[:, None] - 1,
                (comp_beacon_first + row_ids % compressed_page)[:, None].expand_as(
                    picks
                ),
                comp_table,
            )
        table = torch.cat((swa_table, comp_table.to(torch.int32)), dim=1)
        lens = (SWA_WINDOW + active).to(torch.int32)
    else:
        comp_pool = (
            swa_pool
            if alias_compressed
            else _random_rows(
                (2, 1, compressed_page, 512), offset=0.25, dtype=dtype, generator=gen
            )
        )
        table = swa_table
        lens = torch.full((rows,), SWA_WINDOW, dtype=torch.int32, device="cuda:0")
    if tile_padded:
        padded = -(-rows // 64) * 64
        parent = torch.full(
            (padded, table.shape[1]), -1, dtype=torch.int32, device="cuda:0"
        )
        parent[:rows] = table
        table = parent[:rows]
        lens_parent = torch.full(
            (padded,), SWA_WINDOW, dtype=torch.int32, device="cuda:0"
        )
        lens_parent[:rows] = lens
        lens = lens_parent[:rows]
    else:
        table = table.contiguous()
        lens = lens.contiguous()
    query = _random_rows((rows, num_heads, 512), offset=0.0, dtype=dtype, generator=gen)
    if layout == "dense":
        query = query.view(batch_size, max_q_len, num_heads, 512)
    sinks = (
        torch.randn((num_heads,), dtype=torch.float32, device="cuda:0", generator=gen)
        * 0.05
    )
    for pool, first, count in beacon_rows:
        pool.reshape(-1, 512)[first : first + count].fill_(BEACON)
    return _Case(
        dtype=dtype,
        num_heads=num_heads,
        query=query,
        swa_kv_cache=swa_pool,
        compressed_kv_cache=comp_pool,
        sparse_indices=table,
        sparse_topk_lens=lens,
        seq_lens=seq_lens_t,
        cum_seq_lens_q=cum,
        max_q_len=max_q_len,
        sinks=sinks,
        sm_scale=512**-0.55,
        poison_rows=poison_rows,
        beacon_rows=beacon_rows,
    )


def _flatten_rows(case: _Case) -> _Case:
    """sglang's draft-token form of a dense ``[B, Q]`` case: ``[B * Q, 1, H, 512]``
    with one causal ``seq_lens`` entry per row (same tables, same answer)."""
    assert not case.ragged
    batch, offset, length = case.row_requests()
    row_lens = (case.seq_lens.long()[batch] - (length - 1 - offset)).to(torch.int32)
    return dataclasses.replace(
        case,
        query=case.query.reshape(case.rows, 1, case.num_heads, 512),
        seq_lens=row_lens.contiguous(),
        max_q_len=1,
    )


def _workspace_128mib() -> torch.Tensor:
    """sglang's fixed, zero-initialised 128 MiB workspace."""
    return torch.zeros(WORKSPACE_128MIB, dtype=torch.uint8, device="cuda:0")


def _relative_rows(
    got: torch.Tensor,
    expected: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalised L2 error per (row, head): ``||got - expected||`` over the scale
    of the mass the row attends (``_Case.reference`` records it: the
    attention-weighted RMS norm of the attendable value rows, times the sink
    factor).  Dividing by ``||expected||`` instead is ill-conditioned on rows
    whose attended values cancel (the SWA pool sits at -0.2 and the compressed
    pool at +0.25, so a row mixing both can sum to nearly zero) and turns the
    FP8 precision of P into a spurious validity failure; a dropped or leaked
    column still moves the error by its full share of the attended mass."""
    err = (got.float() - expected.float()).norm(dim=-1)
    denominator = expected.float().norm(dim=-1) if scale is None else scale.float()
    return err / denominator.clamp_min(1e-3)


def _reference_scale(case: "_Case") -> torch.Tensor:
    """The attended-mass scale of ``case`` (computes the reference if needed)."""
    if getattr(case, "_reference_scale", None) is None:
        case.reference()
    return case._reference_scale


def _row_detail(case: _Case, row: int) -> str:
    return (
        f"row {row}: active={int(case.sparse_topk_lens[row])} "
        f"visible={int(case.visible()[row])}"
    )


def _stock_relative_error(
    case: _Case, expected: torch.Tensor, **run_kwargs
) -> Optional[float]:
    """The stock backend's relative error on the same inputs (None without its cubins).

    Attribution only: a Cake error far above it is a Cake defect, one in the
    same class is the precision level of the shape. The tolerances do not move.
    """
    try:
        stock = case.run(
            workspace=_workspace_128mib(),
            out=case.nan_out(),
            backend="trtllm-gen",
            **run_kwargs,
        )
        torch.cuda.synchronize()
    except (
        Exception
    ):  # diagnostics only: the stock backend must never block the assertion
        torch.cuda.synchronize()
        return None
    return _relative_rows(stock, expected, _reference_scale(case)).max().item()


def _assert_valid(
    got: torch.Tensor,
    case: _Case,
    expected: Optional[torch.Tensor] = None,
    *,
    stock_rel: Optional[float] = None,
):
    """Absolute tolerance of the stock test plus a relative metric (normalised L2
    per row / head); the message names the worst row's active / visible counts
    and, when known, the stock backend's own relative error."""
    expected = case.reference() if expected is None else expected
    rel_rows = _relative_rows(got, expected, _reference_scale(case))
    rel = rel_rows.max().item()
    row, head = divmod(int(rel_rows.argmax().item()), case.num_heads)
    detail = (
        f"route={case.requirement().route} "
        f"max_abs={(got.float() - expected.float()).abs().max().item():.4g} "
        f"rows_over_rel_tol={int((rel_rows.amax(dim=1) > _REL_TOL[case.dtype]).sum())}"
        f"/{case.rows} worst head {head} {_row_detail(case, row)}"
        + (f" stock_rel={stock_rel:.4g}" if stock_rel is not None else "")
    )
    try:
        _assert_close(got, expected, case.dtype)
    except AssertionError as exc:
        raise AssertionError(f"{exc}\n{detail}") from None
    assert rel <= _REL_TOL[case.dtype], (
        f"relative error {rel:.4g} exceeds {_REL_TOL[case.dtype]}; {detail}"
    )
    return expected


def _assert_poison_inert(
    case: _Case, workspace: torch.Tensor, **run_kwargs
) -> torch.Tensor:
    """Run with the poison page at its random values and at POISON: bit-identical
    outputs (masked columns contribute exactly nothing) that match the reference."""
    clean = case.run(workspace=workspace, out=case.nan_out(), **run_kwargs)
    torch.cuda.synchronize()
    case.fill_poison(POISON)
    poisoned = case.run(workspace=workspace, out=case.nan_out(), **run_kwargs)
    torch.cuda.synchronize()
    expected = case.reference()
    stock_rel = _stock_relative_error(case, expected, **run_kwargs)
    delta = (poisoned.float() - clean.float()).abs().amax(dim=(1, 2))
    leaking = delta > 0
    assert torch.equal(clean, poisoned), (
        f"masked columns leak into the output: estimated weight "
        f"{delta.max().item() / POISON:.3g}, {int(leaking.sum())}/{case.rows} rows, "
        f"worst {_row_detail(case, int(delta.argmax().item()))} "
        f"(route {case.requirement().route}"
        + (f", stock_rel={stock_rel:.4g})" if stock_rel is not None else ")")
    )
    _assert_valid(poisoned, case, expected, stock_rel=stock_rel)
    return poisoned


def _assert_boundary_columns_attended(case: _Case, workspace: torch.Tensor) -> None:
    """Flip the sign of the beacon rows behind the boundary columns: a row whose
    output does not move dropped its last visible SWA slot / last active
    compressed column (window or active length applied too short, at any
    granularity). Tolerance-free; every flipped state also matches the reference."""
    base = _assert_poison_inert(case, workspace)
    swa_beacon, *comp_beacon = case.beacon_rows
    pool, first, count = swa_beacon
    pool.reshape(-1, 512)[first : first + count].fill_(-BEACON)
    flipped_swa = case.run(workspace=workspace, out=case.nan_out())
    torch.cuda.synchronize()
    moved = (flipped_swa.float() - base.float()).abs().amax(dim=(1, 2)) > 0
    assert bool(moved.all()), (
        "the last visible SWA slot is not attended in rows "
        f"{(~moved).nonzero().flatten().tolist()} ({case.requirement().route}); "
        + "; ".join(
            _row_detail(case, r) for r in (~moved).nonzero().flatten().tolist()[:4]
        )
    )
    _assert_valid(flipped_swa, case)
    if comp_beacon:
        pool, first, count = comp_beacon[0]
        pool.reshape(-1, 512)[first : first + count].fill_(-BEACON)
        flipped_comp = case.run(workspace=workspace, out=case.nan_out())
        torch.cuda.synchronize()
        moved = (flipped_comp.float() - flipped_swa.float()).abs().amax(dim=(1, 2)) > 0
        has_compressed = case.sparse_topk_lens > SWA_WINDOW
        missing = (~moved & has_compressed).nonzero().flatten().tolist()
        assert not missing, (
            "the last active compressed column is not attended in rows "
            f"{missing} ({case.requirement().route}); "
            + "; ".join(_row_detail(case, r) for r in missing[:4])
        )
        _assert_valid(flipped_comp, case)


def _compressed_width(h_q: int) -> int:
    # Combined widths of the production rows: 384 / 640 / 1152 (page 64).
    return {16: 256, 32: 256, 64: 512, 128: 1024}[h_q]


_LONG_SEQ = 2048


def _layout_lens(layout: str, *, short: bool):
    """(seq_lens, q_lens) per layout; ``short`` puts requests below the 128 window."""
    if layout == "dense-q1":
        seq = (
            [1, 5, 64, 127, 128, 300]
            if short
            else [_LONG_SEQ + 37 * i for i in range(6)]
        )
        return seq, [1] * len(seq)
    if layout == "dense-q4":
        seq = (
            [4, 9, 64, 127, 130, 300]
            if short
            else [_LONG_SEQ + 37 * i for i in range(4)]
        )
        return seq, [4] * len(seq)
    if layout == "ragged":
        q = [1, 3, 7, 12, 5, 2]
        seq = (
            [1, 3, 70, 127, 130, 300]
            if short
            else [_LONG_SEQ + 37 * i for i in range(6)]
        )
        return seq, q
    raise ValueError(layout)


def _mixed_compressed_lens(rows: int, width: int) -> list:
    """Non-tile-aligned active counts (not multiples of 128, 64 or 4), incl. 0 and the width."""
    pattern = [37, 0, 131, 1, 253, 65, width - 1, 129, 3, width, 255, 63]
    return [min(width, pattern[i % len(pattern)]) for i in range(rows)]


def _layout_kind(layout: str) -> str:
    return "ragged" if layout == "ragged" else "dense"


_VALIDITY_CASES = [
    pytest.param(h, d, id=f"h{h}-{'bf16' if d == torch.bfloat16 else 'fp8'}")
    for h in (32, 64, 128)
    for d in DTYPES
]
_LAYOUTS = ["dense-q1", "dense-q4", "ragged"]


@pytest.mark.parametrize("h_q,dtype", _VALIDITY_CASES)
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_poisoned_tail_is_bit_identical(h_q, dtype, layout):
    """Columns at or beyond the active length hold valid indices into a poison page:
    the output is bit-identical with the page at random values and at 400.0."""
    _skip_unless_cake_gpu()
    seq, q = _layout_lens(layout, short=False)
    width = _compressed_width(h_q)
    rows = sum(q) if layout == "ragged" else len(seq) * q[0]
    case = _build_case(
        num_heads=h_q,
        dtype=dtype,
        seq_lens=seq,
        q_lens=q,
        layout=_layout_kind(layout),
        compressed_width=width,
        compressed_lens=[width // 2 + (i % 3) for i in range(rows)],
    )
    _assert_poison_inert(case, _workspace_128mib())


@pytest.mark.parametrize("h_q,dtype", _VALIDITY_CASES)
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_minus_one_inside_the_compressed_active_length(h_q, dtype, layout):
    """``-1`` slots inside the compressed active length are neither attended nor dereferenced."""
    _skip_unless_cake_gpu()
    seq, q = _layout_lens(layout, short=False)
    width = _compressed_width(h_q)
    rows = sum(q) if layout == "ragged" else len(seq) * q[0]
    case = _build_case(
        num_heads=h_q,
        dtype=dtype,
        seq_lens=seq,
        q_lens=q,
        layout=_layout_kind(layout),
        compressed_width=width,
        compressed_lens=[width - (i % 5) for i in range(rows)],
        minus_one_inside=True,
    )
    assert bool((case.sparse_indices[:, SWA_WINDOW:] == -1).any())
    _assert_poison_inert(case, _workspace_128mib())


@pytest.mark.parametrize("h_q,dtype", _VALIDITY_CASES)
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_non_tile_aligned_active_lengths(h_q, dtype, layout):
    """Active lengths that are no multiple of 128 / 64 / 4 (incl. 0 and the full width)
    are honoured at column granularity."""
    _skip_unless_cake_gpu()
    seq, q = _layout_lens(layout, short=False)
    width = _compressed_width(h_q)
    rows = sum(q) if layout == "ragged" else len(seq) * q[0]
    case = _build_case(
        num_heads=h_q,
        dtype=dtype,
        seq_lens=seq,
        q_lens=q,
        layout=_layout_kind(layout),
        compressed_width=width,
        compressed_lens=_mixed_compressed_lens(rows, width),
    )
    _assert_poison_inert(case, _workspace_128mib())


_WINDOW_FORMS = [
    pytest.param("dense-q1", False, id="dense-q1-swa"),
    pytest.param("dense-q1", True, id="dense-q1-c4"),
    pytest.param("dense-q4", True, id="dense-q4-c4"),
    pytest.param("ragged", False, id="ragged-swa"),
    pytest.param("ragged", True, id="ragged-c4"),
]


def _short_window_case(
    h_q, dtype, layout, compressed, *, swa_tail: str, beacon: bool = False
) -> _Case:
    """Requests below the 128 window (q_len 1 / 4 / ragged), mixed active lengths."""
    seq, q = _layout_lens(layout, short=True)
    width = _compressed_width(h_q) if compressed else 0
    rows = sum(q) if layout == "ragged" else len(seq) * q[0]
    case = _build_case(
        num_heads=h_q,
        dtype=dtype,
        seq_lens=seq,
        q_lens=q,
        layout=_layout_kind(layout),
        compressed_width=width,
        compressed_lens=_mixed_compressed_lens(rows, width) if width else None,
        swa_tail=swa_tail,
        beacon=beacon,
    )
    visible = case.visible()
    assert int(visible.min()) >= 1 and int(visible.max()) == SWA_WINDOW
    assert int((visible < SWA_WINDOW).sum()) > 0
    return case


@pytest.mark.parametrize("h_q,dtype", _VALIDITY_CASES)
@pytest.mark.parametrize("layout,compressed", _WINDOW_FORMS)
def test_short_seq_lens_apply_the_causal_swa_window(h_q, dtype, layout, compressed):
    """``seq_lens < 128`` with stale valid SWA slots beyond the window: only the first
    ``visible(t)`` SWA columns are attended, per token (causal offset inside MTP /
    prefill requests). The test of the window itself: the slots beyond
    ``visible`` are valid rows of the poison page, so a missing window leaks
    (needs the CAKE-957 exports)."""
    _skip_unless_cake_gpu()
    case = _short_window_case(h_q, dtype, layout, compressed, swa_tail="stale")
    _assert_poison_inert(case, _workspace_128mib())


@pytest.mark.parametrize("h_q,dtype", _VALIDITY_CASES)
@pytest.mark.parametrize("layout,compressed", _WINDOW_FORMS)
def test_minus_one_padding_beyond_the_window_is_masked(h_q, dtype, layout, compressed):
    """sglang's table convention for short requests: ``-1`` in every SWA slot beyond
    the window. The ``-1`` slots are masked and never dereferenced. This does not
    observe the window (the padding already implies it); the window test is the
    stale-slot one above."""
    _skip_unless_cake_gpu()
    case = _short_window_case(h_q, dtype, layout, compressed, swa_tail="minus_one")
    _assert_poison_inert(case, _workspace_128mib())


@pytest.mark.parametrize("h_q,dtype", _VALIDITY_CASES)
@pytest.mark.parametrize("layout,compressed", _WINDOW_FORMS)
def test_boundary_columns_are_attended(h_q, dtype, layout, compressed):
    """The last visible SWA slot and the last active compressed column of every row
    must be attended: flipping the beacon rows behind them must move every row.
    Catches a window or an active length applied too short at any granularity
    (a tile-rounded length drops the boundary column and the output stays put),
    independently of the tolerances. Slots beyond the window are ``-1``, columns
    beyond the active length point at the poison page."""
    _skip_unless_cake_gpu()
    case = _short_window_case(
        h_q, dtype, layout, compressed, swa_tail="minus_one", beacon=True
    )
    assert len(case.beacon_rows) == (2 if compressed else 1)
    _assert_boundary_columns_attended(case, _workspace_128mib())


_SGLANG_HEADS = [pytest.param(h, id=f"h{h}") for h in (16, 32, 64, 128)]


def _sglang_case(h_q: int, form: str, *, compressed: bool, tail: str) -> _Case:
    """sglang's DSv4 trtllm-backend call shapes (FP8 query, float scales).

    decode: ``[B, 1, H, 512]`` with one ``seq_lens`` entry per row; mtp-dense:
    ``[B, 4, H, 512]``; mtp-wide: the same with 17 requests (68 rows); mtp-flat:
    the same batch as mtp-dense as sglang passes it (one row per draft token,
    per-row causal ``seq_lens``); prefill: ragged rows with ``cum_seq_lens_q`` /
    ``max_q_len`` and 64-row tile-padded tables.
    """
    width = _compressed_width(h_q) if compressed else 0
    kwargs = dict(
        num_heads=h_q,
        dtype=torch.float8_e4m3fn,
        compressed_width=width,
        compressed_page=64,
        swa_tail=tail,
        alias_compressed=not compressed,
    )
    if form == "decode":
        seq = [1, 5, 64, 127, 128, 300, 2048, 4097]
        rows = len(seq)
        return _build_case(
            seq_lens=seq,
            q_lens=[1] * rows,
            layout="dense",
            compressed_lens=_mixed_compressed_lens(rows, width) if width else None,
            **kwargs,
        )
    if form in ("mtp-dense", "mtp-flat", "mtp-wide"):
        seq = [4, 70, 127, 300, 2048]
        if form == "mtp-wide":
            seq = seq * 3 + [513, 1025]
        rows = 4 * len(seq)
        case = _build_case(
            seq_lens=seq,
            q_lens=[4] * len(seq),
            layout="dense",
            compressed_lens=_mixed_compressed_lens(rows, width) if width else None,
            **kwargs,
        )
        return _flatten_rows(case) if form == "mtp-flat" else case
    if form == "prefill":
        q = [1, 3, 40, 130, 12]
        seq = [1, 3, 70, 130, 300]
        rows = sum(q)
        return _build_case(
            seq_lens=seq,
            q_lens=q,
            layout="ragged",
            tile_padded=True,
            compressed_lens=_mixed_compressed_lens(rows, width) if width else None,
            **kwargs,
        )
    raise ValueError(form)


@pytest.mark.parametrize("h_q", _SGLANG_HEADS)
@pytest.mark.parametrize("form", ["decode", "mtp-dense", "mtp-flat", "prefill"])
@pytest.mark.parametrize("tail", ["minus_one", "stale"])
def test_sglang_swa_only_convention(h_q, form, tail):
    """Constant ``sparse_topk_lens = 128``, ``compressed_kv_cache is swa_kv_cache``,
    page-256 HND pool, FP8 query, float scales; ``-1`` or stale slots beyond the
    window of short requests."""
    _skip_unless_cake_gpu()
    case = _sglang_case(h_q, form, compressed=False, tail=tail)
    assert case.compressed_kv_cache is case.swa_kv_cache
    assert bool((case.sparse_topk_lens == SWA_WINDOW).all())
    assert case.sparse_indices.shape[1] == SWA_WINDOW
    _assert_poison_inert(case, _workspace_128mib(), float_scales=True)


@pytest.mark.parametrize("form", ["decode", "mtp-dense", "mtp-wide"])
def test_enable_pdl_lowhead_prefill_matches_and_waits_for_the_query_producer(
    monkeypatch, form
):
    """The SWA-only FP8/H64 rows of one wave run the low-head prefill route: on
    SM107 its V-block sibling up to 27 rows (8 and 20 rows) and the route's
    program above the early-PV sibling's 64 rows (68 rows), elsewhere the
    route's program.  As a programmatic dependent each is bitwise equal to the
    plain launch and, inside a graph, reads the query the copy right before it
    writes."""
    _skip_unless_cake_gpu(sm107=True)
    import flashinfer.mla.cake_dsv4 as cake

    launched: list = []
    original = cake._prepare_variant

    def spy(variant, **kwargs):
        launched.append(variant)
        return original(variant, **kwargs)

    monkeypatch.setattr(cake, "_prepare_variant", spy)
    case = _sglang_case(64, form, compressed=False, tail="minus_one")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    route = _route(
        dtype=case.dtype,
        num_heads=case.num_heads,
        max_q_len=case.max_q_len,
        ragged=case.ragged,
        sparse_topk=SWA_WINDOW,
        batch_size=case.batch_size,
        compressed_page_size=int(case.compressed_kv_cache.shape[-2]),
        num_query_tokens=case.rows,
        sm_count=sm_count,
    )
    assert route == "fp8_lowhead_prefill", route
    workspace = _workspace_128mib()
    plain = case.run(workspace=workspace, out=case.nan_out(), float_scales=True)
    assert set(launched) == {_lowhead_prefill_variant(case.rows)}, launched
    pdl = case.run(
        workspace=workspace, out=case.nan_out(), float_scales=True, enable_pdl=True
    )
    torch.cuda.synchronize()
    assert torch.equal(plain, pdl)

    source = case.query.clone()
    case.query.zero_()
    out = case.nan_out()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.query.copy_(source)
        case.run(workspace=workspace, out=out, float_scales=True, enable_pdl=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.reshape(plain.shape), plain)


def test_enable_pdl_h64_cluster_body_matches_and_waits_for_the_query_producer(
    monkeypatch,
):
    """12 FP8/H64 tokens of sparse_topk 260 run the H64 cluster program (the
    route on SM100; SM103 routes them to the persistent body, so the test
    selects the program there); as a programmatic dependent it is bitwise
    equal to the plain launch and, inside a graph, reads the query the copy
    right before it writes."""
    _skip_unless_cake_gpu()
    import flashinfer.mla.cake_dsv4 as cake

    monkeypatch.setattr(cake, "_route", lambda **kwargs: "fp8_lowhead_h64")
    calls = _count_dispatches(monkeypatch)
    case = _build_case(
        num_heads=64,
        dtype=torch.float8_e4m3fn,
        seq_lens=[_LONG_SEQ + 37 * i for i in range(3)],
        q_lens=[5, 4, 3],
        layout="ragged",
        compressed_width=132,
        compressed_page=2,
    )
    workspace = _workspace_128mib()
    plain = case.run(workspace=workspace, out=case.nan_out(), float_scales=True)
    torch.cuda.synchronize()
    assert {route for route, _ in calls} == {"fp8_lowhead_h64"}, calls
    assert torch.isfinite(plain).all()
    _assert_valid(plain, case)
    pdl = case.run(
        workspace=workspace, out=case.nan_out(), float_scales=True, enable_pdl=True
    )
    torch.cuda.synchronize()
    assert torch.equal(plain.view(torch.int16), pdl.view(torch.int16))

    source = case.query.clone()
    case.query.zero_()
    out = case.nan_out()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.query.copy_(source)
        case.run(workspace=workspace, out=out, float_scales=True, enable_pdl=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        out.reshape(plain.shape).view(torch.int16), plain.view(torch.int16)
    )


@pytest.mark.parametrize("h_q", _SGLANG_HEADS)
@pytest.mark.parametrize("form", ["decode", "mtp-flat", "prefill"])
def test_sglang_compressed_c4_convention(h_q, form):
    """``sparse_indices[:, 128:] = extra_indices``, ``sparse_topk_lens = extra_topk_lengths
    + 128`` (non-tile-aligned), page-64 compressed pool, ``-1`` SWA padding beyond
    the window, poisoned rows behind the inactive compressed columns."""
    _skip_unless_cake_gpu()
    case = _sglang_case(h_q, form, compressed=True, tail="minus_one")
    assert case.sparse_indices.shape[1] == SWA_WINDOW + _compressed_width(h_q)
    _assert_poison_inert(case, _workspace_128mib(), float_scales=True)


# --------------------------------------------------------------------------- CAKE-939: workspace / counter contract
def _count_dispatches(monkeypatch) -> list:
    """Record every (route, rows) the host dispatches (one entry per launch chunk)."""
    import flashinfer.mla.cake_dsv4 as cake

    calls: list = []
    original = cake._dispatch_route

    def spy(route, launcher):
        calls.append((route, launcher.values["num_query_tokens"]))
        return original(route, launcher)

    monkeypatch.setattr(cake, "_dispatch_route", spy)
    return calls


def _tiling_case(*, layout: str, short: bool = False) -> _Case:
    """bf16 H32 topk128x (page 2, three KV tiles): the split-merge counter route and
    the only route whose single launch can outgrow a fixed buffer."""
    if layout == "ragged":
        seq = (
            [1, 3, 70, 127, 130, 300, 2048, 4097]
            if short
            else [_LONG_SEQ + 37 * i for i in range(8)]
        )
        q = [1, 3, 12, 12, 12, 12, 12, 12] if short else [12] * 8
    else:
        q = [4] * 24
        seq = (
            ([4, 9, 64, 127, 130, 300] * 4)
            if short
            else [_LONG_SEQ + 37 * i for i in range(24)]
        )
    rows = sum(q)
    return _build_case(
        num_heads=32,
        dtype=torch.bfloat16,
        seq_lens=seq,
        q_lens=q,
        layout=layout,
        compressed_width=132,
        compressed_page=2,
        compressed_lens=_mixed_compressed_lens(rows, 132),
        swa_tail="stale" if short else "minus_one",
    )


def test_fixed_128mib_workspace_admits_1024_fp8_h128_rows(monkeypatch):
    """1024 dense FP8/H128 rows (persistent one-split route) run in one launch inside
    sglang's fixed buffer: the carve is the LSE region only (CAKE-939)."""
    _skip_unless_cake_gpu()
    calls = _count_dispatches(monkeypatch)
    seq = [200 + (i % 7) for i in range(1024)]
    case = _build_case(
        num_heads=128,
        dtype=torch.float8_e4m3fn,
        seq_lens=seq,
        q_lens=[1] * 1024,
        layout="dense",
        alias_compressed=True,
    )
    req = case.requirement()
    # SM103 keeps SWA-only decode launches on the uniform-gather program.
    assert req.route in (
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
    )
    assert req.num_splits == 1
    assert req.single_launch_bytes <= WORKSPACE_128MIB
    assert req.rows_per_launch(WORKSPACE_128MIB) == 1024
    workspace = _workspace_128mib()
    out = case.run(workspace=workspace, out=case.nan_out(), float_scales=True)
    torch.cuda.synchronize()
    assert calls == [(req.route, 1024)]
    _assert_valid(out, case)


@pytest.mark.parametrize("layout", ["ragged", "dense"])
def test_row_tiled_launches_match_the_single_launch(monkeypatch, layout):
    """A workspace below the single-launch carve tiles the rows (CAKE-939): same
    result as one launch, no allocation after warm-up, graph-replayable."""
    _skip_unless_cake_gpu()
    calls = _count_dispatches(monkeypatch)
    case = _tiling_case(layout=layout)
    req = case.requirement()
    assert req.route == "bf16_h32_topk128x_early_v47" and req.num_splits == 3
    rows = case.rows
    chunk = 37
    small_bytes = cake_dsv4_workspace_layout(
        chunk, 32, req.num_splits, num_query_offsets=case.batch_size + 1
    ).total_bytes
    assert req.rows_per_launch(small_bytes) == chunk
    full = torch.zeros(req.single_launch_bytes, dtype=torch.uint8, device="cuda:0")
    small = torch.zeros(small_bytes, dtype=torch.uint8, device="cuda:0")
    expected = case.reference()
    single = case.run(workspace=full, out=case.nan_out())
    torch.cuda.synchronize()
    assert calls == [(req.route, rows)]
    _assert_valid(single, case, expected)
    calls.clear()
    tiled = case.run(workspace=small, out=case.nan_out())
    torch.cuda.synchronize()
    assert calls == [(req.route, n) for n in [chunk] * (rows // chunk) + [rows % chunk]]
    _assert_valid(tiled, case, expected)
    torch.testing.assert_close(tiled.float(), single.float(), rtol=1e-2, atol=1e-2)
    # Warm (dense request offsets cached), then no allocation on the call path.
    out = case.nan_out()
    case.run(workspace=small, out=out)
    torch.cuda.synchronize()
    requests = _allocation_requests()
    case.run(workspace=small, out=out)
    torch.cuda.synchronize()
    assert _allocation_requests() == requests, "tiled call allocated"
    # Capture + replay: the per-chunk offset writes and launches are recorded.
    graph = torch.cuda.CUDAGraph()
    out.fill_(0.0)
    with torch.cuda.graph(graph):
        case.run(workspace=small, out=out)
    torch.cuda.synchronize()
    replay_requests = _allocation_requests()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert _allocation_requests() == replay_requests, "graph replay allocated"
    _assert_valid(out.reshape(-1, 32, 512), case, expected)


def test_row_tiled_launches_apply_the_window_in_every_chunk():
    """Chunks after the first resolve their requests through the shifted offsets:
    short requests with stale SWA slots keep the causal window across chunk
    boundaries (needs the CAKE-957 exports)."""
    _skip_unless_cake_gpu()
    case = _tiling_case(layout="ragged", short=True)
    req = case.requirement()
    small_bytes = cake_dsv4_workspace_layout(
        29, 32, req.num_splits, num_query_offsets=case.batch_size + 1
    ).total_bytes
    assert case.rows == 76 and req.rows_per_launch(small_bytes) == 29
    assert int((case.visible() < SWA_WINDOW).sum()) > 0
    small = torch.zeros(small_bytes, dtype=torch.uint8, device="cuda:0")
    _assert_poison_inert(case, small)
    dense = _tiling_case(layout="dense", short=True)
    req = dense.requirement()
    small_bytes = cake_dsv4_workspace_layout(
        29, 32, req.num_splits, num_query_offsets=dense.batch_size + 1
    ).total_bytes
    _assert_poison_inert(
        dense, torch.zeros(small_bytes, dtype=torch.uint8, device="cuda:0")
    )


def test_counters_are_zeroed_inside_capture_without_priming():
    """A workspace never seen eagerly can be captured: the graph carries the zero fill
    of the counters it uses, replays are correct and allocation-free, and the
    workspace stays unregistered until an eager call primes it (CAKE-939)."""
    import flashinfer.mla.cake_dsv4 as cake

    _skip_unless_cake_gpu()
    case = _build_case(
        num_heads=32,
        dtype=torch.bfloat16,
        seq_lens=[_LONG_SEQ + 37 * i for i in range(3)],
        q_lens=[3, 4, 5],
        layout="ragged",
        compressed_width=132,
        compressed_page=2,
        compressed_lens=_mixed_compressed_lens(12, 132),
    )
    req = case.requirement()
    assert (
        req.route == "bf16_h32_topk128x_early_v47"
        and req.plan.merge_groups_per_row == 4
    )
    expected = case.reference()
    # Warm the JIT module and the descriptor storage of these tensors through
    # another workspace; the workspace under test is never used eagerly.
    warm = torch.zeros(req.single_launch_bytes, dtype=torch.uint8, device="cuda:0")
    _assert_valid(case.run(workspace=warm, out=case.nan_out()), case, expected)
    workspace = torch.empty(req.single_launch_bytes, dtype=torch.uint8, device="cuda:0")
    workspace.fill_(0xFF)  # garbage counters
    raw = cake._workspace_bytes(workspace)
    assert not cake._counters_primed(workspace, raw)
    out = case.nan_out()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        case.run(workspace=workspace, out=out)
    torch.cuda.synchronize()
    assert not cake._counters_primed(workspace, raw)
    assert torch.all(raw[1024 : 1024 + 12 * 4 * 4] == 0xFF), "capture executed nothing"
    for _ in range(3):
        out.fill_(float("nan"))
        requests = _allocation_requests()
        graph.replay()
        torch.cuda.synchronize()
        assert _allocation_requests() == requests, "graph replay allocated"
        _assert_valid(out.reshape(-1, 32, 512), case, expected)
    assert torch.all(raw[1024 : 1024 + 12 * 4 * 4].view(torch.uint32) == 0)
    # Eager first use through the still-unregistered workspace primes it.
    workspace.fill_(0xFF)
    _assert_valid(case.run(workspace=workspace, out=case.nan_out()), case, expected)
    torch.cuda.synchronize()
    assert cake._counters_primed(workspace, raw)
    assert torch.all(raw[1024 : cake._PARTIAL_OFFSET] == 0)


def test_workspace_requirement_matches_the_launch(monkeypatch):
    """The exact single-launch carve runs in one launch, the minimum in one-row
    launches, one byte less is rejected (CAKE-939)."""
    _skip_unless_cake_gpu()
    calls = _count_dispatches(monkeypatch)
    case = _build_case(
        num_heads=64,
        dtype=torch.bfloat16,
        seq_lens=[_LONG_SEQ + 37 * i for i in range(3)],
        q_lens=[3, 4, 5],
        layout="ragged",
        compressed_width=512,
        compressed_lens=_mixed_compressed_lens(12, 512),
    )
    req = case.requirement()
    assert req.route == "bf16_h64_compressed_q8_v38" and req.num_splits == 5
    expected = case.reference()
    exact = torch.zeros(req.single_launch_bytes, dtype=torch.uint8, device="cuda:0")
    _assert_valid(case.run(workspace=exact, out=case.nan_out()), case, expected)
    torch.cuda.synchronize()
    assert calls == [(req.route, 12)]
    calls.clear()
    minimum = torch.zeros(req.minimum_bytes, dtype=torch.uint8, device="cuda:0")
    _assert_valid(case.run(workspace=minimum, out=case.nan_out()), case, expected)
    torch.cuda.synchronize()
    assert calls == [(req.route, 1)] * 12
    with pytest.raises(ValueError, match="needs at least"):
        case.run(workspace=minimum[: req.minimum_bytes - 128], out=case.nan_out())


# ---------------------------------------------------------- caller out off the store alignment
_BF16 = torch.bfloat16
_FP8 = torch.float8_e4m3fn


def _align_case(route, heads, dtype, q_lens, layout="dense", width=0, page=64, form=""):
    """One call of ``route`` (SWA-only when ``width`` is 0)."""
    args = (route, heads, dtype, q_lens, layout, width, page)
    return pytest.param(*args, id=f"{route}-{form}" if form else route)


# ---------------------------------------------------------- index tables off the 16-byte row loads
# The SWA-only FP8 8-head rows of the low-head prefill route on SM107's early-PV
# sibling (48 rows: above the V-block sibling's 26 rows on 212 SMs and 27 on
# 216, at most 62 rows on 212 SMs and 64 on 216) and on the route's own program
# above it (68 rows); elsewhere both run the route's program.  Then
# the persistent FP8/H128 route, whose topk4x rows every architecture builds and
# routes there, and two routes SM107 does not build (the tests skip there).
_INDEX_ALIGN_CASES = [
    _align_case("fp8_lowhead_prefill", 8, _FP8, [4] * 12, form="48-rows"),
    _align_case("fp8_lowhead_prefill", 8, _FP8, [4] * 17, form="68-rows"),
    _align_case("fp8_h128_prefill_source_persistent", 128, _FP8, [64, 64], width=512),
    _align_case("fp8_lowhead_one_partition", 8, _FP8, [4] * 6, width=128),
    _align_case(
        "bf16_h128_topk128x", 128, _BF16, [20] * 2, width=256, page=2, form="row_first"
    ),
]
_SM107_INDEX_ALIGN_ROUTES = (
    "fp8_lowhead_prefill",
    "fp8_h128_prefill_source_persistent",
)


@pytest.mark.parametrize(
    "route,num_heads,dtype,q_lens,layout,width,page", _INDEX_ALIGN_CASES
)
def test_index_tables_off_the_row_loads_raise_before_launch(
    monkeypatch, route, num_heads, dtype, q_lens, layout, width, page
):
    """The BF16 / FP8 programs load index rows 16 bytes at a time.  A table 4
    bytes into its parent (``parent[:, 1:]`` of a ``[T, W + 1]`` parent) or one
    with an odd row stride (``parent[:, :W]``) raises ValueError before any
    launch and leaves ``out`` untouched; a view 16 bytes into a ``[T, W + 4]``
    parent runs in place (same launches and programs, nothing allocated) with
    bitwise the output of the contiguous table."""
    _skip_unless_cake_gpu(sm107=route in _SM107_INDEX_ALIGN_ROUTES)
    import flashinfer.mla.cake_dsv4 as cake

    calls = _count_dispatches(monkeypatch)
    launched: list = []
    original = cake._prepare_variant

    def spy(variant, **kwargs):
        launched.append(variant)
        return original(variant, **kwargs)

    monkeypatch.setattr(cake, "_prepare_variant", spy)
    case = _build_case(
        num_heads=num_heads,
        dtype=dtype,
        seq_lens=[_LONG_SEQ + 37 * i for i in range(len(q_lens))],
        q_lens=q_lens,
        layout=layout,
        compressed_width=width,
        compressed_page=page,
    )
    table = case.sparse_indices
    assert table.is_contiguous() and table.data_ptr() % 16 == 0
    rows, cols = table.shape
    workspace = _workspace_128mib()
    expected = case.run(workspace=workspace, out=case.nan_out(), float_scales=True)
    torch.cuda.synchronize()
    assert calls and {r for r, _ in calls} == {route}
    if route == "fp8_lowhead_prefill":
        assert set(launched) == {_lowhead_prefill_variant(rows)}, launched
    contiguous = (list(calls), set(launched))

    off = torch.full((rows, cols + 1), -1, dtype=torch.int32, device="cuda:0")
    off[:, 1:] = table
    odd = torch.full((rows, cols + 1), -1, dtype=torch.int32, device="cuda:0")
    odd[:, :cols] = table
    calls.clear()
    launched.clear()
    for bad, message in (
        (off[:, 1:], r"^sparse_indices must start at a 16-byte aligned address"),
        (odd[:, :cols], r"^sparse_indices must have a row stride that is a multiple"),
    ):
        out = case.nan_out()
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(case, sparse_indices=bad).run(
                workspace=workspace, out=out, float_scales=True
            )
        torch.cuda.synchronize()
        assert not calls and not launched, (calls, launched)
        assert torch.isnan(out).all()

    wide = torch.full((rows, cols + 4), -1, dtype=torch.int32, device="cuda:0")
    wide[:, 4:] = table
    view = wide[:, 4:]
    assert view.data_ptr() == wide.data_ptr() + 16 and view.stride(0) == cols + 4
    in_place = dataclasses.replace(case, sparse_indices=view)
    out = case.nan_out()
    # Only the allocation count needs the native caching allocator.
    native = torch.cuda.get_allocator_backend() == "native"
    requests = _allocation_requests() if native else None
    got = in_place.run(workspace=workspace, out=out, float_scales=True)
    torch.cuda.synchronize()
    assert (list(calls), set(launched)) == contiguous, (calls, launched)
    if native:
        assert _allocation_requests() == requests, "an aligned index view was copied"
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


def test_row_tiled_launches_bind_an_aligned_index_view_in_place(monkeypatch):
    """Row-tiled launches slice the index table by row.  A view 16 bytes into a
    ``[T, W + 4]`` parent binds every chunk's rows in place, including chunks
    that start at a row that is not a multiple of 4, with bitwise the tiled
    output of the contiguous table; an odd row stride raises before the first
    chunk."""
    _skip_unless_cake_gpu()
    import flashinfer.mla.cake_dsv4 as cake

    bound: list = []
    original = cake._dispatch_route

    def spy(route, launcher):
        values = launcher.values
        bound.append(
            (route, values["num_query_tokens"], values["swa_indices"].data_ptr())
        )
        return original(route, launcher)

    monkeypatch.setattr(cake, "_dispatch_route", spy)
    case = _tiling_case(layout="dense")
    req = case.requirement()
    assert req.route == "bf16_h32_topk128x_early_v47" and req.num_splits == 3
    rows = case.rows
    chunk = 37
    small_bytes = cake_dsv4_workspace_layout(
        chunk, 32, req.num_splits, num_query_offsets=case.batch_size + 1
    ).total_bytes
    assert req.rows_per_launch(small_bytes) == chunk
    small = torch.zeros(small_bytes, dtype=torch.uint8, device="cuda:0")
    firsts = list(range(0, rows, chunk))
    assert len(firsts) > 1 and any(first % 4 for first in firsts)
    expected = case.run(workspace=small, out=case.nan_out())
    torch.cuda.synchronize()
    tiles = [(req.route, min(chunk, rows - first)) for first in firsts]
    assert [(r, n) for r, n, _ in bound] == tiles

    table = case.sparse_indices
    cols = int(table.shape[1])
    wide = torch.full((rows, cols + 4), -1, dtype=torch.int32, device="cuda:0")
    wide[:, 4:] = table
    view = wide[:, 4:]
    assert view.data_ptr() == wide.data_ptr() + 16 and view.stride(0) == cols + 4
    bound.clear()
    got = dataclasses.replace(case, sparse_indices=view).run(
        workspace=small, out=case.nan_out()
    )
    torch.cuda.synchronize()
    row_bytes = view.stride(0) * view.element_size()
    assert bound == [
        (r, n, view.data_ptr() + first * row_bytes)
        for (r, n), first in zip(tiles, firsts, strict=True)
    ]
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))

    odd = torch.full((rows, cols + 1), -1, dtype=torch.int32, device="cuda:0")
    odd[:, :cols] = table
    bound.clear()
    out = case.nan_out()
    with pytest.raises(
        ValueError, match=r"^sparse_indices must have a row stride that is a multiple"
    ):
        dataclasses.replace(case, sparse_indices=odd[:, :cols]).run(
            workspace=small, out=out
        )
    torch.cuda.synchronize()
    assert not bound and torch.isnan(out).all()


# One call per route, and per O-writing program of the multi-program routes.
_OUT_ALIGN_CASES = [
    _align_case(
        "fp8_lowhead_one_partition",
        32,
        _FP8,
        [4, 2, 3],
        "ragged",
        256,
        form="q4-ragged",
    ),
    _align_case("fp8_lowhead_one_partition", 32, _FP8, [4] * 3, width=256, form="q4"),
    _align_case("fp8_lowhead_one_partition", 32, _FP8, [1] * 3, width=256, form="q1"),
    _align_case("fp8_lowhead_prefill", 32, _FP8, [4] * 3),
    # 66 tokens: above SM107's early-PV sibling ceiling, the route's own program there.
    _align_case("fp8_lowhead_prefill", 32, _FP8, [22] * 3, form="66-rows"),
    # 12 tokens, sparse_topk 260: two complete tiles and a partial third.
    _align_case("fp8_lowhead_h64", 64, _FP8, [5, 4, 3], "ragged", 132, page=2),
    _align_case("fp8_h64_prefill_source_persistent_m64", 32, _FP8, [257]),
    _align_case(
        "fp8_h64_prefill_source_persistent_m64_multi_tile", 32, _FP8, [257], width=256
    ),
    _align_case(
        "fp8_h128_prefill_source_persistent_uniform", 128, _FP8, [4] * 3, width=1024
    ),
    _align_case("fp8_h128_prefill_source_persistent", 128, _FP8, [64, 64], width=512),
    _align_case("fp8_h64_source_exact", 64, _FP8, [257, 100], "ragged", 512),
    _align_case("fp8_h8_h16_source_exact", 16, _FP8, [5, 3, 4], "ragged", 128),
    _align_case("bf16_h8_h16_source_exact", 16, _BF16, [5, 3, 4], "ragged", 128),
    _align_case("bf16_h8_swa128_v43", 8, _BF16, [4] * 3),
    _align_case("bf16_h16_h32_swa128_v44", 32, _BF16, [4] * 3),
    _align_case("bf16_h32_topk128x_early_v47", 32, _BF16, [4] * 3, width=256),
    _align_case("bf16_h64_prefill", 64, _BF16, [257]),
    _align_case("bf16_h64_guard_q_tma_batch_r25", 64, _BF16, [8, 8]),
    _align_case("bf16_swa128_single_cta", 64, _BF16, [4] * 3),
    _align_case("bf16_h64_compressed_q8_v38", 64, _BF16, [4] * 3, width=512),
    _align_case("bf16_h128_prefill_v42", 128, _BF16, [64]),
    _align_case("bf16_h128_swa128", 128, _BF16, [4] * 3),
    _align_case("bf16_h128_topk4x_v52", 128, _BF16, [4] * 3, width=1024),
    # Split owners + reducer (12 tokens), V-split row-first owners (24), row-first (40).
    _align_case(
        "bf16_h128_topk128x", 128, _BF16, [4] * 3, width=256, page=2, form="split4"
    ),
    _align_case(
        "bf16_h128_topk128x", 128, _BF16, [8] * 3, width=256, page=2, form="vsplit"
    ),
    _align_case(
        "bf16_h128_topk128x", 128, _BF16, [20] * 2, width=256, page=2, form="row_first"
    ),
]


@pytest.mark.parametrize("offset_bytes", [2, 16])
@pytest.mark.parametrize(
    "route,num_heads,dtype,q_lens,layout,width,page", _OUT_ALIGN_CASES
)
def test_misaligned_caller_out_matches_aligned(
    monkeypatch, route, num_heads, dtype, q_lens, layout, width, page, offset_bytes
):
    """Every route stores O with 16- or 32-byte vectors or a TMA store
    (``_OUT_ALIGN_BY_ROUTE``). A contiguous caller ``out`` 2 or 16 bytes off a
    32-byte boundary gets bitwise the aligned call's result through an aligned
    staging buffer, the bytes around it stay untouched, and an aligned ``out``
    allocates nothing.  On SM107 the low-head prefill cases run the V-block
    sibling up to its row ceiling (12 rows) and the route's program above the
    early-PV sibling's (66 rows); elsewhere they run the route's program."""
    _skip_unless_cake_gpu(sm107=route == "fp8_lowhead_prefill")
    import flashinfer.mla.cake_dsv4 as cake

    if route == "fp8_lowhead_h64":
        # The route on SM100; SM103 runs these rows on the persistent body.
        monkeypatch.setattr(cake, "_route", lambda **kwargs: route)
    calls = _count_dispatches(monkeypatch)
    launched: list = []
    original = cake._prepare_variant

    def spy(variant, **kwargs):
        launched.append(variant)
        return original(variant, **kwargs)

    monkeypatch.setattr(cake, "_prepare_variant", spy)
    case = _build_case(
        num_heads=num_heads,
        dtype=dtype,
        seq_lens=[_LONG_SEQ + 37 * i for i in range(len(q_lens))],
        q_lens=q_lens,
        layout=layout,
        compressed_width=width,
        compressed_page=page,
    )
    workspace = _workspace_128mib()
    aligned = case.nan_out()
    assert aligned.data_ptr() % 32 == 0
    case.run(workspace=workspace, out=aligned, float_scales=True)
    torch.cuda.synchronize()
    assert calls and {r for r, _ in calls} == {route}
    if route == "fp8_lowhead_prefill":
        assert set(launched) == {_lowhead_prefill_variant(sum(q_lens))}, launched

    again = case.nan_out()
    after_alloc = _allocation_requests()
    case.run(workspace=workspace, out=again, float_scales=True)
    torch.cuda.synchronize()
    assert _allocation_requests() == after_alloc, "an aligned out must not be staged"
    assert torch.equal(again.view(torch.int16), aligned.view(torch.int16))

    numel = aligned.numel()
    pad = offset_bytes // aligned.element_size()
    raw = torch.full((numel + 32,), float("nan"), dtype=torch.bfloat16, device="cuda:0")
    start = ((-raw.data_ptr()) % 32) // raw.element_size() + pad
    view = raw[start : start + numel].view(aligned.shape)
    assert view.is_contiguous() and view.data_ptr() % 32 == offset_bytes
    case.run(workspace=workspace, out=view, float_scales=True)
    torch.cuda.synchronize()
    assert torch.equal(view.view(torch.int16), aligned.view(torch.int16))
    assert torch.isnan(raw[:start]).all() and torch.isnan(raw[start + numel :]).all()


# --------------------------------------------------------------------------- DeepSeek-V4 layer rows
#
# The rows DeepSeek-V4 serves, FP8 cache: every layer has the 128-token window
# (sglang's table: column j holds position pos - j, newest first); a C4 layer
# adds the top 512 (Flash) or 1024 (Pro) compressed rows, a C128 layer one row
# per 128 positions (page 2), 128 + context / 128 slots wide; in decode the
# C128 table is as wide as the server's context allows (padded to 64), so a
# 16K server's rows hold 128 + 128 live slots of 320.  Flash runs 64
# heads per rank under DP attention and 32 / 16 / 8 at TP2 / TP4 / TP8; Pro
# runs 128, and 64 / 32 / 16 at TP2 / TP4 / TP8.  Decode batches reach past one
# wave of two CTAs per token; MTP rows take 4 tokens per request; prefill is
# two ragged requests.

# layer -> (compressed width, compressed page size, live compressed slots or
# None: every slot the position has produced)
_DSV4_LAYERS = {
    "swa": (0, 64, None),
    "c4flash": (512, 64, None),
    "c4pro": (1024, 64, None),
    "c128-16k": (128, 2, None),
    "c128-16k-server": (192, 2, 128),
    "c128-64k": (512, 2, None),
    "c128-128k": (1024, 2, None),
}
_DSV4_FLASH_LAYERS = (
    "swa",
    "c4flash",
    "c128-16k",
    "c128-16k-server",
    "c128-64k",
    "c128-128k",
)
_DSV4_ROWS = [
    pytest.param(h, layer, id=f"h{h}-{layer}")
    for h in (64, 32, 16, 8)
    for layer in _DSV4_FLASH_LAYERS
] + [pytest.param(h, "c4pro", id=f"h{h}-c4pro") for h in (128, 64, 32, 16)]
_DSV4_FORMS = ["decode-b8", "decode-b64", "decode-b256", "mtp-b32q4", "prefill"]


def _dsv4_layer_case(h_q: int, layer: str, form: str, width: Optional[int] = None):
    """One DeepSeek-V4 layer row (newest-first window table, FP8 cache).

    C4 rows attend their whole selection; a C128 row attends the compressed
    rows its position has produced, so its active count grows with the
    position inside a request (``width`` overrides the layer's compressed
    width)."""
    compressed, page, live = _DSV4_LAYERS[layer]
    if width is not None:
        compressed = width - SWA_WINDOW
    if form.startswith("decode"):
        batch = int(form.split("-b")[1])
        q_lens, seq = [1] * batch, [640 + 37 * i % 1024 for i in range(batch)]
        layout = "dense"
    elif form == "mtp-b32q4":
        q_lens, seq = [4] * 32, [640 + 37 * i % 1024 for i in range(32)]
        layout = "dense"
    elif form == "prefill":
        q_lens, seq = [171, 341], [1024 + 171, 2048 + 341]
        layout = "ragged"
    else:
        raise ValueError(form)
    lens = None
    rows = sum(q_lens)
    if live is not None:
        lens = [live] * rows
    elif compressed and page == 2:
        lens = []
        for q in q_lens:
            lens += [max(compressed - (q - 1 - i) // 128, 1) for i in range(q)]
    return _build_case(
        num_heads=h_q,
        dtype=torch.float8_e4m3fn,
        seq_lens=seq,
        q_lens=q_lens,
        layout=layout,
        compressed_width=compressed,
        compressed_page=page,
        compressed_pages=max(64, rows * compressed // page // 4),
        compressed_lens=lens,
        alias_compressed=not compressed,
        newest_first=True,
    )


@pytest.mark.parametrize("h_q,layer", _DSV4_ROWS)
@pytest.mark.parametrize("form", _DSV4_FORMS)
def test_dsv4_layer_rows(h_q, layer, form):
    """Every DeepSeek-V4 layer row the FP8 route serves: the masked columns
    contribute nothing and the output matches the reference.  Covers the
    low-head decode rows wider than one producer partition and the 8-head
    prefill rows, which run the persistent or the M64 body.  SM107 runs every
    row too: its low-head decode rows over compressed items take the M64 body."""
    _skip_unless_cake_gpu(sm107=True)
    case = _dsv4_layer_case(h_q, layer, form)
    _assert_poison_inert(case, _workspace_128mib(), float_scales=True)


@pytest.mark.parametrize("h_q", [128, 64, 32, 16, 8])
@pytest.mark.parametrize("form", ["decode-b64", "mtp-b32q4", "prefill"])
def test_dsv4_c128_rows_past_128k_raise_before_any_launch(h_q, form):
    """A C128 layer past 128K context (128 + 2048 slots at 256K) is wider than
    any FP8 body takes: the call raises before it launches anything and leaves
    ``out`` untouched."""
    _skip_unless_cake_gpu(sm107=True)
    case = _dsv4_layer_case(h_q, "c128-128k", form, width=2176)
    out = case.nan_out()
    with pytest.raises(ValueError, match=r"sparse_topk 2176: .* at most 1152") as e:
        case.run(workspace=_workspace_128mib(), out=out, float_scales=True)
    torch.cuda.synchronize()
    assert "nvfp4" not in str(e.value)
    assert torch.isnan(out.float()).all()


@pytest.mark.parametrize("h_q", [128, 64, 32, 16, 8])
@pytest.mark.parametrize("width", [320, 704], ids=["c128-16k", "c128-64k"])
@pytest.mark.parametrize("batch", [8, 32])
@pytest.mark.parametrize("minus_one_inside", [False, True], ids=["", "minus-one"])
def test_dsv4_c128_mtp_rows_at_the_compression_boundary(
    h_q, width, batch, minus_one_inside
):
    """C128 MTP rows whose active counts come from their positions, across a
    128-position boundary: request ``b`` ends at position 640 + 128 b, so its
    four rows attend 4 + b, 4 + b, 4 + b and 5 + b compressed rows of a table
    padded to the server's context (320 or 704 slots).  Columns past each
    row's count point at a poison page (bit-identical output), every seventh
    active slot is ``-1`` with ``minus_one_inside``, and the last visible
    window slot and the last active compressed column read beacon rows whose
    sign flip must move every row.  Runs the persistent and the M64 bodies
    (and the one-partition producer at 8/16/32 heads, width 320, 8
    requests)."""
    _skip_unless_cake_gpu(sm107=True)
    seq = [640 + 128 * b for b in range(batch)]
    lens = [max((s - 4 + i + 1) // 128, 1) for s in seq for i in range(4)]
    assert lens[:4] == [4, 4, 4, 5]
    case = _build_case(
        num_heads=h_q,
        dtype=torch.float8_e4m3fn,
        seq_lens=seq,
        q_lens=[4] * batch,
        layout="dense",
        compressed_width=width - SWA_WINDOW,
        compressed_page=2,
        compressed_pages=max(64, 4 * batch * (width - SWA_WINDOW) // 8),
        compressed_lens=lens,
        minus_one_inside=minus_one_inside,
        newest_first=True,
        beacon=True,
    )
    _assert_boundary_columns_attended(case, _workspace_128mib())


# ---------------------------------------------------------- padded rows on the source-exact routes
# The routes whose programs launch one CTA per request position (max_q_len x
# batch_size, whatever the metadata rows), at every head count and sparse
# width they take.  The H8/H16 routes run request lengths 5, 3, 4 and 1, 5, 5:
# with a one-row first request, T = 1 ends at a request boundary and both
# later requests start at or past T.
_SOURCE_EXACT_PADDED_CASES = [
    _align_case(route, heads, dtype, q_lens, "ragged", width, page, form=form + tag)
    for route, dtype in (
        ("bf16_h8_h16_source_exact", _BF16),
        ("fp8_h8_h16_source_exact", _FP8),
    )
    for heads, width, page, form in (
        (8, 64, 64, "h8-192"),
        (16, 128, 64, "h16-256"),
        (8, 132, 2, "h8-260"),
        (16, 132, 2, "h16-260"),
    )
    for q_lens, tag in (([5, 3, 4], ""), ([1, 5, 5], "-q1-5-5"))
] + [_align_case("fp8_h64_source_exact", 64, _FP8, [257, 100], "ragged", 512)]


@pytest.mark.parametrize("swa", ["long", "short-stale"])
@pytest.mark.parametrize(
    "route,num_heads,dtype,q_lens,layout,width,page", _SOURCE_EXACT_PADDED_CASES
)
def test_source_exact_padded_rows_match_the_unpadded_call(
    monkeypatch, route, num_heads, dtype, q_lens, layout, width, page, swa
):
    """A padded ragged call on a source-exact route: the metadata of the first
    ``T`` rows alone (tables and lengths of exactly ``T`` rows) under the
    query and ``cum_seq_lens_q`` of every row, for ``T`` = 1, the first
    request boundary and one row short of the end.  Rows below ``T`` get
    bitwise the unpadded call's output, in an aligned ``out`` and in one 2
    bytes off its store alignment (the program writes a ``T``-row staging
    buffer, which the host copies into ``out``).  In the aligned ``out``
    every row at or past ``T`` keeps its NaN: the program wrote none of them.
    In the staged call those rows are out of the program's reach, so the same
    check holds by construction.  The bytes around ``out`` stay untouched.
    ``T = 1`` also runs with its one row viewed out of a ``[1, W + 1]``
    parent (odd row stride), whose next row would be a misaligned 16-byte
    index load.  These checks see what lands in ``out`` only: a read of a
    metadata row at or past ``T``, or a write past the end of the staging
    buffer, leaves ``out`` as it is.  Those show under ``compute-sanitizer
    --tool memcheck`` with ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` (each
    ``T``-row table, the lengths and the staging buffer an allocation of its
    own).  ``swa="short-stale"`` gives every request
    an SWA cache a few tokens longer than its query, with valid stale slots
    past the causal window: the rows of the requests ``T`` cuts see fewer than
    128 SWA slots, so a window taken from the cut length instead of the
    request's would attend a stale slot and move a row below ``T``."""
    _skip_unless_cake_gpu()
    calls = _count_dispatches(monkeypatch)
    short = swa == "short-stale"
    case = _build_case(
        num_heads=num_heads,
        dtype=dtype,
        seq_lens=(
            [q + 3 + 7 * i for i, q in enumerate(q_lens)]
            if short
            else [_LONG_SEQ + 37 * i for i in range(len(q_lens))]
        ),
        q_lens=q_lens,
        layout=layout,
        compressed_width=width,
        compressed_page=page,
        swa_tail="stale" if short else "minus_one",
    )
    if short:
        # The rows below T of a request T cuts see fewer than 128 SWA slots:
        # the first row (T = 1 cuts a first request longer than one row) and
        # the last row below T = cum[B] - 1 (which cuts the last request).
        visible = case.visible()
        assert int(visible[0]) < SWA_WINDOW and int(visible[-2]) < SWA_WINDOW
    workspace = _workspace_128mib()
    expected = case.run(workspace=workspace, out=case.nan_out(), float_scales=True)
    torch.cuda.synchronize()
    assert calls == [(route, case.rows)], calls
    assert not torch.isnan(expected).any()

    table, lens = case.sparse_indices, case.sparse_topk_lens
    cols = int(table.shape[1])
    ends = case.cum_seq_lens_q.tolist()
    one_row = torch.full((1, cols + 1), -1, dtype=torch.int32, device="cuda:0")
    one_row[:, :cols] = table[:1]
    assert one_row[:, :cols].stride(0) % 4 != 0
    metadata = [
        (rows, table[:rows].clone()) for rows in sorted({1, ends[1], ends[-1] - 1})
    ]
    metadata.append((1, one_row[:, :cols]))
    numel = expected.numel()
    for rows, rows_table in metadata:
        padded = dataclasses.replace(
            case, sparse_indices=rows_table, sparse_topk_lens=lens[:rows].clone()
        )
        for offset_bytes in (0, 2):
            raw = torch.full(
                (numel + 32,), float("nan"), dtype=torch.bfloat16, device="cuda:0"
            )
            start = ((-raw.data_ptr()) % 32 + offset_bytes) // raw.element_size()
            out = raw[start : start + numel].view(case.query.shape)
            assert out.data_ptr() % 32 == offset_bytes
            calls.clear()
            got = padded.run(workspace=workspace, out=out, float_scales=True)
            torch.cuda.synchronize()
            where = f"T={rows} of {case.rows}, stride {rows_table.stride(0)}, out +{offset_bytes} B"
            assert calls == [(route, rows)], (where, calls)
            assert torch.equal(
                got[:rows].view(torch.int16), expected[:rows].view(torch.int16)
            ), where
            assert torch.isnan(got[rows:]).all(), where
            assert torch.isnan(raw[:start]).all(), where
            assert torch.isnan(raw[start + numel :]).all(), where
