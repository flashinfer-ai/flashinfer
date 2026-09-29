"""GPU hardening tests for the CAKE DeepSeek-V4 sparse-MLA host (flashinfer#4671).

Covers padded query rows, separate SWA/compressed tables, length offsets,
all-invalid rows, CUDA graph capture/replay with mutated inputs, and the
no-allocation contract. Reference and generator helpers are imported from the
TRTLLM-GEN DSv4 test module. Skips without an SM100/SM103 GPU.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

import pytest
import torch

from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from flashinfer.mla.cake_dsv4 import (
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


def _skip_unless_cake_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CAKE DSv4 hardening tests")
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability not in ((10, 0), (10, 3)):
        pytest.skip(
            "CAKE DeepSeek V4 sparse MLA requires SM100/SM103, "
            f"got SM{compute_capability[0]}{compute_capability[1]}"
        )


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
):
    global _seed_counter
    if seed is None:
        seed = SEED_BASE + _seed_counter
        _seed_counter += 1
    param = ref.RawTestParamForDecode(
        b=BATCH,
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
            enable_pdl=False,
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
    ref._assert_close(got[:rows], expected[:rows], dtype)
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
    ref._assert_close(_rows(combined, inputs), inputs.reference_rows(), dtype)
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
    ref._assert_close(_rows(baseline, inputs), inputs.reference_rows(), dtype)
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
    ref._assert_close(got, inputs.reference_rows(), dtype)


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
    baseline_allocated = torch.cuda.memory_allocated()
    static.run(out=out, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == baseline_allocated, "eager call allocated"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static.run(out=out, workspace=workspace)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    ref._assert_close(_rows(out, static), static.reference_rows(), dtype)

    # Mutate every input in place (same shapes) and replay.
    p2, tc2 = _make_case(h_q, dtype, s_q, varlen=True, seed=p.seed + 5_000)
    mutated = _Inputs(p2, tc2)
    assert mutated.query.shape == static.query.shape
    static.copy_from(mutated)
    torch.cuda.synchronize()
    replay_allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == replay_allocated, "graph replay allocated"

    eager_out = _out_like(static, fill=0.0)
    static.run(out=eager_out, workspace=_workspace(static))
    torch.cuda.synchronize()
    # The reference must be evaluated on the mutated testcase's own tensors.
    ref._assert_close(_rows(eager_out, static), mutated.reference_rows(), dtype)
    assert torch.equal(out, eager_out)


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
    ref._assert_close(got[:rows], expected[:rows], dtype)
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
        ref._assert_close(_rows(out, inputs), expected, dtype)
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
    ref._assert_close(_rows(out, inputs), fresh.reference_rows(), dtype)
