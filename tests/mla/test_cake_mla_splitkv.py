"""Cake's stateless split-KV decode of Blackwell MLA (domain ``mla_bf16_splitkv``).

The split-KV program serves aligned BF16 decode with long KV and few query rows on
the devices ``_SPLITKV_RANGE`` lists. Its mainloop derives each query row's request,
causal KV length and pages from the caller's ``seq_lens`` and ``block_tables`` on
the device and cuts the row's KV tiles into ``plan_kv_split`` equal ranges; its
reducer merges the BF16 partials in a fixed order and applies the sink and
``bmm2_scale``. The binding launches one of two reducer builds that write the same
bits: the prefetch_all build up to ``CAKE_SPLITKV_PREFETCH_ALL_MAX_ROWS`` query rows
split at least ``CAKE_SPLITKV_PREFETCH_ALL_MIN_SPLITS`` ways, the prefetch_any build
otherwise; the row counts and lengths below lie on both sides of that bound. These tests check, on every Blackwell device (the fixture lists the
current device from max_seq_len 1025, so the split-KV program is also checked below
the max_seq_len 2049 where the listed ranges of B200 and GB300 start):

- accuracy against an FP64 reference at the level of TRTLLM-GEN (its maximum
  absolute error plus one BF16 rounding of the partials), with sinks,
  ``bmm2_scale != 1``, ``q_len == 2``, page 64, padded and short rows;
- padded rows (no causal KV token) output exact zeros;
- run-to-run bitwise determinism, eager and under CUDA-graph replay, and equal
  bits in the default and the stateless mode;
- CUDA-graph replay after in-place updates of ``seq_lens``, ``block_tables`` and
  the query;
- empty splits (rows much shorter than ``max_seq_len``) ignore stale shared memory;
- the split-KV scratch is keyed by the split count and never cached under stream
  capture or in the stateless mode;
- the ``max_seq_len`` field of the ``backend="auto"`` decision memo keeps the side
  of the split-KV range start, also a start inside a bucket (no GPU needed);
- the split-count rule: up to 32 splits, up to 64 for one query row whose
  max_seq_len KV tiles hold at least the device's start (65536 tokens on B200,
  131072 on GB300; an unlisted device keeps 32), every other launch unchanged,
  and the binding's caps equal the host's (no GPU needed);
- more than 32 splits (the wide reducer build), forced at 33, 37, 38, 48 and 64
  splits and at the counts the rule picks: on live rows accuracy against FP64 at
  the level of TRTLLM-GEN (as above, three draws), padded rows are exact zeros,
  and the bits are run-to-run identical, eager and under CUDA-graph replay; 65
  splits have no dispatch route.
"""

import math
from collections import OrderedDict

import pytest
import torch

import flashinfer
from flashinfer.mla import _core as core
from flashinfer.mla import cake_trtllm_mla_blackwell as cake_module
from flashinfer.utils import get_compute_capability

NUM_HEADS, QK_DIM, LATENT = 128, 576, 512
BMM1 = 1.0 / (192**0.5)
TOLERANCE = 1e-2
SEEDS = range(5)

_workspace_buffer = None


def _workspace() -> torch.Tensor:
    global _workspace_buffer
    if _workspace_buffer is None:
        _workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
    return _workspace_buffer


def _require_blackwell() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("Cake Blackwell MLA decode requires SM100a or SM103a")


@pytest.fixture
def splitkv_device(monkeypatch):
    """List the current device in ``_SPLITKV_RANGE`` from max_seq_len 1025, up
    to 18 query rows, so these tests reach the split-KV program from max_seq_len
    1025 as well (the listed ranges of B200 and GB300 start at 2049)."""
    _require_blackwell()
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    monkeypatch.setattr(cake_module, "_SPLITKV_RANGE", {(arch, num_sms): (1025, 18)})
    return arch, num_sms


@pytest.fixture
def splitkv_domains(monkeypatch):
    """The domains ``trtllm_mla_blackwell_decode`` selects, in call order."""
    seen: list = []
    select = cake_module.select_domain

    def spy(**kwargs):
        domain = select(**kwargs)
        seen.append(domain)
        return domain

    monkeypatch.setattr(cake_module, "select_domain", spy)
    return seen


def _inputs(batch, q_len, page_size, max_seq_len, pattern, seed):
    """Query, KV cache, page table and KV lengths of a paged decode request.

    ``pattern``: ``uniform`` (every request ``max_seq_len``), ``varied`` (half
    to all of ``max_seq_len``, the last request at ``max_seq_len``), ``padded``
    (``varied`` with every third request but the last at 0 and, with
    ``q_len > 1``, the next one but the last at 1: its first query row has no
    causal KV token; a single request stays live), ``all_padded`` and
    ``short`` (every request at most a quarter of ``max_seq_len``: most splits of
    a row are empty).
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    width = -(-max_seq_len // page_size)
    pages = batch * width
    query = torch.randn(
        batch, q_len, NUM_HEADS, QK_DIM, device="cuda", generator=generator
    )
    kv_cache = torch.randn(pages, page_size, QK_DIM, device="cuda", generator=generator)
    block_tables = torch.randperm(pages, device="cuda", generator=generator)
    if pattern == "uniform":
        seq_lens = torch.full((batch,), max_seq_len, device="cuda")
    elif pattern == "short":
        seq_lens = torch.randint(
            1,
            max(2, max_seq_len // 4 + 1),
            (batch,),
            device="cuda",
            generator=generator,
        )
    else:
        seq_lens = torch.randint(
            max(q_len, max_seq_len // 2),
            max_seq_len + 1,
            (batch,),
            device="cuda",
            generator=generator,
        )
        seq_lens[-1] = max_seq_len
        if pattern == "padded":
            # The last request keeps max_seq_len: a padded call still has a row
            # with every split busy.
            seq_lens[:-1][::3] = 0
            if q_len > 1:
                seq_lens[1:-1:3] = 1
        elif pattern == "all_padded":
            seq_lens.zero_()
    return (
        (query * 0.5).to(torch.bfloat16),
        (kv_cache * 0.5).to(torch.bfloat16),
        block_tables.to(torch.int32).view(batch, width),
        seq_lens.to(torch.int32),
    )


def _cake(query, kv_cache, block_tables, seq_lens, max_seq_len, *, sink=None, **kwargs):
    """Explicit ``backend="cake"``: the Blackwell module's decode entry."""
    arguments = {
        "qk_nope_head_dim": 128,
        "kv_lora_rank": LATENT,
        "qk_rope_head_dim": QK_DIM - LATENT,
        "sparse_mla_top_k": 0,
        "out": None,
        "bmm1_scale": BMM1,
        "bmm2_scale": 1.0,
        "sinks": None if sink is None else [sink],
        "skip_softmax_threshold_scale_factor": None,
        "enable_pdl": None,
        "uses_shared_paged_kv_idx": True,
        "lse": None,
        "return_lse": False,
        "cum_seq_lens_q": None,
        "max_q_len": None,
        "multi_ctas_kv_counter_buffer": None,
        "sparse_mla_top_k_lens": None,
        "enable_dcp": False,
        "backend": "cake",
    }
    arguments.update(kwargs)
    return cake_module.trtllm_mla_blackwell_decode(
        query, kv_cache, block_tables, seq_lens, max_seq_len, **arguments
    )


def _trtllm_gen(
    query, kv_cache, block_tables, seq_lens, max_seq_len, *, sink, bmm2_scale, out
):
    return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=_workspace(),
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=QK_DIM - LATENT,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        out=out,
        bmm1_scale=BMM1,
        bmm2_scale=bmm2_scale,
        # The explicit TRTLLM-GEN entry takes the sink as one tensor.
        sinks=sink,
        backend="trtllm-gen",
    )


def _row_lengths(seq_lens, q_len):
    """[batch, q_len] causal KV length of every query row."""
    offsets = torch.arange(1 - q_len, 1, dtype=torch.int32, device=seq_lens.device)
    return seq_lens[:, None] + offsets[None, :]


def _reference_fp64(query, kv_cache, block_tables, seq_lens, sink, bmm2_scale):
    """FP64 attention of every query row; a row without causal KV token is zero."""
    batch, q_len = query.shape[:2]
    page_size = kv_cache.shape[-2]
    kv_flat = kv_cache.reshape(-1, QK_DIM).double()
    out = torch.zeros(
        batch, q_len, NUM_HEADS, LATENT, dtype=torch.float64, device="cuda"
    )
    tokens = torch.arange(page_size, device="cuda")
    lengths = _row_lengths(seq_lens, q_len).tolist()
    for b in range(batch):
        longest = max(lengths[b])
        if longest <= 0:
            continue
        pages = block_tables[b, : -(-longest // page_size)].long()
        keys = kv_flat[(pages[:, None] * page_size + tokens[None, :]).reshape(-1)]
        keys = keys[:longest]
        for i in range(q_len):
            length = lengths[b][i]
            if length <= 0:
                continue
            logits = query[b, i].double() @ keys[:length].T * BMM1
            if sink is not None:
                logits = torch.cat([logits, sink.double()[:, None]], dim=-1)
            probabilities = torch.softmax(logits, dim=-1)[:, :length]
            out[b, i] = probabilities @ keys[:length, :LATENT] * bmm2_scale
    return out


def _nan_out(query):
    return torch.full(
        (*query.shape[:-1], LATENT), float("nan"), dtype=torch.bfloat16, device="cuda"
    )


def _bf16_ulp(value: float) -> float:
    """One BF16 unit in the last place at ``value`` (8 significand bits)."""
    if value <= 0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(value)) - 7)


# name: (q_len, page_size, sink, bmm2_scale, seq_lens pattern)
_VARIANTS = {
    "plain": (1, 32, False, 1.0, "varied"),
    "sinks": (1, 32, True, 1.0, "varied"),
    "bmm2_scale": (1, 32, False, 0.5, "varied"),
    "q_len2": (2, 32, False, 1.0, "varied"),
    "padded": (1, 32, False, 1.0, "padded"),
    "padded_q_len2": (2, 32, False, 1.0, "padded"),
    "page64": (1, 64, False, 1.0, "varied"),
    "page64_sinks_bmm2_q_len2_padded": (2, 64, True, 0.5, "padded"),
    "short": (1, 32, False, 1.0, "short"),
    "all_padded": (1, 32, False, 1.0, "all_padded"),
}


@pytest.mark.parametrize("rows", [1, 4, 9, 18])
@pytest.mark.parametrize("max_seq_len", [2048, 4096, 16384])
@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_splitkv_matches_trtllm_gen_and_fp64(
    splitkv_device, splitkv_domains, variant, max_seq_len, rows
):
    """On live rows the split-KV output is within TRTLLM-GEN's maximum absolute
    error against FP64 plus one BF16 unit of the largest reference value (the BF16
    partials round once more), and within the suite's tolerance; padded rows are
    exact zeros (the output is NaN-filled first). Five draws per cell; the default
    and the stateless mode give the same bits."""
    q_len, page_size, has_sink, bmm2_scale, pattern = _VARIANTS[variant]
    if rows % q_len:
        pytest.skip("the row count is not a multiple of q_len")
    batch = rows // q_len
    for seed in SEEDS:
        query, kv_cache, block_tables, seq_lens = _inputs(
            batch, q_len, page_size, max_seq_len, pattern, seed
        )
        sink = None
        if has_sink:
            generator = torch.Generator(device="cuda").manual_seed(100 + seed)
            sink = torch.randn(NUM_HEADS, device="cuda", generator=generator) * 2.0
        cake = _cake(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            sink=sink,
            bmm2_scale=bmm2_scale,
            out=_nan_out(query),
        )
        stateless = _cake(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            sink=sink,
            bmm2_scale=bmm2_scale,
            out=_nan_out(query),
            stateless=True,
        )
        torch.cuda.synchronize()
        label = f"{variant} kv{max_seq_len} rows{rows} seed{seed}"
        assert torch.equal(cake, stateless), label
        live = _row_lengths(seq_lens, q_len) > 0
        padded = cake[~live]
        assert torch.equal(padded, torch.zeros_like(padded)), label
        if not bool(live.any()):
            continue
        reference = _reference_fp64(
            query, kv_cache, block_tables, seq_lens, sink, bmm2_scale
        )[live]
        trtllm_gen = _trtllm_gen(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            sink=sink,
            bmm2_scale=bmm2_scale,
            out=_nan_out(query),
        )
        torch.cuda.synchronize()
        cake_live = cake[live].double()
        cake_error = float((cake_live - reference).abs().max())
        trtllm_gen_error = float((trtllm_gen[live].double() - reference).abs().max())
        ulp = _bf16_ulp(float(reference.abs().max()))
        rel_l2 = float(
            (cake_live - reference).norm() / reference.norm().clamp_min(1e-30)
        )
        print(
            f"{label}: max_abs cake {cake_error:.3e} trtllm-gen "
            f"{trtllm_gen_error:.3e} ulp {ulp:.3e} rel_l2 {rel_l2:.3e}"
        )
        assert cake_error <= trtllm_gen_error + ulp, label
        torch.testing.assert_close(
            cake_live.float(),
            reference.float(),
            rtol=TOLERANCE,
            atol=TOLERANCE,
            msg=lambda message: f"{label} vs FP64: {message}",
        )
    assert splitkv_domains and set(splitkv_domains) == {cake_module.DOMAIN_BF16_SPLITKV}


# name: (batch, q_len, page_size, max_seq_len, sink)
_DETERMINISM_CASES = {
    "rows1_kv16384": (1, 1, 32, 16384, False),
    "rows8_q_len2_page64_sink_kv4096": (4, 2, 64, 4096, True),
    "rows18_kv2048": (18, 1, 32, 2048, False),
}


@pytest.mark.parametrize("case", sorted(_DETERMINISM_CASES))
def test_splitkv_is_deterministic(splitkv_device, splitkv_domains, case):
    """200 eager calls and 1000 replays of a captured graph give identical bits;
    the default and the stateless mode launch the same kernels and agree."""
    batch, q_len, page_size, max_seq_len, has_sink = _DETERMINISM_CASES[case]
    query, kv_cache, block_tables, seq_lens = _inputs(
        batch, q_len, page_size, max_seq_len, "padded", 0
    )
    # The request at max_seq_len is live: the bits compared cover every split.
    assert int(seq_lens[-1]) == max_seq_len
    assert bool((_row_lengths(seq_lens, q_len) > 0).any())
    sink = None
    if has_sink:
        generator = torch.Generator(device="cuda").manual_seed(7)
        sink = torch.randn(NUM_HEADS, device="cuda", generator=generator) * 2.0

    def decode(output, **kwargs):
        return _cake(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            sink=sink,
            out=output,
            **kwargs,
        )

    first = decode(_nan_out(query)).clone()
    torch.cuda.synchronize()
    assert bool(first.isfinite().all())
    for call in range(200):
        assert torch.equal(decode(_nan_out(query)), first), call
    assert torch.equal(decode(_nan_out(query), stateless=True), first)

    out = _nan_out(query)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode(out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(out)
    mismatches = torch.zeros((), dtype=torch.int64, device="cuda")
    for _ in range(1000):
        out.fill_(float("nan"))
        graph.replay()
        mismatches += (out != first).any()
    torch.cuda.synchronize()
    assert int(mismatches) == 0
    assert set(splitkv_domains) == {cake_module.DOMAIN_BF16_SPLITKV}


@pytest.mark.parametrize("stateless", [False, True])
@pytest.mark.parametrize(
    "batch,q_len,page_size,max_seq_len",
    [(2, 2, 32, 16384), (3, 1, 64, 4096), (9, 2, 32, 2048)],
    ids=["rows4_kv16384", "rows3_page64_kv4096", "rows18_q_len2_kv2048"],
)
def test_splitkv_graph_replay_with_updated_inputs(
    splitkv_device, splitkv_domains, batch, q_len, page_size, max_seq_len, stateless
):
    """A graph captured once follows later in-place changes of ``seq_lens``,
    ``block_tables`` and the query: every replay equals an eager call on fresh
    tensors holding the same values, matches the FP64 reference on live rows and
    outputs zeros on padded rows (empty, short and full rows included)."""
    query, kv_cache, block_tables, seq_lens = _inputs(
        batch, q_len, page_size, max_seq_len, "varied", 0
    )
    out = _nan_out(query)

    def decode(lens, table, output):
        return _cake(
            query, kv_cache, table, lens, max_seq_len, out=output, stateless=stateless
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode(seq_lens, block_tables, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(seq_lens, block_tables, out)

    generator = torch.Generator(device="cuda").manual_seed(1)
    width = block_tables.shape[1]
    updates = [
        ("captured", seq_lens.clone()),
        ("all_zero", torch.zeros_like(seq_lens)),
        ("all_max", torch.full_like(seq_lens, max_seq_len)),
        ("padded", _inputs(batch, q_len, page_size, max_seq_len, "padded", 2)[3]),
        ("short", _inputs(batch, q_len, page_size, max_seq_len, "short", 3)[3]),
    ]
    for label, values in updates:
        seq_lens.copy_(values)
        permutation = torch.randperm(batch * width, device="cuda", generator=generator)
        block_tables.copy_(block_tables.reshape(-1)[permutation].reshape(batch, width))
        out.fill_(float("nan"))
        graph.replay()
        eager = decode(seq_lens.clone(), block_tables.clone(), None)
        torch.cuda.synchronize()
        assert torch.equal(out, eager), label
        _check_rows(out, query, kv_cache, block_tables, seq_lens, label)

    query.copy_(_inputs(batch, q_len, page_size, max_seq_len, "varied", 4)[0])
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _check_rows(out, query, kv_cache, block_tables, seq_lens, "query update")
    assert set(splitkv_domains) == {cake_module.DOMAIN_BF16_SPLITKV}


def _check_rows(out, query, kv_cache, block_tables, seq_lens, label):
    live = _row_lengths(seq_lens, query.shape[1]) > 0
    padded = out[~live]
    assert torch.equal(padded, torch.zeros_like(padded)), label
    if bool(live.any()):
        reference = _reference_fp64(query, kv_cache, block_tables, seq_lens, None, 1.0)
        torch.testing.assert_close(
            out[live].float(),
            reference[live].float(),
            rtol=TOLERANCE,
            atol=TOLERANCE,
            msg=lambda message: f"{label}: {message}",
        )


_SMEM_FILL_SOURCE = r"""
#include <torch/extension.h>

__global__ void fill_shared_memory(int bytes) {
  extern __shared__ unsigned char smem[];
  for (int i = threadIdx.x; i < bytes; i += blockDim.x) smem[i] = 0xFF;
  __syncthreads();
}

void fill_shared_memory_of_every_sm() {
  int device = 0, sms = 0, bytes = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaFuncSetAttribute(fill_shared_memory, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  fill_shared_memory<<<4 * sms, 1024, bytes>>>(bytes);
}
"""


def test_splitkv_empty_splits_ignore_stale_shared_memory(
    splitkv_device, splitkv_domains
):
    """Rows much shorter than ``max_seq_len`` leave most of their splits empty.

    Every SM's shared memory is filled with 0xFF bytes (a BF16 and an FP32 NaN)
    right before each of eight calls, so an empty split that reads stale shared
    memory instead of writing its LSE -inf and zero partial turns its row
    non-finite. Every call gives the same bits and matches the FP64 reference.
    """
    from torch.utils.cpp_extension import load_inline

    smem_fill = load_inline(
        name="test_cake_mla_splitkv_smem_fill",
        cpp_sources=["void fill_shared_memory_of_every_sm();"],
        cuda_sources=[_SMEM_FILL_SOURCE],
        functions=["fill_shared_memory_of_every_sm"],
        verbose=False,
    )
    max_seq_len, page_size, q_len = 16384, 32, 1
    query, kv_cache, block_tables, _ = _inputs(
        4, q_len, page_size, max_seq_len, "varied", 0
    )
    seq_lens = torch.tensor([1, 129, 0, 300], dtype=torch.int32, device="cuda")
    assert cake_module.plan_kv_split(4, max_seq_len, splitkv_device[1]) >= 8
    first = None
    for call in range(8):
        smem_fill.fill_shared_memory_of_every_sm()
        out = _cake(
            query, kv_cache, block_tables, seq_lens, max_seq_len, out=_nan_out(query)
        )
        torch.cuda.synchronize()
        if first is None:
            first = out.clone()
            _check_rows(out, query, kv_cache, block_tables, seq_lens, "empty splits")
        assert torch.equal(out, first), call
    assert set(splitkv_domains) == {cake_module.DOMAIN_BF16_SPLITKV}


def test_splitkv_scratch_keyed_by_split_count_and_never_cached_in_capture(
    splitkv_device, monkeypatch
):
    """The default mode caches the scratch per (rows, split count); a stream
    capture and the stateless mode allocate theirs per call and cache nothing."""
    arch, num_sms = splitkv_device
    # An empty workspace cache: entries of earlier tests do not count.
    monkeypatch.setattr(cake_module, "_WORKSPACE_CACHE", OrderedDict())

    def keys():
        return {
            key
            for key in cake_module._WORKSPACE_CACHE
            if isinstance(key, tuple) and key and key[0] == "splitkv"
        }

    query, kv_cache, block_tables, seq_lens = _inputs(2, 1, 32, 4096, "varied", 0)
    before = keys()
    _cake(query, kv_cache, block_tables, seq_lens, 4096)
    split = cake_module.plan_kv_split(2, 4096, num_sms)
    assert ("splitkv", query.device, 2, split) in keys() - before

    # Same rows, other max_seq_len and split count: its own entry.
    query, kv_cache, block_tables, seq_lens = _inputs(2, 1, 32, 2048, "varied", 0)
    _cake(query, kv_cache, block_tables, seq_lens, 2048)
    other = cake_module.plan_kv_split(2, 2048, num_sms)
    assert other != split
    assert ("splitkv", query.device, 2, other) in keys()

    query, kv_cache, block_tables, seq_lens = _inputs(3, 1, 32, 8192, "varied", 1)
    before = keys()
    _cake(query, kv_cache, block_tables, seq_lens, 8192, stateless=True)
    assert keys() == before
    out = _nan_out(query)
    torch.cuda.synchronize()
    # The stateless call above loaded the family and created the FP8 decode table
    # (the placeholder sink pointer), so the capture builds nothing.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _cake(query, kv_cache, block_tables, seq_lens, 8192, out=out)
    assert keys() == before
    graph.replay()
    eager = _cake(query, kv_cache, block_tables, seq_lens, 8192, stateless=True)
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize("start", [2049, 3000, 4097])
def test_auto_memo_seq_len_follows_the_splitkv_range(monkeypatch, start):
    """The field that stands for max_seq_len in the ``backend="auto"`` decision
    memo key (``_mla_auto_memo_seq_len``) keeps the side of the device's
    split-KV range start a max_seq_len is on, also for a start inside a bucket
    and a KV tile (3000), so all values of one field select one domain: split-KV
    for 4 query rows from the start on, VQUARTER below it. A device without a
    range has no side (``None``). No GPU needed."""
    facts = ("sm_100a", 148)
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: facts)
    monkeypatch.setattr(cake_module, "_SPLITKV_RANGE", {facts: (start, 18)})
    domains = {}
    for max_seq_len in range(1, 8193):
        field = core._mla_auto_memo_seq_len(max_seq_len, 0)
        assert field[2] == (max_seq_len >= start), max_seq_len
        domain = cake_module.select_domain(
            dtype=torch.bfloat16,
            num_heads=NUM_HEADS,
            qk_dim=QK_DIM,
            value_dim=LATENT,
            qk_nope_head_dim=128,
            page_size=32,
            topk=0,
            uses_shared_paged_kv_idx=True,
            ragged_query=False,
            enable_sink=False,
            skip_softmax=False,
            wants_lse=False,
            bmm2_scale=1.0,
            total_q=4,
            max_seq_len=max_seq_len,
            batch_size=4,
            q_len=1,
            table_width=256,
            kv_lens=None,
            num_sms=facts[1],
            arch=facts[0],
        )
        assert domains.setdefault(field, domain) == domain, max_seq_len
    assert set(domains.values()) == {
        cake_module.DOMAIN_BF16_SPLITKV,
        cake_module.DOMAIN_BF16_VQUARTER,
    }
    below, at = (core._mla_auto_memo_seq_len(value, 0) for value in (start - 1, start))
    assert below != at
    if start == 3000:
        # Same bucket and KV tile count: only the side tells them apart.
        assert below[:2] == at[:2]
    monkeypatch.setattr(cake_module, "_SPLITKV_RANGE", {})
    assert core._mla_auto_memo_seq_len(4096, 0)[2] is None


# --- More than 32 splits: the wide reducer build -----------------------------------


def _narrow_plan(total_q, max_seq_len, num_sms):
    """The split count of every launch outside the wide rule: at most 32 splits."""
    resident = max(1, num_sms // 2)
    tiles = max(1, -(-max_seq_len // 128))
    split = min(32, max(1, resident // max(1, total_q)), tiles)
    return -(-tiles // -(-tiles // split))


def _wide_plan(total_q, max_seq_len, num_sms):
    """The split count of one query row of long KV: at most 64 splits."""
    resident = max(1, num_sms // 2)
    tiles = max(1, -(-max_seq_len // 128))
    split = min(64, max(1, resident // max(1, total_q)), tiles)
    return -(-tiles // -(-tiles // split))


def test_splitkv_plan_wide_rule():
    """``plan_kv_split`` allows up to 64 splits only for one query row whose
    max_seq_len KV tiles hold at least the device's start (65536 tokens on B200,
    131072 on GB300); every other launch, an unlisted device and a call without
    ``arch`` keep the 32-split count they had. One wave of clusters at most, and
    the count depends on max_seq_len only through its KV tile count. No GPU
    needed."""
    assert cake_module._SPLITKV_MAX_SPLIT == 64
    assert cake_module._SPLITKV_NARROW_MAX_SPLIT == 32
    starts = {("sm_100a", 148): 65536, ("sm_103a", 152): 131072}
    assert starts == cake_module._SPLITKV_WIDE_MIN_SEQ_LEN
    assert cake_module._SPLITKV_WIDE_MAX_ROWS == 1
    for start in starts.values():
        assert start % cake_module._KV_TILE == 0
    plan = cake_module.plan_kv_split
    devices = (("sm_100a", 148), ("sm_103a", 152))
    # (rows, max_seq_len) -> splits on B200 (sm_100a, 148 SMs) and GB300
    # (sm_103a, 152 SMs).
    expected = {
        (1, 32768): (32, 32),
        (1, 33280): (29, 29),
        (1, 65408): (32, 32),
        (1, 65409): (64, 32),
        (1, 65536): (64, 32),
        (1, 65568): (57, 31),
        (1, 130944): (64, 32),
        (1, 130945): (64, 64),
        (1, 131072): (64, 64),
        (1, 131200): (61, 61),
        (1, 262144): (64, 64),
        (2, 32768): (32, 32),
        (2, 65536): (32, 32),
        (2, 131072): (32, 32),
        (3, 65536): (24, 25),
        (3, 262144): (24, 25),
    }
    for (rows, max_seq_len), splits in expected.items():
        for (arch, num_sms), split in zip(devices, splits, strict=True):
            assert plan(rows, max_seq_len, num_sms, arch) == split, (
                arch,
                num_sms,
                rows,
                max_seq_len,
            )
    lengths = (
        1025,
        2048,
        4096,
        8192,
        16384,
        16385,
        24576,
        32640,
        32641,
        32767,
        32768,
        33280,
        49152,
        65408,
        65409,
        65536,
        65568,
        130944,
        130945,
        131072,
        131200,
        262144,
    )
    # The listed devices, the other arch at the same SM count, and no arch.
    for arch, num_sms in devices + (("sm_103a", 148), ("sm_100a", 152), (None, 148)):
        start = starts.get((arch, num_sms))
        for rows in range(1, 39):
            for max_seq_len in lengths:
                split = plan(rows, max_seq_len, num_sms, arch)
                case = (arch, num_sms, rows, max_seq_len)
                tiles = -(-max_seq_len // 128)
                if start is None or rows > 1 or tiles * 128 < start:
                    assert split == _narrow_plan(rows, max_seq_len, num_sms), case
                else:
                    assert split == _wide_plan(rows, max_seq_len, num_sms), case
                if arch is None:
                    assert split == plan(rows, max_seq_len, num_sms), case
                assert 1 <= split <= 64, case
                assert rows * split <= num_sms // 2, case
                # Every max_seq_len of one KV tile count plans one count.
                assert split == plan(rows, tiles * 128, num_sms, arch), case


def test_splitkv_binding_caps_match_the_host():
    """The binding accepts at most ``_SPLITKV_MAX_SPLIT`` splits and launches
    the wide reducer build above ``_SPLITKV_NARROW_MAX_SPLIT``; no module
    overrides either. The wide build's opt-in and launch use one dynamic
    shared-memory size above 48 KiB and its own kernel. No GPU needed."""
    import re

    text = cake_module._source_path(cake_module._SPLITKV_BINDING).read_text()
    assert f"#define CAKE_SPLITKV_MAX_SPLIT {cake_module._SPLITKV_MAX_SPLIT}\n" in text
    assert (
        f"#define CAKE_SPLITKV_NARROW_MAX_SPLIT "
        f"{cake_module._SPLITKV_NARROW_MAX_SPLIT}\n"
    ) in text
    assert "(num_split <= CAKE_SPLITKV_MAX_SPLIT)" in text
    assert "if (num_split > CAKE_SPLITKV_NARROW_MAX_SPLIT) {" in text
    start = text.index("namespace CAKE_VARIANT_NS_3 {")
    block = text[start : text.index("}  // namespace CAKE_VARIANT_NS_3", start)]
    opt_in = re.findall(
        r"CakeSetMaxDynamicSmem\(\s*reinterpret_cast<const void\*>\((kernel_\w+)\),"
        r"\s*(\d+)\)",
        block,
    )
    launch = re.findall(r"config\.dynamicSmemBytes = (\d+)u;", block)
    kernels = set(re.findall(r"reinterpret_cast<const void\*>\((kernel_\w+)\)", block))
    assert len(opt_in) == 1 and len(launch) == 1, (opt_in, launch)
    assert int(opt_in[0][1]) == int(launch[0]) > 48 * 1024
    for name, family in cake_module.FAMILIES.items():
        if family["binding"] != cake_module._SPLITKV_BINDING:
            continue
        assert kernels == {f"kernel_{family['dependencies'][3]}"}, name
        for flag in family.get("binding_flags", []):
            assert not flag.startswith("-DCAKE_SPLITKV_MAX_SPLIT"), name
            assert not flag.startswith("-DCAKE_SPLITKV_NARROW_MAX_SPLIT"), name


@pytest.fixture
def forced_split(monkeypatch):
    """Force the split count ``plan_kv_split`` returns to the launch (``None``:
    the rule's own count); records every count the launch used."""
    used: list = []
    plan = cake_module.plan_kv_split

    def force(value):
        def spy(total_q, max_seq_len, num_sms, arch=None):
            split = (
                plan(total_q, max_seq_len, num_sms, arch) if value is None else value
            )
            used.append(split)
            return split

        monkeypatch.setattr(cake_module, "plan_kv_split", spy)
        return used

    return force


# name: (batch, q_len, page_size, sink, bmm2_scale, seq_lens pattern, max_seq_len)
_WIDE_CASES = {
    "rows1_page32_varied": (1, 1, 32, False, 1.0, "varied", 32768),
    "rows2_q_len2_page64_sink_bmm2": (1, 2, 64, True, 0.5, "varied", 32768),
    "rows2_page32_padded": (2, 1, 32, False, 1.0, "padded", 32768),
}
_WIDE_SPLITS = (33, 37, 38, 48, 64)
_WIDE_DRAWS = range(3)


def _wide_accuracy(case, label):
    """Run the cells of ``case`` (``_WIDE_CASES`` layout) and check every draw as
    ``test_splitkv_matches_trtllm_gen_and_fp64`` does: on live rows the maximum
    absolute error against FP64 is within TRTLLM-GEN's plus one BF16 unit of the
    largest reference value, and within the suite's tolerance; padded rows are
    exact zeros and the default and the stateless mode agree."""
    batch, q_len, page_size, has_sink, bmm2_scale, pattern, max_seq_len = case
    for seed in _WIDE_DRAWS:
        query, kv_cache, block_tables, seq_lens = _inputs(
            batch, q_len, page_size, max_seq_len, pattern, seed
        )
        sink = None
        if has_sink:
            generator = torch.Generator(device="cuda").manual_seed(100 + seed)
            sink = torch.randn(NUM_HEADS, device="cuda", generator=generator) * 2.0
        outputs = [
            _cake(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                sink=sink,
                bmm2_scale=bmm2_scale,
                out=_nan_out(query),
                **mode,
            )
            for mode in ({}, {"stateless": True})
        ]
        torch.cuda.synchronize()
        cell = f"{label} seed{seed}"
        assert torch.equal(outputs[0], outputs[1]), cell
        live = _row_lengths(seq_lens, q_len) > 0
        padded = outputs[0][~live]
        assert torch.equal(padded, torch.zeros_like(padded)), cell
        assert bool(live.any()), cell
        reference = _reference_fp64(
            query, kv_cache, block_tables, seq_lens, sink, bmm2_scale
        )[live]
        trtllm_gen = _trtllm_gen(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            sink=sink,
            bmm2_scale=bmm2_scale,
            out=_nan_out(query),
        )
        torch.cuda.synchronize()
        cake_live = outputs[0][live].double()
        assert bool(cake_live.isfinite().all()), cell
        cake_error = float((cake_live - reference).abs().max())
        trtllm_gen_error = float((trtllm_gen[live].double() - reference).abs().max())
        ulp = _bf16_ulp(float(reference.abs().max()))
        rel_l2 = float(
            (cake_live - reference).norm() / reference.norm().clamp_min(1e-30)
        )
        print(
            f"{cell}: max_abs cake {cake_error:.3e} trtllm-gen "
            f"{trtllm_gen_error:.3e} ulp {ulp:.3e} rel_l2 {rel_l2:.3e}"
        )
        assert cake_error <= trtllm_gen_error + ulp, cell
        torch.testing.assert_close(
            cake_live.float(),
            reference.float(),
            rtol=TOLERANCE,
            atol=TOLERANCE,
            msg=lambda message: f"{cell}: {message}",
        )


@pytest.mark.parametrize("num_split", _WIDE_SPLITS)
@pytest.mark.parametrize("case", sorted(_WIDE_CASES))
def test_splitkv_wide_split_counts_match_trtllm_gen_and_fp64(
    splitkv_device, splitkv_domains, forced_split, case, num_split
):
    """33 to 64 splits (the wide reducer build), forced: accuracy against FP64 at
    the level of TRTLLM-GEN, padded rows exact zeros."""
    used = forced_split(num_split)
    _wide_accuracy(_WIDE_CASES[case], f"{case} split{num_split}")
    assert used and set(used) == {num_split}
    assert set(splitkv_domains) == {cake_module.DOMAIN_BF16_SPLITKV}


# name: (batch, q_len, page_size, sink, bmm2_scale, seq_lens pattern, max_seq_len)
_WIDE_PLAN_CASES = {
    "rows1_kv131072": (1, 1, 32, False, 1.0, "varied", 131072),
    "rows1_kv131200": (1, 1, 32, False, 1.0, "varied", 131200),
    "rows1_q_len1_page64_sink_bmm2_kv196736": (1, 1, 64, True, 0.5, "varied", 196736),
}


@pytest.mark.parametrize("case", sorted(_WIDE_PLAN_CASES))
def test_splitkv_wide_plan_cases_match_trtllm_gen_and_fp64(
    splitkv_device, splitkv_domains, forced_split, case
):
    """The split counts the rule picks above 32 on B200 and GB300 (64 at one row
    of KV 131072, 61 at KV 131200, 62 at KV 196736): accuracy against FP64 at
    the level of TRTLLM-GEN."""
    batch, q_len, max_seq_len = (
        _WIDE_PLAN_CASES[case][0],
        _WIDE_PLAN_CASES[case][1],
        _WIDE_PLAN_CASES[case][6],
    )
    arch, num_sms = splitkv_device
    expected = cake_module.plan_kv_split(batch * q_len, max_seq_len, num_sms, arch)
    if expected <= cake_module._SPLITKV_NARROW_MAX_SPLIT:
        pytest.skip(f"this device plans {expected} splits for {case}")
    used = forced_split(None)
    _wide_accuracy(_WIDE_PLAN_CASES[case], f"{case} split{expected}")
    assert used and set(used) == {expected}
    assert set(splitkv_domains) == {cake_module.DOMAIN_BF16_SPLITKV}


def _check_deterministic(decode, query):
    """200 eager calls and 1000 replays of a captured graph give the first
    call's bits; the stateless mode gives them too."""
    first = decode(_nan_out(query)).clone()
    torch.cuda.synchronize()
    assert bool(first.isfinite().all())
    for call in range(200):
        assert torch.equal(decode(_nan_out(query)), first), call
    assert torch.equal(decode(_nan_out(query), stateless=True), first)
    out = _nan_out(query)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode(out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode(out)
    mismatches = torch.zeros((), dtype=torch.int64, device="cuda")
    for _ in range(1000):
        out.fill_(float("nan"))
        graph.replay()
        mismatches += (out != first).any()
    torch.cuda.synchronize()
    assert int(mismatches) == 0


@pytest.mark.parametrize(
    "case,num_split",
    [("rows2_q_len2_page64_sink_kv32768", split) for split in _WIDE_SPLITS]
    + [("rows1_kv131072", None)],
)
def test_splitkv_wide_split_counts_are_deterministic(
    splitkv_device, splitkv_domains, forced_split, case, num_split
):
    """More than 32 splits (forced, or the rule's 64 at one row of KV 131072):
    200 eager calls and 1000 graph replays give identical bits, and the
    stateless mode agrees."""
    batch, q_len, page_size, max_seq_len, has_sink = {
        "rows2_q_len2_page64_sink_kv32768": (1, 2, 64, 32768, True),
        "rows1_kv131072": (1, 1, 32, 131072, False),
    }[case]
    if num_split is None:
        arch, num_sms = splitkv_device
        expected = cake_module.plan_kv_split(batch * q_len, max_seq_len, num_sms, arch)
        if expected <= cake_module._SPLITKV_NARROW_MAX_SPLIT:
            pytest.skip(f"this device plans {expected} splits for {case}")
    used = forced_split(num_split)
    query, kv_cache, block_tables, seq_lens = _inputs(
        batch, q_len, page_size, max_seq_len, "padded", 0
    )
    assert int(seq_lens[-1]) == max_seq_len
    sink = None
    if has_sink:
        generator = torch.Generator(device="cuda").manual_seed(7)
        sink = torch.randn(NUM_HEADS, device="cuda", generator=generator) * 2.0

    def decode(output, **kwargs):
        return _cake(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            sink=sink,
            bmm2_scale=0.5 if has_sink else 1.0,
            out=output,
            **kwargs,
        )

    _check_deterministic(decode, query)
    assert used and min(used) > cake_module._SPLITKV_NARROW_MAX_SPLIT
    if num_split is not None:
        assert set(used) == {num_split}
    assert set(splitkv_domains) == {cake_module.DOMAIN_BF16_SPLITKV}


def test_splitkv_binding_rejects_more_splits_than_it_merges(
    splitkv_device, forced_split
):
    """65 splits, one more than the reducers merge: the binding raises before
    any launch."""
    forced_split(cake_module._SPLITKV_MAX_SPLIT + 1)
    query, kv_cache, block_tables, seq_lens = _inputs(1, 1, 32, 32768, "varied", 0)
    with pytest.raises(ValueError, match="no dispatch route"):
        _cake(query, kv_cache, block_tables, seq_lens, 32768, stateless=True)
        torch.cuda.synchronize()
