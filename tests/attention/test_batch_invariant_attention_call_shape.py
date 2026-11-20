# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The call shapes of serving engines on the batch-invariant attention wrappers: the arguments the class does not consume
are accepted at the values that ask for nothing and refused (typed error) at every other value, and no accepted value changes
an output bit. Device-free tests of each rule and of the delegate's plan / run, then replays on the device of the eager
``constructor -> plan -> run`` sequences of vLLM and SGLang against the minimal call."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.attention.batch_invariant import neutral, wrappers
from flashinfer.page import get_seq_lens

from .test_batch_invariant_attention import Problem, bits, needs_device, workspace

CLASS = wrappers.CLASS
BF16 = torch.bfloat16
Unsupported = bi.BatchInvariantUnsupportedError


# ------------------------------------------------------------------------------------------------------ the rules
@pytest.mark.parametrize(
    "scales",
    [
        (),
        (None, None, None),
        (1.0, 1.0, 1.0),
        (1, 1.0, None, 1.0),
        (np.float32(1.0), torch.tensor(1.0), torch.ones(1, dtype=torch.float64)),
    ],
)
def test_unit_scales_are_accepted(scales):
    neutral.check_unit_scales(CLASS, scales)


def test_scales_a_class_takes_as_operands_are_not_checked():
    """A class with per-tensor K / V scales names them ``free``; the query and the output scale stay unit-only."""
    free = ("k_scale", "v_scale")
    neutral.check_unit_scales(CLASS, (1.0, 0.0371, 2.5, None), free=free)
    neutral.check_unit_scales(CLASS, (None, torch.tensor(0.5), 0.25, 1.0), free=free)
    with pytest.raises(Unsupported, match="q_scale=0.5"):
        neutral.check_unit_scales(CLASS, (0.5, 0.0371, 2.5), free=free)
    with pytest.raises(Unsupported, match="o_scale=2.0"):
        neutral.check_unit_scales(CLASS, (1.0, 0.0371, 2.5, 2.0), free=free)


@pytest.mark.parametrize(
    "scales,named",
    [
        ((0.5, None, None), "q_scale=0.5"),
        ((None, 2.0, None), "k_scale=2.0"),
        ((1.0, 1.0, 1.0000001), "v_scale"),
        ((None, None, None, 0.0), "o_scale=0.0"),
        ((None, float("nan"), None), "k_scale=nan"),
        ((None, torch.tensor(0.25), None), "k_scale=0.25"),
        ((None, None, torch.ones(2)), "v_scale as a tensor of 2 elements"),
        ((None, "1.0x", None), "k_scale='1.0x'"),
    ],
)
def test_non_unit_scales_raise(scales, named):
    with pytest.raises(Unsupported) as e:
        neutral.check_unit_scales(CLASS, scales)
    assert named in str(e.value) and "exactly 1.0" in str(e.value)


@pytest.mark.parametrize(
    "bad,named",
    [
        (True, "q_scale=True"),
        (torch.tensor(1), "q_scale as a tensor of dtype torch.int64"),
        (torch.tensor(True), "q_scale as a tensor of dtype torch.bool"),
    ],
)
def test_scale_forms_that_are_not_numbers_raise(bad, named):
    with pytest.raises(Unsupported) as e:
        neutral.check_unit_scales(CLASS, (bad, None, None))
    assert named in str(e.value)


@needs_device
def test_device_scale_tensors_are_read_once_per_version(monkeypatch):
    """A one-element device tensor as a scale is read back once and remembered for the tensor object and its version: the
    same tensor again costs no readback, an in-place write is read again (and refused when it is no longer one), another
    tensor of the same value is its own entry, and nothing is read inside a CUDA-graph capture."""
    reads = []
    read = neutral._read_device_scale
    monkeypatch.setattr(
        neutral, "_read_device_scale", lambda t: reads.append(1) or read(t)
    )
    dev = torch.device("cuda")
    one = torch.ones((), device=dev)
    for _ in range(5):
        neutral.check_unit_scales(CLASS, (one, one, None, one))
    assert len(reads) == 1
    assert neutral.resolve_scale(CLASS, "k_scale", one) == 1.0 and len(reads) == 1
    other = torch.ones(1, device=dev, dtype=torch.float64)
    neutral.check_unit_scales(CLASS, (one, other, other))
    assert len(reads) == 2
    one.mul_(0.5)  # an in-place write: a new version
    with pytest.raises(Unsupported, match="q_scale=0.5"):
        neutral.check_unit_scales(CLASS, (one,))
    assert len(reads) == 3
    one.fill_(1.0)
    neutral.check_unit_scales(CLASS, (one,))
    neutral.check_unit_scales(CLASS, (one,))
    assert len(reads) == 4
    # a class that owns the scale gets its value from the same resolver
    quarter = torch.full((1,), 0.25, device=dev)
    neutral.check_unit_scales(CLASS, (None, quarter), free=("k_scale",))
    assert neutral.resolve_scale(CLASS, "k_scale", quarter) == 0.25
    for bad, named in (
        (torch.ones(2, device=dev), "as a tensor of 2 elements"),
        (torch.ones(1, device=dev, dtype=torch.int32), "of dtype torch.int32"),
    ):
        with pytest.raises(Unsupported, match=named):
            neutral.check_unit_scales(CLASS, (bad,))
    # the entry goes with the tensor
    key = id(other)
    assert key in neutral._DEVICE_SCALES
    del other
    assert key not in neutral._DEVICE_SCALES
    # inside a capture: a tensor seen before is served from memory, a new one raises without a readback
    fresh = torch.ones((), device=dev)
    before = len(reads)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        neutral.check_unit_scales(CLASS, (one,))
        with pytest.raises(Unsupported, match="inside a CUDA-graph capture"):
            neutral.check_unit_scales(CLASS, (fresh,))
        torch.zeros(1, device=dev).add_(1)
    assert len(reads) == before


@needs_device
@pytest.mark.parametrize("kind", ["decode", "paged", "ragged"])
def test_wrappers_accept_device_unit_scales(kind, monkeypatch):
    """Unit scales as one-element device tensors on the three wrappers: the bits of the call without scales, one readback
    per tensor however many ``run()`` calls follow, and a non-unit value after an in-place write refused."""
    reads = []
    read = neutral._read_device_scale
    monkeypatch.setattr(
        neutral, "_read_device_scale", lambda t: reads.append(1) or read(t)
    )
    pr = Problem([3, 1, 40], [700, 37, 40], 32, 8, 128, 16, seed=5)
    dev = pr.device
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    ws = workspace(dev)
    if kind == "decode":
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16, **kw)
        args = (pr.q[:3].contiguous(), pr.cache("NHD"))
    elif kind == "paged":
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16, causal=True, **kw)  # fmt: skip
        args = (pr.q, pr.cache("NHD"))
    else:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(pr.qo_indptr, pr.kv_indptr_rag, 32, 8, 128, causal=True, **kw)
        args = (pr.q, *pr.ragged("NHD"))
    ref = bits(w.run(*args)).clone()
    scales = {n: torch.ones((), device=dev) for n in ("q_scale", "k_scale", "v_scale")}
    for _ in range(4):
        assert torch.equal(bits(w.run(*args, **scales)), ref)
    assert len(reads) == 3
    scales["v_scale"].mul_(2.0)
    with pytest.raises(Unsupported, match="v_scale=2.0"):
        w.run(*args, **scales)
    assert len(reads) == 4


@pytest.mark.parametrize(
    "size,disable,fused",
    [
        (None, False, False),
        (-1, False, False),
        (2048, False, False),
        (4096, True, True),
        (1, False, False),
        (None, True, True),
    ],
)
def test_split_requests_the_class_satisfies(size, disable, fused):
    assert neutral.check_split_request(CLASS, size, disable) is fused


@pytest.mark.parametrize(
    "size,disable,named",
    [
        (0, False, "fixed_split_size=0"),
        (-2, False, "fixed_split_size=-2"),
        (2048.0, False, "fixed_split_size=2048.0"),
        (True, False, "fixed_split_size=True"),
        (None, "yes", "disable_split_kv='yes'"),
        (None, None, "disable_split_kv=None"),
    ],
)
def test_split_requests_without_a_meaning_raise(size, disable, named):
    with pytest.raises(Unsupported) as e:
        neutral.check_split_request(CLASS, size, disable)
    assert named in str(e.value)


def test_lengths_rule():
    planned = [700, 0, 33]
    for given in (
        torch.tensor(planned, dtype=torch.int32),
        torch.tensor(planned, dtype=torch.int64).reshape(3, 1),
        planned,
    ):
        neutral.check_lengths(CLASS, "seq_lens", given, planned, "the page table")
    # the second convention of a pageless request (`get_seq_lens` reports its last-page entry)
    neutral.check_lengths(
        CLASS,
        "seq_lens",
        torch.tensor([700, 5, 33]),
        planned,
        "the page table",
        [700, 5, 33],
    )
    for given, named in (
        (torch.tensor([700, 0, 32]), "seq_lens[2] = 32 where the page table gives 33"),
        (torch.tensor([700, 5, 33]), "seq_lens[1] = 5"),
        (torch.tensor([700, 0]), "seq_lens of 2 entries for 3 requests"),
        (torch.tensor([700.0, 0.0, 33.0]), "seq_lens of dtype torch.float32"),
    ):
        with pytest.raises(Unsupported) as e:
            neutral.check_lengths(CLASS, "seq_lens", given, planned, "the page table")
        assert named in str(e.value)


def _csr(spans, start=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    total = start + sum(spans)
    indices = torch.randperm(total + 5, generator=g)[:total].to(torch.int32)
    indptr = torch.tensor(
        [start] + [start + sum(spans[: i + 1]) for i in range(len(spans))]
    )
    return indptr, indices


def _table(indptr, indices, width, fill=-7):
    rows = []
    for i in range(indptr.numel() - 1):
        row = indices[int(indptr[i]) : int(indptr[i + 1])].tolist()
        rows.append(row + [fill] * (width - len(row)))
    return torch.tensor(rows, dtype=torch.int32).reshape(indptr.numel() - 1, width)


@pytest.mark.parametrize("start", [0, 4])
def test_block_table_rule(start):
    spans = [3, 0, 5, 1]
    indptr, indices = _csr(spans, start)
    kv_spans = indptr[1:] - indptr[:-1]
    for width in (5, 9):
        table = _table(indptr, indices, width)
        neutral.check_block_table_shape(CLASS, table, kv_spans)
        assert (
            int(neutral.block_table_mismatches(table, indices, indptr, kv_spans)) == 0
        )
    table = _table(indptr, indices, 5)
    table[2, 4] += 1
    table[0, 0] += 1
    table[1, 0] = 99  # beyond the request's pages: not compared
    assert int(neutral.block_table_mismatches(table, indices, indptr, kv_spans)) == 2
    assert "differ from kv_indices at 2 page entries" in str(
        neutral.block_table_mismatch_error(CLASS, 2)
    )
    for bad, named in (
        (_table(indptr, indices, 5)[:3], "block_tables of shape (3, 5) for 4 requests"),
        (_table(indptr, indices, 5)[:, :4], "of up to 5 pages"),
        (_table(indptr, indices, 5).flatten(), "block_tables of shape (20,)"),
        (_table(indptr, indices, 5).float(), "block_tables of dtype torch.float32"),
        ([[1, 2]], "block_tables of type list"),
    ):
        with pytest.raises(Unsupported) as e:
            neutral.check_block_table_shape(CLASS, bad, kv_spans)
        assert named in str(e.value)


# ------------------------------------------------------------------------------------------------------ the delegate
class _Delegates:
    """Device-free delegates: the device gate, the planner and the launch are replaced by recorders, so everything the
    wrappers' ``plan`` / ``run`` decide before the engine runs on the host."""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(wrappers, "check_device", lambda *a, **k: None)

        def plan_attention(
            qo,
            kv,
            last,
            Hq,
            Hkv,
            D,
            page,
            window_left,
            config,
            ws,
            device,
            ragged=False,
            has_sinks=None,
        ):
            self.calls.append(
                (
                    "plan",
                    qo.tolist(),
                    kv.tolist(),
                    None if last is None else last.tolist(),
                    (Hq, Hkv, D, page, window_left, ragged),
                    config,
                )
            )
            # what the delegate reads off the engine's planned record
            # a plan record without tensors: nothing for a run's buffers to overlap
            return SimpleNamespace(
                has_sinks=has_sinks,
                **dict.fromkeys(
                    (
                        "kv_indptr",
                        "workspace",
                        "ws",
                        "scratch",
                        "desc",
                        "groups",
                        "fixup_groups",
                        "fold_counters",
                        "counts",
                    )
                ),
            )

        def launch(
            planned,
            q,
            k,
            v,
            kv_indptr,
            kv_indices,
            c,
            window_left,
            layout,
            out,
            config,
            ragged,
            enable_pdl=None,
            lse=None,
            lse_mode=0,
            sinks=None,
        ):
            self.calls.append(("run", planned, c, window_left, layout, config, ragged))

        monkeypatch.setattr(wrappers.engine, "plan_attention", plan_attention)
        monkeypatch.setattr(wrappers.engine, "_run_attention_validated", launch)

    def make(self, kind):
        ws = torch.empty(1 << 16, dtype=torch.uint8)
        return wrappers.BatchInvariantAttention(ws, "NHD", kind, "auto", None)


@pytest.fixture
def delegates(monkeypatch):
    return _Delegates(monkeypatch)


def _problem():
    return Problem([3, 1, 40], [700, 37, 40], 8, 8, 128, 16, seed=3, device="cpu")


def _plan_kwargs(kind, pr, **over):
    kw = dict(
        qo_indptr=None if kind == "decode" else pr.qo_indptr,
        kv_indptr=pr.kv_indptr_rag if kind == "ragged" else pr.kv_indptr,
        kv_indices=None if kind == "ragged" else pr.kv_indices,
        kv_last_page_len=None if kind == "ragged" else pr.kv_last,
        num_qo_heads=pr.Hq,
        num_kv_heads=pr.Hkv,
        head_dim_qk=pr.D,
        head_dim_vo=None,
        page_size=pr.page,
        causal=True,
        window_left=-1,
        logits_soft_cap=None,
        pos_encoding_mode="NONE",
        q_data_type=BF16,
        kv_data_type=BF16,
        o_data_type=None,
        sm_scale=None,
    )
    kw.update(over)
    return kw


def _table_of(pr):
    width = int((pr.kv_indptr[1:] - pr.kv_indptr[:-1]).max())
    return _table(pr.kv_indptr, pr.kv_indices, width)


def _neutral_plan_arguments(kind, pr):
    kv_lens = torch.tensor(pr.kv_lens, dtype=torch.int32)
    cases = {
        "fixed_split_size=-1": dict(fixed_split_size=-1),
        "fixed_split_size=2048": dict(fixed_split_size=2048),
        "fixed_split_size=4096": dict(fixed_split_size=4096),
        "disable_split_kv=False": dict(disable_split_kv=False),
        "seq_lens": dict(seq_lens=kv_lens),
        "seq_lens int64": dict(seq_lens=kv_lens.to(torch.int64)),
        "o_data_type": dict(o_data_type=BF16),
        "o_data_type str": dict(o_data_type="bfloat16"),
        "non_blocking=False": dict(non_blocking=False),
        "non_blocking=True": dict(non_blocking=True),
        "logits_soft_cap=0.0": dict(logits_soft_cap=0.0),
        "rope None": dict(rope_scale=None, rope_theta=None),
    }
    if kind != "ragged":
        cases["seq_lens get_seq_lens"] = dict(
            seq_lens=get_seq_lens(pr.kv_indptr, pr.kv_last, pr.page)
        )
        cases["block_tables"] = dict(block_tables=_table_of(pr))
    if kind != "decode":
        cases["seq_lens_q"] = dict(
            seq_lens_q=torch.tensor(pr.q_lens, dtype=torch.int32)
        )
    return cases


KINDS = ("decode", "prefill", "ragged")


@pytest.mark.parametrize("kind", KINDS)
def test_plan_neutral_arguments_reach_the_same_plan(delegates, kind):
    pr = _problem()
    d = delegates.make(kind)
    d.plan(**_plan_kwargs(kind, pr))
    minimal = delegates.calls.pop()
    state = (
        d._sm_scale,
        d._scale_log2,
        d._window_left,
        d._total_q,
        d._heads,
        d._page,
        d._max_page,
    )
    for name, extra in _neutral_plan_arguments(kind, pr).items():
        d.plan(**_plan_kwargs(kind, pr, **extra))
        assert delegates.calls.pop() == minimal, name
        assert (
            d._sm_scale,
            d._scale_log2,
            d._window_left,
            d._total_q,
            d._heads,
            d._page,
            d._max_page,
        ) == state, name
    # disable_split_kv=True is honoured: the same plan arguments with fused units
    for extra in (
        dict(disable_split_kv=True),
        dict(disable_split_kv=True, fixed_split_size=4096),
    ):
        d.plan(**_plan_kwargs(kind, pr, **extra))
        fused = delegates.calls.pop()
        assert fused[:5] == minimal[:5]
        assert fused[5].split_policy == "fused" and minimal[5].split_policy == "auto"
        from dataclasses import replace

        assert replace(fused[5], split_policy="auto") == minimal[5]


def _non_neutral_plan_arguments(kind, pr):
    kv_lens = torch.tensor(pr.kv_lens, dtype=torch.int32)
    off = kv_lens.clone()
    off[0] -= 1
    cases = {
        "fixed_split_size=0": (dict(fixed_split_size=0), "fixed_split_size=0"),
        "fixed_split_size=-4096": (
            dict(fixed_split_size=-4096),
            "fixed_split_size=-4096",
        ),
        "fixed_split_size float": (
            dict(fixed_split_size=2048.0),
            "fixed_split_size=2048.0",
        ),
        "disable_split_kv=1": (dict(disable_split_kv=1), "disable_split_kv=1"),
        "seq_lens off by one": (dict(seq_lens=off), "seq_lens[0] = 699"),
        "seq_lens short": (
            dict(seq_lens=kv_lens[:2]),
            "seq_lens of 2 entries for 3 requests",
        ),
        "o_data_type fp16": (
            dict(o_data_type=torch.float16),
            "o_data_type torch.float16",
        ),
        "o_data_type fp32": (dict(o_data_type="float32"), "o_data_type torch.float32"),
        "logits_soft_cap": (dict(logits_soft_cap=30.0), "logits_soft_cap 30.0"),
    }
    if kind == "decode":
        cases["seq_lens_q"] = (
            dict(seq_lens_q=torch.ones(3, dtype=torch.int32)),
            "seq_lens_q on the decode wrapper",
        )
    else:
        q_off = torch.tensor(pr.q_lens, dtype=torch.int32)
        q_off[2] += 1
        cases["seq_lens_q off by one"] = (
            dict(seq_lens_q=q_off),
            "seq_lens_q[2] = 41 where qo_indptr gives 40",
        )
    if kind == "ragged":
        cases["block_tables"] = (
            dict(block_tables=torch.zeros(3, 4, dtype=torch.int32)),
            "block_tables on the ragged wrapper",
        )
    else:
        table = _table_of(pr)
        wrong = table.clone()
        wrong[0, 1] += 1
        cases["block_tables wrong page"] = (
            dict(block_tables=wrong),
            "differ from kv_indices at 1 page entries",
        )
        cases["block_tables narrow"] = (
            dict(block_tables=table[:, :-1]),
            "block_tables of shape",
        )
        cases["block_tables batch"] = (dict(block_tables=table[:2]), "for 3 requests")
    return cases


@pytest.mark.parametrize("kind", KINDS)
def test_plan_non_neutral_arguments_raise_and_keep_the_plan(delegates, kind):
    pr = _problem()
    d = delegates.make(kind)
    d.plan(**_plan_kwargs(kind, pr))
    planned = d._planned
    delegates.calls.clear()
    for name, (extra, named) in _non_neutral_plan_arguments(kind, pr).items():
        with pytest.raises(Unsupported) as e:
            d.plan(**_plan_kwargs(kind, pr, **extra))
        assert named in str(e.value), (name, str(e.value))
        assert "never falls back" in str(e.value)
        assert d._planned is planned and not delegates.calls, name


@pytest.mark.parametrize("kind", KINDS)
def test_run_neutral_arguments_reach_the_same_launch(delegates, kind):
    pr = _problem()
    d = delegates.make(kind)
    d.plan(**_plan_kwargs(kind, pr))
    delegates.calls.clear()
    q = pr.q if kind != "decode" else pr.q[:3].contiguous()
    if kind == "decode":
        d._total_q = 3
    k, v = pr.ragged("NHD") if kind == "ragged" else pr.cache("NHD")
    base = dict(out=None, return_lse=False, lse=None, window_left=None)
    d.run(q, k, v, **base)
    minimal = delegates.calls.pop()
    out = torch.empty(q.shape[0] + 4, *q.shape[1:], dtype=BF16)
    for extra in (
        dict(scales=(None, None, None)),
        dict(scales=(1.0, 1.0, 1.0)),
        dict(scales=(1.0, 1.0, 1.0, 1.0)),
        dict(scales=(1, torch.tensor(1.0), np.float64(1.0))),
        dict(sinks=None, kv_cache_sf=None),
        dict(scales=(1.0, 1.0, 1.0), sinks=None, kv_cache_sf=None, window_left=-1),
        dict(out=out[4:]),
        dict(out=out[: q.shape[0]]),
        dict(skip_softmax_threshold_scale_factor=None, use_fp16_softmax=None),
    ):
        got = d.run(q, k, v, **{**base, **extra})
        assert delegates.calls.pop() == minimal, extra
        if extra.get("out") is not None:
            assert got is extra["out"]
    for extra, named in (
        (dict(scales=(0.5, None, None)), "q_scale=0.5"),
        (dict(scales=(1.0, 0.0625, 1.0)), "k_scale=0.0625"),
        (dict(scales=(1.0, 1.0, 2.0)), "v_scale=2.0"),
        (dict(scales=(None, None, None, 3.0)), "o_scale=3.0"),
        (dict(kv_cache_sf=torch.zeros(1)), "KV cache"),
        (dict(window_left=5), "window_left 5"),
    ):
        with pytest.raises(Unsupported) as e:
            d.run(q, k, v, **{**base, **extra})
        assert named in str(e.value) and not delegates.calls, extra


@needs_device
def test_fast_decode_plan_replans_a_batch_invariant_wrapper():
    """``fast_decode_plan`` on a batch-invariant wrapper goes to the class's own planner (the wrapper has no cached module):
    outside CUDA graphs it is ``plan()``, with the serving arguments at their neutral values."""
    from flashinfer.decode import fast_decode_plan

    pr = Problem([1, 1, 1], [700, 40, 1300], 32, 8, 128, 16, seed=3)
    q = pr.q.contiguous()
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    args = (pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(*args, **kw)
    ref = bits(w.run(q, pr.cache("NHD"))).clone()
    for extra in (
        dict(),
        dict(fixed_split_size=-1, disable_split_kv=False),
        dict(fixed_split_size=2048, disable_split_kv=True),
        dict(global_override_indptr_cpu=pr.kv_indptr.cpu()),
    ):
        fast_decode_plan(w, *args, **kw, **extra)
        assert w._logits_soft_cap == 0.0
        assert torch.equal(bits(w.run(q, pr.cache("NHD"))), ref), extra
    with pytest.raises(Unsupported):
        fast_decode_plan(w, *args, logits_soft_cap=30.0, **kw)


# ------------------------------------------------------------------------------------------------------ on the device
def _vllm_layer_cache(pr):
    """A vLLM layer cache: K and V packed in the content dimension of one tensor, handed to FlashInfer as the two views of
    a split along it."""
    packed = torch.cat([pr.k_nhd, pr.v_nhd], dim=-1).contiguous()
    return packed.split(pr.D, dim=-1)


def _vllm_fast_plan_decode(self, indptr_cpu, indices, last_page_len_cpu, seq_lens_cpu, num_qo_heads, num_kv_heads, head_dim, page_size, pos_encoding_mode="NONE", window_left=-1, logits_soft_cap=None, q_data_type="float16", kv_data_type=None, o_data_type=None, data_type=None, sm_scale=None, rope_scale=None, rope_theta=None, non_blocking=True, fixed_split_size=-1, disable_split_kv=False, q_len_per_req=1):  # fmt: skip
    """The eager branch of vLLM's ``fast_plan_decode`` (its argument list, verbatim)."""
    assert not self.is_cuda_graph_enabled
    self.plan(
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        o_data_type=o_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        block_tables=None,
        seq_lens=seq_lens_cpu,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
        q_len_per_req=q_len_per_req,
    )


#: vLLM's split arguments: the default mode, and VLLM_BATCH_INVARIANT (decode size, prefill size, disable_split_kv)
VLLM_SPLIT_FLAGS = {"default": (-1, -1, False), "batch_invariant": (2048, 4096, True)}


def _vllm_step(pr, num_decodes, flags, window_left=-1, pin=True, ws=None):
    """One vLLM forward of a mixed batch through the FLASHINFER backend's wrapper route: the first ``num_decodes`` requests
    decode one token each, the rest prefill; host CSR slices of the builder's buffers (the prefill slice of ``kv_indptr`` is
    not rebased), device page indices, the KV lengths restated as ``seq_lens``, unit scales, outputs written into slices of
    one buffer. Returns the output and the two wrappers."""
    dev = pr.device
    decode_split, prefill_split, disable_split_kv = VLLM_SPLIT_FLAGS[flags]
    ws = ws if ws is not None else workspace(dev)
    sm_scale = 1.0 / math.sqrt(pr.D)
    logits_soft_cap = None
    kv_cache_tuple = _vllm_layer_cache(pr)
    qo_cpu = pr.qo_indptr.cpu()
    indptr_cpu, last_cpu = pr.kv_indptr.cpu(), pr.kv_last.cpu()
    num_reqs = len(pr.q_lens)
    num_decode_tokens = int(qo_cpu[num_decodes])
    assert num_decode_tokens == num_decodes
    output = torch.empty(pr.q.shape, dtype=BF16, device=dev)

    def staged(t):
        return t.pin_memory() if pin else t

    prefill_wrapper = decode_wrapper = None
    if num_decodes < num_reqs:
        prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="auto", batch_invariant=True
        )
        qo_prefill = staged(qo_cpu[num_decodes:] - qo_cpu[num_decodes])
        indptr_prefill = staged(indptr_cpu[num_decodes : num_reqs + 1])
        last_prefill = staged(last_cpu[num_decodes:num_reqs])
        kv_lens_prefill = staged(get_seq_lens(indptr_prefill, last_prefill, pr.page))
        prefill_wrapper.plan(
            qo_indptr=qo_prefill,
            paged_kv_indptr=indptr_prefill,
            paged_kv_indices=pr.kv_indices,
            paged_kv_last_page_len=last_prefill,
            seq_lens=kv_lens_prefill,
            num_qo_heads=pr.Hq,
            num_kv_heads=pr.Hkv,
            head_dim_qk=pr.D,
            page_size=pr.page,
            causal=True,
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=BF16,
            kv_data_type=BF16,
            o_data_type=BF16,
            fixed_split_size=prefill_split,
            disable_split_kv=disable_split_kv,
        )
    if num_decodes:
        decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws,
            "NHD",
            use_cuda_graph=False,
            paged_kv_indptr_buffer=None,
            paged_kv_indices_buffer=None,
            paged_kv_last_page_len_buffer=None,
            use_tensor_cores=True,
            backend="auto",
            batch_invariant=True,
        )
        indptr_decode = staged(indptr_cpu[: num_decodes + 1])
        last_decode = staged(last_cpu[:num_decodes])
        kv_lens_decode = staged(get_seq_lens(indptr_decode, last_decode, pr.page))
        _vllm_fast_plan_decode(
            decode_wrapper,
            indptr_cpu=indptr_decode,
            indices=pr.kv_indices,
            last_page_len_cpu=last_decode,
            seq_lens_cpu=kv_lens_decode,
            num_qo_heads=pr.Hq,
            num_kv_heads=pr.Hkv,
            head_dim=pr.D,
            page_size=pr.page,
            pos_encoding_mode="NONE",
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=BF16,
            kv_data_type=BF16,
            o_data_type=BF16,
            fixed_split_size=decode_split,
            disable_split_kv=disable_split_kv,
            q_len_per_req=1,
        )
    # forward(): the state vLLM asserts on the planned wrappers, then the two run() calls
    if prefill_wrapper is not None:
        assert prefill_wrapper._window_left == window_left
        assert prefill_wrapper._logits_soft_cap == (logits_soft_cap or 0.0)
        assert prefill_wrapper._sm_scale == sm_scale
        assert prefill_wrapper._causal is True
        prefill_wrapper.run(
            pr.q[num_decode_tokens:],
            kv_cache_tuple,
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=output[num_decode_tokens:],
            kv_cache_sf=None,
        )
    if decode_wrapper is not None:
        assert decode_wrapper._window_left == window_left
        assert decode_wrapper._logits_soft_cap == (logits_soft_cap or 0.0)
        assert decode_wrapper._sm_scale == sm_scale
        decode_wrapper.run(
            pr.q[:num_decode_tokens],
            kv_cache_tuple,
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=output[:num_decode_tokens],
            kv_cache_sf=None,
            sinks=None,
        )
    return output, prefill_wrapper, decode_wrapper


def _sub(pr, lo, hi):
    """Requests ``lo:hi`` of ``pr`` as CSR metadata rebased to zero (the minimal call's inputs) and their query rows."""
    qo = pr.qo_indptr[lo : hi + 1] - pr.qo_indptr[lo]
    start, stop = int(pr.kv_indptr[lo]), int(pr.kv_indptr[hi])
    indptr = pr.kv_indptr[lo : hi + 1] - start
    q = pr.q[int(pr.qo_indptr[lo]) : int(pr.qo_indptr[hi])].contiguous()
    return qo, indptr, pr.kv_indices[start:stop].contiguous(), pr.kv_last[lo:hi], q


def _minimal_step(pr, num_decodes, window_left=-1):
    """The same mixed batch through the minimal calls: no argument beyond the class's own inputs, separate contiguous K /
    V, rebased device metadata, outputs allocated by the wrappers."""
    outs = []
    ws = workspace(pr.device)
    kw = dict(q_data_type=BF16, kv_data_type=BF16, window_left=window_left)
    num_reqs = len(pr.q_lens)
    if num_decodes:
        _, indptr, indices, last, q = _sub(pr, 0, num_decodes)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(indptr, indices, last, pr.Hq, pr.Hkv, pr.D, pr.page, **kw)
        outs.append(w.run(q, pr.cache("NHD")))
    if num_decodes < num_reqs:
        qo, indptr, indices, last, q = _sub(pr, num_decodes, num_reqs)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(
            qo, indptr, indices, last, pr.Hq, pr.Hkv, pr.D, pr.page, causal=True, **kw
        )
        outs.append(w.run(q, pr.cache("NHD")))
    return torch.cat(outs)


def _mixed(D, page, seed=21, Hq=32, Hkv=8):
    # three decodes (one beyond a chunk, one short, one beyond several chunks), then three prefills (a fresh prompt, a
    # prefix-cache hit, a long prompt that the default plan may split)
    q_lens = [1, 1, 1, 40, 17, 600]
    kv_lens = [900, 37, 2500, 40, 517, 1800]
    return Problem(q_lens, kv_lens, Hq, Hkv, D, page, seed=seed)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("flags", list(VLLM_SPLIT_FLAGS))
def test_vllm_eager_call_sequence_equals_the_minimal_call(D, flags):
    pr = _mixed(D, 16)
    want = _minimal_step(pr, 3)
    got, _, _ = _vllm_step(pr, 3, flags)
    assert torch.equal(bits(got), bits(want))


@needs_device
@pytest.mark.parametrize("page", [1, 32])
@pytest.mark.parametrize("window_left", [-1, 100])
def test_vllm_eager_call_sequence_pages_window_and_pageable_metadata(page, window_left):
    pr = _mixed(128, page, seed=22)
    want = _minimal_step(pr, 3, window_left=window_left)
    for flags in VLLM_SPLIT_FLAGS:
        got, _, _ = _vllm_step(pr, 3, flags, window_left=window_left, pin=False)
        assert torch.equal(bits(got), bits(want)), flags


@needs_device
def test_vllm_pure_decode_and_pure_prefill_steps():
    pr = Problem([1] * 5, [3000, 1, 129, 512, 77], 32, 8, 128, 16, seed=23)
    for flags in VLLM_SPLIT_FLAGS:
        got, prefill, decode = _vllm_step(pr, 5, flags)
        assert prefill is None and torch.equal(bits(got), bits(_minimal_step(pr, 5)))
    pr = Problem([64, 700, 5], [64, 700, 4101], 32, 8, 128, 16, seed=24)
    for flags in VLLM_SPLIT_FLAGS:
        got, prefill, decode = _vllm_step(pr, 0, flags)
        assert decode is None and torch.equal(bits(got), bits(_minimal_step(pr, 0)))


@needs_device
@pytest.mark.parametrize("flags", list(VLLM_SPLIT_FLAGS))
def test_vllm_flags_batch_composition(flags):
    """A prefill batch with decode-like mates (one query token over a long context) under vLLM's flags: every request's
    bits are those of the request served alone, whatever its mates."""
    q_lens = [300, 1, 1, 45, 1]
    kv_lens = [300, 4100, 9, 1045, 700]
    pr = Problem(q_lens, kv_lens, 32, 8, 128, 16, seed=25)
    together, _, _ = _vllm_step(pr, 0, flags)
    row = 0
    for i, ql in enumerate(q_lens):
        alone = Problem.__new__(Problem)
        alone.__dict__.update(pr.__dict__)
        alone.q_lens, alone.kv_lens = [ql], [kv_lens[i]]
        qo, indptr, indices, last, q = _sub(pr, i, i + 1)
        alone.qo_indptr, alone.kv_indptr, alone.kv_indices, alone.kv_last, alone.q = (
            qo,
            indptr,
            indices,
            last,
            q,
        )
        got, _, _ = _vllm_step(alone, 0, flags)
        assert torch.equal(bits(got), bits(together[row : row + ql])), i
        row += ql


@needs_device
@pytest.mark.parametrize("flags", list(VLLM_SPLIT_FLAGS))
@pytest.mark.parametrize("tail", [1, 7, 127, 128, 129, 300])
def test_vllm_flags_prefix_cache_tail(flags, tail):
    """A prefix-cache hit under vLLM's flags: the last ``tail`` tokens of a prompt prefilled over the cached prefix have the
    bits they have in the prefill of the whole prompt, also when the cached length is not a multiple of the tile."""
    L = 1500
    full = Problem([L], [L], 32, 8, 128, 16, seed=26)
    whole, _, _ = _vllm_step(full, 0, flags)
    hit = Problem.__new__(Problem)
    hit.__dict__.update(full.__dict__)
    hit.q_lens = [tail]
    hit.q = full.q[L - tail :].contiguous()
    hit.qo_indptr = torch.tensor([0, tail], dtype=torch.int32, device=full.device)
    got, _, _ = _vllm_step(hit, 0, flags)
    assert torch.equal(bits(got), bits(whole[L - tail :]))


def _sglang_decode(pr, split, scale):
    """SGLang's eager decode: device CSR metadata at page size 1 with a buffer of ones as last-page lengths, ``begin_forward``
    with its positional / keyword list, then the deprecated ``forward`` with the layer's options."""
    bs = len(pr.kv_lens)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device),
        "NHD",
        backend="auto",
        use_tensor_cores=True,
        batch_invariant=True,
    )
    kv_last_page_len = torch.ones(bs + 3, dtype=torch.int32, device=pr.device)
    wrapper.begin_forward(
        pr.kv_indptr,
        pr.kv_indices,
        kv_last_page_len[:bs],
        pr.Hq,
        pr.Hkv,
        pr.D,
        1,
        data_type=BF16,
        q_data_type=BF16,
        non_blocking=True,
        fixed_split_size=split,
        disable_split_kv=False,
    )
    last = (pr.qo_indptr[1:] - 1).long()
    q = pr.q[last].reshape(bs, pr.Hq * pr.D)
    return wrapper.forward(
        q.contiguous().view(-1, pr.Hq, pr.D),
        pr.cache("NHD"),
        sm_scale=pr.D**-0.5,
        logits_soft_cap=0.0,
        k_scale=scale,
        v_scale=scale,
    )


def _sglang_prefill(pr, split, scale, **begin):
    bs = len(pr.kv_lens)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", backend="auto", batch_invariant=True
    )
    kv_last_page_len = torch.ones(bs + 3, dtype=torch.int32, device=pr.device)
    wrapper.begin_forward(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        kv_last_page_len[:bs],
        pr.Hq,
        pr.Hkv,
        pr.D,
        1,
        q_data_type=BF16,
        kv_data_type=BF16,
        custom_mask=None,
        non_blocking=True,
        fixed_split_size=split,
        prefix_len_ptr=None,
        token_pos_in_items_ptr=None,
        token_pos_in_items_len=0,
        max_item_len_ptr=None,
        **begin,
    )
    return wrapper.forward(
        pr.q.reshape(-1, pr.Hq * pr.D).view(-1, pr.Hq, pr.D),
        pr.cache("NHD"),
        causal=True,
        sm_scale=pr.D**-0.5,
        window_left=-1,
        logits_soft_cap=0.0,
        k_scale=scale,
        v_scale=scale,
    )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sglang_eager_call_sequence_equals_the_minimal_call(D):
    pr = Problem([1, 12, 300, 1], [900, 37, 1300, 2500], 32, 8, D, 1, seed=31)
    ws = workspace(pr.device)
    kw = dict(q_data_type=BF16, kv_data_type=BF16)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    w.plan(pr.kv_indptr, pr.kv_indices, pr.kv_last, pr.Hq, pr.Hkv, D, 1, **kw)
    last = (pr.qo_indptr[1:] - 1).long()
    want_decode = w.run(pr.q[last].contiguous(), pr.cache("NHD"))
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        D,
        1,
        causal=True,
        **kw,
    )
    want_prefill = w.run(pr.q, pr.cache("NHD"))
    # (decode, prefill) split tiles: the default mode, then --enable-deterministic-inference; the layer's scales are None
    # for an unquantized model, 1.0 where a caller fills them in
    for decode_split, prefill_split in ((None, None), (2048, 4096)):
        for scale in (None, 1.0):
            got = _sglang_decode(pr, decode_split, scale)
            assert torch.equal(bits(got), bits(want_decode)), (decode_split, scale)
            got = _sglang_prefill(pr, prefill_split, scale, causal=True)
            assert torch.equal(bits(got), bits(want_prefill)), (prefill_split, scale)


@needs_device
def test_sglang_prefill_needs_causal_at_begin_forward():
    """SGLang's paged ``begin_forward`` leaves ``causal`` at its default (False) and names it at ``forward``; the class plans
    the mask, so the plan raises and the caller has to name ``causal=True`` there."""
    pr = Problem([12, 300], [37, 1300], 32, 8, 128, 1, seed=32)
    with pytest.raises(Unsupported) as e:
        _sglang_prefill(pr, 4096, None)
    assert "non-causal" in str(e.value)


@needs_device
@pytest.mark.parametrize("kind", ["decode", "prefill"])
@pytest.mark.parametrize("where", ["cpu", "cuda"])
def test_restated_metadata_on_the_device(kind, where):
    """``seq_lens`` / ``seq_lens_q`` / ``block_tables`` as host or device tensors, with host or device CSR metadata: the
    consistent ones change nothing, the inconsistent ones raise before the plan is replaced."""
    pr = Problem([1, 1, 1], [900, 37, 2500], 32, 8, 128, 16, seed=41)
    cls = (
        flashinfer.BatchDecodeWithPagedKVCacheWrapper
        if kind == "decode"
        else flashinfer.BatchPrefillWithPagedKVCacheWrapper
    )
    kw = dict(q_data_type=BF16, kv_data_type=BF16)
    head = () if kind == "decode" else (pr.qo_indptr,)
    if kind == "prefill":
        kw["causal"] = True
    w = cls(workspace(pr.device), "NHD", batch_invariant=True)
    w.plan(
        *head,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        **kw,
    )
    want = w.run(pr.q, pr.cache("NHD"))
    table = _table(pr.kv_indptr.cpu(), pr.kv_indices.cpu(), 200)
    seq_lens = torch.tensor(pr.kv_lens, dtype=torch.int32)
    for meta in ("cpu", "cuda"):
        csr = (pr.kv_indptr.to(meta), pr.kv_indices.to(meta), pr.kv_last.to(meta))
        extra = dict(seq_lens=seq_lens.to(where), block_tables=table.to(where))
        if kind == "prefill":
            extra["seq_lens_q"] = torch.ones(3, dtype=torch.int32, device=where)
        w.plan(*head, *csr, pr.Hq, pr.Hkv, pr.D, pr.page, **kw, **extra)
        assert torch.equal(bits(w.run(pr.q, pr.cache("NHD"))), bits(want)), meta
        wrong = table.clone()
        wrong[2, 150] += 1
        short = seq_lens.clone()
        short[1] -= 1
        for bad, named in (
            (
                dict(block_tables=wrong.to(where)),
                "differ from kv_indices at 1 page entries",
            ),
            (
                dict(seq_lens=short.to(where)),
                "seq_lens[1] = 36 where the page table gives 37",
            ),
        ):
            with pytest.raises(Unsupported) as e:
                w.plan(*head, *csr, pr.Hq, pr.Hkv, pr.D, pr.page, **kw, **bad)
            assert named in str(e.value)
            assert torch.equal(bits(w.run(pr.q, pr.cache("NHD"))), bits(want))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_disable_split_kv_plans_fused_units_with_the_same_bits(D):
    """Long KV at a small batch, where the default plan splits: ``disable_split_kv=True`` keeps every unit fused (no partials
    in the workspace) and the bits are the same, on the three wrappers."""
    pr = Problem([1, 300], [9000, 5000], 32, 8, D, 16, seed=51)
    kw = dict(q_data_type=BF16, kv_data_type=BF16)
    outs = {}
    for disable in (False, True):
        ws = workspace(pr.device)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, pr.Hq, pr.Hkv, D, 16, causal=True, disable_split_kv=disable, **kw)  # fmt: skip
        assert (w._batch_invariant._config.split_policy == "fused") is disable
        paged = w.run(pr.q, pr.cache("NHD"))
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr_rag,
            pr.Hq,
            pr.Hkv,
            D,
            causal=True,
            disable_split_kv=disable,
            **kw,
        )
        ragged = w.run(pr.q, *pr.ragged("NHD"))
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            pr.Hq,
            pr.Hkv,
            D,
            16,
            disable_split_kv=disable,
            **kw,
        )
        last = (pr.qo_indptr[1:] - 1).long()
        decode = w.run(pr.q[last].contiguous(), pr.cache("NHD"))
        outs[disable] = (paged, ragged, decode)
        assert torch.equal(bits(paged), bits(ragged))
        assert torch.equal(bits(paged[last]), bits(decode))
    for a, b in zip(outs[False], outs[True], strict=True):
        assert torch.equal(bits(a), bits(b))


@needs_device
def test_ragged_and_forward_accept_unit_scales():
    pr = Problem([40, 300], [40, 300], 32, 8, 128, 16, seed=61)
    kw = dict(q_data_type=BF16, kv_data_type=BF16, causal=True)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(pr.qo_indptr, pr.kv_indptr_rag, pr.Hq, pr.Hkv, pr.D, **kw)
    k, v = pr.ragged("NHD")
    want = w.run(pr.q, k, v)
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr_rag,
        pr.Hq,
        pr.Hkv,
        pr.D,
        seq_lens=torch.tensor(pr.kv_lens),
        seq_lens_q=torch.tensor(pr.q_lens),
        fixed_split_size=4096,
        **kw,
    )
    got = w.run(
        pr.q, k, v, q_scale=1.0, k_scale=1.0, v_scale=1.0, o_scale=1.0, kv_cache_sf=None
    )
    assert torch.equal(bits(got), bits(want))
    with pytest.raises(Unsupported, match="o_scale=0.5"):
        w.run(pr.q, k, v, o_scale=0.5)
    got = w.run(pr.q, k, v, k_scale=torch.ones(1, device=pr.device))
    assert torch.equal(bits(got), bits(want))
    with pytest.raises(Unsupported, match="k_scale=0.5"):
        w.run(pr.q, k, v, k_scale=torch.full((1,), 0.5, device=pr.device))
    paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    paged.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        **kw,
    )
    got = paged.forward(pr.q, pr.cache("NHD"), causal=True, k_scale=1.0, v_scale=1.0)
    assert torch.equal(bits(got), bits(want))
    with pytest.raises(Unsupported, match="v_scale=0.5"):
        paged.forward(pr.q, pr.cache("NHD"), causal=True, k_scale=1.0, v_scale=0.5)
