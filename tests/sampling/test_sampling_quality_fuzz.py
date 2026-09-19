"""Sampling quality / contract fuzzer for ``flashinfer.sampling`` (gh #3605).

Contract first: every covered API's legal inputs, returned shape/dtype, ``valid`` semantics,
threshold-equality and tie rules, filter order and RNG boundary are fixed from the source at the
pinned revision, encoded in ``tests/sampling/sampling_reference.py`` (an independent CPU oracle
that implements softmax and every filter itself) and asserted here against the real kernels.
The facts that are easy to get wrong, and that this suite pins:

* retention is value based for every filter, so ties at a boundary stay together (top-k keeps
  every token equal to the k-th largest value) -- ``torch.topk``'s tie order is not a contract;
* top-p excludes a boundary token whose strictly-larger tail mass equals ``top_p`` exactly,
  while min-p is inclusive at ``max(p) * min_p``;
* ``top_k_first`` and ``joint`` are different filters (different retained sets, different
  normalizers) and are checked against separate references;
* per-row filter parameters are keyed by the probability row they describe, also under
  ``indices``, and the parameter length is validated against ``probs.size(0)``;
* ``seed``/``offset`` are the RNG authority; the generator is a stream allocator whose offset
  advances by ``ceil(rows_per_draw * batch / 4) * 4`` per call;
* ``valid`` means "this row had a legal token", which is a per-API fact, not a uniform one.

Layers (each case declares its layer in ``tests/sampling/sampling_cases.py``): deterministic /
one-hot, structural (support sets, tie and boundary membership, filter values, fast path vs
slow path), RNG, and distribution (bounded-Bernoulli comparison of observed class frequencies
against the oracle's target over the declared N/K/alpha budget).

Run (one GPU; the CPU layer needs none):
  CUDA_HOME=<cuda> pytest -q tests/sampling/
Env: FLASHINFER_SAMPLING_FUZZ_ONLY_CASE=<case id>  replay exactly one case (printed on failure)
     FLASHINFER_SAMPLING_FUZZ_SHARD=<i>/<n>        run shard i of n; shards are independent
                                                   single-GPU tasks (see the CPU layer's plan)
     FLASHINFER_SAMPLING_FUZZ_SEED=<int>           base seed for the structural layer
     FLASHINFER_SAMPLING_FUZZ_TRIALS=<int>         override N (the half width follows it)

Determinism / repro: a case is a fixed declaration plus a seed derived from its id, and the
distribution layer records the ``(seed, offset)`` of every trial stream, so a failure reproduces
bit-for-bit from the case id alone.  Failures dump a JSON ledger entry (case, input rule,
thresholds, indices, reference support/target, observed output or frequencies, versions, code
path taken, repro command) beside a printed repro line.  Seeds are never re-rolled to make a
failure pass: a failure is shrunk to a fixed regression case, listed in ``REGRESSIONS`` and
replayable with ``FLASHINFER_SAMPLING_FUZZ_ONLY_CASE``.
"""

from __future__ import annotations

import json
import os
import zlib
from typing import Dict, List, Optional, Tuple

import pytest
import torch

import flashinfer
from tests.sampling import sampling_cases as cases
from tests.sampling import sampling_reference as ref
from tests.sampling import sampling_stats as stats
from tests.test_helpers.fuzz_ledger import Finding, FuzzLedger

N_TRIALS = int(os.environ.get("FLASHINFER_SAMPLING_FUZZ_TRIALS", str(stats.N_TRIALS)))
# APIs that add the per-row `valid` mask; the logits entry points have no such output.
VALID_APIS = (
    "sampling_from_probs",
    "top_k_sampling_from_probs",
    "top_p_sampling_from_probs",
    "min_p_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
)
LOGITS_APIS = ("sampling_from_logits", "top_k_top_p_sampling_from_logits")

# Frozen regressions: a shrunk failure is added here as (case id, note) and replayed by the
# normal case lists.  Empty because the round has no filed sampling failure yet.
REGRESSIONS: Tuple[Tuple[str, str], ...] = ()

_SKIP = None if torch.cuda.is_available() else "CUDA not available"
pytestmark = [
    pytest.mark.long_running,
    pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP)),
]

# Known-bug ledger (mechanism: tests/test_helpers/fuzz_ledger.py).  quarantine=False entries
# still RUN: a wrong answer is tolerated as an xfail, and an unexpected pass hard-fails, so an
# entry cannot silently outlive its bug.
LEDGER = FuzzLedger(
    "sampling-quality",
    findings=(
        Finding(
            match=lambda case: case.cid == "det/degenerate/min_p",
            reason=(
                "min_p_sampling_from_probs returns (vocab_size-1, valid=True) on a row with no "
                "positive probability: its `p >= max(p) * min_p` predicate accepts zeros, so the "
                "never-crossed scan falls back to the last valid index, while every other "
                "sampling API returns (0, valid=False). Probe: run the case; a pass means this "
                "entry is stale. gh #3605"
            ),
        ),
    ),
)

SUPPORT_CASES = cases.select(cases.SUPPORT_CASES, ids=list(cases.ALL_IDS))
FILTER_CASES = cases.select(cases.FILTER_CASES, ids=list(cases.ALL_IDS))
DIST_CASES = cases.select(cases.DIST_CASES, ids=list(cases.ALL_IDS))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _dev() -> torch.device:
    return torch.device("cuda:0")


def _case_seed(case) -> int:
    """Per-case seed, derived from the id: stable across processes and shards."""
    return (zlib.crc32(case.cid.encode()) + cases.BASE_SEED * 1_000_003) % (2**31 - 1)


def _value(case, dev: torch.device) -> torch.Tensor:
    """The tensor the case feeds its API: probabilities, or logits for the logits APIs."""
    base = (
        torch.tensor([case.logits], dtype=torch.float32, device=dev)
        if case.logits is not None
        else torch.tensor(case.probs, dtype=torch.float32, device=dev)
    )
    if case.api in LOGITS_APIS and case.logits is None:
        return torch.log(base)
    return base


def _param(value):
    if value is None or isinstance(value, (int, float)):
        return value
    return torch.tensor(value, dtype=torch.float64)


def _call(case, value, indices, seed: int, offset: int):
    """Call the API under test; returns ``(samples, valid_or_None)``."""
    api, spec = case.api, case.spec
    top_k, top_p, min_p = spec.top_k, spec.top_p, spec.min_p
    fn = getattr(flashinfer.sampling, api)
    want_valid = api in VALID_APIS
    common = dict(indices=indices, seed=seed, offset=offset)
    if api == "sampling_from_probs":
        out = fn(value, return_valid=want_valid, **common)
    elif api == "sampling_from_logits":
        out = fn(value, **common)
    elif api == "top_k_sampling_from_probs":
        out = fn(value, _param(top_k), return_valid=want_valid, **common)
    elif api == "top_p_sampling_from_probs":
        out = fn(value, _param(top_p), return_valid=want_valid, **common)
    elif api == "min_p_sampling_from_probs":
        out = fn(value, _param(min_p), return_valid=want_valid, **common)
    elif api == "top_k_top_p_sampling_from_probs":
        out = fn(
            value,
            _param(top_k),
            _param(top_p),
            filter_apply_order=spec.order,
            return_valid=want_valid,
            **common,
        )
    elif api == "top_k_top_p_sampling_from_logits":
        out = fn(value, _param(top_k), _param(top_p), filter_apply_order=spec.order, **common)
    else:
        raise ValueError(f"unhandled API {api}")
    return (out[0], out[1]) if want_valid else (out, None)


def _open_stream(gen: torch.Generator, api: str, rows: int, dev: torch.device) -> Tuple[int, int]:
    """Advance the generator by the API's documented reservation and return (seed, offset)."""
    return flashinfer.sampling.get_seed_and_offset(rows * cases.ROWS_PER_DRAW[api], gen, dev)


def _stream_offset(gen: torch.Generator) -> int:
    """The stream allocator's offset, read without consuming randomness (increment 0)."""
    return flashinfer.sampling.get_seed_and_offset(0, gen, _dev())[1]


def _expected(case) -> Tuple[torch.Tensor, torch.Tensor]:
    """(support, target) per output row of one call, fp64 on the CPU."""
    if case.logits is not None:
        probs = ref.reference_softmax(torch.tensor([case.logits], dtype=torch.float64))
    else:
        probs = torch.tensor(case.probs, dtype=torch.float64)
    indices = None if case.indices is None else torch.tensor(case.indices)
    return ref.target_distribution(probs, case.spec, indices=indices, keying=case.keying)


def _draw(case, dev: torch.device, gen: torch.Generator):
    """Run the case's calls; returns (samples, valid_or_None, stream log)."""
    value = _value(case, dev)
    indices = (
        None
        if case.indices is None
        else torch.tensor(case.indices, dtype=case.index_dtype, device=dev)
    )
    samples, valids, log = [], [], []
    for _ in range(case.calls):
        seed, offset = _open_stream(gen, case.api, case.batch, dev)
        log.append({"seed": seed, "offset": offset, "rows": case.batch})
        got, valid = _call(case, value, indices, seed, offset)
        samples.append(got)
        valids.append(valid)
    valid = None if valids[0] is None else torch.cat(valids)
    return torch.cat(samples), valid, log


def _versions() -> Dict[str, str]:
    return {
        "torch": torch.__version__,
        "flashinfer": str(getattr(flashinfer, "__version__", "unknown")),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "capability": str(torch.cuda.get_device_capability(0)),
    }


def _code_path(case, dev: torch.device) -> str:
    """Which internal path the case exercises (the fast path needs a scalar top_k on a large
    vocabulary), so a ledger entry names the code that actually ran."""
    if case.api not in ("top_k_top_p_sampling_from_probs", "top_k_top_p_sampling_from_logits"):
        return "direct-kernel"
    if case.spec.order == ref.ORDER_JOINT:
        return "joint-kernel"
    indices = (
        None
        if case.indices is None
        else torch.tensor(case.indices, dtype=case.index_dtype, device=dev)
    )
    fast = flashinfer.sampling._top_k_first_fast_path_applicable(
        _value(case, dev), case.spec.top_k, indices
    )
    return "top_k_first-fast-path" if fast else "top_k_first-mask-renorm"


def _ledger(case, tag: str, why: str, extra: Dict) -> None:
    """Dump a JSON ledger entry and fail with the same content in the message."""
    entry = {
        "case": case.cid,
        "api": case.api,
        "layer": case.kind,
        "filter": {
            "top_k": case.spec.top_k,
            "top_p": case.spec.top_p,
            "min_p": case.spec.min_p,
            "apply_order": case.spec.order,
            "keying": case.keying,
        },
        "input_rule": {
            "probs": getattr(case, "probs", None),
            "logits": case.logits,
            "indices": case.indices,
            "index_dtype": str(case.index_dtype),
            "rows_per_call": case.batch,
            "calls": case.calls,
        },
        "code_path": _code_path(case, _dev()),
        "versions": _versions(),
        "repro": cases.replay_command(case.cid),
        "diagnosis": why,
    }
    entry.update(extra)
    path = os.path.join(
        os.environ.get("FLASHINFER_SAMPLING_FUZZ_DUMP_DIR", "/tmp"),
        f"flashinfer_sampling_fuzz_{case.cid.replace('/', '_')}.json",
    )
    with open(path, "w") as handle:
        json.dump(entry, handle, indent=2, sort_keys=True, default=str)
    pytest.fail(f"{tag}: {why}\n{json.dumps(entry, indent=2, sort_keys=True, default=str)}")


def _ids(case_list) -> List[str]:
    return [c.cid for c in case_list]


# --------------------------------------------------------------------------------------
# deterministic layer: one-hot / unique candidate (A2-02)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", SUPPORT_CASES, ids=_ids(SUPPORT_CASES))
def test_one_hot_and_singular_support(case):
    """A row with a single legal token must return exactly that token, every time."""
    LEDGER.xfail_if_quarantined(case)
    dev = _dev()
    support, _ = _expected(case)
    singular = support.sum(dim=-1) == 1
    if not bool(singular.any()):
        pytest.skip("case has no singular-support row")
    gen = torch.Generator(device=dev).manual_seed(_case_seed(case))
    samples, valid, log = _draw(case, dev, gen)
    rows = torch.arange(samples.numel(), device=dev)
    only = singular.repeat(case.calls)[rows]
    want = support.argmax(dim=-1).repeat(case.calls)
    if not torch.equal(samples[only], want[only]):
        _ledger(
            case,
            "one-hot",
            "a singular-support row returned a token outside its only legal class",
            {"streams": log, "reference_support": support.tolist(), "observed": samples.tolist()},
        )
    if valid is not None and not bool(valid[only].all()):
        _ledger(case, "one-hot", "a legal singleton row reported valid=False", {"streams": log})
    print(f"{case.cid}: api={case.api} filter={case.spec.label()} note={case.note}")
    print(cases.replay_command(case.cid))


# --------------------------------------------------------------------------------------
# structural layer: support membership, ties, boundaries (A2-03/A2-04/A2-05/A2-06/A2-11)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", SUPPORT_CASES, ids=_ids(SUPPORT_CASES))
def test_support_set_and_zero_probability_tokens(case):
    """Every draw lands in the oracle's retained set; zero-probability tokens never appear."""
    LEDGER.xfail_if_quarantined(case)
    dev = _dev()
    support, target = _expected(case)
    gen = torch.Generator(device=dev).manual_seed(_case_seed(case))
    samples, valid, log = _draw(case, dev, gen)
    rows = torch.arange(samples.numel(), device=dev)
    allowed = support.repeat(case.calls, 1)
    reachable = target.repeat(case.calls, 1) > 0
    picked = allowed[rows, samples.long()]
    if not bool(picked.all()):
        _ledger(
            case,
            "support",
            "sampled token outside the oracle's retained set",
            {
                "streams": log,
                "reference_support": support.tolist(),
                "reference_target": target.tolist(),
                "observed": samples.tolist(),
                "offending_rows": rows[~picked].tolist()[:32],
            },
        )
    if not bool(reachable[rows, samples.long()].all()):
        _ledger(
            case,
            "zero-probability",
            "a token with zero target mass was sampled",
            {"streams": log, "reference_target": target.tolist(), "observed": samples.tolist()},
        )
    if valid is not None and not torch.equal(valid, support.any(dim=-1).repeat(case.calls)):
        _ledger(
            case,
            "valid",
            "valid mask disagrees with the oracle's legal-token set",
            {"streams": log, "reference_support": support.tolist(), "valid": valid.tolist()},
        )
    print(
        f"{case.cid}: api={case.api} filter={case.spec.label()} draws={samples.numel()} "
        f"support={[int(x) for x in support.sum(dim=-1)]} note={case.note}"
    )
    print(cases.replay_command(case.cid))


@pytest.mark.parametrize("case", FILTER_CASES, ids=_ids(FILTER_CASES))
def test_filter_values(case):
    """Exact support for ``top_k_renorm_probs`` / ``top_p_renorm_probs`` / ``top_k_mask_logits``.

    A filtered-out token must be exactly zero (renormalize) or exactly -inf (mask), the
    surviving values must match the oracle's renormalization, and the boundary and tie cases
    must follow the value-based rules rather than ``torch.topk``.
    """
    LEDGER.xfail_if_quarantined(case)
    dev = _dev()
    values = torch.tensor(case.values, dtype=torch.float32, device=dev)
    spec = (
        ref.FilterSpec(top_p=case.spec.top_p)
        if case.api == "top_p_renorm_probs"
        else ref.FilterSpec(top_k=case.spec.top_k)
    )
    support, target = ref.target_distribution(
        torch.tensor(case.values, dtype=torch.float64), spec
    )
    support, target = support.to(dev), target.to(dev)
    fn = getattr(flashinfer.sampling, case.api)
    if case.api == "top_k_mask_logits":
        got = fn(values, _param(case.spec.top_k))
        if not torch.equal(torch.isneginf(got), ~support) or not torch.equal(
            got[support], values[support]
        ):
            _ledger(
                case,
                "mask",
                "masked positions or surviving logits disagree with the oracle",
                {"reference_support": support.tolist(), "observed": got.tolist()},
            )
    else:
        got = fn(
            values,
            _param(case.spec.top_p if case.api == "top_p_renorm_probs" else case.spec.top_k),
        )
        if not torch.equal(got > 0, support):
            _ledger(
                case,
                "renorm-support",
                "renormalized support disagrees with the oracle",
                {"reference_support": support.tolist(), "observed": got.tolist()},
            )
        torch.testing.assert_close(got.double(), target, atol=1e-6, rtol=1e-6)
    print(f"{case.cid}: api={case.api} filter={case.spec.label()} note={case.note}")


def test_logits_and_probs_agree_in_joint_order():
    """A2-07/A2-08: with the same stream the logits and probs entry points return the same
    tokens in ``joint`` order -- both run the same kernel on the same values."""
    dev = _dev()
    logits = torch.log(torch.tensor([cases.SKEW8], dtype=torch.float32, device=dev))
    top_k, top_p = 4, 0.85
    stream = torch.Generator(device=dev).manual_seed(11)
    seed, offset = _open_stream(stream, "top_k_top_p_sampling_from_logits", 64, dev)
    from_logits = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, filter_apply_order="joint", seed=seed, offset=offset
    )
    from_probs = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        torch.softmax(logits, dim=-1),
        top_k,
        top_p,
        filter_apply_order="joint",
        seed=seed,
        offset=offset,
    )
    assert torch.equal(from_logits, from_probs)
    support, target = ref.target_distribution(
        torch.tensor([cases.SKEW8], dtype=torch.float64),
        ref.FilterSpec(top_k=top_k, top_p=top_p, order=ref.ORDER_JOINT),
    )
    assert bool(support[0, from_logits.long()].all())
    print(
        f"joint order: logits == probs for {from_logits.numel()} rows, "
        f"support={int(support.sum())}, target={[round(float(x), 4) for x in target[0] if x > 0]}"
    )


def test_top_k_first_fast_path_and_slow_path_differ():
    """A2-04: the large-vocabulary fast path and the mask/renorm slow path are different filters.

    A vocabulary at the fast-path threshold with a tie group straddling the k-th value: the slow
    path keeps the whole tie group (the value-based rule), while the fast path selects exactly
    ``k`` tokens with the radix top-k and therefore cannot.  Both must stay inside the oracle's
    support, and the row's exactly-zero tokens must stay unreachable.
    """
    dev = _dev()
    vocab, rows, calls, top_k, top_p = 65536, 256, 4, 8, 1.0
    row = torch.zeros(vocab, dtype=torch.float32)
    row[:4] = 0.125  # four tokens strictly above the boundary
    row[4:12] = 0.0625  # eight tokens tied exactly at the k-th value
    probs = row.unsqueeze(0).expand(rows, vocab).contiguous().to(dev)
    support, _ = ref.target_distribution(
        row.double().unsqueeze(0),
        ref.FilterSpec(top_k=top_k, top_p=top_p, order=ref.ORDER_TOP_K_FIRST),
    )
    support = support.to(dev)
    stream = torch.Generator(device=dev).manual_seed(7)
    fast, slow = [], []
    for _ in range(calls):
        seed, offset = _open_stream(stream, "top_k_top_p_sampling_from_probs", rows, dev)
        fast.append(
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k, top_p, seed=seed, offset=offset
            )
        )
    k_tensor = torch.full((rows,), top_k, dtype=torch.int32, device=dev)
    for _ in range(calls):
        seed, offset = _open_stream(stream, "top_k_top_p_sampling_from_probs", rows, dev)
        slow.append(
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, k_tensor, top_p, seed=seed, offset=offset
            )
        )
    fast, slow = torch.cat(fast), torch.cat(slow)
    assert flashinfer.sampling._top_k_first_fast_path_applicable(probs, top_k, None)
    assert not flashinfer.sampling._top_k_first_fast_path_applicable(probs, k_tensor, None)
    legal = int(support.sum())
    for name, got in (("fast", fast), ("slow", slow)):
        assert bool(support[0, got.long()].all()), f"{name} path left the value-based support"
        assert int(got.max()) < legal, f"{name} path sampled an exactly-zero token"
    distinct_fast = int(fast.unique().numel())
    distinct_slow = int(slow.unique().numel())
    assert distinct_fast <= top_k < distinct_slow, (
        f"fast path returned {distinct_fast} distinct tokens, slow path {distinct_slow}: the two "
        f"paths must not be the same filter (the value-based support holds {legal} tokens)"
    )
    print(
        f"fast path: {distinct_fast} distinct tokens (<= k={top_k}); slow path: {distinct_slow} "
        f"distinct tokens (value-based support {legal})"
    )


# --------------------------------------------------------------------------------------
# RNG layer: seed / offset / generator boundary (A2-09)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", SUPPORT_CASES, ids=_ids(SUPPORT_CASES))
def test_rng_stream_contract(case):
    """Same (seed, offset) reproduces bitwise; the generator only allocates streams.

    Two calls with an explicit ``(seed, offset)`` must agree exactly; restoring the generator
    state must reproduce the same ``(seed, offset)`` and the same samples; and the generator's
    offset must advance by ``ceil(rows_per_draw * batch / 4) * 4`` per call -- the documented
    reservation that makes consecutive calls independent rather than a replay of one draw.
    """
    LEDGER.xfail_if_quarantined(case)
    dev = _dev()
    value = _value(case, dev)
    indices = (
        None
        if case.indices is None
        else torch.tensor(case.indices, dtype=case.index_dtype, device=dev)
    )
    rows = case.batch
    step = (rows * cases.ROWS_PER_DRAW[case.api] + 3) // 4 * 4

    gen = torch.Generator(device=dev).manual_seed(_case_seed(case))
    before = _stream_offset(gen)
    seed, offset = _open_stream(gen, case.api, rows, dev)
    first, _ = _call(case, value, indices, seed, offset)
    after = _stream_offset(gen)
    if after - before != step:
        _ledger(
            case,
            "rng-step",
            f"generator offset moved by {after - before}, documented reservation is {step}",
            {
                "streams": [{"seed": seed, "offset": offset}],
                "rows_per_draw": cases.ROWS_PER_DRAW[case.api],
            },
        )
    again, _ = _call(case, value, indices, seed, offset)
    if not torch.equal(first, again):
        _ledger(case, "rng-replay", "the same (seed, offset) did not reproduce bitwise", {})
    gen.set_state(gen.get_state())
    seed2, offset2 = _open_stream(gen, case.api, rows, dev)
    if (seed2, offset2) != (seed, offset):
        _ledger(
            case,
            "rng-state",
            f"restored generator produced {(seed2, offset2)}, expected {(seed, offset)}",
            {},
        )
    third, _ = _call(case, value, indices, seed2, offset2)
    if not torch.equal(first, third):
        _ledger(case, "rng-state", "restored generator state did not reproduce the samples", {})
    support, _ = _expected(case)
    shifted = ""
    if int(support.sum(dim=-1).max()) > 1:
        other, _ = _call(case, value, indices, seed, offset + 4)
        shifted = f" shifted_offset_changed={int((first != other).sum())}/{first.numel()}"
    print(f"{case.cid}: step={step} rows_per_draw={cases.ROWS_PER_DRAW[case.api]}{shifted}")


# --------------------------------------------------------------------------------------
# indices layer: repeated / reordered rows (A2-08)
# --------------------------------------------------------------------------------------
def test_indices_map_outputs_to_the_declared_source_rows():
    """Repeated and reordered ``indices`` must sample the row they name, for every covered API."""
    dev = _dev()
    rows = ((0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    probs = torch.tensor(rows, dtype=torch.float32, device=dev)
    logits = torch.log(probs)
    pattern = [1, 0, 1, 0]
    want_tokens = [3, 1, 3, 1]
    per_api = (
        ("sampling_from_probs", ref.FilterSpec()),
        ("top_k_sampling_from_probs", ref.FilterSpec(top_k=1)),
        ("top_p_sampling_from_probs", ref.FilterSpec(top_p=0.5)),
        ("min_p_sampling_from_probs", ref.FilterSpec(min_p=1.0)),
        (
            "top_k_top_p_sampling_from_probs",
            ref.FilterSpec(top_k=1, top_p=0.5, order=ref.ORDER_JOINT),
        ),
        ("sampling_from_logits", ref.FilterSpec()),
        (
            "top_k_top_p_sampling_from_logits",
            ref.FilterSpec(top_k=1, top_p=0.5, order=ref.ORDER_JOINT),
        ),
    )
    for index_dtype in (torch.int32, torch.int64):
        indices = torch.tensor(pattern, dtype=index_dtype, device=dev)
        want = torch.tensor(want_tokens, dtype=index_dtype, device=dev)
        for api, spec in per_api:
            case = cases.SupportCase(
                f"det/indices/{api}",
                api,
                rows,
                spec,
                indices=tuple(pattern),
                index_dtype=index_dtype,
            )
            seed, offset = _open_stream(torch.Generator(device=dev).manual_seed(5), api, 4, dev)
            got, _ = _call(case, logits if api in LOGITS_APIS else probs, indices, seed, offset)
            if got.dtype != index_dtype or not torch.equal(got, want):
                _ledger(
                    case,
                    "indices",
                    f"outputs {got.tolist()} do not map to the named rows (want {want.tolist()})",
                    {"streams": [{"seed": seed, "offset": offset}]},
                )
    print("indices row mapping verified for 7 APIs x {int32, int64}")


# --------------------------------------------------------------------------------------
# contract layer: check_nan / return_valid / degenerate rows (A2-12)
# --------------------------------------------------------------------------------------
def test_check_nan_is_per_function():
    """``check_nan`` exists on the 7 sampling entry points and on none of the filters.

    With ``check_nan=True`` a NaN input raises ``ValueError``; the default (False) is a no-op,
    and the affected row is then merely illegal (``valid=False``, sample 0) rather than an error.
    """
    dev = _dev()
    good = (0.125,) * 8
    bad = (float("nan"),) * 8
    probs = torch.tensor([good, bad], dtype=torch.float32, device=dev)
    logits = torch.zeros((2, 8), dtype=torch.float32, device=dev)
    logits[1] = float("nan")
    nan_probs_case = cases.SupportCase("det/nan/probs", "sampling_from_probs", (good, bad))
    nan_logits_case = cases.SupportCase(
        "det/nan/logits", "sampling_from_logits", (good, bad), logits=bad
    )
    calls = (
        ("sampling_from_probs", probs, nan_probs_case, {}),
        ("top_k_sampling_from_probs", probs, nan_probs_case, {"top_k": 4}),
        ("top_p_sampling_from_probs", probs, nan_probs_case, {"top_p": 0.9}),
        ("min_p_sampling_from_probs", probs, nan_probs_case, {"min_p": 0.1}),
        (
            "top_k_top_p_sampling_from_probs",
            probs,
            nan_probs_case,
            {"top_k": 4, "top_p": 0.9, "filter_apply_order": "joint"},
        ),
        ("sampling_from_logits", logits, nan_logits_case, {}),
        (
            "top_k_top_p_sampling_from_logits",
            logits,
            nan_logits_case,
            {"top_k": 4, "top_p": 0.9, "filter_apply_order": "joint"},
        ),
    )
    for api, value, case, kwargs in calls:
        fn = getattr(flashinfer.sampling, api)
        with pytest.raises(ValueError, match="NaN"):
            fn(value, check_nan=True, **kwargs)
        out = fn(value, **kwargs)
        samples, valid = out if isinstance(out, tuple) else (out, None)
        if valid is not None and (bool(valid[1]) or int(samples[1]) != 0):
            _ledger(
                case,
                "check_nan",
                "with check_nan off a NaN row must give valid=False and sample 0",
                {"api": api, "valid": valid.tolist(), "samples": samples.tolist()},
            )
        if valid is None:
            assert 0 <= int(samples[1]) < 8
    filt = torch.full((2, 8), 0.125, dtype=torch.float32, device=dev)
    for api, args in (
        ("top_k_renorm_probs", (4,)),
        ("top_p_renorm_probs", (0.9,)),
        ("top_k_mask_logits", (4,)),
        ("softmax", ()),
    ):
        with pytest.raises(TypeError):
            getattr(flashinfer.sampling, api)(filt, *args, check_nan=True)
    print("check_nan verified on 7 entry points; the 4 filter APIs reject the flag")


def test_return_valid_and_output_dtype_contract():
    """``return_valid`` is on the 5 probs entry points only; the output follows ``indices``.

    Without ``indices`` the output is int32 of shape ``(batch_size,)``; with ``indices`` it takes
    the index dtype, and ``return_valid=True`` adds a bool mask of the same length.  The logits
    entry points have no such output, so the flag is a TypeError there.
    """
    dev = _dev()
    probs = torch.full((3, 8), 0.125, dtype=torch.float32, device=dev)
    logits = torch.zeros((3, 8), dtype=torch.float32, device=dev)
    for api, value, kwargs in (
        ("sampling_from_probs", probs, {}),
        ("top_k_sampling_from_probs", probs, {"top_k": 4}),
        ("top_p_sampling_from_probs", probs, {"top_p": 0.9}),
        ("min_p_sampling_from_probs", probs, {"min_p": 0.1}),
        (
            "top_k_top_p_sampling_from_probs",
            probs,
            {"top_k": 4, "top_p": 0.9, "filter_apply_order": "joint"},
        ),
    ):
        fn = getattr(flashinfer.sampling, api)
        samples, valid = fn(value, return_valid=True, **kwargs)
        assert samples.dtype == torch.int32 and samples.shape == (3,)
        assert valid.dtype == torch.bool and valid.shape == (3,) and bool(valid.all())
        typed = fn(value, indices=torch.arange(3, dtype=torch.int64, device=dev), **kwargs)
        assert typed.dtype == torch.int64 and typed.shape == (3,)
        plain = fn(value, **kwargs)
        assert plain.dtype == torch.int32 and plain.shape == (3,)
    for api, value, kwargs in (
        ("sampling_from_logits", logits, {}),
        (
            "top_k_top_p_sampling_from_logits",
            logits,
            {"top_k": 4, "top_p": 0.9, "filter_apply_order": "joint"},
        ),
    ):
        fn = getattr(flashinfer.sampling, api)
        assert fn(value, **kwargs).shape == (3,)
        with pytest.raises(TypeError):
            fn(value, return_valid=True, **kwargs)
    print("return_valid / output dtype contract verified")


def test_degenerate_row_contract():
    """A row with no legal token: per-API ``valid`` semantics.

    ``sampling_from_probs``, the top-k/top-p variants and the joint kernel write ``(0, False)``
    for an all-zero row.  ``min_p_sampling_from_probs`` is the exception tracked in the ledger:
    its predicate accepts zeros, so the never-crossed scan falls back to the last valid index and
    reports ``(vocab - 1, True)``.  An empty retained set on a normal row (``min_p`` above 1) is
    ``(0, False)`` for every API.
    """
    dev = _dev()
    zero_row = (0.0,) * 8
    normal_row = (0.125,) * 8
    checks = (
        ("det/degenerate/from_probs", "sampling_from_probs", ref.FilterSpec(), zero_row),
        ("det/degenerate/top_k", "top_k_sampling_from_probs", ref.FilterSpec(top_k=4), zero_row),
        ("det/degenerate/top_p", "top_p_sampling_from_probs", ref.FilterSpec(top_p=0.9), zero_row),
        ("det/degenerate/min_p", "min_p_sampling_from_probs", ref.FilterSpec(min_p=0.1), zero_row),
        (
            "det/degenerate/joint",
            "top_k_top_p_sampling_from_probs",
            ref.FilterSpec(top_k=4, top_p=0.9, order=ref.ORDER_JOINT),
            zero_row,
        ),
        (
            "det/degenerate/min_p_gt_one",
            "min_p_sampling_from_probs",
            ref.FilterSpec(min_p=2.0),
            normal_row,
        ),
    )
    for cid, api, spec, row in checks:
        case = cases.SupportCase(cid, api, (row,), spec)
        value = torch.tensor([row], dtype=torch.float32, device=dev)
        fn = getattr(flashinfer.sampling, api)
        kwargs = {"return_valid": True}
        if spec.top_k is not None:
            kwargs["top_k"] = spec.top_k
        if spec.top_p is not None:
            kwargs["top_p"] = spec.top_p
        if spec.min_p is not None:
            kwargs["min_p"] = spec.min_p
        if api == "top_k_top_p_sampling_from_probs":
            kwargs["filter_apply_order"] = spec.order
        samples, valid = fn(value, **kwargs)
        want = ref.degenerate_expectation(value[0].double().cpu(), api, spec)
        got = (bool(valid[0]), int(samples[0]))
        finding = LEDGER.find(case)
        if got != want:
            if finding is not None:
                LEDGER.report_expected_failures(
                    [(finding, cid)], context=f"{cid}: got {got}, contract expects {want}"
                )
            _ledger(
                case,
                "degenerate",
                f"degenerate row returned {got}, the contract expects {want}",
                {"samples": samples.tolist(), "valid": valid.tolist()},
            )
        elif finding is not None:
            LEDGER.flag_xpass(finding, cid)
    print("degenerate-row contract checked for 6 (API, filter) combinations")


# --------------------------------------------------------------------------------------
# distribution layer: bounded-Bernoulli comparison (A2-03, A2-10)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", DIST_CASES, ids=_ids(DIST_CASES))
def test_distribution_matches_the_oracle(case):
    """N independent trials per case, judged against the declared K and alpha.

    Each call draws ``TRIAL_ROWS`` rows, every row on its own Philox subsequence, and the
    generator hands out a fresh ``(seed, offset)`` per call -- so the N trials are N independent
    draws, recorded in the ledger, not one draw replayed.  A class is accepted when
    ``|p_hat - p| <= sqrt(log(2K/alpha) / (2N))`` with K and alpha declared up front; classes with
    exactly zero probability get a structural check instead and must never appear.
    """
    k, per_case = cases.round_comparisons()
    hw = stats.half_width(N_TRIALS, k, stats.ALPHA)
    dev = _dev()
    support, target = cases.dist_target(case)
    support, p = support[0], target[0]
    vocab = int(p.numel())
    base = torch.tensor([case.probs], dtype=torch.float32, device=dev)
    value = torch.log(base) if case.api in LOGITS_APIS else base
    gen = torch.Generator(device=dev).manual_seed(_case_seed(case))
    counts = torch.zeros(vocab, dtype=torch.int64)
    streams: List[Dict[str, int]] = []
    first_chunk: Optional[torch.Tensor] = None
    done = 0
    while done < N_TRIALS:
        rows = min(cases.TRIAL_ROWS, N_TRIALS - done)
        indices = torch.zeros(rows, dtype=torch.int32, device=dev)
        seed, offset = _open_stream(gen, case.api, rows, dev)
        streams.append({"seed": seed, "offset": offset, "rows": rows})
        samples, _ = _call(case, value, indices, seed, offset)
        assert samples.dtype == torch.int32 and samples.numel() == rows
        counts += torch.bincount(samples.long().cpu(), minlength=vocab)
        if first_chunk is None:
            first_chunk = samples.clone()
        done += rows
    p_hat = counts.double() / N_TRIALS
    effective = p > 0
    zero_hits = int(counts[~effective].sum())
    deviation = (p_hat - p).abs()
    bad = [int(c) for c in range(vocab) if bool(effective[c]) and float(deviation[c]) > hw]
    assert int(effective.sum()) == per_case[case.cid], "declared K disagrees with the oracle"
    print(
        f"{case.cid}: api={case.api} filter={case.spec.label()} N={N_TRIALS} K={k} "
        f"hw={hw:.6f} classes={int(effective.sum())} max|dp|="
        f"{float(deviation[effective].max()):.6f} zero_class_hits={zero_hits} streams={len(streams)}"
    )
    print(cases.replay_command(case.cid))
    if zero_hits:
        _ledger(
            case,
            "distribution",
            f"{zero_hits} draws landed on an exactly-zero-probability class",
            {
                "streams": streams,
                "reference_target": p.tolist(),
                "observed_frequencies": p_hat.tolist(),
            },
        )
    if bad:
        _ledger(
            case,
            "distribution",
            f"{len(bad)} classes outside the declared band (half width {hw:.6f})",
            {
                "K": k,
                "alpha": stats.ALPHA,
                "N": N_TRIALS,
                "half_width": hw,
                "streams": streams,
                "reference_support": support.tolist(),
                "reference_target": p.tolist(),
                "observed_frequencies": p_hat.tolist(),
                "offending_classes": bad,
            },
        )
    offsets = [s["offset"] for s in streams]
    if len(set(offsets)) != len(offsets):
        _ledger(
            case,
            "distribution",
            "a trial stream was re-used (replay padding)",
            {"streams": streams},
        )
    replay, _ = _call(
        case,
        value,
        torch.zeros(streams[0]["rows"], dtype=torch.int32, device=dev),
        streams[0]["seed"],
        streams[0]["offset"],
    )
    if not torch.equal(replay, first_chunk):
        _ledger(
            case,
            "distribution",
            "a recorded stream did not replay bitwise",
            {"streams": streams[:1]},
        )
