"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Host tests of the candidate-support entry points (flashinfer-ai/flashinfer#7636) of the Cake LM-head package:
the argument contract, the validation stage's error word, the canonical FP32 sequence of the row stages and the
reference backend against the masked FP64 oracle (gate: 1.05x the masked PyTorch baseline's error plus a floor).
The package is reached through ``tests.test_helpers.cake_package`` only.
"""

import math
import random
import re
import subprocess
import sys
import textwrap
from fractions import Fraction

import pytest
import torch

from tests.test_helpers.cake_lm_head_loss_reference import Inputs as LossInputs
from tests.test_helpers.cake_lm_head_loss_reference import reference_fp64
from tests.test_helpers.cake_lm_head_support_reference import (
    IGNORE_INDEX,
    gate_failures,
    label_positions,
    make_support,
    make_support_inputs,
    reference_support_b0,
    reference_support_fp64,
    support_error_report,
    topp_support,
)
from tests.test_helpers.cake_package import cake_package

cake_support = cake_package("cake_lm_head_loss", "cake_support")
cake_backend = cake_package("cake_lm_head_loss", "cake_backend")
support_logprob = cake_support.chunked_lm_head_support_logprob
support_kl = cake_support.chunked_lm_head_support_kl
Engine = cake_support.SupportReferenceEngine
SupportRows = cake_support.SupportRows
NO_ERROR = cake_support.NO_ERROR
word_of = cake_support.support_error_word

DEV = torch.device("cpu")
SEED = 20261028
REF = "reference"
H_HOST, V_HOST = 256, 512


def run_support(inp, outputs, *, upstream=None, need_dx=True, need_dw=True, **kw):
    """Call the entry point that forms ``outputs`` on the reference backend and back-propagate the upstreams
    (default ``inp.g`` / ``inp.d`` / ``inp.e``).  Returns the outputs and ``dX`` / ``dW``."""
    X = inp.X.detach().requires_grad_(need_dx)
    W = inp.W.detach().requires_grad_(need_dw)
    kwargs = dict(inp.kwargs(), backend=REF)
    kwargs.update(kw)
    res = dict(packed=None, selected=None, kl=None)
    if "kl" in outputs:
        out = support_kl(
            X,
            W,
            inp.support_ids,
            inp.support_offsets,
            inp.teacher,
            return_logprobs="packed" in outputs,
            **kwargs,
        )
        if "packed" in outputs:
            res["kl"], res["packed"] = out
        else:
            res["kl"] = out
    elif "selected" in outputs:
        res["selected"] = support_logprob(
            X, W, inp.support_ids, inp.support_offsets, inp.labels, **kwargs
        )
    else:
        res["packed"] = support_logprob(
            X, W, inp.support_ids, inp.support_offsets, **kwargs
        )
    up = dict(packed=inp.g, selected=inp.d, kl=inp.e) if upstream is None else upstream
    tensors = [res[n] for n in outputs if up.get(n) is not None]
    grads = [up[n] for n in outputs if up.get(n) is not None]
    if tensors and (need_dx or need_dw):
        torch.autograd.backward(tensors, grads)
    res["dX"] = X.grad if need_dx else None
    res["dW"] = W.grad if need_dw else None
    return res


def check_gates(inp, outputs, *, upstream=None, **kw):
    res = run_support(inp, outputs, upstream=upstream, **kw)
    oracle = reference_support_fp64(inp, outputs=outputs, upstream=upstream)
    b0 = reference_support_b0(inp, outputs=outputs, upstream=upstream)
    err = support_error_report(res, oracle, inp)
    b0err = support_error_report(b0, oracle, inp)
    assert not err["nan"], err
    # every requested output and both gradients are gated (no metric silently missing on either side)
    gated = {f"{n}_max_abs" for n in outputs} | {"dX_rel_l2", "dW_rel_l2"}
    assert gated <= err.keys() and gated <= b0err.keys(), (err, b0err)
    bad = gate_failures(err, b0err)
    assert not bad, f"{bad}\nerrors {err}\nB0 {b0err}"
    return res, err, b0err


def assert_same(a, b):
    for name in ("packed", "selected", "kl", "dX", "dW"):
        x, y = a.get(name), b.get(name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert torch.equal(
                x.view(torch.int16)
                if x.dtype == torch.bfloat16
                else x.view(torch.int32),
                y.view(torch.int16)
                if y.dtype == torch.bfloat16
                else y.view(torch.int32),
            ), name


def small(T=37, **kw):
    kw.setdefault("seed", SEED)
    kw.setdefault("device", DEV)
    return make_support_inputs(T, **kw)


# ---------------------------------------------------------------------------
# Host argument rules
# ---------------------------------------------------------------------------


def _base(T=4, M=6):
    X = torch.zeros(T, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    ids = torch.arange(M, dtype=torch.int64)
    off = torch.tensor([0, 2, 3, 6, 6][: T + 1], dtype=torch.int64)
    return X, W, ids, off


def _bad_cases():
    X, W, ids, off = _base()
    lab = torch.full((4,), IGNORE_INDEX, dtype=torch.int64)
    teacher = torch.zeros(6)
    meta = torch.device("meta")
    return {
        "X fp32": (dict(X=X.float()), "X must be a BF16"),
        "X 1-D": (dict(X=X[0]), "X must be a BF16"),
        "W fp32": (dict(W=W.float()), "W must be a BF16"),
        "W strided": (dict(W=W.t().contiguous().t()), "W must be contiguous"),
        "H mismatch": (
            dict(W=torch.zeros(V_HOST, 2 * H_HOST, dtype=torch.bfloat16)),
            "hidden sizes differ",
        ),
        "H 128": (
            dict(
                X=torch.zeros(4, 128, dtype=torch.bfloat16),
                W=torch.zeros(V_HOST, 128, dtype=torch.bfloat16),
            ),
            "H must be a positive multiple of 256",
        ),
        "V 384": (
            dict(W=torch.zeros(384, H_HOST, dtype=torch.bfloat16)),
            "V must be a positive multiple of 256",
        ),
        "V over max": (
            dict(
                X=torch.zeros(4, H_HOST, dtype=torch.bfloat16, device=meta),
                W=torch.zeros(
                    cake_support.SUPPORT_V_MAX + 256,
                    H_HOST,
                    dtype=torch.bfloat16,
                    device=meta,
                ),
                support_ids=ids.to(meta),
                support_offsets=off.to(meta),
            ),
            "V must not exceed",
        ),
        "X col stride": (
            dict(X=torch.zeros(H_HOST, 4, dtype=torch.bfloat16).t()),
            "stride\\(1\\) == 1",
        ),
        "ids 2-D": (
            dict(support_ids=ids.view(2, 3)),
            "support_ids must be a 1-D tensor",
        ),
        "ids fp32": (
            dict(support_ids=ids.float()),
            "support_ids must be int32 or int64",
        ),
        "offsets length": (
            dict(support_offsets=off[:4]),
            "support_offsets must have 5 entries",
        ),
        "offsets int16": (
            dict(support_offsets=off.to(torch.int16)),
            "support_offsets must be int32 or int64",
        ),
        "M over 2^31": (
            dict(
                X=torch.zeros(4, H_HOST, dtype=torch.bfloat16, device=meta),
                W=torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16, device=meta),
                support_ids=torch.zeros(2**31, dtype=torch.int32, device=meta),
                support_offsets=off.to(meta),
                k_cap=2**31,
            ),
            "fewer than 2\\^31",
        ),
        "k_cap 0": (dict(k_cap=0), "k_cap must be a positive integer"),
        "k_cap bool": (dict(k_cap=True), "k_cap must be a positive integer"),
        "k_cap float": (dict(k_cap=2.0), "k_cap must be a positive integer"),
        "M > T k_cap": (dict(k_cap=1), "more than T \\* k_cap"),
        "T 0 with pairs": (
            dict(
                X=torch.zeros(0, H_HOST, dtype=torch.bfloat16),
                support_offsets=torch.zeros(1, dtype=torch.int64),
            ),
            "must be empty when X has no rows",
        ),
        "labels int32": (dict(labels=lab.int()), "labels must be int64"),
        "labels length": (dict(labels=lab[:3]), "labels must be a \\[T\\] tensor"),
        "chunk 0": (dict(chunk_size=0), "chunk_size must be a positive integer"),
        "chunk bool": (dict(chunk_size=True), "chunk_size must be a positive integer"),
        "backend": (dict(backend="triton"), "backend must be one of"),
        "check_inputs": (dict(check_inputs=1), "check_inputs must be a bool"),
        "W device": (dict(W=W.to(meta)), "W must be on the device of X"),
        "ids device": (
            dict(support_ids=ids.to(meta)),
            "support_ids must be on the device of X",
        ),
        "teacher fp64": (
            dict(teacher=teacher.double()),
            "teacher_logprobs must be FP32",
        ),
        "teacher length": (dict(teacher=teacher[:5]), "aligned with support_ids"),
        "teacher grad": (
            dict(teacher=teacher.clone().requires_grad_()),
            "must not require grad",
        ),
    }


@pytest.mark.parametrize("case", sorted(_bad_cases()))
def test_host_rules_raise(case):
    X, W, ids, off = _base()
    args = dict(X=X, W=W, support_ids=ids, support_offsets=off, k_cap=3, backend=REF)
    change, match = _bad_cases()[case]
    args.update(change)
    teacher = args.pop("teacher", None)
    labels = args.pop("labels", None)
    with pytest.raises(ValueError, match=match):
        if teacher is not None:
            support_kl(
                args.pop("X"),
                args.pop("W"),
                args.pop("support_ids"),
                args.pop("support_offsets"),
                teacher,
                **args,
            )
        else:
            support_logprob(
                args.pop("X"),
                args.pop("W"),
                args.pop("support_ids"),
                args.pop("support_offsets"),
                labels,
                **args,
            )


@pytest.mark.parametrize(
    "temperature, match",
    [
        (True, "positive number or a float32 \\[T\\] tensor, got bool"),
        (0.0, "positive finite"),
        (-1.0, "positive finite"),
        (math.nan, "positive finite"),
        (math.inf, "positive finite"),
        (10**400, "an int too large for a float"),
        (1e-50, "rounds to 0.0"),
        (1e39, "rounds to inf"),
        ("1.0", "positive number or a float32 \\[T\\] tensor, got str"),
        (torch.tensor(0.5, requires_grad=True), "not differentiable"),
        (torch.ones(4, requires_grad=True), "not differentiable"),
        (torch.ones(4, 1), "got a tensor of shape \\(4, 1\\)"),
        (torch.ones(4, dtype=torch.float64), "must be a float32 tensor"),
        (torch.ones(5), "must have shape \\(4,\\)"),
        (torch.tensor(0.5, device="meta"), "Python number or a CPU scalar"),
        (torch.ones(4, device="meta"), "must be on X's device"),
        (torch.tensor([1.0, 0.0, 1.0, 1.0]), "positive and finite on every row"),
        (torch.tensor([1.0, 1.0, math.nan, 1.0]), "positive and finite on every row"),
        (torch.tensor([1.0, 1.0, 1.0, math.inf]), "positive and finite on every row"),
        (torch.tensor([1.0, -2.0, 1.0, 1.0]), "positive and finite on every row"),
    ],
)
@pytest.mark.parametrize("check_inputs", [False, True])
def test_temperature_rules(temperature, match, check_inputs):
    # row 3 is empty and every label is -100: the strict rule still checks those rows
    X, W, ids, off = _base()
    lab = torch.full((4,), IGNORE_INDEX, dtype=torch.int64)
    with pytest.raises(ValueError, match=match):
        support_logprob(
            X,
            W,
            ids,
            off,
            lab,
            k_cap=3,
            temperature=temperature,
            check_inputs=check_inputs,
            backend=REF,
        )


def test_temperature_forms_resolve():
    # the entry points use the package's shared resolver (no private copy)
    assert cake_support._resolve_temperature is cake_backend._resolve_temperature
    resolve = cake_support._resolve_temperature
    assert resolve(None, 4, DEV) is None
    assert resolve(1.0, 4, DEV) is None
    assert resolve(1, 4, DEV) is None
    assert resolve(torch.tensor(1.0), 4, DEV) is None
    assert resolve(torch.tensor(1.0, dtype=torch.float64), 4, DEV) is None
    assert resolve(0.7, 4, DEV) == (
        "scalar",
        float(torch.tensor(0.7, dtype=torch.float32)),
    )
    kind, tau = resolve(torch.ones(8)[::2], 4, DEV)
    assert kind == "tensor" and tau.is_contiguous() and torch.equal(tau, torch.ones(4))
    assert resolve(torch.ones(0), 0, DEV)[0] == "tensor"


def test_cake_backend_not_built():
    X, W, ids, off = _base()
    with pytest.raises(NotImplementedError, match="backend='reference'"):
        support_logprob(X, W, ids, off, k_cap=3, backend="cake")


# ---------------------------------------------------------------------------
# Validation stage (K1): error word, label positions
# ---------------------------------------------------------------------------

E = cake_support


def _check(ids, off, labels=None, *, k_cap=3, V=V_HOST, dtype=torch.int64):
    ids = torch.tensor(ids, dtype=dtype)
    off = torch.tensor(off, dtype=dtype)
    lab = None if labels is None else torch.tensor(labels, dtype=torch.int64)
    chk = Engine.support_check(
        ids, off, lab, num_rows=int(off.numel()) - 1, vocab=V, k_cap=k_cap
    )
    return int(chk.word.item()), chk.lpos


# (ids, offsets, labels, k_cap, expected (code, row) or None)
K1_CASES = {
    "valid": ([5, 9, 7, 5, 1, 0], [0, 2, 3, 6, 6], [9, -100, 0, -100], 3, None),
    "valid ids repeat across rows": ([5, 9, 5, 9, 5], [0, 2, 3, 5, 5], None, 3, None),
    "offsets first": ([5, 9, 7, 5], [1, 2, 3, 4, 4], None, 3, (E.ERR_OFFSETS_FIRST, 0)),
    "offsets decreasing": (
        [5, 9, 7, 5],
        [0, 3, 1, 4, 4],
        None,
        3,
        (E.ERR_OFFSETS_DECREASING, 1),
    ),
    "offsets last": ([5, 9, 7, 5], [0, 1, 2, 3, 3], None, 3, (E.ERR_OFFSETS_LAST, 4)),
    "over cap": ([5, 9, 7, 5, 1], [0, 1, 5, 5, 5], None, 3, (E.ERR_OVER_CAP, 1)),
    "id = V": ([5, 9, 512, 5], [0, 1, 2, 4, 4], None, 3, (E.ERR_ID_RANGE, 2)),
    "id < 0": ([5, 9, 7, -1], [0, 1, 2, 3, 4], None, 3, (E.ERR_ID_RANGE, 3)),
    "duplicate": ([5, 9, 7, 7], [0, 1, 2, 2, 4], None, 3, (E.ERR_DUPLICATE_ID, 3)),
    "label on empty row": (
        [5, 9, 7],
        [0, 0, 1, 3, 3],
        [3, 5, -100, -100],
        3,
        (E.ERR_LABEL_EMPTY_ROW, 0),
    ),
    "label not in support": (
        [5, 9, 7],
        [0, 1, 2, 3, 3],
        [-100, 7, -100, -100],
        3,
        (E.ERR_LABEL_NOT_IN_SUPPORT, 1),
    ),
    "label -1": (
        [5, 9, 7],
        [0, 1, 2, 3, 3],
        [-100, -100, -1, -100],
        3,
        (E.ERR_LABEL_NOT_IN_SUPPORT, 2),
    ),
    "label >= V": (
        [5, 9, 7],
        [0, 1, 2, 3, 3],
        [600, -100, -100, -100],
        3,
        (E.ERR_LABEL_NOT_IN_SUPPORT, 0),
    ),
    # priorities: band 0 beats a content violation at a smaller row; then the smaller row; then the smaller code
    "band 0 first": (
        [5, 5, 7, 8],
        [0, 2, 3, 4, 3],
        None,
        3,
        (E.ERR_OFFSETS_DECREASING, 3),
    ),
    "smaller row first": (
        [5, 9, 600, 7, 7],
        [0, 1, 3, 5, 5],
        None,
        3,
        (E.ERR_ID_RANGE, 1),
    ),
    "cap before duplicate": (
        [5, 5, 5, 5],
        [0, 4, 4, 4, 4],
        None,
        3,
        (E.ERR_OVER_CAP, 0),
    ),
    "range before duplicate": (
        [600, 600, 1],
        [0, 0, 3, 3, 3],
        None,
        3,
        (E.ERR_ID_RANGE, 1),
    ),
    "duplicate before label": (
        [4, 4, 1],
        [0, 3, 3, 3, 3],
        [9, -100, -100, -100],
        3,
        (E.ERR_DUPLICATE_ID, 0),
    ),
}


@pytest.mark.parametrize("case", sorted(K1_CASES))
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_support_check_word(case, dtype):
    ids, off, labels, k_cap, expected = K1_CASES[case]
    word, lpos = _check(ids, off, labels, k_cap=k_cap, dtype=dtype)
    if expected is None:
        assert word == NO_ERROR
    else:
        assert word == word_of(*expected), (hex(word), hex(word_of(*expected)))
        assert cake_support.decode_support_error(word) == expected
    if labels is not None:
        assert lpos.dtype == torch.int32
    if case == "valid":
        assert lpos.tolist() == [1, -1, 5, -1]


@pytest.mark.parametrize(
    "case", sorted(c for c, v in K1_CASES.items() if v[4] is not None)
)
@pytest.mark.parametrize("check_inputs", [False, True])
def test_support_check_raises(case, check_inputs):
    ids, off, labels, k_cap, (code, row) = K1_CASES[case]
    T = len(off) - 1
    X = torch.zeros(T, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    lab = None if labels is None else torch.tensor(labels)
    rule = cake_support.SUPPORT_ERRORS[code][1]
    with pytest.raises(ValueError, match=re.escape(f"at row {row}: {rule}")):
        support_logprob(
            X,
            W,
            torch.tensor(ids),
            torch.tensor(off),
            lab,
            k_cap=k_cap,
            check_inputs=check_inputs,
            backend=REF,
        )


def test_error_word_layout():
    words = [
        word_of(code, row)
        for code in cake_support.SUPPORT_ERRORS
        for row in (0, 1, 7, 2**40)
    ]
    assert all(w < NO_ERROR for w in words)
    for code in cake_support.SUPPORT_ERRORS:
        for row in (0, 5, 2**40):
            assert cake_support.decode_support_error(word_of(code, row)) == (code, row)
    band0 = [
        word_of(c, 2**40) for c, (b, _) in cake_support.SUPPORT_ERRORS.items() if b == 0
    ]
    band1 = [
        word_of(c, 0) for c, (b, _) in cake_support.SUPPORT_ERRORS.items() if b == 1
    ]
    assert max(band0) < min(band1)


def test_support_check_random_matches_rules():
    # random small supports with injected violations: the word is the minimum of the expected violations
    rng = random.Random(SEED)
    for _ in range(300):
        T, V, k_cap = rng.randint(1, 6), 256, rng.randint(1, 4)
        rows = [
            [
                rng.randrange(-2, V + 2) if rng.random() < 0.05 else rng.randrange(V)
                for _ in range(rng.randint(0, k_cap + (rng.random() < 0.1)))
            ]
            for _ in range(T)
        ]
        if rng.random() < 0.3:
            r = rng.randrange(T)
            if rows[r]:
                rows[r].append(rows[r][0])
        ids = [v for r in rows for v in r]
        off = [0]
        for r in rows:
            off.append(off[-1] + len(r))
        labels = []
        for r in rows:
            u = rng.random()
            labels.append(
                -100
                if u < 0.3
                else (rng.choice(r) if r and u < 0.8 else rng.randrange(-1, V))
            )
        expected = []
        for t, r in enumerate(rows):
            if len(r) > k_cap:
                expected.append(word_of(E.ERR_OVER_CAP, t))
            if any(v < 0 or v >= V for v in r):
                expected.append(word_of(E.ERR_ID_RANGE, t))
            if len(set(r)) != len(r):
                expected.append(word_of(E.ERR_DUPLICATE_ID, t))
            if labels[t] != -100 and not r:
                expected.append(word_of(E.ERR_LABEL_EMPTY_ROW, t))
            if labels[t] != -100 and r and labels[t] not in r:
                expected.append(word_of(E.ERR_LABEL_NOT_IN_SUPPORT, t))
        word, lpos = _check(ids, off, labels, k_cap=k_cap, V=V)
        assert word == min(expected, default=NO_ERROR)
        if word == NO_ERROR:
            assert lpos.tolist() == [
                off[t] + rows[t].index(labels[t]) if labels[t] != -100 else -1
                for t in range(T)
            ]


# ---------------------------------------------------------------------------
# Canonical FP32 primitives
# ---------------------------------------------------------------------------


def _round_f32(x: Fraction) -> float:
    """``x`` rounded to the nearest FP32 value (ties to even), as a Python float."""
    if x == 0:
        return 0.0
    sign = -1.0 if x < 0 else 1.0
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    q = max(e, -126) - 23
    scaled = a / Fraction(2) ** q
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
        n += 1
    val = Fraction(n) * Fraction(2) ** q
    if val >= Fraction(2) ** 128:
        return sign * math.inf
    return sign * float(val)


def _random_f32(gen, n, lo, hi):
    mant = torch.rand(n, generator=gen, dtype=torch.float64) + 1.0
    exp = torch.randint(lo, hi, (n,), generator=gen).double()
    sign = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0).double()
    return (sign * mant * torch.pow(2.0, exp)).float()


def test_fma_rn_exact():
    gen = torch.Generator().manual_seed(SEED)
    a = torch.cat([_random_f32(gen, 1500, -20, 20), _random_f32(gen, 300, -75, -60)])
    b = torch.cat([_random_f32(gen, 1500, -20, 20), _random_f32(gen, 300, -75, -60)])
    c = _random_f32(gen, 1800, -30, 30)
    # half the cases cancel: c ~ -a * b
    cancel = torch.arange(1800) % 2 == 0
    c = torch.where(cancel, (-(a.double() * b.double()) * (1 + 2.0**-30)).float(), c)
    # the double-rounding trap: a * b = 1 + 2^-11 + 2^-24 (an FP32 midpoint), c = +-2^-80 moves it off the midpoint
    trap_a = torch.tensor([1 + 2.0**-12, -(1 + 2.0**-12)], dtype=torch.float32)
    trap_c = torch.tensor([2.0**-80, -(2.0**-80)], dtype=torch.float32)
    a, b, c = (
        torch.cat([a, trap_a]),
        torch.cat([b, trap_a.abs()]),
        torch.cat([c, trap_c]),
    )
    got = cake_support.fma_rn(a, b, c)
    for i in range(int(a.numel())):
        exact = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
        if exact == 0:
            continue
        assert float(got[i]) == _round_f32(exact), (
            i,
            float(a[i]),
            float(b[i]),
            float(c[i]),
        )
    naive = (a[-2:].double() * b[-2:].double() + c[-2:].double()).float()
    assert not torch.equal(
        naive, got[-2:]
    )  # the trap is live: one FP64 evaluation rounds twice


def test_div_rn_exact():
    gen = torch.Generator().manual_seed(SEED + 1)
    a = _random_f32(gen, 2000, -40, 40)
    b = _random_f32(gen, 2000, -5, 5).abs()
    got = cake_support.div_rn(a, b)
    for i in range(2000):
        assert float(got[i]) == _round_f32(
            Fraction(float(a[i])) / Fraction(float(b[i]))
        )


def _f32_add(x: float, y: float) -> float:
    return _round_f32(Fraction(x) + Fraction(y))


def _lane_sum_scalar(values):
    """The canonical lane sum of one row, in exact FP32 arithmetic."""
    acc = [0.0] * 32
    for i, v in enumerate(values):
        acc[i % 32] = _f32_add(acc[i % 32], float(v))
    for o in (16, 8, 4, 2, 1):
        acc = [_f32_add(acc[lane], acc[lane ^ o]) for lane in range(32)]
    return acc[0]


def test_lane_sum_order():
    # entries 0 and 32 share lane 0: 2^25 and -2^25 cancel there before the small terms of other lanes are added
    vals = [0.75] * 70
    vals[0], vals[32], vals[1], vals[33] = 2.0**25, -(2.0**25), 1.0, 3.0
    gen = torch.Generator().manual_seed(SEED)
    vals[40:70] = (torch.randn(30, generator=gen) * 1e3).tolist()
    v = torch.tensor(vals, dtype=torch.float32)
    expected = _lane_sum_scalar(v.tolist())
    seq = 0.0
    for x in v.tolist():
        seq = _f32_add(seq, x)
    assert expected != seq  # the order is observable
    ids = torch.arange(70)
    off = torch.tensor([0, 70])
    rows = SupportRows.build(ids, off, num_rows=1, vocab=V_HOST, k_cap=128)
    assert float(cake_support.lane_sum(rows, v)[0]) == expected


@pytest.mark.parametrize("budget", [None, 64])
def test_row_stages_follow_the_canonical_sequence(monkeypatch, budget):
    """The vectorized row stages equal a per-row scalar evaluation of CANONICAL_SEQUENCE bit for bit (the
    transcendental values come from the same PyTorch calls on the same vectors)."""
    if budget is not None:
        monkeypatch.setattr(
            cake_support, "_LANE_BUDGET", budget
        )  # many lane-fold blocks
    inp = small(
        T=9,
        k_cap=80,
        k_max=80,
        labels=True,
        teacher=True,
        tau_form="hetero",
        x_scale=3.0,
    )
    T, M = inp.T, inp.M
    rows = SupportRows.build(
        inp.support_ids, inp.support_offsets, num_rows=T, vocab=inp.V, k_cap=inp.k_cap
    )
    z = Engine.support_logits(inp.X, inp.W, rows, 4096)
    tau = inp.tau_rows()
    chk = Engine.support_check(
        inp.support_ids,
        inp.support_offsets,
        inp.labels,
        num_rows=T,
        vocab=inp.V,
        k_cap=inp.k_cap,
    )
    out = Engine.row_finalize(
        z, tau, rows, chk.lpos, inp.teacher, packed=True, selected=True, kl=True
    )
    off = inp.support_offsets.tolist()
    q = cake_support.div_rn(z.float(), tau[rows.row])
    m = torch.tensor(
        [max(q[off[t] : off[t + 1]].tolist(), default=-math.inf) for t in range(T)]
    )
    ex = torch.exp(q - m[rows.row])
    s = torch.tensor(
        [_lane_sum_scalar(ex[off[t] : off[t + 1]].tolist()) for t in range(T)],
        dtype=torch.float32,
    )
    K = torch.tensor([off[t + 1] - off[t] for t in range(T)])
    lse = torch.where(K > 0, m + torch.log(s), torch.zeros(T))
    assert torch.equal(out["lse"], lse)
    logp = torch.where((K >= 2)[rows.row], q - lse[rows.row], torch.zeros(M))
    assert torch.equal(out["packed"], logp)
    lq = inp.teacher
    Q = torch.exp(lq)
    diff = lq - logp
    kl = []
    for t in range(T):
        acc = [0.0] * 32
        for i, j in enumerate(range(off[t], off[t + 1])):
            if float(Q[j]) > 0:
                acc[i % 32] = _round_f32(
                    Fraction(float(Q[j])) * Fraction(float(diff[j]))
                    + Fraction(acc[i % 32])
                )
        for o in (16, 8, 4, 2, 1):
            acc = [_f32_add(acc[lane], acc[lane ^ o]) for lane in range(32)]
        kl.append(acc[0] if off[t + 1] - off[t] >= 2 else 0.0)
    assert torch.equal(out["kl"], torch.tensor(kl, dtype=torch.float32))
    lp = chk.lpos.long()
    sel = torch.where(lp >= 0, logp[lp.clamp_min(0)], torch.zeros(T))
    assert torch.equal(out["selected"], sel)
    # backward with all three upstreams
    dz = Engine.row_grad(
        z, tau, out["lse"], rows, chk.lpos, inp.teacher, inp.g, inp.d, inp.e
    )
    p = torch.exp(cake_support.div_rn(z.float(), tau[rows.row]) - out["lse"][rows.row])
    u = inp.g.clone()
    is_label = lp[rows.row] == torch.arange(M)
    u = torch.where(is_label, u + inp.d[rows.row], u)
    u = cake_support.fma_rn(-inp.e[rows.row], Q, u)
    U = torch.tensor(
        [_lane_sum_scalar(u[off[t] : off[t + 1]].tolist()) for t in range(T)],
        dtype=torch.float32,
    )
    ref = cake_support.div_rn(
        cake_support.fma_rn(-p, U[rows.row], u), tau[rows.row]
    ).to(torch.bfloat16)
    ref = torch.where((K >= 2)[rows.row], ref, torch.zeros_like(ref))
    assert torch.equal(dz.view(torch.int16), ref.view(torch.int16))


# ---------------------------------------------------------------------------
# Numerics against the masked FP64 oracle
# ---------------------------------------------------------------------------

NUMERIC_CASES = {
    # name: (outputs, make_support_inputs kwargs, entry kwargs)
    "packed ind": (("packed",), dict(T=257, k_cap=8), {}),
    "packed shr hetero": (
        ("packed",),
        dict(T=255, k_cap=128, kind="shr", tau_form="hetero"),
        {},
    ),
    "packed T1 scalar": (
        ("packed",),
        dict(T=1, k_cap=8, edge_rows=False, k_max=8, tau_form="scalar"),
        {},
    ),
    "packed int32": (
        ("packed",),
        dict(T=256, k_cap=32, ids_dtype=torch.int32, offsets_dtype=torch.int32),
        {},
    ),
    "packed k_cap > V": (
        ("packed",),
        dict(T=129, k_cap=4097, k_max=96, tau_form="hetero"),
        {},
    ),
    "packed full": (("packed",), dict(T=33, k_cap=512, kind="full", x_scale=2.0), {}),
    "selected ind": (("selected",), dict(T=257, k_cap=8, labels=True), {}),
    "selected heavy hetero": (
        ("selected",),
        dict(T=256, k_cap=128, kind="heavy", labels=True, tau_form="hetero"),
        {},
    ),
    "selected k_cap 1": (
        ("selected",),
        dict(T=255, k_cap=1, labels=True, tau_form="scalar"),
        {},
    ),
    "selected chunked": (
        ("selected",),
        dict(T=257, k_cap=8, labels=True, tau_form="hetero"),
        dict(chunk_size=64),
    ),
    "selected strided X": (("selected",), dict(T=130, k_cap=16, labels=True), {}),
    "kl ind": (("kl",), dict(T=257, k_cap=8, teacher=True), {}),
    "kl shr hetero": (
        ("kl",),
        dict(T=256, k_cap=128, kind="shr", teacher=True, tau_form="hetero"),
        {},
    ),
    "kl+packed heavy constant": (
        ("packed", "kl"),
        dict(T=257, k_cap=32, kind="heavy", teacher=True, tau_form="constant"),
        dict(chunk_size=100),
    ),
    "kl large logits small tau": (
        ("kl",),
        dict(
            T=200,
            k_cap=64,
            teacher=True,
            x_scale=2.0,
            tau_form="scalar",
            tau_scalar=0.25,
        ),
        {},
    ),
    "kl two chunks": (
        ("kl", "packed"),
        dict(T=4097, k_cap=8, teacher=True, tau_form="hetero"),
        {},
    ),
    "selected T4095 chunk 256": (
        ("selected",),
        dict(T=4095, k_cap=8, labels=True, tau_form="hetero"),
        dict(chunk_size=256),
    ),
    "kl+packed T16231": (
        ("packed", "kl"),
        dict(T=16231, k_cap=4, teacher=True, tau_form="hetero"),
        {},
    ),
    "packed k_cap 2048": (
        ("packed",),
        dict(T=48, V=2048, k_cap=2048, tau_form="hetero"),
        {},
    ),
}


@pytest.mark.parametrize("case", sorted(NUMERIC_CASES))
def test_numerics_against_oracle(case):
    outputs, mk, kw = NUMERIC_CASES[case]
    inp = small(**mk)
    if case == "selected strided X":
        padded = torch.zeros(inp.T, inp.H + 64, dtype=torch.bfloat16)
        padded[:, : inp.H] = inp.X
        inp.X = padded[:, : inp.H]
        assert inp.X.stride(0) == inp.H + 64
    check_gates(inp, outputs, **kw)


@pytest.mark.parametrize("scale", [0.0, -1.0, 1e4])
def test_upstream_values(scale):
    inp = small(T=129, k_cap=16, labels=True, teacher=True, tau_form="hetero")
    up = dict(packed=inp.g * scale, kl=inp.e * scale)
    if scale == 0.0:
        res = run_support(inp, ("packed", "kl"), upstream=up)
        assert not res["dX"].any() and not res["dW"].any()
    else:
        check_gates(inp, ("packed", "kl"), upstream=up)


def test_teacher_minus_inf_entries():
    inp = small(T=257, k_cap=32, teacher=True, teacher_inf_frac=0.3)
    assert bool(torch.isinf(inp.teacher).any())
    res, _, _ = check_gates(inp, ("kl",))
    assert bool(torch.isfinite(res["kl"]).all())


# ---------------------------------------------------------------------------
# Bitwise properties of the reference backend
# ---------------------------------------------------------------------------


def _with_tau(inp, temperature):
    inp2 = make_support_inputs(
        inp.T,
        seed=inp.seed,
        H=inp.H,
        V=inp.V,
        k_cap=inp.k_cap,
        device=DEV,
        labels=True,
        teacher=True,
    )
    inp2.temperature = temperature
    return inp2


def test_default_temperature_forms_bitwise():
    base = small(T=97, k_cap=16, labels=True, teacher=True)
    runs = []
    for temperature in (None, 1.0, 1, torch.tensor(1.0), torch.ones(base.T)):
        inp = _with_tau(base, temperature)
        runs.append(
            (run_support(inp, ("packed", "kl")), run_support(inp, ("selected",)))
        )
    for a, b in runs[1:]:
        assert_same(runs[0][0], a)
        assert_same(runs[0][1], b)


@pytest.mark.parametrize("s", [0.5, 0.7, 1.3])
def test_scalar_equals_constant_tensor_bitwise(s):
    base = small(T=97, k_cap=16, labels=True, teacher=True)
    a = _with_tau(base, s)
    b = _with_tau(base, torch.full((base.T,), s))
    assert_same(run_support(a, ("packed", "kl")), run_support(b, ("packed", "kl")))
    assert_same(run_support(a, ("selected",)), run_support(b, ("selected",)))
    check_gates(a, ("packed", "kl"))


def test_frozen_inputs_bitwise():
    inp = small(T=129, k_cap=16, labels=True, teacher=True, tau_form="hetero")
    both = run_support(inp, ("packed", "kl"))
    only_x = run_support(inp, ("packed", "kl"), need_dw=False)
    only_w = run_support(inp, ("packed", "kl"), need_dx=False)
    none = run_support(inp, ("packed", "kl"), need_dx=False, need_dw=False)
    assert torch.equal(both["dX"], only_x["dX"]) and only_x["dW"] is None
    assert torch.equal(both["dW"], only_w["dW"]) and only_w["dX"] is None
    for name in ("packed", "kl"):
        assert torch.equal(both[name], none[name])
        assert not none[name].requires_grad


def test_rerun_bitwise():
    inp = small(
        T=300, k_cap=32, kind="heavy", labels=True, teacher=True, tau_form="hetero"
    )
    assert_same(
        run_support(inp, ("packed", "kl"), chunk_size=64),
        run_support(inp, ("packed", "kl"), chunk_size=64),
    )
    assert_same(
        run_support(inp, ("selected",), chunk_size=64),
        run_support(inp, ("selected",), chunk_size=64),
    )


def test_retain_graph_repeat():
    inp = small(T=97, k_cap=16, teacher=True, tau_form="hetero")
    X = inp.X.detach().requires_grad_()
    W = inp.W.detach().requires_grad_()
    kl, logp = support_kl(
        X,
        W,
        inp.support_ids,
        inp.support_offsets,
        inp.teacher,
        return_logprobs=True,
        backend=REF,
        **inp.kwargs(),
    )
    loss = (kl * inp.e).sum() + (logp * inp.g).sum()
    g1 = torch.autograd.grad(loss, [X, W], retain_graph=True)
    g2 = torch.autograd.grad(loss, [X, W])
    assert torch.equal(g1[0], g2[0]) and torch.equal(g1[1], g2[1])


def test_unused_output_has_no_upstream():
    # only the KL feeds the loss: the packed output's upstream is absent (skipped), equal to a KL-only call
    inp = small(T=97, k_cap=16, teacher=True)
    a = run_support(inp, ("packed", "kl"), upstream=dict(packed=None, kl=inp.e))
    b = run_support(inp, ("kl",), upstream=dict(kl=inp.e))
    assert torch.equal(a["dX"], b["dX"]) and torch.equal(a["dW"], b["dW"])


def test_nan_upstream_on_inactive_entries():
    inp = small(T=129, k_cap=8, labels=True, teacher=True)
    lab = inp.labels != IGNORE_INDEX
    small_row = inp.counts <= 1
    pair_small = small_row[torch.repeat_interleave(torch.arange(inp.T), inp.counts)]
    assert bool((~lab).any()) and bool(small_row.any())
    d_nan = torch.where(lab, inp.d, torch.full_like(inp.d, math.nan))
    d_zero = torch.where(lab, inp.d, torch.zeros_like(inp.d))
    assert_same(
        run_support(inp, ("selected",), upstream=dict(selected=d_nan)),
        run_support(inp, ("selected",), upstream=dict(selected=d_zero)),
    )
    e_nan = torch.where(small_row, torch.full_like(inp.e, math.nan), inp.e)
    e_zero = torch.where(small_row, torch.zeros_like(inp.e), inp.e)
    g_nan = torch.where(pair_small, torch.full_like(inp.g, math.nan), inp.g)
    g_zero = torch.where(pair_small, torch.zeros_like(inp.g), inp.g)
    a = run_support(inp, ("packed", "kl"), upstream=dict(packed=g_nan, kl=e_nan))
    b = run_support(inp, ("packed", "kl"), upstream=dict(packed=g_zero, kl=e_zero))
    assert bool(torch.isfinite(a["dX"].float()).all()) and bool(
        torch.isfinite(a["dW"].float()).all()
    )
    assert_same(a, b)


def test_singleton_and_empty_rows_exact():
    inp = small(T=200, k_cap=4, labels=True, teacher=True, tau_form="hetero")
    small_row = inp.counts <= 1
    assert bool((inp.counts == 0).any()) and bool((inp.counts == 1).any())
    pair_rows = torch.repeat_interleave(torch.arange(inp.T), inp.counts)
    # an unnormalized teacher on singleton rows: the KL is still exactly 0 there
    inp.teacher = torch.where(
        small_row[pair_rows], torch.full_like(inp.teacher, 0.3), inp.teacher
    )
    res = run_support(inp, ("packed", "kl"))
    assert bool((res["packed"][small_row[pair_rows]] == 0).all())
    assert bool((res["kl"][small_row] == 0).all())
    assert bool((res["dX"][small_row] == 0).all())
    sel = run_support(inp, ("selected",))
    assert bool((sel["selected"][small_row] == 0).all())
    assert bool((sel["selected"][inp.labels == IGNORE_INDEX] == 0).all())
    assert bool((sel["dX"][small_row | (inp.labels == IGNORE_INDEX)] == 0).all())


def test_empty_calls():
    W = torch.randn(V_HOST, H_HOST).to(torch.bfloat16).requires_grad_()
    X0 = torch.zeros(0, H_HOST, dtype=torch.bfloat16, requires_grad=True)
    ids0 = torch.zeros(0, dtype=torch.int64)
    off0 = torch.zeros(1, dtype=torch.int64)
    kl, logp = support_kl(
        X0, W, ids0, off0, torch.zeros(0), k_cap=4, return_logprobs=True, backend=REF
    )
    assert kl.shape == (0,) and logp.shape == (0,)
    (kl.sum() + logp.sum()).backward()
    assert X0.grad.shape == (0, H_HOST) and not W.grad.any()
    # M == 0 with rows: every label must be -100
    X = torch.randn(3, H_HOST).to(torch.bfloat16).requires_grad_()
    off = torch.zeros(4, dtype=torch.int64)
    sel = support_logprob(
        X, W, ids0, off, torch.full((3,), IGNORE_INDEX), k_cap=4, backend=REF
    )
    assert torch.equal(sel, torch.zeros(3))
    sel.sum().backward()
    assert not X.grad.any()
    with pytest.raises(ValueError, match="at row 1: the row has an empty support"):
        support_logprob(
            X, W, ids0, off, torch.tensor([-100, 4, -100]), k_cap=4, backend=REF
        )


def test_no_grad_teacher_preparation():
    inp = small(T=97, k_cap=16, tau_form="hetero")
    with torch.no_grad():
        teacher = support_logprob(
            inp.X,
            inp.W,
            inp.support_ids,
            inp.support_offsets,
            backend=REF,
            **inp.kwargs(),
        )
    assert not teacher.requires_grad
    live = run_support(inp, ("packed",))
    assert torch.equal(teacher, live["packed"].detach())
    # a teacher prepared this way feeds the KL directly
    support_kl(
        inp.X,
        inp.W,
        inp.support_ids,
        inp.support_offsets,
        teacher,
        backend=REF,
        **inp.kwargs(),
    )


def test_saved_state_is_packed_sized():
    """What the forward saves for the backward is the inputs plus O(M) / O(T) packed state, never a
    vocabulary-sized activation."""
    inp = small(T=300, k_cap=16, labels=True, teacher=True, tau_form="hetero")
    X = inp.X.detach().requires_grad_()
    W = inp.W.detach().requires_grad_()
    shapes = []

    def pack(t):
        shapes.append((tuple(t.shape), t.data_ptr()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        support_kl(
            X,
            W,
            inp.support_ids,
            inp.support_offsets,
            inp.teacher,
            return_logprobs=True,
            backend=REF,
            **inp.kwargs(),
        )
    allowed = {(inp.M,), (inp.T,), (inp.T + 1,)}
    for shape, ptr in shapes:
        assert shape in allowed or ptr in (X.data_ptr(), W.data_ptr()), shape


# ---------------------------------------------------------------------------
# Full support = the full-vocabulary problem; replay of a recorded support
# ---------------------------------------------------------------------------


def test_full_support_oracle_matches_full_vocabulary_oracle():
    inp = small(T=65, k_cap=V_HOST, kind="full", labels=True, x_scale=2.0)
    ours = reference_support_fp64(inp, outputs=("selected",))
    loss_inp = LossInputs(
        X=inp.X,
        W=inp.W,
        labels=inp.labels,
        objective="ce",
        loss_div=1.0,
        infer_logp=None,
        loss_weights=None,
        dlogp=inp.d,
        regime=None,
        seed=inp.seed,
    )
    theirs = reference_fp64(loss_inp, entry="logprob")
    assert float((ours["selected"] - theirs["logp"]).abs().max()) < 1e-12
    assert float((ours["dX"] - theirs["dX"]).norm() / theirs["dX"].norm()) < 1e-12
    assert float((ours["dW"] - theirs["dW"]).norm() / theirs["dW"].norm()) < 1e-12
    check_gates(inp, ("selected",))


def test_replay_uses_the_recorded_support():
    T, V, H = 96, V_HOST, H_HOST
    base = small(T=T, k_cap=16, x_scale=4.0)
    ids, off = topp_support(base.X, base.W, p=0.9, k_cap=16)
    gen = torch.Generator().manual_seed(SEED + 5)
    W1 = (base.W.float() + 0.5 * torch.randn(V, H, generator=gen) / math.sqrt(H)).to(
        torch.bfloat16
    )
    ids1, off1 = topp_support(base.X, W1, p=0.9, k_cap=16)
    sets0 = [set(ids[off[t] : off[t + 1]].tolist()) for t in range(T)]
    sets1 = [set(ids1[off1[t] : off1[t + 1]].tolist()) for t in range(T)]
    changed = [t for t in range(T) if sets0[t] != sets1[t]]
    assert changed, "the update must change some top-p set"
    rec = make_support_inputs(
        T, seed=SEED, H=H, V=V, k_cap=16, device=DEV, W=W1, x_scale=4.0
    )
    rec.support_ids, rec.support_offsets = ids, off
    rec.g = torch.randn(int(ids.numel()), generator=gen) * 0.05
    res, _, _ = check_gates(rec, ("packed",))
    new = make_support_inputs(
        T, seed=SEED, H=H, V=V, k_cap=16, device=DEV, W=W1, x_scale=4.0
    )
    new.support_ids, new.support_offsets = ids1, off1
    lse_rec = reference_support_fp64(
        rec, outputs=("packed",), need_dx=False, need_dw=False
    )["lse"]
    lse_new = reference_support_fp64(
        new, outputs=("packed",), need_dx=False, need_dw=False
    )["lse"]
    assert (
        float((lse_rec - lse_new)[changed].abs().max()) > 1e-3
    )  # recomputing top-p would be detected


def test_label_positions_helper_rejects_foreign_labels():
    ids, off = make_support(4, 64, 3, seed=1)
    with pytest.raises(ValueError):
        label_positions(ids, off, torch.full((4,), 70))


# ---------------------------------------------------------------------------
# Device paths (CUDA only; skipped on a host)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("what", ["support", "temperature"])
def test_device_assert_ends_the_context(what):
    """``check_inputs=False`` on CUDA: no readback; the queued device assertion fails the stream."""
    script = textwrap.dedent(
        f"""
        import torch
        from tests.test_helpers.cake_package import cake_package
        cs = cake_package("cake_lm_head_loss", "cake_support")
        X = torch.zeros(4, 256, dtype=torch.bfloat16, device="cuda")
        W = torch.zeros(512, 256, dtype=torch.bfloat16, device="cuda")
        ids = torch.tensor([1, 2, 3, 4], device="cuda")
        off = torch.tensor([0, 4, 4, 4, 4], device="cuda") if "{what}" == "support" else torch.tensor(
            [0, 1, 2, 3, 4], device="cuda")
        tau = torch.tensor([1.0, 0.0, 1.0, 1.0], device="cuda") if "{what}" == "temperature" else 1.0
        out = cs.chunked_lm_head_support_logprob(X, W, ids, off, k_cap=2, temperature=tau, backend="reference")
        print("QUEUED", flush=True)
        torch.cuda.synchronize()
        print("NO_ASSERT", flush=True)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert "QUEUED" in proc.stdout and "NO_ASSERT" not in proc.stdout, (
        proc.stdout + proc.stderr
    )
    assert proc.returncode != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_device_reference_matches_oracle():
    inp = make_support_inputs(
        257,
        seed=SEED,
        k_cap=32,
        kind="heavy",
        labels=True,
        teacher=True,
        tau_form="hetero",
        device="cuda",
    )
    check_gates(inp, ("packed", "kl"))
    check_gates(inp, ("selected",))
