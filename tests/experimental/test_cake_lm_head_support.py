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

Candidate-support entries (https://github.com/flashinfer-ai/flashinfer/issues/7636) of the non-deterministic LM-head
package ``flashinfer.experimental.cake.cake_lm_head_loss``:
``chunked_lm_head_support_logprob`` / ``chunked_lm_head_support_kl`` take ``deterministic=False`` only, raise
PyTorch's deterministic-algorithms alert in their backward, and are held to the deterministic suite's gate
(``tests/test_helpers/cake_lm_head_support_reference.py``: 1.05x the masked PyTorch baseline's error against the
FP64 oracle plus a floor).  Their programs are the deterministic package's, so the outputs, the sparse dX and the
weight gradient equal the deterministic entries' bits on the same inputs; the dense dX adds its vocabulary blocks into
the chunk sum through the K-sliced reduction (``cake_support.support_dx_form``: one slice keeps the deterministic
bits, more slices -- only inside the rule's measured cells, or forced by a test -- add in the order they finish).  The
gradients carry no run-to-run promise in this package and are checked over repeated runs against the gate.  The
package is reached through ``tests.test_helpers.cake_package``.
"""

import contextlib
import dataclasses

import pytest
import torch

from tests.test_helpers.cake_lm_head_support_reference import (
    gate_failures,
    make_support_inputs,
    reference_support_b0,
    reference_support_fp64,
    support_error_report,
)
from tests.test_helpers.cake_package import cake_package

cake_support = cake_package("cake.cake_lm_head_loss", "cake_support")
cake_backend = cake_package("cake.cake_lm_head_loss", "cake_backend")
det_support = cake_package("cake_lm_head_det_loss", "cake_support")
support_logprob = cake_support.chunked_lm_head_support_logprob
support_kl = cake_support.chunked_lm_head_support_kl

DEV = torch.device("cpu")
CUDA = torch.device("cuda")
SEED = 20261031
REF = "reference"
ND = dict(deterministic=False)
REPEATS = 20


def run_support(
    inp, outputs, *, package="nd", upstream=None, need_dx=True, need_dw=True, **kw
):
    """Call the entry that forms ``outputs`` (``package``: ``"nd"`` this package, ``"det"`` the deterministic one;
    the reference backend unless ``backend`` is given) and back-propagate the upstreams (default ``inp.g`` /
    ``inp.d`` / ``inp.e``).  Returns the outputs and ``dX`` / ``dW``."""
    X = inp.X.detach().requires_grad_(need_dx)
    W = inp.W.detach().requires_grad_(need_dw)
    kwargs = dict(inp.kwargs(), backend=REF)
    kwargs.update(kw)
    if package == "nd":
        logprob_fn, kl_fn = support_logprob, support_kl
        kwargs.update(ND)
    else:
        logprob_fn = det_support.chunked_lm_head_det_support_logprob
        kl_fn = det_support.chunked_lm_head_det_support_kl
    res = dict(packed=None, selected=None, kl=None)
    if "kl" in outputs:
        out = kl_fn(
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
        res["selected"] = logprob_fn(
            X, W, inp.support_ids, inp.support_offsets, inp.labels, **kwargs
        )
    else:
        res["packed"] = logprob_fn(X, W, inp.support_ids, inp.support_offsets, **kwargs)
    up = dict(packed=inp.g, selected=inp.d, kl=inp.e) if upstream is None else upstream
    tensors = [res[n] for n in outputs if up.get(n) is not None]
    grads = [up[n] for n in outputs if up.get(n) is not None]
    if tensors and (need_dx or need_dw):
        torch.autograd.backward(tensors, grads)
    res["dX"] = X.grad if need_dx else None
    res["dW"] = W.grad if need_dw else None
    return res


def check_gates(inp, outputs, *, oracle=None, b0err=None, **kw):
    """One call of this package against the FP64 oracle with the deterministic suite's gate."""
    res = run_support(inp, outputs, **kw)
    if oracle is None:
        oracle = reference_support_fp64(inp, outputs=outputs)
    if b0err is None:
        b0err = support_error_report(
            reference_support_b0(inp, outputs=outputs), oracle, inp
        )
    err = support_error_report(res, oracle, inp)
    assert not err["nan"], err
    gated = {f"{n}_max_abs" for n in outputs} | {"dX_rel_l2", "dW_rel_l2"}
    assert gated <= err.keys() and gated <= b0err.keys(), (err, b0err)
    bad = gate_failures(err, b0err)
    assert not bad, f"{bad}\nerrors {err}\nB0 {b0err}"
    return res, err, b0err


def _bits(t):
    t = t.detach().reshape(-1).contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def assert_same(a, b):
    for name in ("packed", "selected", "kl", "dX", "dW"):
        x, y = a.get(name), b.get(name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert torch.equal(_bits(x), _bits(y)), name


def _bf16_ulp(t):
    """One BF16 unit in the last place of each element of ``t``: ``2^(e - 7)`` for ``2^e <= |t| < 2^(e + 1)`` in the
    normal range, the subnormal spacing ``2^-133`` below ``2^-126`` and at 0 (``frexp(0)`` has exponent 0, so 0 is
    set apart)."""
    t = t.float()
    e = (torch.frexp(t.abs())[1] - 1).clamp_min(-126)
    ulp = torch.ldexp(torch.ones_like(t), e - 7)
    return torch.where(t == 0, torch.full_like(t, 2.0**-133), ulp)


def _within_rounding_of_det(nd, det, zero_rows=None):
    """A K-sliced (unordered) dX against the deterministic (ordered) one, element by element:
    ``|nd - det| <= ulp(det) + 2^-14 rowmax |det|``.  Two FP32 sums less than one BF16 ulp apart round to BF16 values
    at most one ulp apart; the two FP32 sums differ by a few FP32 roundings of partial sums of about the size of the
    row's elements, far below 2^-14 of the row's largest one.  A lost or doubled slice moves an element by that
    slice's whole share.  A row with no nonzero product (an ignored label, a single candidate; ``zero_rows``, by
    default the rows of the deterministic dX that are all 0) is 0 here too, compared by value (``-0.0`` and ``+0.0``
    are equal): the ``-0.0`` fill plus zero products, so a value a missing fill leaves in such a row fails.  A single 0
    in another row keeps the bound (``ulp(0) = 2^-133``): a sum that cancels to exactly 0 in one order, or rounds to 0
    at the BF16 underflow, may end one rounding away in the other.  Returns the largest ``|nd - det| / bound``."""
    d, ref = nd.float(), det.float()
    if zero_rows is None:
        zero_rows = (ref == 0).all(dim=-1)
    stale = d[zero_rows] != 0
    assert not bool(stale.any()), (
        int(stale.sum()),
        float(d[zero_rows].abs().max()),
    )
    floor = ref.abs().amax(dim=-1, keepdim=True) * 2.0**-14
    excess = (d - ref).abs() / (_bf16_ulp(ref) + floor)
    assert bool((excess <= 1).all()), float(excess.max())
    return float(excess.max())


def small(T=37, **kw):
    kw.setdefault("seed", SEED)
    kw.setdefault("device", DEV)
    return make_support_inputs(T, **kw)


@contextlib.contextmanager
def _torch_deterministic(enabled: bool, warn_only: bool = False):
    was, was_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn)


# ---------------------------------------------------------------------------
# Host layer: the mode contract and the reference backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, None, 1, 0])
def test_entries_refuse_any_mode_but_false(value):
    inp = small(T=8, k_cap=4, labels=True, teacher=True)
    with pytest.raises(ValueError, match="deterministic=False"):
        support_logprob(
            inp.X,
            inp.W,
            inp.support_ids,
            inp.support_offsets,
            inp.labels,
            **inp.kwargs(),
            backend=REF,
            deterministic=value,
        )
    with pytest.raises(ValueError, match="cake_lm_head_det_loss"):
        support_kl(
            inp.X,
            inp.W,
            inp.support_ids,
            inp.support_offsets,
            inp.teacher,
            **inp.kwargs(),
            backend=REF,
            deterministic=value,
        )
    with pytest.raises(
        ValueError, match="deterministic=False"
    ):  # the default is refused too
        support_logprob(
            inp.X,
            inp.W,
            inp.support_ids,
            inp.support_offsets,
            **inp.kwargs(),
            backend=REF,
        )


def test_names_are_the_deterministic_names_without_det():
    assert (
        cake_support.SupportLogprobFunction is not det_support.SupportDetLogprobFunction
    )
    assert support_logprob is not det_support.chunked_lm_head_det_support_logprob
    assert set(cake_support.__all__) >= {
        "SupportLogprobFunction",
        "chunked_lm_head_support_logprob",
        "chunked_lm_head_support_kl",
    }
    assert not [
        name for name in cake_support.__all__ if "_det_" in name or "Det" in name
    ]


def test_backward_raises_the_alert_and_records_the_mode():
    """The support backward raises PyTorch's alert under torch's deterministic algorithms (``RuntimeError``; a
    ``UserWarning`` under ``warn_only``); a forward-only call (``torch.no_grad()``, teacher preparation) does not."""
    inp = small(T=33, k_cap=8, labels=True)
    X = inp.X.detach().requires_grad_(True)
    out = support_logprob(
        X,
        inp.W,
        inp.support_ids,
        inp.support_offsets,
        inp.labels,
        **inp.kwargs(),
        backend=REF,
        **ND,
    )
    assert out.grad_fn.deterministic is False
    with _torch_deterministic(True):
        with pytest.raises(
            RuntimeError, match="does not have a deterministic implementation"
        ):
            out.backward(inp.d, retain_graph=True)
        with torch.no_grad():
            support_logprob(
                inp.X,
                inp.W,
                inp.support_ids,
                inp.support_offsets,
                inp.labels,
                **inp.kwargs(),
                backend=REF,
                **ND,
            )
    with (
        _torch_deterministic(True, warn_only=True),
        pytest.warns(UserWarning, match="does not have a deterministic implementation"),
    ):
        out.backward(inp.d)
    assert X.grad is not None


SUPPORT_CASES = {
    # name: (outputs, make_support_inputs kwargs, entry kwargs)
    "packed ind": (("packed",), dict(T=257, k_cap=8), {}),
    "packed shr hetero": (
        ("packed",),
        dict(T=255, k_cap=128, kind="shr", tau_form="hetero"),
        {},
    ),
    "selected heavy hetero": (
        ("selected",),
        dict(T=256, k_cap=128, kind="heavy", labels=True, tau_form="hetero"),
        {},
    ),
    "selected chunked": (
        ("selected",),
        dict(T=257, k_cap=8, labels=True, tau_form="hetero"),
        dict(chunk_size=64),
    ),
    "kl+packed heavy constant": (
        ("packed", "kl"),
        dict(T=257, k_cap=32, kind="heavy", teacher=True, tau_form="constant"),
        dict(chunk_size=100),
    ),
    "kl two chunks": (
        ("kl", "packed"),
        dict(T=4097, k_cap=8, teacher=True, tau_form="hetero"),
        {},
    ),
}


@pytest.mark.parametrize("case", sorted(SUPPORT_CASES))
def test_reference_matches_the_oracle_and_the_deterministic_reference(case):
    outputs, mk, kw = SUPPORT_CASES[case]
    inp = small(**mk)
    res, _, _ = check_gates(inp, outputs, **kw)
    assert_same(res, run_support(inp, outputs, package="det", **kw))


# ---------------------------------------------------------------------------
# Device (the package's rendered support programs)
# ---------------------------------------------------------------------------

CAKE_FORMS = {
    "auto": {},
    "sparse": dict(path="sparse"),
    "dense": dict(path="dense"),
    "dense vb256": dict(path="dense", vocab_block=256),
}


def _require_cake():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    from pathlib import Path

    if not (Path(cake_support.__file__).resolve().parent / "build.json").is_file():
        pytest.skip(
            "the package's programs are not rendered in this checkout (no build.json)"
        )
    if not cake_support.support_program_available(CUDA):
        pytest.skip("the candidate-support programs are not registered for this device")


def cuda_inputs(T, **kw):
    """A problem on the device; ``H`` defaults to 512 (the dense dX / dW form needs whole 512-column items)."""
    kw.setdefault("seed", SEED)
    kw.setdefault("H", 512)
    return make_support_inputs(T, device="cuda", **kw)


@pytest.mark.parametrize("form", sorted(CAKE_FORMS))
@pytest.mark.parametrize("case", sorted(SUPPORT_CASES))
def test_cake_matches_the_oracle_and_the_deterministic_package(case, form, monkeypatch):
    """Every output and both gradients pass the gate in each pass form and equal the deterministic entries' bits on
    the same inputs (the same programs).  A dense dX adds its vocabulary blocks into the chunk sum through the
    reduction form at the rule's :func:`cake_support.support_dx_form`: every shape here is outside the rule's measured
    cells, so every block runs one slice and dX has the deterministic bits too."""
    _require_cake()
    outputs, mk, kw = SUPPORT_CASES[case]
    mk = dict(mk)
    inp = cuda_inputs(mk.pop("T"), **mk)
    kw = dict(kw, backend="cake", **CAKE_FORMS[form])
    forms = _record_dx_forms(monkeypatch)
    res, _, _ = check_gates(inp, outputs, **kw)
    det = run_support(inp, outputs, package="det", **kw)
    dense = CAKE_FORMS[form].get("path") == "dense"
    assert bool(forms) == dense, forms
    assert all(s == 1 for s, _ in forms), forms
    assert_same(res, det)


@pytest.mark.parametrize("case", ["kl+packed heavy constant", "selected heavy hetero"])
def test_cake_worst_case_over_repeated_runs(case):
    """``REPEATS`` runs: each passes the gate (the worst case is the gate's)."""
    _require_cake()
    outputs, mk, kw = SUPPORT_CASES[case]
    mk = dict(mk)
    inp = cuda_inputs(mk.pop("T"), **mk)
    kw = dict(kw, backend="cake")
    oracle = reference_support_fp64(inp, outputs=outputs)
    b0err = support_error_report(
        reference_support_b0(inp, outputs=outputs), oracle, inp
    )
    for _ in range(REPEATS):
        check_gates(inp, outputs, oracle=oracle, b0err=b0err, **kw)


def test_cake_frozen_inputs_and_retain_graph():
    _require_cake()
    inp = cuda_inputs(
        257, k_cap=32, kind="heavy", teacher=True, labels=True, tau_form="hetero"
    )
    both = run_support(inp, ("packed", "kl"), backend="cake")
    x_only = run_support(inp, ("packed", "kl"), backend="cake", need_dw=False)
    w_only = run_support(inp, ("packed", "kl"), backend="cake", need_dx=False)
    assert x_only["dW"] is None and w_only["dX"] is None
    assert torch.equal(_bits(x_only["dX"]), _bits(both["dX"]))
    assert torch.equal(_bits(w_only["dW"]), _bits(both["dW"]))
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    out = support_logprob(
        X,
        W,
        inp.support_ids,
        inp.support_offsets,
        inp.labels,
        **inp.kwargs(),
        backend="cake",
        **ND,
    )
    g1 = torch.autograd.grad(out, (X, W), inp.d, retain_graph=True)
    g2 = torch.autograd.grad(out, (X, W), inp.d)
    for a, b in zip(g1, g2, strict=True):
        assert torch.equal(_bits(a), _bits(b))


# ---------------------------------------------------------------------------
# Dense dX through the reduction GEMM (this package's dense dX form)
# ---------------------------------------------------------------------------


def _rule_cell(rows, hidden, k_iters, num_sms, arch):
    """The measured cell holding the shape, if any (the table read directly)."""
    pairs = ((rows + 127) // 128 + 1) // 2
    for cell in cake_support.SUPPORT_DX_SLICE_CELLS.get(arch, ()):
        lo, hi = cell.k_iters
        key = (cell.num_sms, cell.hidden, cell.row_pairs)
        if key == (num_sms, hidden, pairs) and lo <= k_iters <= hi:
            return cell
    return None


def _split_tail(rows, hidden, slices, num_sms, arch):
    """The split tail recomputed: ``r`` = the last wave's items when ``0 < r <= P / 2`` on a split architecture."""
    pairs = num_sms // 2
    r = (((rows + 127) // 128 + 1) // 2) * (hidden // 512) * slices % pairs
    return r if arch in cake_backend.DX_SPLIT_ARCHES and 0 < r <= pairs // 2 else 0


# (rows_c, H, K steps, SMs, arch) -> (S, F); B200: 74 SM pairs, split tail for 0 < items % 74 <= 37
DX_RULE_EXPECTED = {
    # H 5120: 16 row pairs x 10 column items x 2 slices = 320 items, r = 24; its tail chunk / block
    (4096, 5120, 240, 148, "sm_100a"): (2, 24),
    (3943, 5120, 216, 148, "sm_100a"): (2, 24),
    (3841, 5120, 228, 148, "sm_100a"): (2, 24),
    # outside: 15 row pairs (150 items), wider / narrower blocks (160), another SM count (66 pairs)
    (3840, 5120, 240, 148, "sm_100a"): (1, 2),
    (4096, 5120, 241, 148, "sm_100a"): (1, 12),
    (4096, 5120, 215, 148, "sm_100a"): (1, 12),
    (4096, 5120, 240, 132, "sm_100a"): (1, 28),
    # the one-chunk cells (blocks of 224 to 244 K steps): 3 x 48 = 144 items (r = 70) at H 2048 / 3072 / 6144;
    # H 4096: 2 row pairs x 8 x 2 = 32 items, r = 32
    (3072, 2048, 244, 148, "sm_100a"): (3, 0),
    (2817, 2048, 224, 148, "sm_100a"): (3, 0),
    (2048, 3072, 242, 148, "sm_100a"): (3, 0),
    (1793, 3072, 224, 148, "sm_100a"): (3, 0),
    (512, 4096, 242, 148, "sm_100a"): (2, 32),
    (257, 4096, 244, 148, "sm_100a"): (2, 32),
    (1024, 6144, 242, 148, "sm_100a"): (3, 0),
    (769, 6144, 244, 148, "sm_100a"): (3, 0),
    # outside them: one row pair less / more, blocks of 223 / 245 K steps, another SM count
    (768, 6144, 242, 148, "sm_100a"): (1, 36),
    (1025, 6144, 242, 148, "sm_100a"): (1, 0),
    (256, 4096, 242, 148, "sm_100a"): (1, 8),
    (1024, 6144, 223, 148, "sm_100a"): (1, 0),
    (1024, 6144, 245, 148, "sm_100a"): (1, 0),
    (512, 4096, 242, 132, "sm_100a"): (1, 16),
    # H 4096 at 16 row pairs (two slices measured equal to one), H 6144 and 8192 at full chunks, a 1808-row chunk
    (4096, 4096, 242, 148, "sm_100a"): (1, 0),
    (4096, 6144, 242, 148, "sm_100a"): (1, 0),
    (4096, 8192, 244, 148, "sm_100a"): (1, 34),
    (1808, 6144, 242, 148, "sm_100a"): (1, 22),
    (257, 512, 8, 148, "sm_100a"): (1, 2),
    # no cell on sm_103a (split forms); sm_107a: no cell, no split forms
    (4096, 5120, 240, 148, "sm_103a"): (1, 12),
    (1024, 6144, 242, 148, "sm_103a"): (1, 0),
    (512, 4096, 242, 148, "sm_103a"): (1, 16),
    (4096, 5120, 240, 148, "sm_107a"): (1, 0),
    (1808, 6144, 242, 148, "sm_107a"): (1, 0),
}


def test_support_dx_form_cells():
    """The dense dX rule: ``S > 1`` only inside a measured cell of ``SUPPORT_DX_SLICE_CELLS`` (B200, 148 SMs: H 5120,
    chunks of 3841 to 4096 rows, vocabulary blocks of 216 to 240 K steps -> S = 2; one chunk of 2817 to 3072 rows at H
    2048, 1793 to 2048 at H 3072, 769 to 1024 at H 6144 -> S = 3 and 257 to 512 at H 4096 -> S = 2, vocabulary blocks
    of 224 to 244 K steps), ``S = 1`` at every other shape, SM count and architecture (H 4096 at full chunks and H 8192
    included); the split tail only on an architecture with split forms, for a last wave of at most half the SM
    pairs."""
    form, split = cake_support.support_dx_form, cake_support.support_dx_split
    Cell, table = cake_support.SupportDxCell, cake_support.SUPPORT_DX_SLICE_CELLS
    assert table == {
        "sm_100a": (
            Cell(148, 2048, 12, (224, 244), 3),
            Cell(148, 3072, 8, (224, 244), 3),
            Cell(148, 4096, 2, (224, 244), 2),
            Cell(148, 5120, 16, (216, 240), 2),
            Cell(148, 6144, 4, (224, 244), 3),
        )
    }
    for cells in table.values():
        for cell in cells:  # every slice holds at least one K step anywhere in the cell
            assert 1 < cell.slices <= cake_support.SUPPORT_DX_SLICES_MAX
            fit = cake_backend.dx_slices_fit(cell.slices, cell.k_iters[0])
            assert fit == cell.slices
    for args, want in DX_RULE_EXPECTED.items():
        assert form(*args) == want, (args, form(*args), want)
    rows_all = (1, 256, 257, 512, 768, 769, 1024, 1300, 1792, 1793, 1808, 2048)
    rows_all += (2816, 2817, 3072, 3840, 3841, 3943, 4096)
    k_all = (1, 4, 8, 215, 216, 223, 224, 240, 241, 244, 245, 256)
    shapes = [
        (rows, hidden, k_iters)
        for rows in rows_all
        for hidden in range(512, 8193, 512)
        for k_iters in k_all
    ]
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for num_sms in (132, 148, 160):
            for rows, hidden, k_iters in shapes:
                s, f = form(rows, hidden, k_iters, num_sms, arch)
                cell = _rule_cell(rows, hidden, k_iters, num_sms, arch)
                assert s == (cell.slices if cell else 1)
                assert f == split(rows, hidden, s, num_sms, arch)
                assert f == _split_tail(rows, hidden, s, num_sms, arch)


def test_dense_stages_follow_the_rule():
    """The dense pass form requires the split-tail programs only on an architecture whose rule can choose them; every
    registered record carries the programs its rule reaches."""
    stages = cake_support.support_dense_stages
    split = set(cake_support.SUPPORT_DENSE_SPLIT_STAGES)
    assert split == {"gemm_dx_split", "gemm_dx_red_split"}
    for arch in cake_backend.DX_SPLIT_ARCHES:
        assert split <= set(stages(arch))
    assert not split & set(stages("sm_107a"))
    assert set(cake_support.SUPPORT_DENSE_STAGES) <= set(stages("sm_107a"))
    # the later blocks of a chunk add through the reduction form at one slice too
    assert "gemm_dx_red" in cake_support.SUPPORT_DENSE_STAGES
    for name, record in cake_support.MODULES.items():
        need = cake_support.SUPPORT_STAGES + stages(record["arch"])
        assert [s for s in need if s not in record] == [], name


def _record_dx_forms(monkeypatch):
    """The ``(S, F)`` of every dense dX block GEMM of the calls that follow (the rule itself unchanged)."""
    rule, seen = cake_support.support_dx_form, []

    def spy(*args):
        seen.append(rule(*args))
        return seen[-1]

    monkeypatch.setattr(cake_support, "support_dx_form", spy)
    return seen


def _force_dx_form(monkeypatch, form):
    """Run every block's dX GEMM at ``form = (S, F)``: ``S`` requested slices cut to the block's K steps
    (``dx_slices_fit``: every slice holds at least one K step, so 3 over a 4-step block run as 2) and ``F`` split-tail
    items (``None``: the rule's split tail at that slice count)."""
    slices, split = form

    def forced(rows_c, hidden, k_iters, num_sms, arch):
        s = cake_backend.dx_slices_fit(slices, k_iters)
        if split is None:
            return s, cake_support.support_dx_split(rows_c, hidden, s, num_sms, arch)
        return s, split

    monkeypatch.setattr(cake_support, "support_dx_form", forced)


# (slices, split-tail items): S = 1 keeps the deterministic bits (with or without the split tail); S > 1 adds the
# slices in the order they finish
DX_FORMS = [(1, 0), (1, 1), (2, 0), (2, 1), (4, 0), (4, 2)]
DX_CASES = ["kl two chunks", "selected chunked", "packed shr hetero"]


@pytest.mark.parametrize("form", DX_FORMS, ids=lambda f: f"S{f[0]}F{f[1]}")
@pytest.mark.parametrize("case", DX_CASES)
@pytest.mark.parametrize("vocab_block", [None, 256])
def test_cake_dense_dx_reduction_forms(case, form, vocab_block, monkeypatch):
    """The dense dX pass at every slice / split-tail form passes the gate; one slice equals the deterministic
    package's bits (outputs, dX and dW); above one slice the outputs and dW stay bitwise and dX differs from the
    deterministic one by rounding only."""
    _require_cake()
    outputs, mk, kw = SUPPORT_CASES[case]
    mk = dict(mk)
    inp = cuda_inputs(mk.pop("T"), **mk)
    kw = dict(kw, backend="cake", path={"X": "dense"}, vocab_block=vocab_block)
    _force_dx_form(monkeypatch, form)
    res, err, _ = check_gates(inp, outputs, **kw)
    det = run_support(inp, outputs, package="det", **kw)
    if form[0] == 1:
        assert_same(res, det)
        return
    assert_same(dict(res, dX=None), dict(det, dX=None))
    _within_rounding_of_det(res["dX"], det["dX"])


def _repeat_dense_dx(label, inp, outputs, kw, *, oracle, b0err, det):
    """``REPEATS`` runs of this package against the FP64 oracle: each passes the gate, keeps the outputs and dW
    bitwise and dX within rounding of the deterministic package.  Prints the worst dX error (beside the deterministic
    package's and the masked PyTorch baseline's, B0), the largest ``|dX_nd - dX_det|``, the most BF16 flips against
    the deterministic dX and the number of distinct dX results."""
    det_err = support_error_report(det, oracle, inp)["dX_rel_l2"]
    worst, max_diff, flips, excess, distinct = 0.0, 0.0, 0, 0.0, set()
    for _ in range(REPEATS):
        res, err, _ = check_gates(inp, outputs, oracle=oracle, b0err=b0err, **kw)
        assert_same(dict(res, dX=None), dict(det, dX=None))
        excess = max(excess, _within_rounding_of_det(res["dX"], det["dX"]))
        worst = max(worst, err["dX_rel_l2"])
        max_diff = max(
            max_diff, float((res["dX"].float() - det["dX"].float()).abs().max())
        )
        flips = max(flips, int((_bits(res["dX"]) != _bits(det["dX"])).sum()))
        distinct.add(_bits(res["dX"]).cpu().numpy().tobytes().__hash__())
    print(
        f"DENSE_DX {label} worst-of-{REPEATS}: dX_rel_l2 {worst:.3e} (det {det_err:.3e}, B0 "
        f"{b0err['dX_rel_l2']:.3e}) max|nd-det| {max_diff:.3e} flips {flips}/{det['dX'].numel()} "
        f"bound use {excess:.3f} distinct {len(distinct)} zero rows "
        f"{int((det['dX'] == 0).all(dim=-1).sum())}"
    )


def _oracle_and_b0(inp, outputs):
    oracle = reference_support_fp64(inp, outputs=outputs)
    b0err = support_error_report(
        reference_support_b0(inp, outputs=outputs), oracle, inp
    )
    return oracle, b0err


@pytest.mark.parametrize("case", DX_CASES)
def test_cake_dense_dx_worst_case_over_repeated_runs(case, monkeypatch):
    """``REPEATS`` runs of a K-sliced dense dX (S = 4, split tail) per entry (KL, selected, packed) against the FP64
    oracle (:func:`_repeat_dense_dx`)."""
    _require_cake()
    outputs, mk, kw = SUPPORT_CASES[case]
    mk = dict(mk)
    inp = cuda_inputs(mk.pop("T"), **mk)
    kw = dict(kw, backend="cake", path={"X": "dense"})
    oracle, b0err = _oracle_and_b0(inp, outputs)
    det = run_support(inp, outputs, package="det", **kw)
    _force_dx_form(monkeypatch, (4, 2))
    _repeat_dense_dx(
        f"S4F2 {case}", inp, outputs, kw, oracle=oracle, b0err=b0err, det=det
    )


# Production-like dense-X geometry (the cases above run H = 512: one 512-column item, at most 8 K steps and two
# vocabulary blocks): H = 1536 (three column items, split halves on column tiles past the first), V = 1280 in
# vocabulary blocks of 512 (8, 8 and 4 K steps: different forms per block of one chunk; 3 requested slices run as 2 on
# the 4-step block), T = 1300 in chunks of 1024 + 276 rows and T = 4097 in chunks of 4096 + 1.
PROD_CASES = {
    "kl+packed t1300": (
        ("kl", "packed"),
        dict(T=1300, k_cap=256, kind="heavy", teacher=True, tau_form="hetero"),
        dict(chunk_size=1024),
    ),
    "selected t4097": (
        ("selected",),
        dict(T=4097, k_cap=256, kind="heavy", labels=True),
        {},
    ),
}
PROD_FORMS = [(1, 0), (1, 2), (2, 1), (3, 0), (4, 2)]


def _prod_case(case):
    outputs, mk, kw = PROD_CASES[case]
    mk = dict(mk, H=1536, V=1280)
    inp = cuda_inputs(mk.pop("T"), **mk)
    return outputs, inp, dict(kw, backend="cake", path={"X": "dense"}, vocab_block=512)


@pytest.mark.parametrize("form", PROD_FORMS, ids=lambda f: f"S{f[0]}F{f[1]}")
@pytest.mark.parametrize("case", sorted(PROD_CASES))
def test_cake_dense_dx_production_geometry(case, form, monkeypatch):
    """Dense dX at production-like geometry with every block forced to ``form``: one slice equals the deterministic
    package's bits (outputs, dX and dW); 2 to 4 slices pass the gate, keep the outputs and dW bitwise and stay within
    rounding of the deterministic dX."""
    _require_cake()
    outputs, inp, kw = _prod_case(case)
    _force_dx_form(monkeypatch, form)
    res, _, _ = check_gates(inp, outputs, **kw)
    det = run_support(inp, outputs, package="det", **kw)
    if form[0] == 1:
        assert_same(res, det)
    else:
        assert_same(dict(res, dX=None), dict(det, dX=None))
        _within_rounding_of_det(res["dX"], det["dX"])


@pytest.mark.parametrize("slices", [2, 3, 4])
@pytest.mark.parametrize("case", sorted(PROD_CASES))
def test_cake_dense_dx_production_geometry_repeated(case, slices, monkeypatch):
    """``REPEATS`` runs at ``slices`` K slices with the rule's split tail at production-like geometry
    (:func:`_repeat_dense_dx`)."""
    _require_cake()
    outputs, inp, kw = _prod_case(case)
    oracle, b0err = _oracle_and_b0(inp, outputs)
    det = run_support(inp, outputs, package="det", **kw)
    _force_dx_form(monkeypatch, (slices, None))
    _repeat_dense_dx(
        f"S{slices} {case}", inp, outputs, kw, oracle=oracle, b0err=b0err, det=det
    )


# (T, H, V, vocab_block, S on B200): one chunk inside a cell of the rule, two vocabulary blocks
RULE_CELL_CASES = {
    # V = 29184 in blocks of 15360 + 13824 columns (240 + 216 K steps), 3943 rows (16 row pairs): S2 F24
    "h5120 t3943": (3943, 5120, 29184, 15360, 2),
    # V = 29952 in blocks of 15616 + 14336 columns (244 + 224 K steps), 3072 rows (12 row pairs): S3 F0
    "h2048 t3072": (3072, 2048, 29952, 15616, 3),
}


@pytest.mark.parametrize("case", sorted(RULE_CELL_CASES))
def test_cake_dense_dx_rule_cell(case, monkeypatch):
    """A call inside a measured cell of the rule (B200, 148 SMs): the rule itself runs the cell's slices on every
    block (one slice on any other device: then the deterministic bits), and ``REPEATS`` runs pass
    :func:`_repeat_dense_dx`."""
    _require_cake()
    T, H, V, vb, slices = RULE_CELL_CASES[case]
    inp = cuda_inputs(T, H=H, V=V, k_cap=64, kind="heavy", teacher=True)
    kw = dict(backend="cake", path={"X": "dense"}, vocab_block=vb)
    cx = cake_support._cake_context(CUDA)
    want = slices if (cx.arch, cx.num_sms) == ("sm_100a", 148) else 1
    oracle, b0err = _oracle_and_b0(inp, ("kl",))
    det = run_support(inp, ("kl",), package="det", **kw)
    forms = _record_dx_forms(monkeypatch)
    res, _, _ = check_gates(inp, ("kl",), oracle=oracle, b0err=b0err, **kw)
    assert len(forms) == 2 and {s for s, _ in forms} == {want}, forms
    if want == 1:
        assert_same(res, det)
        return
    _repeat_dense_dx(
        f"rule cell {case} {forms}",
        inp,
        ("kl",),
        kw,
        oracle=oracle,
        b0err=b0err,
        det=det,
    )


def test_cake_half_item_hidden_runs_x_and_w_sparse(monkeypatch):
    """H % 512 == 256 (H = 768): the dense dX / dW forms need whole 512-column items, so a call whose mean support
    reaches every dense threshold runs the forward dense and X / W sparse (the deterministic package's bits); a
    forced dense X pass raises."""
    _require_cake()
    inp = cuda_inputs(64, H=768, V=1280, k_cap=1280, kind="full", teacher=True)
    arch = cake_support._cake_context(CUDA).arch
    plan = cake_support.support_memory_report(
        inp.T, inp.H, inp.V, inp.M, outputs=("kl",), arch=arch
    )["plan"]
    assert (plan["forward"], plan["dx"], plan["dw"]) == ("dense", "sparse", "sparse")
    with pytest.raises(ValueError, match="the dense X pass needs H"):
        run_support(inp, ("kl",), backend="cake", path={"X": "dense"})
    forms = _record_dx_forms(monkeypatch)
    res, _, _ = check_gates(inp, ("kl",), backend="cake")
    assert forms == []
    assert_same(res, run_support(inp, ("kl",), package="det", backend="cake"))


def test_cake_dense_dx_subnormal_products_are_kept(monkeypatch):
    """``W`` scaled by 2^-120 makes most of dX subnormal (FP32 / BF16 values below 2^-126): the K-sliced reduction
    keeps them (round to nearest, no flush), as the deterministic package's adds do.  Values, not only counts: every
    element within rounding of the deterministic dX (one BF16 ulp, 2^-133 in the subnormal range), and the subnormal
    elements' summed difference at most 2^-7 of their summed magnitude (a flush loses each whole).  The rows with no
    nonzero product are those of the unscaled problem's deterministic dX that are all 0 (at this scale whole rows of
    tiny products also round to 0, and their sums may round to 2^-133 in another order)."""
    _require_cake()
    inp = cuda_inputs(1100, k_cap=32, kind="heavy", teacher=True)
    kw = dict(backend="cake", path={"X": "dense"})
    unscaled = run_support(inp, ("kl",), package="det", need_dw=False, **kw)["dX"]
    no_product = (unscaled == 0).all(dim=-1)
    assert bool(no_product.any())
    inp = dataclasses.replace(inp, W=(inp.W.float() * 2.0**-120).to(torch.bfloat16))
    det = run_support(inp, ("kl",), package="det", need_dw=False, **kw)
    _force_dx_form(monkeypatch, (4, 2))
    nd = run_support(inp, ("kl",), need_dw=False, **kw)
    d, r = nd["dX"].float(), det["dX"].float()
    tiny = 2.0**-126
    sub = (r != 0) & (r.abs() < tiny)
    sub_det = int(sub.sum())
    sub_nd = int(((d != 0) & (d.abs() < tiny)).sum())
    assert sub_det > r.numel() // 4, sub_det
    assert abs(sub_nd - sub_det) <= max(16, sub_det // 1000), (sub_nd, sub_det)
    excess = _within_rounding_of_det(nd["dX"], det["dX"], zero_rows=no_product)
    lost = float((d - r).abs()[sub].sum()) / float(r.abs()[sub].sum())
    assert lost <= 2.0**-7, lost
    print(
        f"DENSE_DX subnormal S4F2: subnormal det {sub_det} nd {sub_nd} of {r.numel()}, "
        f"bound use {excess:.3f}, summed |nd-det| / |det| over subnormals {lost:.3e}, rows with no "
        f"nonzero product {int(no_product.sum())} (all-0 rows at this scale {int((r == 0).all(dim=-1).sum())})"
    )


@pytest.mark.parametrize("form", [(1, 1), (2, 1)], ids=lambda f: f"S{f[0]}F{f[1]}")
def test_cake_dense_dx_graph_streams_and_reused_buffers(form, monkeypatch):
    """The dense dX form under CUDA graph capture (replayed twice over the graph pool's reused, nonzero buffers), on
    two concurrent streams, and after a larger call left nonzero blocks in the caching allocator: the KL and dW equal
    the eager deterministic bits every time, dX too at one slice; K slices keep dX within rounding of them and pass
    the gate."""
    _require_cake()
    _force_dx_form(monkeypatch, form)
    exact = form[0] == 1
    kw = dict(backend="cake", path="dense", **ND)

    def inputs(T, seed):
        inp = cuda_inputs(T, k_cap=32, kind="heavy", teacher=True, seed=seed)
        return inp, dict(kw, **inp.kwargs())

    def step(inp, kwargs, X, W):
        kl = support_kl(
            X, W, inp.support_ids, inp.support_offsets, inp.teacher, **kwargs
        )
        return (kl, *torch.autograd.grad(kl, (X, W), inp.e))

    def same(a, b):
        (kl, dx, dw), (kl_ref, dx_ref, dw_ref) = a, b
        assert torch.equal(_bits(kl), _bits(kl_ref))
        assert torch.equal(_bits(dw), _bits(dw_ref))
        if exact:
            assert torch.equal(_bits(dx), _bits(dx_ref))
        else:
            _within_rounding_of_det(dx, dx_ref)

    big, big_kw = inputs(4097, SEED + 1)
    small_inp, small_kw = inputs(1300, SEED + 2)
    leaf = lambda t: t.detach().clone().requires_grad_(True)  # noqa: E731
    det_kw = {k: v for k, v in small_kw.items() if k != "deterministic"}
    det_ref = run_support(small_inp, ("kl",), package="det", **det_kw)
    ref = (det_ref["kl"], det_ref["dX"], det_ref["dW"])
    # a larger call first: its freed blocks (nonzero) are reused by the smaller call
    step(big, big_kw, leaf(big.X), leaf(big.W))
    same(step(small_inp, small_kw, leaf(small_inp.X), leaf(small_inp.W)), ref)
    # two streams, each with its own problem, against the sequential results
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    seq_big = step(big, big_kw, leaf(big.X), leaf(big.W))
    s1.wait_stream(torch.cuda.current_stream())
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s1):
        par_big = step(big, big_kw, leaf(big.X), leaf(big.W))
    with torch.cuda.stream(s2):
        par_small = step(small_inp, small_kw, leaf(small_inp.X), leaf(small_inp.W))
    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)
    same(par_small, ref)
    same(par_big, seq_big)
    # CUDA graph capture of the whole step, replayed twice
    X, W = leaf(small_inp.X), leaf(small_inp.W)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step(small_inp, small_kw, X, W)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = step(small_inp, small_kw, X, W)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        same(static, ref)
    if not exact:
        oracle = reference_support_fp64(small_inp, outputs=("kl",))
        b0err = support_error_report(
            reference_support_b0(small_inp, outputs=("kl",)), oracle, small_inp
        )
        err = support_error_report(
            dict(kl=static[0], dX=static[1], dW=static[2]), oracle, small_inp
        )
        assert not gate_failures(err, b0err), (err, b0err)


def test_sparse_dw_zero_fill_bits():
    """The global sparse dW's zero fill (FP32 zero words viewed as BF16) gives +0 BF16 bits: ids in no support keep
    +0, equal to the deterministic package's dW."""
    _require_cake()
    inp = cuda_inputs(257, k_cap=8, V=4096, teacher=True)
    kw = dict(backend="cake", path="sparse", dw_mode="global")
    nd = run_support(inp, ("kl",), **kw)
    det = run_support(inp, ("kl",), package="det", **kw)
    assert torch.equal(_bits(nd["dW"]), _bits(det["dW"]))
    used = torch.zeros(inp.W.shape[0], dtype=torch.bool, device=inp.W.device)
    used[inp.support_ids.long()] = True
    assert int((_bits(nd["dW"][~used]) != 0).sum()) == 0
