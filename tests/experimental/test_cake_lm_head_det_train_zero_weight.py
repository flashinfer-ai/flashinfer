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

The zero-weight split of the deterministic LM-head package's loss entry (``cake_backend.gradient_rows``,
``gradient_index``, ``set_zero_weight_min_rows``, ``set_zero_weight_loss_mode``, ``_zero_weight_exact_loss``): the
split indices, the floor, the loss mode and the not-applied list on the host, the exact loss's term placement against
the plain path's reduction bit for bit, and on the device the bits the split keeps against the same call without it
(``loss`` in the exact mode, ``logp`` of every row, ``dX`` of the gradient rows whose chunk keeps its dX form), the
exact ``+0`` ``dX`` rows of the zero-weight rows, the deterministic contract (the same bits on every run), the
accuracy gates of ``test_cake_lm_head_det_train.py`` and the peak memory against the call without the split.
"""

import contextlib
import dataclasses
import gc
import inspect
import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from tests.experimental.test_helpers.cake_lm_head_loss_reference import (
    error_report,
    make_inputs,
    reference_fp64,
    reference_unchunked,
    rel_l2,
)
from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_lm_head_det_train", "cake_backend")
opt_in = cake_package("cake.cake_lm_head_det_train", "cake_backend_opt_in")
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
generated_program_available = cake_backend.generated_program_available

SEED = 20261105
CUDA = torch.device("cuda")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Host-layer shapes (the reference backend runs at any multiple of 256).
H_HOST, V_HOST = 256, 512
# The accuracy gates of the deterministic suite (``test_cake_lm_head_det_train.py``), copied unchanged: at most 1.05x
# the error of the unchunked PyTorch path on the same inputs plus a floor per metric, and twice the production-geometry
# unchunked error as a sanity ceiling.
GATE_MARGIN = 1.05
GATE_TINY = dict(loss_rel=1e-6, logp_max_abs=1e-5, dX_rel_l2=1e-3, dW_rel_l2=1e-3)
PRODUCTION_B0 = dict(dX_rel_l2=1.85e-3, dW_rel_l2=1.88e-3, logp_max_abs=4.1e-3)
# Runs of one call compared bit for bit (the deterministic contract).
RUNS = 3
# A floor above every call of this file: the plain path (no split).
NO_SPLIT = 1 << 62


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program(*, entry: str = "loss"):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda"), entry=entry):
        pytest.skip(
            f"generated chunked LM-head program ({entry} entry) not registered for this device"
        )


def _check_against_references(result, inp, *, ceiling=False):
    """``test_cake_lm_head_det_train.py``'s gate, unchanged."""
    oracle = reference_fp64(inp)
    b0 = reference_unchunked(inp)
    errors = error_report(result, oracle, inp.labels)
    b0_errors = error_report(b0, oracle, inp.labels)
    assert not errors["nan"]
    for key, tiny in GATE_TINY.items():
        if key in errors:
            assert errors[key] <= GATE_MARGIN * b0_errors[key] + tiny, (
                f"{key}: {errors[key]:.3e} vs unchunked {b0_errors[key]:.3e}"
            )
    if ceiling:
        for key, measured in PRODUCTION_B0.items():
            if key in errors:
                assert errors[key] <= 2.0 * measured, (
                    f"{key}: {errors[key]:.3e} above twice the production-geometry unchunked error"
                )
    assert errors["logp_ignored_zero"]
    if result.get("dX") is not None:
        assert errors["dX_ignored_zero"]
    return errors, b0_errors


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The bit pattern of a float tensor as integers (``torch.equal`` on floats equates +0 and -0)."""
    t = t.detach().reshape(-1).contiguous()
    return t.view(
        {
            torch.bfloat16: torch.int16,
            torch.float32: torch.int32,
            torch.float64: torch.int64,
        }[t.dtype]
    )


def _same_bits(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _all_plus_zero(t: torch.Tensor) -> bool:
    """Every element is ``+0`` (bit pattern zero)."""
    return bool((_bits(t) == 0).all())


@contextlib.contextmanager
def _zero_weight_floor(rows):
    """Run under the split's floor ``rows`` (:data:`NO_SPLIT`: never), then restore the previous setting."""
    previous = cake_backend.set_zero_weight_min_rows(rows)
    try:
        yield
    finally:
        cake_backend.set_zero_weight_min_rows(previous)


@contextlib.contextmanager
def _loss_mode(mode):
    """Run under the split's loss mode, then restore the previous one."""
    previous = cake_backend.set_zero_weight_loss_mode(mode)
    try:
        yield
    finally:
        cake_backend.set_zero_weight_loss_mode(previous)


def _zero_weight_set(inp, kind, *, seed=SEED):
    """bool ``[T]`` of the valid rows whose weight becomes zero: ``"none"``, ``"one"`` (the first valid row),
    ``("group", f)`` (whole groups of 512 consecutive rows, ``round(f * groups)`` of them), ``("random", f)``
    (``round(f * T_v)`` random valid rows) or ``"all"``."""
    valid = inp.labels >= 0
    T = int(valid.numel())
    zs = torch.zeros(T, dtype=torch.bool, device=valid.device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + T)
    if kind == "none":
        return zs
    if kind == "all":
        return valid.clone()
    if kind == "one":
        zs[int(valid.nonzero()[0])] = True
        return zs
    pattern, frac = kind
    if pattern == "group":
        groups = -(-T // 512)
        for g in torch.randperm(groups, generator=gen)[
            : int(round(frac * groups))
        ].tolist():
            zs[g * 512 : min(T, (g + 1) * 512)] = True
    else:
        rows = valid.nonzero().squeeze(1).cpu()
        pick = rows[
            torch.randperm(rows.numel(), generator=gen)[
                : int(round(frac * rows.numel()))
            ]
        ]
        zs[pick.to(zs.device)] = True
    return zs & valid


def _with_zero_weights(inp, zero, *, negative_zero=False):
    """``inp`` with ``loss_weights`` 0 on ``zero`` (``-0.0`` on every other zero row with ``negative_zero``)."""
    w = inp.loss_weights.clone()
    w[zero] = 0.0
    if negative_zero:
        rows = zero.nonzero().squeeze(1)
        w[rows[::2]] = -0.0
    return dataclasses.replace(inp, loss_weights=w.contiguous())


def _host_inputs(T, *, seed=SEED, **kw):
    return make_inputs(
        T, objective="policy", seed=seed + T, H=H_HOST, V=V_HOST, device=DEV, **kw
    )


def _loss_call(
    inp,
    C,
    *,
    backend="cake",
    return_logp=True,
    train_x=True,
    train_w=True,
    X=None,
    module=None,
    **kw,
):
    """One autograd loss call (forward + backward) of the package's loss entry (``module``: the opt-in module for
    its keywords); returns loss / logp (``None`` without ``return_logp``) / dX / dW.  ``X`` replaces the inputs' ``X``
    (the same values in another layout)."""
    X = (inp.X if X is None else X).detach().requires_grad_(train_x)
    W = inp.W.detach().requires_grad_(train_w)
    mod = module if module is not None else cake_backend
    out = mod.cake_lm_head_det_train_loss(
        X,
        W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=C,
        return_logp=return_logp,
        backend=backend,
        **kw,
    )
    loss, logp = out if return_logp else (out, None)
    if train_x or train_w:
        loss.backward()
    if X.is_cuda:
        torch.cuda.synchronize()
    return dict(
        loss=loss.detach(),
        logp=None if logp is None else logp.detach(),
        dX=X.grad,
        dW=W.grad,
    )


@contextlib.contextmanager
def _recording_loss_plans(module=None):
    """Record the plan of every loss-entry chunk loop the remembered bindings run (the logp-only pass's
    log-probability bindings are left out)."""
    mod = module if module is not None else cake_backend
    plans = []
    original = mod._Binding.forward

    def recording(self, *a, **k):
        if self.plan.problem.entry == "loss":
            plans.append(self.plan)
        return original(self, *a, **k)

    mod._Binding.forward = recording
    try:
        yield plans
    finally:
        mod._Binding.forward = original


def _g_rows_of_equal_dx_forms(labels, zero, C, split_plan, plain_plan):
    """The gradient rows whose chunk runs the same dX form and K-slice count with the split (the gradient rows only)
    and without it (every valid row): their ``dX`` row is the same arithmetic, so the same bits."""
    rows = (labels >= 0).nonzero().squeeze(1)
    grad = ~zero[rows]
    plain_chunk = torch.arange(rows.numel(), device=rows.device) // C
    rank = torch.cumsum(grad.to(torch.int64), 0) - 1
    split_chunk = rank.clamp(min=0) // C
    forms: dict[tuple, int] = {}

    def form_ids(plan, chunk):
        ids = [
            forms.setdefault((plan.dx_stage_of(i), plan.dx_slices_of(i)), len(forms))
            for i in range(plan.num_chunks)
        ]
        table = torch.tensor(ids, dtype=torch.int64, device=rows.device)
        return table[chunk.clamp(max=plan.num_chunks - 1)]

    same = form_ids(split_plan, split_chunk) == form_ids(plain_plan, plain_chunk)
    return rows[grad & same]


def _policy_problem(T, C, *, entry="loss", objective="policy"):
    X = torch.zeros((T, H_HOST), dtype=torch.bfloat16)
    W = torch.zeros((V_HOST, H_HOST), dtype=torch.bfloat16)
    labels = torch.zeros((T,), dtype=torch.int64)
    if entry == "logprob":
        return cake_backend.validate_lm_head_det_inputs(
            X, W, labels, chunk_size=C, entry="logprob"
        )
    if objective == "ce":
        return cake_backend.validate_lm_head_det_inputs(
            X, W, labels, objective="ce", loss_div=1.0, chunk_size=C
        )
    return cake_backend.validate_lm_head_det_inputs(
        X,
        W,
        labels,
        objective="policy",
        infer_logp=torch.zeros((T,)),
        loss_weights=torch.ones((T,)),
        chunk_size=C,
    )


# ---------------------------------------------------------------------------
# Host layer (any device)
# ---------------------------------------------------------------------------


def test_gradient_rows_split_the_valid_rows():
    """``gradient_rows``: the gradient rows (weight != 0, NaN included) and the zero-weight rows (+0 and -0), both
    ascending, ``None`` for an empty side; every row in ``G``: the uncompacted path."""
    labels = torch.tensor([3, -100, 5, 7, 1, 2, -100, 4, 0], dtype=torch.int64)
    w = torch.tensor([1.0, 0.0, 0.0, -0.0, float("nan"), -2.0, 3.0, 0.0, 0.5])
    g, mask, z = cake_backend.gradient_rows(labels, w, mask=True)
    assert g.tolist() == [0, 4, 5, 8] and z.tolist() == [2, 3, 7]
    assert g.dtype == torch.int64 and z.dtype == torch.int64
    assert mask.tolist() == [True, False, False, False, True, True, False, False, True]
    g, mask, z = cake_backend.gradient_rows(labels, torch.ones(9))
    assert g.tolist() == [0, 2, 3, 4, 5, 7, 8] and mask is None and z is None
    assert cake_backend.gradient_rows(torch.arange(4), torch.ones(4)) == (
        None,
        None,
        None,
    )
    g, _, z = cake_backend.gradient_rows(torch.arange(4), torch.zeros(4))
    assert g.numel() == 0 and z.tolist() == [0, 1, 2, 3]
    assert cake_backend.gradient_rows(
        torch.zeros(0, dtype=torch.int64), torch.zeros(0)
    ) == (None, None, None)


@pytest.mark.parametrize("T", [1, 37, 1500])
def test_gradient_index_scans_and_counts(T):
    """``gradient_index``: the two masks, their separate inclusive int32 scans (both counts at ``pos[T - 1]`` /
    ``pos[2T - 1]``, one strided copy) and the gradient-row gather index (``T`` past the count); the indices of
    ``gradient_rows`` follow from the scans."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + T)
    labels = torch.randint(0, 50, (T,), generator=gen)
    labels[torch.rand(T, generator=gen) < 0.1] = -100
    w = torch.randn(T, generator=gen)
    w[torch.rand(T, generator=gen) < 0.4] = 0.0
    w[torch.rand(T, generator=gen) < 0.1] = -0.0
    w[torch.rand(T, generator=gen) < 0.05] = float("nan")
    grad = torch.empty(T, dtype=torch.bool)
    zero = torch.empty(T, dtype=torch.bool)
    pos = torch.empty(2 * T, dtype=torch.int32)
    idx = torch.empty(T, dtype=torch.int64)
    ar = torch.arange(1, T + 1, dtype=torch.int32)
    cake_backend.gradient_index(
        labels, w, grad=grad, zero=zero, pos=pos, ar=ar, idx_full=idx
    )
    valid = labels >= 0
    want_g = valid & (w != 0)
    want_z = valid & (w == 0)
    assert torch.equal(grad, want_g) and torch.equal(zero, want_z)
    assert pos.dtype == torch.int32
    assert pos[:T].tolist() == torch.cumsum(want_g.to(torch.int32), 0).tolist()
    assert pos[T:].tolist() == torch.cumsum(want_z.to(torch.int32), 0).tolist()
    n_g, n_z = pos[T - 1 :: T].tolist()
    assert n_g == int(want_g.sum()) and n_z == int(want_z.sum())
    g_rows = want_g.nonzero().squeeze(1)
    assert idx[:n_g].tolist() == g_rows.tolist()
    assert bool((idx[n_g:] == T).all())
    g, mask, z = cake_backend.gradient_rows(labels, w, mask=True)
    if n_g == T:
        assert (g, mask, z) == (None, None, None)
    else:
        assert g.tolist() == g_rows.tolist() and torch.equal(mask, want_g)
        assert (z is None) == (n_z == 0)
        if z is not None:
            assert z.tolist() == want_z.nonzero().squeeze(1).tolist()


def test_zero_weight_floor_and_mode_switches():
    """The programmatic switches of the split: the floor (default one default chunk, ``4096`` rows ``T``) and the loss
    mode (default ``"exact"``), the previous value returned, bad values refused, the floor applied to the split
    decision; no environment variable and no binding depend on either."""
    assert cake_backend.ZERO_WEIGHT_MIN_ROWS_DEFAULT == 4096
    assert cake_backend.zero_weight_min_rows() == 4096
    assert cake_backend.ZERO_WEIGHT_LOSS_MODES == ("exact", "regroup")
    assert cake_backend.zero_weight_loss_mode() == "exact"
    problem = _policy_problem(200, 64)
    assert not cake_backend.zero_weight_split(problem, True)  # below the default floor
    assert cake_backend.zero_weight_split(problem, True, min_rows=200)
    assert not cake_backend.zero_weight_split(problem, True, min_rows=201)
    with _zero_weight_floor(0):
        assert cake_backend.zero_weight_min_rows() == 0
        assert cake_backend.zero_weight_split(problem, True)
        assert not cake_backend.zero_weight_split(problem, False)  # compact_rows=False
        assert not cake_backend.zero_weight_split(
            _policy_problem(200, 64, objective="ce"), True
        )
        assert not cake_backend.zero_weight_split(
            _policy_problem(200, 64, entry="logprob"), True
        )
        with _zero_weight_floor(256):
            assert cake_backend.zero_weight_min_rows() == 256
            assert not cake_backend.zero_weight_split(problem, True)
        assert cake_backend.zero_weight_min_rows() == 0
    assert cake_backend.zero_weight_min_rows() == 4096
    for bad in (-1, 2.5, False, True, "4096", None):
        with pytest.raises(ValueError, match="min rows"):
            cake_backend.set_zero_weight_min_rows(bad)
    assert cake_backend.zero_weight_min_rows() == 4096
    with _loss_mode("regroup"):
        assert cake_backend.zero_weight_loss_mode() == "regroup"
        assert cake_backend.set_zero_weight_loss_mode("regroup") == "regroup"
    assert cake_backend.zero_weight_loss_mode() == "exact"
    for bad in ("Exact", "", None, 1, "plain"):
        with pytest.raises(ValueError, match="loss mode"):
            cake_backend.set_zero_weight_loss_mode(bad)
    assert cake_backend.zero_weight_loss_mode() == "exact"


def test_term_rows_keep_every_chunk():
    """``_term_rows``: without ``keep_terms`` every chunk binds the chunk-local ``term`` buffer; with it chunk 0 keeps
    ``term`` and chunk ``c >= 1`` binds the compact ``loss_weights`` rows of chunk ``c - 1`` (only the base moves:
    a contiguous view of the chunk's row count)."""
    plan = SimpleNamespace(
        chunks=cake_backend.plan_chunks(10, 4)
    )  # (0, 4) (4, 4) (8, 2)
    t = dict(term=torch.zeros(4), loss_weights=torch.zeros(10))
    for index in range(3):
        assert cake_backend._term_rows(t, plan, index) is t["term"]
    t["keep_terms"] = True
    assert cake_backend._term_rows(t, plan, 0) is t["term"]
    base = t["loss_weights"].data_ptr()
    for index, (offset, rows) in ((1, (0, 4)), (2, (4, 2))):
        view = cake_backend._term_rows(t, plan, index)
        assert view.is_contiguous() and view.numel() == rows
        assert view.data_ptr() == base + offset * 4


@pytest.mark.parametrize(
    "n_v, frac, C",
    [(1, 1.0, 4), (37, 0.4, 16), (300, 0.3, 64), (1000, 0.9, 256), (513, 0.0, 256)],
)
def test_exact_loss_reproduces_the_plain_reduction(n_v, frac, C):
    """``_zero_weight_exact_loss`` (reference backend) against the plain path's reduction over the valid rows: the
    gradient rows' terms kept as the split's chunk loop keeps them (chunk 0 in ``term``, chunk ``c >= 1`` in the
    ``loss_weights`` rows of chunk ``c - 1``), the zero-weight slots holding the weights (``+0`` / ``-0``); the same
    ``loss_reduce`` over the plain chunks gives the same bits, the sign of an all-zero sum included."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + n_v)
    T = n_v + n_v // 3
    labels = torch.full((T,), -100, dtype=torch.int64)
    labels[torch.randperm(T, generator=gen)[:n_v]] = 1
    valid = labels >= 0
    zero = valid & (torch.rand(T, generator=gen) < frac)
    if frac == 1.0:
        zero = valid.clone()
    w = torch.randn(T, generator=gen)
    w[zero] = 0.0
    w[zero.nonzero().squeeze(1)[::2]] = -0.0
    # the plain path's terms per valid row: the gradient rows' values, the zero-weight rows' w * clipped = +-0
    term_full = (torch.randn(T, generator=gen) * 3).float()
    term_full[zero] = w[zero]
    vrows = valid.nonzero().squeeze(1)
    plain_terms = term_full[vrows]
    problem = SimpleNamespace(chunk=C, mode=cake_backend.MODE_POLICY, loss_div=None)
    acc = torch.empty(1, dtype=torch.float64)
    plain = torch.empty(1, dtype=torch.float32)
    chunks = cake_backend.plan_chunks(n_v, C)
    for i, (row0, rows_c) in enumerate(chunks):
        cake_backend.ReferenceEngine.loss_reduce(
            dict(
                term=plain_terms[row0 : row0 + rows_c],
                loss_acc=acc,
                loss_out=plain,
                rows_c=rows_c,
                first_chunk=int(i == 0),
                last_chunk=int(i == len(chunks) - 1),
                mode=cake_backend.MODE_POLICY,
                loss_div=1.0,
            )
        )
    g = (valid & ~zero).nonzero().squeeze(1)
    z = zero.nonzero().squeeze(1)
    n_g = int(g.numel())
    tg = term_full[g]
    head = min(C, n_g)
    term_buf = torch.full((max(head, 1),), float("nan"))
    term_buf[:head] = tg[:head]
    w_buf = torch.full((max(n_g, 1),), float("nan"))
    w_buf[: n_g - head] = tg[head:]
    got = cake_backend._zero_weight_exact_loss(
        problem,
        dict(term=term_buf, loss_weights=w_buf) if n_g else None,
        g,
        z,
        w,
        backend="reference",
    )
    assert _same_bits(got, plain), (float(got), float(plain))
    if n_g:
        # the regrouped sum of the same terms is the same value to FP64 rounding, not necessarily the same bits
        assert math.isclose(
            float(got), -float(tg.double().sum()), rel_tol=1e-6, abs_tol=1e-6
        )


ZERO_KINDS_HOST = ["one", ("random", 0.166), ("group", 0.5), "all"]


@pytest.mark.parametrize("mode", ["exact", "regroup"])
@pytest.mark.parametrize("kind", ZERO_KINDS_HOST, ids=str)
@pytest.mark.parametrize("T, C", [(37, 16), (1200, 256)])
def test_reference_zero_weight_rows(kind, T, C, mode):
    """The reference backend with zero-weight rows (``+0`` and ``-0``, ignored rows mixed in, ragged multi-chunk
    calls): the deterministic suite's gates against the FP64 oracle; the zero-weight rows' ``dX`` rows are exact
    ``+0``; ``logp`` / ``loss`` / ``dX`` / ``dW`` agree with the call without the split at the reference GEMMs'
    precision (the reference GEMMs may round differently for another ``M``); the memory report records the split."""
    base = _host_inputs(T)
    if kind == ("group", 0.5) and T < 1024:
        kind = ("random", 0.5)  # one 512-row group: the fraction rounds to no group
    zero = _zero_weight_set(base, kind)
    inp = _with_zero_weights(base, zero, negative_zero=True)
    with _zero_weight_floor(0), _loss_mode(mode):
        split = _loss_call(inp, C, backend="reference")
        fr = cake_backend.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=C,
            backend="reference",
        )
    with _zero_weight_floor(NO_SPLIT):
        plain = _loss_call(inp, C, backend="reference")
    valid = inp.labels >= 0
    assert fr.memory["zero_weight_rows"] == int(zero.sum())
    assert fr.memory["zero_weight_loss_mode"] == mode
    assert ("zero_weight_exact_loss_temporary_bytes" in fr.memory) == (mode == "exact")
    assert _all_plus_zero(split["dX"][zero])
    assert bool((split["logp"][~valid] == 0).all())
    torch.testing.assert_close(split["logp"], plain["logp"], atol=2e-5, rtol=1e-5)
    if bool(zero[valid].all()):
        assert float(split["loss"]) == 0.0 and _same_bits(split["loss"], plain["loss"])
        assert not bool(split["dX"].any()) and not bool(split["dW"].any())
        return
    _check_against_references(split, inp)
    assert math.isclose(
        float(split["loss"]), float(plain["loss"]), rel_tol=1e-5, abs_tol=1e-6
    )
    assert rel_l2(split["dX"].float(), plain["dX"].float()) <= 1e-3
    assert rel_l2(split["dW"].float(), plain["dW"].float()) <= 1e-3


def test_reference_zero_weight_rows_without_return_logp():
    """``return_logp=False``: the logp-only pass does not run (``forward_loss`` leaves the zero-weight rows' ``logp``
    at 0); ``loss`` and the gradients are the ``return_logp=True`` call's bits."""
    base = _host_inputs(200)
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero)
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
        backend="reference",
    )
    with _zero_weight_floor(0):
        with_logp = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
        without = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, return_logp=False, **kw
        )
        a = _loss_call(inp, 64, backend="reference")
        b = _loss_call(inp, 64, backend="reference", return_logp=False)
    assert bool((without.logp[zero] == 0).all()) and bool(
        (with_logp.logp[zero] != 0).all()
    )
    assert _same_bits(without.logp[~zero], with_logp.logp[~zero])
    assert _same_bits(without.loss, with_logp.loss)
    assert with_logp.memory["zero_weight_rows"] == int(zero.sum())
    assert without.memory["zero_weight_logp_temporary_bytes"] == 0
    for key in ("loss", "dX", "dW"):
        assert _same_bits(a[key], b[key]), key


def test_reference_zero_weight_rows_do_not_propagate_non_finite_values():
    """A non-finite ``X`` row or ``infer_logp`` in a zero-weight row does not reach ``loss`` / ``dX`` / ``dW`` (the
    row is treated like an ignored row for the gradients; its exact-loss slot is its weight); a NaN weight keeps its
    row in the gradient path."""
    base = _host_inputs(200)
    zero = _zero_weight_set(base, ("random", 0.2))
    clean = _with_zero_weights(base, zero)
    rows = zero.nonzero().squeeze(1)
    X = clean.X.clone()
    X[rows[0]] = float("inf")
    infer = clean.infer_logp.clone()
    infer[rows[1]] = float("nan")
    poisoned = dataclasses.replace(clean, X=X, infer_logp=infer)
    with _zero_weight_floor(0):
        a = _loss_call(clean, 64, backend="reference")
        b = _loss_call(poisoned, 64, backend="reference")
        for key in ("loss", "dW"):
            assert _same_bits(a[key], b[key]), key
        assert _same_bits(a["dX"][~zero], b["dX"][~zero]) and _all_plus_zero(
            b["dX"][zero]
        )
        assert _same_bits(a["logp"][rows[2:]], b["logp"][rows[2:]])
        w = clean.loss_weights.clone()
        w[int((~zero & (clean.labels >= 0)).nonzero()[0])] = float("nan")
        nan_row = _loss_call(
            dataclasses.replace(clean, loss_weights=w), 64, backend="reference"
        )
    assert math.isnan(float(nan_row["loss"]))


def test_zero_weight_split_not_applied():
    """The plain path runs (every valid row in the chunk loop, no split in the memory report): ``compact_rows=False``,
    a call below the floor, the opt-in module's caller-given ``num_valid_rows``; the log-probability entry and the
    prepared runner take no weights; the loss entry has no temperature keyword (no temperature form of the split)."""
    base = _host_inputs(200)
    inp = _with_zero_weights(base, _zero_weight_set(base, ("random", 0.3)))
    valid = int((inp.labels >= 0).sum())
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
        backend="reference",
    )
    with _zero_weight_floor(0):
        unc = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, compact_rows=False, **kw
        )
        assert "zero_weight_rows" not in unc.memory and unc.row_index is None
        given = opt_in.forward_loss(
            inp.X, inp.W, inp.labels, num_valid_rows=valid, **kw
        )
        assert "zero_weight_rows" not in given.memory
        assert given.row_index is None or int(given.row_index.numel()) == valid
        split = opt_in.forward_loss(inp.X, inp.W, inp.labels, **kw)
        assert split.memory["zero_weight_rows"] == int(
            (inp.loss_weights == 0)[inp.labels >= 0].sum()
        )
    with _zero_weight_floor(201):
        below = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
        assert "zero_weight_rows" not in below.memory
        assert int(below.row_index.numel()) == valid
    for fn in (
        cake_backend.cake_lm_head_det_train_loss,
        cake_backend.forward_loss,
        opt_in.cake_lm_head_det_train_loss,
        opt_in.forward_loss,
    ):
        assert "temperature" not in inspect.signature(fn).parameters
    assert (
        "loss_weights" not in inspect.signature(cake_backend.forward_logprob).parameters
    )


def test_zero_weight_rows_report_the_combined_peak():
    """A loss call with zero-weight rows reports ``peak_temporary_bytes`` as the larger of the chunk loop's
    temporaries and the logp-only pass's phase -- its own temporaries plus the compaction outputs it coexists with,
    plus chunk 0's buffers when it ran while the hidden valid-row count held them -- and, in the exact mode, the exact
    loss's temporaries (``4 T_v`` term array, then the larger of ``8 |G|`` and ``12 |Z|``)."""
    temporary = dict(logits=1000, stats=100, x_c=200, row_index=8, row_valid=4, d=16)
    memory = dict(temporary=temporary, temporary_bytes=1328, peak_temporary_bytes=1328)
    zero_index = torch.tensor([1, 3])
    row_index = torch.tensor([0, 2, 4])

    def report(zero_temp, **kw):
        result = SimpleNamespace(
            memory=dict(memory), logp=torch.zeros(5), row_index=row_index
        )
        logp = torch.tensor([-1.0, -2.0])
        out = cake_backend._with_zero_weight_rows(
            result, zero_index, (logp, {"temporary_bytes": zero_temp}), **kw
        )
        assert out.logp.tolist() == [0.0, -1.0, 0.0, -2.0, 0.0]
        return out.memory

    after = report(1250)
    assert (
        after["zero_weight_rows"] == 2 and after["zero_weight_loss_mode"] == "regroup"
    )
    assert after["zero_weight_logp_temporary_bytes"] == 1250
    assert after["zero_weight_logp_phase_bytes"] == 1250 + 12
    assert after["peak_temporary_bytes"] == 1328 and after["temporary_bytes"] == 1328
    assert "zero_weight_exact_loss_temporary_bytes" not in after
    assert report(2000)["peak_temporary_bytes"] == 2012
    first = report(1250, chunk0_alive=True)
    assert first["zero_weight_logp_phase_bytes"] == 1250 + 12 + 1300
    assert first["peak_temporary_bytes"] == 2562
    deferred = cake_backend._with_zero_weight_rows(
        SimpleNamespace(
            memory=dict(memory, saved_dz_bytes=700),
            logp=torch.zeros(5),
            row_index=row_index,
        ),
        zero_index,
        (torch.tensor([-1.0, -2.0]), {"temporary_bytes": 1250}),
        deferred_alive=True,
    ).memory
    assert deferred["zero_weight_logp_phase_bytes"] == 1250 + 12 + 700
    assert deferred["peak_temporary_bytes"] == 1962
    exact = report(1250, exact=True)
    assert exact["zero_weight_loss_mode"] == "exact"
    assert exact["zero_weight_exact_loss_temporary_bytes"] == 4 * 5 + max(8 * 3, 12 * 2)
    skipped = cake_backend._with_zero_weight_rows(
        SimpleNamespace(memory=dict(memory), logp=torch.zeros(5), row_index=row_index),
        zero_index,
        None,
    ).memory
    assert skipped["zero_weight_logp_temporary_bytes"] == 0
    assert skipped["peak_temporary_bytes"] == 1328
    assert "zero_weight_logp_phase_bytes" not in skipped
    untouched = SimpleNamespace(
        memory=dict(memory), logp=torch.zeros(5), row_index=row_index
    )
    assert cake_backend._with_zero_weight_rows(untouched, None, None) is untouched
    assert untouched.memory == memory


def test_memory_plan_models_the_split():
    """``memory_plan``'s zero-weight rows: the budget counts every valid row while the model's chunk loop runs over
    the gradient rows; the logp-only pass and the exact loss's term phase are phases of the predicted peak (never
    below the plan of the same split without them: ``zero_rows=0``, no exact loss); inconsistent counts are refused
    (the budget's valid rows are at most the gradient rows plus the zero-weight rows)."""
    kw = dict(num_rows=16384, hidden=6144, vocab=154880, chunk=4096, need_dx=True)
    base = opt_in.memory_plan(
        "loss", rows=12000, budget_rows=15000, zero_rows=0, zero_total=3000, **kw
    )
    with_pass = opt_in.memory_plan(
        "loss", rows=12000, budget_rows=15000, zero_rows=3000, zero_total=3000, **kw
    )
    with_exact = opt_in.memory_plan(
        "loss",
        rows=12000,
        budget_rows=15000,
        zero_rows=3000,
        zero_total=3000,
        exact_loss=True,
        **kw,
    )
    assert with_pass.budget_bytes == base.budget_bytes == with_exact.budget_bytes
    assert with_pass.predicted_bytes >= base.predicted_bytes
    assert with_exact.predicted_bytes >= with_pass.predicted_bytes
    with pytest.raises(ValueError):
        opt_in.memory_plan("loss", rows=12000, budget_rows=15001, zero_rows=3000, **kw)
    with pytest.raises(ValueError):
        opt_in.memory_plan("loss", rows=12000, zero_rows=3000, zero_total=2000, **kw)
    with pytest.raises(ValueError):
        opt_in.memory_plan(
            "loss", rows=14000, zero_rows=3000, **kw
        )  # rows + zero_total > num_rows


# The memory rule's two host pieces: the split's dX K-slice cap and the guard (:func:`cake_backend.zero_weight_plan`).
# A production-like dX slice rule: 148 SMs, the two-CTA cluster's co-resident count, every slice count registered.
SLICE_SMS = 148


def _slice_rule_args(hidden, vocab):
    g = cake_backend.DEFAULT_GEOMETRY
    return dict(
        hidden=hidden,
        vocab=vocab,
        num_sms=SLICE_SMS,
        dx_max=cake_backend.DX_K_SLICES_MAX,
        geometry=g,
        resident=cake_backend.cluster_resident(None, g.dx_cluster_ctas, SLICE_SMS),
    )


def test_capped_k_slices():
    """``capped_k_slices``: unchanged without a cap or at / below it; above it, the cap fitted to the K steps (a count
    the reduction form runs, never above the cap)."""
    g = cake_backend.DEFAULT_GEOMETRY
    for vocab in (512, 1024, 151552):
        k_iters = vocab // g.k_block
        for k in range(1, cake_backend.DX_K_SLICES_MAX + 1):
            assert cake_backend.capped_k_slices(k, None, vocab, g) == k
            for cap in range(1, cake_backend.DX_K_SLICES_MAX + 1):
                got = cake_backend.capped_k_slices(k, cap, vocab, g)
                if k <= cap:
                    assert got == k
                else:
                    assert 1 <= got <= cap
                    assert cake_backend.dx_slices_fit(got, k_iters) == got


def test_dx_slices_cap_keeps_the_plain_maximum():
    """The split's chunk loop under ``dx_slices_cap`` = the largest dX slice count of the same call without the split:
    calls whose split tail chunk would pick more slices than any chunk of the plain call exist (a small tail chunk
    takes more K slices), and capped, no chunk exceeds the plain maximum, the K-slice workspace never exceeds the plain
    plan's, the chunks the cap does not touch keep their counts and :func:`dx_slices_max` predicts the plan."""
    C = 4096
    found = 0
    for hidden, vocab in ((4096, 151552), (6144, 154880), (2048, 65536)):
        args = _slice_rule_args(hidden, vocab)
        for n_v in (2 * C, 3 * C, 4 * C - 512, 5 * C):
            plain_max = cake_backend.dx_slices_max(n_v, chunk=C, **args)
            for n_g in range(C + 1, n_v, 97):
                if cake_backend.dx_slices_max(n_g, chunk=C, **args) <= plain_max:
                    continue
                found += 1
                problem = dataclasses.replace(
                    _policy_problem(64, C), num_rows=n_v, hidden=hidden, vocab=vocab
                )
                kw = dict(
                    need_dx=True,
                    need_dw=True,
                    geometry=args["geometry"],
                    dx_max_slices=args["dx_max"],
                    num_sms=SLICE_SMS,
                    dx_resident=args["resident"],
                )
                plain = cake_backend.make_plan(problem, valid_rows=n_v, **kw)
                free = cake_backend.make_plan(problem, valid_rows=n_g, **kw)
                capped = cake_backend.make_plan(
                    problem, valid_rows=n_g, dx_slices_cap=plain_max, **kw
                )
                assert max(plain.dx_slices) == plain_max
                assert max(free.dx_slices) > plain_max
                assert max(capped.dx_slices) <= plain_max
                assert capped.dx_ws_slabs <= plain.dx_ws_slabs
                assert capped.chunks == free.chunks
                # one slice count per chunk in both plans (the same chunks, asserted above)
                for k_free, k_cap in zip(free.dx_slices, capped.dx_slices, strict=True):
                    assert k_cap == cake_backend.capped_k_slices(
                        k_free, plain_max, vocab, args["geometry"]
                    )
                    assert k_cap == k_free or k_free > plain_max
                    assert k_cap <= plain_max
                assert max(capped.dx_slices) == cake_backend.dx_slices_max(
                    n_g, chunk=C, cap=plain_max, **args
                )
    assert found, (
        "no call found whose split would run more dX K slices than the plain call (re-check the grid)"
    )


def test_dx_slices_cap_leaves_an_explicit_knob():
    """An explicit ``k_slices`` knob is the same in the split and the plain call: the cap does not touch it (the
    guard's slice rule, :func:`dx_slices_max`, follows :func:`make_plan`)."""
    args = _slice_rule_args(4096, 151552)
    for rows in (1, 4097, 3 * 4096 + 5):
        assert (
            cake_backend.dx_slices_max(rows, chunk=4096, explicit_k=4, cap=1, **args)
            == 4
        )
        assert cake_backend.dx_slices_max(rows, chunk=4096, explicit_k=1, **args) == 1


# A shape where the logp-only pass ahead of a hidden-count loop (chunk 0's buffers alive) outgrows the plain call: a
# small H leaves the plain call's dW_acc too small to cover the pass's chunk buffers.
GUARD_CASE = dict(
    num_rows=10000,
    grad_rows=6000,
    zero_rows=2000,
    hidden=256,
    vocab=65536,
    chunk=4096,
    need_dx=True,
    need_dw=True,
    fuse_dw_cast=True,
    dx_finalize=True,
    return_logp=True,
    exact_loss=True,
)


def test_zero_weight_memory_guard():
    """The guard's byte model (host arithmetic): the split runs only when its predicted peak stays within the plain
    call's.  The constructed hidden-count case (pass ahead of the loop beside chunk 0's buffers, or after it beside the
    deferred chunk's ``dz`` rows) predicts above the plain peak: no split; the same call on the host count path (the
    pass ahead of the loop, nothing of the loop allocated) splits.  Deterministic: the same inputs, the same answer."""
    hc = cake_backend.zero_weight_memory(hidden_count=True, **GUARD_CASE)
    assert hc["split_bytes"] > hc["plain_bytes"] and hc["split"] is False
    assert hc["split_bytes"] == max(hc["phases"].values())
    assert set(hc["phases"]) == {"loop", "pass", "exact"}
    assert hc == cake_backend.zero_weight_memory(hidden_count=True, **GUARD_CASE)
    host = cake_backend.zero_weight_memory(hidden_count=False, **GUARD_CASE)
    assert host["split"] is True and host["split_bytes"] <= host["plain_bytes"]
    assert host["zero_first"] is True and host["orders"] == {}
    # without return_logp no pass phase; the split's loop stays within the plain loop
    quiet = cake_backend.zero_weight_memory(
        hidden_count=True, **dict(GUARD_CASE, return_logp=False)
    )
    assert "pass" not in quiet["phases"] and quiet["zero_first"] is False
    assert quiet["split"] is True
    # the K-slice slabs: a split that would take more slabs than the plain call predicts more bytes
    more = cake_backend.zero_weight_memory(
        hidden_count=False,
        **dict(GUARD_CASE, exact_loss=False),
        plain_dx_ws_slabs=1,
        split_dx_ws_slabs=3,
    )
    same = cake_backend.zero_weight_memory(
        hidden_count=False,
        **dict(GUARD_CASE, exact_loss=False),
        plain_dx_ws_slabs=1,
        split_dx_ws_slabs=1,
    )
    assert more["phases"]["loop"] - same["phases"]["loop"] == 2 * 4 * 256 * 4096
    for bad in (dict(zero_rows=0), dict(grad_rows=-1), dict(grad_rows=9000)):
        with pytest.raises(ValueError):
            cake_backend.zero_weight_memory(**dict(GUARD_CASE, **bad))


@pytest.mark.parametrize("hidden, first", [(256, True), (8, False)])
def test_zero_weight_memory_orders_the_pass(hidden, first):
    """With the hidden count, a weight gradient and the fused cast the pass runs in the order of the lower predicted
    peak (ahead of the loop on a tie): ahead beside chunk 0's buffers when the accumulators it would meet after the
    loop are larger (a real ``H``), after the loop when chunk 0's statistics and gather buffer are (a synthetic tiny
    ``H``, the branch the rule takes there).  Without the fused cast or without gradient rows it runs after the loop,
    without the hidden count ahead of it while the loop defers its last chunk, else after it."""
    case = dict(GUARD_CASE, hidden=hidden)
    m = cake_backend.zero_weight_memory(hidden_count=True, **case)
    assert set(m["orders"]) == {"first", "after"}
    assert m["zero_first"] is (not m["orders"]["after"] < m["orders"]["first"])
    assert m["zero_first"] is first
    assert m["split_bytes"] == min(m["orders"].values())
    unfused = cake_backend.zero_weight_memory(
        hidden_count=True, **dict(case, fuse_dw_cast=False)
    )
    assert unfused["zero_first"] is False and unfused["orders"] == {}
    empty = cake_backend.zero_weight_memory(
        hidden_count=True, **dict(case, grad_rows=0)
    )
    assert empty["zero_first"] is False and "loop" not in empty["phases"]
    assert (
        cake_backend.zero_weight_memory(hidden_count=False, **case)["zero_first"]
        is True
    )
    # without the hidden count and without a deferred last chunk (no weight gradient, e.g. the gradient-free first
    # forward of a checkpointed step) the pass runs after the loop, in the loop's freed chunk buffers
    for grads in (dict(need_dx=False, need_dw=False), dict(need_dw=False)):
        m = cake_backend.zero_weight_memory(hidden_count=False, **dict(case, **grads))
        assert m["zero_first"] is False and m["orders"] == {}


def test_zero_weight_memory_pass_arena():
    """``pass_arena`` (the templated binding): the pass ahead of a loop that keeps ``dW_acc`` takes its chunk buffers
    from that FP32 accumulator when they fit, so that phase holds ``dW_acc`` and the pass's other buffers (the order
    after the loop is unchanged), with and without the hidden count; they do not fit at ``H`` 256 (``dW_acc`` 64 MiB):
    the same model.  Without a weight gradient the pass has its own buffers."""
    case = dict(GUARD_CASE, hidden=4096, exact_loss=False)
    own = cake_backend.zero_weight_memory(hidden_count=True, **case)
    arena = cake_backend.zero_weight_memory(hidden_count=True, pass_arena=True, **case)
    layout = cake_backend.workspace_layout(
        2000, 65536, 4096, entry="logprob", hidden=4096, compact=True
    )
    carved = cake_backend._carved_bytes(layout)
    dw_acc = 65536 * 4096 * 4
    assert 0 < carved <= dw_acc and own["pass_arena"] is False
    assert arena["orders"]["after"] == own["orders"]["after"]
    assert arena["orders"]["first"] >= own["orders"]["first"]
    assert arena["pass_arena"] is arena["zero_first"]
    if arena["zero_first"] and own["zero_first"]:
        assert arena["phases"]["pass"] - own["phases"]["pass"] == dw_acc - carved
    for other in (GUARD_CASE, dict(case, need_dw=False)):
        plain = cake_backend.zero_weight_memory(hidden_count=True, **other)
        with_arena = cake_backend.zero_weight_memory(
            hidden_count=True, pass_arena=True, **other
        )
        assert with_arena == plain and plain["pass_arena"] is False
    host_own = cake_backend.zero_weight_memory(hidden_count=False, **case)
    host = cake_backend.zero_weight_memory(hidden_count=False, pass_arena=True, **case)
    assert host["pass_arena"] is True and host["zero_first"] is True
    assert host["phases"]["pass"] - host_own["phases"]["pass"] == dw_acc - carved
    assert host["phases"]["loop"] == host_own["phases"]["loop"]


def test_zero_weight_plan_reference_backend():
    """``zero_weight_plan`` on the reference backend (one dX slice): no cap; the guard's answer is the byte model's."""
    problem = _policy_problem(200, 64)
    plan = cake_backend.zero_weight_plan(
        problem,
        torch.device("cpu"),
        grad_rows=120,
        zero_rows=60,
        backend="reference",
        need_dx=True,
        need_dw=True,
        fuse_dw_cast=True,
        dx_finalize=True,
        hidden_count=False,
        return_logp=True,
        exact_loss=True,
        tuning={},
    )
    assert plan["dx_slices_cap"] is None
    assert plan["plain_dx_slices"] == plan["split_dx_slices"] == 1
    assert plan["split"] is (plan["split_bytes"] <= plan["plain_bytes"])


def test_valid_rows_of_count_matches_valid_rows():
    """The declined split's plain index, formed on the device from the known count: :func:`valid_rows`' index and
    mask (``None`` when every row is valid)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED)
    for T in (5, 300, 2000):
        labels = torch.randint(0, 50, (T,), generator=gen).to(DEV)
        labels[torch.rand(T, generator=gen).to(DEV) < 0.2] = -100
        count = int((labels >= 0).sum())
        idx, mask = cake_backend._valid_rows_of_count(labels, count, mask=True)
        ref_idx, ref_mask = cake_backend.valid_rows(labels, mask=True)
        # no ignored row drawn (a short call): both take the uncompacted path
        if ref_idx is None:
            assert idx is None and mask is None and ref_mask is None
            continue
        assert idx.tolist() == ref_idx.tolist() and torch.equal(mask, ref_mask)
    full = torch.arange(7, device=DEV)
    assert cake_backend._valid_rows_of_count(full, 7, mask=True) == (None, None)


@pytest.mark.parametrize("module_name", ["main", "opt_in"])
def test_declined_split_runs_the_plain_call(monkeypatch, module_name):
    """A call the guard declines (the byte model forced to predict a larger split) runs the plain call: no split in
    the memory report, every valid row in the chunk loop, and ``loss`` / ``logp`` / ``dX`` / ``dW`` the bits of the
    same call below the floor (the reference backend: the same launches)."""
    module = cake_backend if module_name == "main" else opt_in
    base = _host_inputs(1200)
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero)
    model = cake_backend.zero_weight_memory

    def declined(**kw):
        out = model(**kw)
        return dict(out, split=False, split_bytes=out["plain_bytes"] + 1)

    monkeypatch.setattr(cake_backend, "zero_weight_memory", declined)
    with _zero_weight_floor(0):
        fr = module.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=256,
            backend="reference",
        )
        declined_call = _loss_call(inp, 256, backend="reference", module=module)
    monkeypatch.setattr(cake_backend, "zero_weight_memory", model)
    with _zero_weight_floor(NO_SPLIT):
        plain = _loss_call(inp, 256, backend="reference", module=module)
    assert "zero_weight_rows" not in fr.memory
    assert int(fr.row_index.numel()) == int((inp.labels >= 0).sum())
    for key in ("loss", "logp", "dX", "dW"):
        assert _same_bits(declined_call[key], plain[key]), key


@pytest.mark.parametrize("module_name", ["main", "opt_in"])
def test_all_zero_weight_split_result(module_name):
    """A split whose valid rows all have zero weight (no gradient GEMM): the result keeps the call's backend (the
    reference backend's backward runs on any device), its ``dw_acc`` holds no FP32 ``[V, H]`` storage, and
    :meth:`ForwardResult.backward` forms ``dW`` in the output dtype with the bits :func:`scale_cast` gives for a zero
    FP32 accumulator (``+0``; ``-0`` for a negative upstream scalar); the autograd entry's ``dX`` / ``dW`` are ``+0``."""
    module = cake_backend if module_name == "main" else opt_in
    base = _host_inputs(1200)
    inp = _with_zero_weights(base, base.labels >= 0)
    with _zero_weight_floor(0):
        fr = module.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=256,
            backend="reference",
        )
        out = _loss_call(inp, 256, backend="reference", module=module)
    assert "zero_weight_rows" in fr.memory and int(fr.row_index.numel()) == 0
    assert fr.backend == "reference"
    assert tuple(fr.dw_acc.shape) == (V_HOST, H_HOST)
    assert fr.dw_acc.untyped_storage().nbytes() <= 4
    assert fr.memory["accumulators"]["dW_acc"] == 4
    assert fr.memory["accumulator_bytes"] == sum(fr.memory["accumulators"].values())
    zero_acc = torch.zeros((V_HOST, H_HOST), dtype=torch.float32, device=DEV)
    f32_max = torch.finfo(torch.float32).max  # finite, beyond the BF16 range
    for g in (None, 2.0, -2.0, f32_max, -f32_max, math.inf, math.nan):
        grad = None if g is None else torch.tensor(g, dtype=torch.float32, device=DEV)
        dx, dw = fr.backward(grad)
        assert dw.dtype == torch.bfloat16 and tuple(dw.shape) == (V_HOST, H_HOST)
        ref = cake_backend.scale_cast(
            zero_acc, grad, torch.bfloat16, backend="reference"
        )
        if g is None or math.isfinite(g):
            negative = g is not None and g < 0
            assert _same_bits(dw, ref), g
            assert bool((dw == 0).all()), g
            assert bool((torch.signbit(dw) == negative).all()), g
            assert _all_plus_zero(dx)
        else:
            assert bool(torch.isnan(dw).all() and torch.isnan(ref).all()), g
    _, dw32 = fr.backward(grad_weight_dtype=torch.float32)
    assert dw32.dtype == torch.float32 and _all_plus_zero(dw32)
    assert _all_plus_zero(out["loss"])
    assert out["dW"].dtype == torch.bfloat16 and _all_plus_zero(out["dW"])
    assert _all_plus_zero(out["dX"])


@pytest.mark.parametrize("module_name", ["main", "opt_in"])
@pytest.mark.parametrize("backend", ["reference", "cake"])
def test_zero_row_result(module_name, backend):
    """A call without rows (no binding, no launch): the result keeps the call's backend, its memory report counts the
    zero ``dw_acc``'s one FP32 element (not an FP32 ``[V, H]`` buffer), and the backward's ``dW`` is the FP32 product
    ``0 * g`` cast once (``+0``; ``-0`` for a negative ``g``, also beyond the BF16 range)."""
    module = cake_backend if module_name == "main" else opt_in
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16, device=DEV)
    X0 = torch.zeros(0, H_HOST, dtype=torch.bfloat16, device=DEV)
    labels0 = torch.zeros(0, dtype=torch.int64, device=DEV)
    fr = module.forward_loss(
        X0, W, labels0, objective="ce", loss_div=1.0, backend=backend
    )
    assert fr.backend == backend
    assert fr.dw_acc.untyped_storage().nbytes() <= 4
    assert fr.memory["accumulators"] == {"dX_acc": 0, "dW_acc": 4}
    assert fr.memory["accumulator_bytes"] == 4
    f32_max = torch.finfo(torch.float32).max
    for g in (None, 2.0, f32_max, -f32_max):
        grad = None if g is None else torch.tensor(g, dtype=torch.float32, device=DEV)
        dx, dw = fr.backward(grad)
        assert tuple(dx.shape) == (0, H_HOST) and dx.dtype == torch.bfloat16
        assert dw.dtype == torch.bfloat16 and tuple(dw.shape) == (V_HOST, H_HOST)
        if g is None or g > 0:
            assert _all_plus_zero(dw), g
        else:
            assert bool((dw == 0).all() and torch.signbit(dw).all()), g


@pytest.mark.parametrize("module_name", ["main", "opt_in"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_empty_autograd_scales_the_zero_weight_gradient(module_name, checkpointed):
    """A loss call without gradient rows (``T == 0``; every row ignored), plain or under a non-reentrant checkpoint:
    the autograd backward's ``dW`` has the bits :meth:`ForwardResult.backward` gives for the same upstream scalar (the
    FP32 product ``0 * g`` cast once: ``+0``; ``-0`` for a negative ``g``, also beyond the BF16 range; NaN for a
    non-finite one), and ``dX`` is ``+0``.  The opt-in arm passes ``num_valid_rows=0`` so its own autograd Function
    runs (without an opt-in keyword the call is ``cake_backend``'s)."""
    from torch.utils.checkpoint import checkpoint

    module = cake_backend if module_name == "main" else opt_in
    extra = {} if module_name == "main" else dict(num_valid_rows=0)
    W0 = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16, device=DEV)
    f32_max = torch.finfo(torch.float32).max
    for T in (0, 37):
        X0 = torch.zeros(T, H_HOST, dtype=torch.bfloat16, device=DEV)
        labels = torch.full((T,), -100, dtype=torch.int64, device=DEV)
        kw = dict(objective="ce", loss_div=1.0, backend="reference", **extra)
        fr = module.forward_loss(X0, W0, labels, **kw)

        def call(X, W):
            return module.cake_lm_head_det_train_loss(X, W, labels, **kw)

        for g in (2.0, -2.0, f32_max, -f32_max, math.inf, math.nan):
            grad = torch.tensor(g, dtype=torch.float32, device=DEV)
            X = X0.clone().requires_grad_()
            W = W0.clone().requires_grad_()
            loss = (
                checkpoint(call, X, W, use_reentrant=False)
                if checkpointed
                else call(X, W)
            )
            assert loss.grad_fn._forward_cls is module.CakeLMHeadDetLossFunction
            loss.backward(grad)
            _, ref = fr.backward(grad)
            assert W.grad.dtype == ref.dtype == torch.bfloat16, (T, g)
            if math.isfinite(g):
                assert _same_bits(W.grad, ref), (T, g)
                assert bool((torch.signbit(W.grad) == (g < 0)).all()), (T, g)
            else:
                both_nan = torch.isnan(W.grad).all() and torch.isnan(ref).all()
                assert bool(both_nan), (T, g)
            assert tuple(X.grad.shape) == (T, H_HOST), (T, g)
            assert _all_plus_zero(X.grad), (T, g)


def test_memory_plan_counts_the_exact_loss_phases():
    """``memory_plan``'s exact-loss phase (``_exact_loss_bytes``): the term array, the zero-weight rows' gathered
    weights and index and, when their pass ran first, their compact ``logp``, plus the larger of the gradient rows'
    positions with the two kept term buffers and the zero-weight rows' positions; the plan's predicted peak takes it."""
    live = opt_in._allocated_bytes
    for R, Cb, Z, Zt in (
        (12000, 4096, 3000, 3000),
        (12000, 4096, 0, 3000),
        (500, 500, 60000, 60000),
    ):
        held = live(4 * (R + Zt)) + live(4 * Zt) + live(8 * Zt) + live(4 * Z)
        expect = held + max(live(4 * R) + live(4 * Cb) + live(8 * R), live(8 * Zt))
        assert opt_in._exact_loss_bytes(R, Cb, zero_rows=Z, zero_total=Zt) == expect
    kw = dict(num_rows=16384, hidden=6144, vocab=154880, chunk=4096, need_dx=True)
    huge = 1 << 50
    original = opt_in._exact_loss_bytes
    try:
        opt_in._exact_loss_bytes = lambda *a, **k: huge
        plan = opt_in.memory_plan(
            "loss",
            rows=12000,
            budget_rows=15000,
            zero_rows=3000,
            zero_total=3000,
            exact_loss=True,
            **kw,
        )
    finally:
        opt_in._exact_loss_bytes = original
    assert plan.predicted_bytes >= huge


# ---------------------------------------------------------------------------
# Device (a supported GPU and the generated program)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def glm_weight():
    _require_program(entry="loss")
    return make_inputs(1, seed=SEED, device=CUDA, ignore_frac=0.0).W


def _device_inputs(T, *, W, seed=SEED, ignore_frac=0.05, ld_pad=0):
    return make_inputs(
        T,
        objective="policy",
        seed=seed + T,
        device=CUDA,
        W=W,
        ignore_frac=ignore_frac,
        ld_pad=ld_pad,
    )


# (T, zero rows): one row, 16.6 % at random and in groups of 512, 30 / 50 % at random and in groups, every valid row;
# multi-chunk calls (C 4096) with ragged tails, 5 % ignored rows mixed in; T > 3 chunks on the hidden valid-row count
# path, T = 4097 / 8191 on the host count path.
ZERO_DEVICE_CASES = [
    (4097, "one"),
    (12289, ("group", 0.166)),
    (12289, ("random", 0.166)),
    (8191, ("random", 0.3)),
    (12289, ("group", 0.5)),
    (4097, ("random", 0.5)),
    (16231, ("random", 0.9)),
    (12289, "all"),
]


@pytest.mark.parametrize("hidden_count", [None, False])
@pytest.mark.parametrize("T, kind", ZERO_DEVICE_CASES, ids=str)
def test_device_exact_loss_and_logp_match_the_plain_path(
    glm_weight, T, kind, hidden_count
):
    """The split in the exact mode against the same call without it: ``loss`` and ``logp`` (every row; the
    zero-weight rows' from the logp-only pass) bit for bit; the zero-weight rows' ``dX`` rows exact ``+0``; the
    gradient rows' ``dX`` rows bit for bit where their chunk runs the same dX form and K-slice count, the rest within
    the gates; ``dW`` within the deterministic suite's gates (the K reduction over other chunk rows).  Every valid row
    of zero weight: the plain loss (``+0``), zero gradients, no gradient GEMM."""
    _require_program(entry="loss")
    base = _device_inputs(T, W=glm_weight)
    zero = _zero_weight_set(base, kind)
    inp = _with_zero_weights(base, zero, negative_zero=True)
    C = 4096
    switches = {} if hidden_count is None else dict(hidden_count=hidden_count)
    cake_backend.BINDING_CACHE.clear()
    with _zero_weight_floor(NO_SPLIT), _recording_loss_plans() as plain_plans:
        plain = _loss_call(inp, C, **switches)
    with _zero_weight_floor(0), _recording_loss_plans() as split_plans:
        split = _loss_call(inp, C, **switches)
    valid = inp.labels >= 0
    assert _same_bits(split["loss"], plain["loss"]), (
        float(split["loss"]),
        float(plain["loss"]),
    )
    assert _same_bits(split["logp"], plain["logp"])
    assert _all_plus_zero(split["dX"][zero])
    if bool(zero[valid].all()):
        assert split_plans == []  # no gradient GEMM
        assert _all_plus_zero(split["loss"])
        assert not bool(split["dX"].any()) and not bool(split["dW"].any())
        return
    assert len(split_plans) == 1 and len(plain_plans) == 1
    rows = _g_rows_of_equal_dx_forms(
        inp.labels, zero, C, split_plans[0], plain_plans[0]
    )
    assert _same_bits(split["dX"][rows], plain["dX"][rows])
    _check_against_references(split, inp, ceiling=True)
    dw_rel = rel_l2(split["dW"].float(), plain["dW"].float())
    print(
        f"\nzero-weight T={T} {kind} zero {int(zero.sum())}/{int(valid.sum())}: dX rows bitwise "
        f"{int(rows.numel())}/{int((valid & ~zero).sum())}, dW rel-L2 vs plain {dw_rel:.2e}"
    )


@pytest.mark.parametrize("T, split_expected", [(4095, False), (4096, True)])
def test_device_floor_boundary(glm_weight, T, split_expected):
    """Just below / at the default floor (4096 rows ``T``): below it the call is the plain path (no split in the
    memory report, the same bits); at it the call splits and keeps the loss / ``logp`` bits."""
    _require_program(entry="loss")
    base = _device_inputs(T, W=glm_weight)
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero)
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=4096,
    )
    assert cake_backend.zero_weight_min_rows() == 4096
    fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
    assert ("zero_weight_rows" in fr.memory) == split_expected
    with _zero_weight_floor(NO_SPLIT):
        ref = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
    torch.cuda.synchronize()
    assert _same_bits(fr.loss, ref.loss) and _same_bits(fr.logp, ref.logp)


@pytest.mark.parametrize("hidden_count", [None, False])
def test_device_runs_repeat_and_both_count_paths(glm_weight, hidden_count):
    """``RUNS`` runs with 16.6 % zero-weight rows: every run repeats the first run's ``loss`` / ``logp`` / ``dX`` /
    ``dW`` bits; the hidden valid-row count and the host count give the same bits; one synchronizing readback per
    call, as the plain path (PyTorch's synchronization debug mode)."""
    _require_program(entry="loss")
    base = _device_inputs(12289, W=glm_weight)
    inp = _with_zero_weights(base, _zero_weight_set(base, ("random", 0.166)))
    switches = {} if hidden_count is None else dict(hidden_count=hidden_count)
    with _zero_weight_floor(0):
        first = _loss_call(inp, 4096, **switches)
        for _ in range(RUNS - 1):
            junk = torch.empty((1 << 30,), dtype=torch.uint8, device=CUDA).fill_(0xFF)
            del junk
            run = _loss_call(inp, 4096, **switches)
            for key in ("loss", "logp", "dX", "dW"):
                assert _same_bits(run[key], first[key]), key
        other = _loss_call(inp, 4096, hidden_count=hidden_count is not None)
    for key in ("loss", "logp", "dX", "dW"):
        assert _same_bits(other[key], first[key]), key

    def syncs():
        counts = []
        for _ in range(3):
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    cake_backend.forward_loss(
                        inp.X,
                        inp.W,
                        inp.labels,
                        objective="policy",
                        infer_logp=inp.infer_logp,
                        loss_weights=inp.loss_weights,
                        chunk_size=4096,
                        **switches,
                    )
            finally:
                torch.cuda.set_sync_debug_mode("default")
            counts.append(sum("synchroniz" in str(w.message) for w in rec))
        return min(counts)

    with _zero_weight_floor(0):
        split_syncs = syncs()
    with _zero_weight_floor(NO_SPLIT):
        plain_syncs = syncs()
    print(
        f"\nzero-weight synchronizations hidden_count={hidden_count}: split {split_syncs} plain {plain_syncs}"
    )
    assert split_syncs == plain_syncs


def test_device_regroup_mode(glm_weight):
    """The ``"regroup"`` mode: ``logp`` / ``dX`` / ``dW`` are the exact mode's bits (the loss mode changes the loss
    only); the loss is the same FP64 sum over the split's chunks, within ``1e-6`` of the plain loss."""
    _require_program(entry="loss")
    base = _device_inputs(12289, W=glm_weight)
    inp = _with_zero_weights(base, _zero_weight_set(base, ("group", 0.3)))
    with _zero_weight_floor(0):
        exact = _loss_call(inp, 4096)
        with _loss_mode("regroup"):
            regroup = _loss_call(inp, 4096)
    for key in ("logp", "dX", "dW"):
        assert _same_bits(regroup[key], exact[key]), key
    rel = abs(float(regroup["loss"]) - float(exact["loss"])) / max(
        abs(float(exact["loss"])), 1e-30
    )
    assert rel <= 1e-6, rel
    _check_against_references(regroup, inp, ceiling=True)


def test_device_noncontiguous_x(glm_weight):
    """An ``X`` with a padded leading stride (the GEMMs read it in place): the split keeps the loss / ``logp`` bits of
    the plain path on the same layout and the bits of a contiguous ``X`` with the same values."""
    _require_program(entry="loss")
    base = _device_inputs(8191, W=glm_weight, ld_pad=64)
    assert not base.X.is_contiguous()
    inp = _with_zero_weights(base, _zero_weight_set(base, ("random", 0.3)))
    with _zero_weight_floor(0):
        split = _loss_call(inp, 4096)
        dense = _loss_call(inp, 4096, X=inp.X.contiguous())
    with _zero_weight_floor(NO_SPLIT):
        plain = _loss_call(inp, 4096)
    assert _same_bits(split["loss"], plain["loss"]) and _same_bits(
        split["logp"], plain["logp"]
    )
    for key in ("loss", "logp", "dX", "dW"):
        assert _same_bits(split[key], dense[key]), key


def test_device_non_finite_zero_weight_rows(glm_weight):
    """An ``inf`` ``X`` row and a NaN ``infer_logp`` in zero-weight rows: the exact loss is the clean call's bits and
    the plain path's on the same inputs (a zero-weight row's term is ``+-0`` for any ratio); ``dX`` / ``dW`` stay
    finite; a NaN weight keeps its row in the gradient path (NaN loss)."""
    _require_program(entry="loss")
    base = _device_inputs(4097, W=glm_weight)
    zero = _zero_weight_set(base, ("random", 0.2))
    clean = _with_zero_weights(base, zero)
    rows = zero.nonzero().squeeze(1)
    X = clean.X.clone()
    X[rows[0]] = float("inf")
    infer = clean.infer_logp.clone()
    infer[rows[1]] = float("nan")
    poisoned = dataclasses.replace(clean, X=X, infer_logp=infer)
    with _zero_weight_floor(0):
        a = _loss_call(clean, 4096)
        b = _loss_call(poisoned, 4096)
    with _zero_weight_floor(NO_SPLIT):
        plain = _loss_call(poisoned, 4096)
    assert _same_bits(a["loss"], b["loss"]) and _same_bits(a["dW"], b["dW"])
    assert _same_bits(b["loss"], plain["loss"])
    assert bool(torch.isfinite(b["dX"].float()).all()) and _all_plus_zero(b["dX"][zero])
    assert bool(torch.isfinite(b["dW"].float()).all())
    w = clean.loss_weights.clone()
    w[int((~zero & (clean.labels >= 0)).nonzero()[0])] = float("nan")
    with _zero_weight_floor(0):
        assert math.isnan(
            float(_loss_call(dataclasses.replace(clean, loss_weights=w), 4096)["loss"])
        )


@pytest.mark.parametrize("train", ["xw", "x_only", "none"])
def test_device_budget_mode(glm_weight, train):
    """The opt-in module's ``memory_mode="budget"`` splits the rows as the fast form (no hidden valid-row count): the
    loss / ``logp`` bits of the budget call without the split, ``+0`` ``dX`` rows on the zero-weight rows, ``dW``
    within the gates; the budget plan counts every valid row."""
    _require_program(entry="loss")
    base = _device_inputs(12289, W=glm_weight)
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero, negative_zero=True)
    kw = dict(
        module=opt_in,
        memory_mode="budget",
        train_x=train in ("xw", "x_only"),
        train_w=train == "xw",
    )
    with _zero_weight_floor(0):
        split = _loss_call(inp, 4096, **kw)
    with _zero_weight_floor(NO_SPLIT):
        plain = _loss_call(inp, 4096, **kw)
    assert _same_bits(split["loss"], plain["loss"]) and _same_bits(
        split["logp"], plain["logp"]
    )
    if split["dX"] is not None:
        assert _all_plus_zero(split["dX"][zero])
    if train == "xw":
        _check_against_references(split, inp, ceiling=True)


def _peaks(call):
    """``(allocated, reserved)`` peak bytes of ``call()`` above its start, after one warm-up call (the binding is
    remembered) with the caching allocator emptied before each measured call.  The warm-up starts from the same
    cleared state for every measured path: both modules' binding caches are cleared and the caching allocator emptied
    first, so the reserved peak does not depend on which allocator segments the remembered bindings of earlier calls
    -- an earlier path's or an earlier test's in the same process -- leave partly used (the measured calls still find
    their own binding remembered)."""
    for mod in (cake_backend, opt_in):
        mod.BINDING_CACHE.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    call()
    out = []
    for _ in range(2):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        start_res = torch.cuda.memory_reserved()
        call()
        torch.cuda.synchronize()
        out.append(
            (
                torch.cuda.max_memory_allocated() - start_alloc,
                torch.cuda.max_memory_reserved() - start_res,
            )
        )
    return max(a for a, _ in out), max(r for _, r in out)


@pytest.mark.parametrize("mode", ["exact", "regroup"])
@pytest.mark.parametrize(
    "train, hidden_count",
    [("xw", None), ("xw", False), ("x_only", None), ("w_only", None), ("none", None)],
)
@pytest.mark.parametrize("frac", [0.01, 0.3])
def test_device_split_peak_memory_within_the_plain_path(
    glm_weight, train, hidden_count, frac, mode
):
    """The split's measured step peak (forward + backward, allocated and reserved) never exceeds the same call's
    without the split: the term array of the exact loss and the split's indices are offset by the gradient rows'
    smaller accumulators and row vectors and by the chunk loop's buffers released before the exact loss runs."""
    _require_program(entry="loss")
    T = 16231
    base = _device_inputs(T, W=glm_weight)
    inp = _with_zero_weights(base, _zero_weight_set(base, ("random", frac)))
    switches = {} if hidden_count is None else dict(hidden_count=hidden_count)

    def call():
        X = inp.X.detach().requires_grad_(train in ("xw", "x_only"))
        W = inp.W.detach().requires_grad_(train in ("xw", "w_only"))
        loss, logp = cake_backend.cake_lm_head_det_train_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
            **switches,
        )
        if X.requires_grad or W.requires_grad:
            loss.backward()
        del loss, logp, X, W

    with _loss_mode(mode):
        with _zero_weight_floor(0):
            split_alloc, split_res = _peaks(call)
        with _zero_weight_floor(NO_SPLIT):
            plain_alloc, plain_res = _peaks(call)
    print(
        f"\nzero-weight peak {train} hidden_count={hidden_count} frac={frac} {mode}: allocated split {split_alloc} "
        f"plain {plain_alloc}; reserved split {split_res} plain {plain_res}"
    )
    assert split_alloc <= plain_alloc
    assert split_res <= plain_res


def _cold_peaks(call, modules):
    """``(allocated, reserved)`` peak bytes of ``call()`` above its start with every binding cache of ``modules``
    cleared before each measured call (each call prepares its bindings), after one warm-up call."""
    call()
    out = []
    for _ in range(2):
        for mod in modules:
            mod.BINDING_CACHE.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        start_res = torch.cuda.memory_reserved()
        call()
        torch.cuda.synchronize()
        out.append(
            (
                torch.cuda.max_memory_allocated() - start_alloc,
                torch.cuda.max_memory_reserved() - start_res,
            )
        )
    return max(a for a, _ in out), max(r for _, r in out)


def _step_call(inp, module, **kw):
    """One forward + backward of the loss entry (``X`` and ``W`` trainable, ``return_logp``), outputs dropped."""

    def call():
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        loss, logp = module.cake_lm_head_det_train_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
            **kw,
        )
        loss.backward()
        del loss, logp, X, W

    return call


@pytest.mark.parametrize("module_name", ["main", "opt_in"])
@pytest.mark.parametrize("hidden_count", [None, False])
@pytest.mark.parametrize("T", [8117, 16231])
def test_device_cold_binding_cache_peak_within_the_plain_path(
    glm_weight, T, hidden_count, module_name
):
    """A split call that misses the binding cache (every call here: the caches are cleared first) prepares its binding
    before the logp-only pass's FP32 accumulator arena exists, so its measured step peak (allocated and reserved)
    stays within the same call's without the split that misses the cache too.  The opt-in module's arm pins
    ``memory_mode="fast"``: its default call's automatic form can plan different forms for the split and the plain
    call."""
    _require_program(entry="loss")
    module = cake_backend if module_name == "main" else opt_in
    base = _device_inputs(T, W=glm_weight)
    inp = _with_zero_weights(base, _zero_weight_set(base, ("random", 0.166)))
    switches = {} if hidden_count is None else dict(hidden_count=hidden_count)
    if module_name == "opt_in":
        switches["memory_mode"] = "fast"
    call = _step_call(inp, module, **switches)
    caches = (cake_backend, opt_in)
    with _zero_weight_floor(0):
        split_alloc, split_res = _cold_peaks(call, caches)
    with _zero_weight_floor(NO_SPLIT):
        plain_alloc, plain_res = _cold_peaks(call, caches)
    print(
        f"\ncold-cache peak {module_name} T={T} hidden_count={hidden_count}: allocated split {split_alloc} "
        f"plain {plain_alloc}; reserved split {split_res} plain {plain_res}"
    )
    assert split_alloc <= plain_alloc
    assert split_res <= plain_res


@pytest.mark.parametrize(
    "module_name, mode", [("main", None), ("opt_in", "fast"), ("opt_in", "budget")]
)
@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("T", [4096, 16231])
def test_device_all_zero_weight_peak_within_the_plain_path(
    glm_weight, T, cold, module_name, mode
):
    """Every valid row of zero weight (one chunk at ``T`` 4096, several at 16231): the split launches no gradient
    GEMM and holds no FP32 weight-gradient accumulator into the backward, so its measured step peak stays within the
    same call's without the split -- allocated with the binding caches warm or cleared before each call, reserved
    with them cleared (a warm call's reserved peak depends on which allocator segments the remembered bindings of
    earlier calls leave partly used, so it is compared from the same cleared state); ``dX`` and ``dW`` are ``+0``."""
    _require_program(entry="loss")
    module = cake_backend if module_name == "main" else opt_in
    base = _device_inputs(T, W=glm_weight)
    inp = _with_zero_weights(base, base.labels >= 0)
    kw = {} if mode is None else dict(memory_mode=mode)
    call = _step_call(inp, module, **kw)
    peaks = (lambda c: _cold_peaks(c, (cake_backend, opt_in))) if cold else _peaks
    with _zero_weight_floor(0):
        split_alloc, split_res = peaks(call)
        out = _loss_call(inp, 4096, module=module, **kw)
    with _zero_weight_floor(NO_SPLIT):
        plain_alloc, plain_res = peaks(call)
    print(
        f"\nall-zero-weight peak {module_name}/{mode} T={T} cold={cold}: allocated split {split_alloc} "
        f"plain {plain_alloc}; reserved split {split_res} plain {plain_res}"
    )
    assert _all_plus_zero(out["dX"]) and _all_plus_zero(out["dW"])
    assert split_alloc <= plain_alloc
    if cold:
        assert split_res <= plain_res
