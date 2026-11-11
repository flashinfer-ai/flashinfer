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

The zero-gradient row skip of the deterministic LM-head package's log-probability backward
(``cake_backend.zero_dlogp_partition``, the ``_dcnt`` forms, ``set_zero_dlogp_min_rows``): the partition
against its reference, the plan and eligibility rules on the host, and on the device the deterministic contract
(the same bits on every run), the accuracy gates of ``test_cake_lm_head_det_train.py`` and the bits that the skip
keeps (``logp``; ``dX`` / ``dW`` of a call without zero-gradient rows).
"""

import contextlib
import copy
import pickle
from dataclasses import replace

import pytest
import torch

from tests.experimental.test_helpers.cake_lm_head_loss_reference import (
    error_report,
    make_inputs,
    reference_fp64,
    reference_unchunked,
)
from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_lm_head_det_train", "cake_backend")
cake_jit = cake_package("cake.cake_lm_head_det_train", "cake_jit")
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
generated_program_available = cake_backend.generated_program_available

SEED = 20260930
CUDA = torch.device("cuda")
# The accuracy gates of the deterministic suite (``test_cake_lm_head_det_train.py``), copied unchanged: at most 1.05x
# the error of the unchunked PyTorch path on the same inputs plus a floor per metric, and twice the production-geometry
# unchunked error as a sanity ceiling.
GATE_MARGIN = 1.05
GATE_TINY = dict(loss_rel=1e-6, logp_max_abs=1e-5, dX_rel_l2=1e-3, dW_rel_l2=1e-3)
PRODUCTION_B0 = dict(dX_rel_l2=1.85e-3, dW_rel_l2=1.88e-3, logp_max_abs=4.1e-3)
# Runs of one call compared bit for bit (the deterministic contract).
RUNS = 5


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


def _require_skip():
    """Skip without a device whose record registers the skip: the counted dz arm and the partition kernel."""
    _require_program(entry="logprob")
    module_name, _record = cake_backend.record_for(CUDA)
    stages = set(cake_jit.registered_stages(module_name))
    if (
        not {
            cake_backend.LOGPROB_DZ_DCNT_STAGE,
            cake_backend.ZERO_DLOGP_PARTITION_STAGE,
        }
        <= stages
    ):
        pytest.skip(
            "this device's record registers no zero-gradient row skip (no counted dz arm or partition kernel)"
        )


def _check_against_references(
    result, inp, *, entry="loss", grad_weight_dtype=torch.bfloat16, ceiling=False
):
    """``test_cake_lm_head_det_train.py``'s gate, unchanged."""
    oracle = reference_fp64(inp, entry=entry)
    b0 = reference_unchunked(inp, entry=entry, grad_weight_dtype=grad_weight_dtype)
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
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _same_bits(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


@contextlib.contextmanager
def _zero_dlogp_floor(rows):
    """Run under the skip's floor ``rows`` (``None``: never), then restore the previous setting."""
    previous = cake_backend.set_zero_dlogp_min_rows(rows)
    try:
        yield
    finally:
        cake_backend.set_zero_dlogp_min_rows(previous)


@contextlib.contextmanager
def _tau_dz_counted(counted):
    """Run with the skip's ``dz`` temperature form counted or not, then restore the previous setting."""
    previous = cake_backend.set_zero_dlogp_temperature_dz_counted(counted)
    try:
        yield
    finally:
        cake_backend.set_zero_dlogp_temperature_dz_counted(previous)


@contextlib.contextmanager
def _counting_partitions(monkeypatch):
    """Count the calls of ``cake_backend.zero_dlogp_partition`` (one per backward that runs the skip)."""
    calls = []
    original = cake_backend.zero_dlogp_partition

    def counting(*args, **kwargs):
        calls.append(int(args[2].numel()))
        return original(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "zero_dlogp_partition", counting)
    yield calls


def _zero_set(labels, kind, *, seed=SEED):
    """bool ``[T]`` of the valid rows whose ``dlogp`` becomes zero: ``"none"``, ``"all"``, ``"one"`` (the first valid
    row), ``("group", f)`` (whole groups of 512 consecutive rows, ``round(f * groups)`` of them) or ``("random", f)``
    (``round(f * T_v)`` random valid rows)."""
    valid = labels >= 0
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
        return zs & valid
    rows = valid.nonzero().squeeze(1).cpu()
    pick = rows[
        torch.randperm(rows.numel(), generator=gen)[: int(round(frac * rows.numel()))]
    ]
    zs[pick.to(zs.device)] = True
    return zs


def _with_zero_dlogp(inp, zero):
    """``inp`` with ``dlogp`` ``+0`` / ``-0`` (alternating) on the rows of ``zero``."""
    d = inp.dlogp.clone()
    d[zero] = 0.0
    d[zero.nonzero().squeeze(1)[::2]] = -0.0
    return replace(inp, dlogp=d.contiguous())


@pytest.fixture(scope="module")
def glm_weight():
    _require_program(entry="logprob")
    return make_inputs(1, seed=SEED, device=CUDA, ignore_frac=0.0).W


def _device_inputs(T, *, W, seed=SEED, temperature=None, ignore_frac=0.05):
    return make_inputs(
        T,
        seed=seed + T,
        device=CUDA,
        W=W,
        ignore_frac=ignore_frac,
        temperature=temperature,
    )


def _logprob_call(inp, C=4096, *, X=None, dlogp=None, **switches):
    """One autograd log-probability call (forward + ``backward(dlogp)``) on the generated program; ``X`` / ``dlogp``
    replace the inputs' tensors (the same values in another layout).  A call with the inputs' temperature runs the
    temperature entry."""
    X = (inp.X if X is None else X).detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    if inp.temperature is not None:
        switches.setdefault("temperature", inp.temperature)
    entry = (
        cake_backend.cake_lm_head_det_train_logprob_temperature
        if "temperature" in switches
        else cake_backend.cake_lm_head_det_train_logprob
    )
    logp = entry(X, W, inp.labels, chunk_size=C, backend="cake", **switches)
    loss = (logp.detach() * inp.dlogp)[inp.valid].sum()
    logp.backward(inp.dlogp if dlogp is None else dlogp)
    torch.cuda.synchronize()
    return dict(loss=loss.detach(), logp=logp.detach(), dX=X.grad, dW=W.grad)


def _backward_plans():
    """The plans of the remembered log-probability backward bindings."""
    return [
        b.plan
        for key, b in list(cake_backend.BINDING_CACHE._bindings.items())
        if key[0] == "bwd"
    ]


def _live_rows_of_equal_dx_forms(labels, zero, C, skip_plan, plain_plan):
    """The live rows whose chunk runs the same dX form and slice count with the skip (the live rows first) and without
    it (the forward's row order): their ``dX`` row is the same arithmetic, so the same bits."""
    rows = (labels >= 0).nonzero().squeeze(1)
    live = ~zero[rows]
    plain_chunk = torch.arange(rows.numel(), device=rows.device) // C
    rank = torch.cumsum(live.to(torch.int64), 0) - 1
    skip_chunk = rank.clamp(min=0) // C
    forms: dict[tuple, int] = {}

    def form_ids(plan, chunk):
        ids = [
            forms.setdefault((plan.dx_stage_of(i), plan.dx_slices_of(i)), len(forms))
            for i in range(plan.num_chunks)
        ]
        table = torch.tensor(ids, dtype=torch.int64, device=rows.device)
        return table[chunk.clamp(max=plan.num_chunks - 1)]

    same = form_ids(skip_plan, skip_chunk) == form_ids(plain_plan, plain_chunk)
    return rows[live & same]


def _problem(T, H=6144, V=154880, C=4096, entry="logprob"):
    return cake_backend.Problem(
        num_rows=T,
        hidden=H,
        vocab=V,
        chunk=C,
        ld_x=H,
        x_copy=False,
        objective="ce",
        loss_div=1.0,
        grad_weight_dtype=torch.bfloat16,
        entry=entry,
    )


def _plan_kw(arch, **kw):
    g = cake_backend.Geometry.from_record(
        {"geometry": {"hidden": 6144, "vocab": 154880}}
    )
    return dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        dx_finalize=True,
        dx_cast=False,
        logprob_dz=True,
        logprob_lbl=True,
        arch=arch,
        **kw,
    )


# ---------------------------------------------------------------------------
# Host layer (any device)
# ---------------------------------------------------------------------------


def test_zero_dlogp_partition_orders_the_live_rows_first():
    """The partition (its reference on the host): the valid rows of a nonzero ``dlogp`` (NaN included) first, then the
    zero-gradient rows (``+0`` and ``-0``), each group in ascending order; ``lse`` in that order; the inclusive scan
    of the live rows over ``[T]``; the per-chunk live counts, at least one in the first chunk."""
    labels = torch.tensor([3, -100, 5, 7, 1, 2, -100, 4, 0], dtype=torch.int64)
    d = torch.tensor([1.0, 0.0, 0.0, -0.0, float("nan"), -2.0, 3.0, 0.0, 0.5])
    index = (labels >= 0).nonzero().squeeze(1)
    lse = torch.arange(7, dtype=torch.float32)
    order, lse_o, scan, counts = cake_backend.zero_dlogp_partition(
        labels, d, index, lse, 4
    )
    assert order.tolist() == [0, 4, 5, 8, 2, 3, 7]
    assert lse_o.tolist() == [0.0, 3.0, 4.0, 6.0, 1.0, 2.0, 5.0]
    assert scan.dtype == torch.int32
    assert scan.tolist() == [1, 1, 1, 1, 2, 3, 3, 3, 4]
    assert counts.dtype == torch.int32 and counts.tolist() == [4, 0]
    order, _, scan, counts = cake_backend.zero_dlogp_partition(
        labels, torch.zeros(9), index, lse, 4
    )
    assert order.tolist() == index.tolist() and int(scan[-1]) == 0
    # the first chunk's weight-gradient GEMM stores one K step (of zeros): its count is at least one
    assert counts.tolist() == [1, 0]
    order, lse_o, _, counts = cake_backend.zero_dlogp_partition(
        labels, torch.ones(9), index, lse, 3
    )
    assert order.tolist() == index.tolist() and lse_o.tolist() == lse.tolist()
    assert counts.tolist() == [3, 3, 1]


@pytest.mark.parametrize(
    "T, chunk, frac", [(1, 4, 0.0), (37, 8, 0.5), (1000, 128, 0.166), (5000, 4096, 0.9)]
)
def test_zero_dlogp_partition_reference_is_a_stable_partition(T, chunk, frac):
    """The reference against a sort on the host: the valid rows ordered by (zero gradient, row), the live-row scan, the
    per-chunk counts of the live rows (clamped to the chunk, at least one in the first chunk) and ``lse`` permuted
    with the rows; two calls give the same bits."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + T)
    labels = torch.randint(0, 50, (T,), generator=gen)
    labels[torch.rand(T, generator=gen) < 0.1] = -100
    d = torch.randn(T, generator=gen)
    d[torch.rand(T, generator=gen) < frac] = 0.0
    index = (labels >= 0).nonzero().squeeze(1)
    if index.numel() == 0:
        labels[0], index = 1, torch.zeros(1, dtype=torch.int64)
    lse = torch.randn(int(index.numel()), generator=gen)
    got = cake_backend.zero_dlogp_partition_reference(labels, d, index, lse, chunk)
    again = cake_backend.zero_dlogp_partition_reference(labels, d, index, lse, chunk)
    for a, b in zip(got, again, strict=True):
        assert torch.equal(
            _bits(a) if a.is_floating_point() else a,
            _bits(b) if b.is_floating_point() else b,
        )
    live = (labels >= 0) & (d != 0)
    keys = sorted(
        range(index.numel()), key=lambda k: (not bool(live[index[k]]), int(index[k]))
    )
    assert got[0].tolist() == [int(index[k]) for k in keys]
    assert _same_bits(got[1], lse[torch.tensor(keys, dtype=torch.int64)])
    assert got[2].tolist() == torch.cumsum(live.to(torch.int32), 0).tolist()
    n_live = int(live.sum())
    want = [
        min(max(n_live - c * chunk, 0), chunk)
        for c in range(-(-int(index.numel()) // chunk))
    ]
    want[0] = max(want[0], 1)
    assert got[3].tolist() == want


def test_zero_dlogp_floor_switch():
    """``set_zero_dlogp_min_rows`` returns the previous floor; ``None`` turns the skip off; a negative or non-integer
    floor is refused."""
    default = cake_backend.zero_dlogp_min_rows()
    assert default == cake_backend.ZERO_DLOGP_MIN_ROWS_DEFAULT == 4097
    with _zero_dlogp_floor(8192):
        assert cake_backend.zero_dlogp_min_rows() == 8192
        with _zero_dlogp_floor(None):
            assert cake_backend.zero_dlogp_min_rows() is None
        assert cake_backend.zero_dlogp_min_rows() == 8192
    assert cake_backend.zero_dlogp_min_rows() == default
    for bad in (-1, 1.5, True, "8192"):
        with pytest.raises((TypeError, ValueError)):
            cake_backend.set_zero_dlogp_min_rows(bad)
    assert cake_backend.zero_dlogp_min_rows() == default


def test_zero_dlogp_floor_per_architecture(monkeypatch):
    """The default floor is per architecture (sm_100a 1946 valid rows, sm_103a 3891, every other 4097), resolved once
    per device index; an override (an integer or ``None``) applies to every device; the setter's returned values
    restore the previous state exactly."""
    sentinel = cake_backend.ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT
    assert cake_backend.ZERO_DLOGP_MIN_ROWS_BY_ARCH == {
        "sm_100a": 1946,
        "sm_103a": 3891,
    }
    assert repr(sentinel) == "ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT"
    archs = {0: "sm_100a", 1: "sm_103a", 2: "sm_107a", 3: None}
    resolved = []

    def arch_for(device=None):
        resolved.append(device.index)
        return archs[device.index]

    monkeypatch.setattr(cake_backend, "arch_for", arch_for)
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_ARCH", {})
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_MIN_ROWS", {"rows": sentinel})
    defaults = {0: 1946, 1: 3891, 2: 4097, 3: 4097}

    def floors():
        return {index: cake_backend.zero_dlogp_min_rows(index) for index in archs}

    assert floors() == defaults and floors() == defaults
    assert sorted(resolved) == [0, 1, 2, 3]  # once per device index
    previous = cake_backend.set_zero_dlogp_min_rows(0)
    assert previous is sentinel
    assert floors() == dict.fromkeys(archs, 0)
    assert cake_backend.set_zero_dlogp_min_rows(None) == 0
    assert floors() == dict.fromkeys(archs)
    assert cake_backend.set_zero_dlogp_min_rows(previous) is None
    assert floors() == defaults
    assert copy.copy(sentinel) is sentinel and copy.deepcopy(sentinel) is sentinel
    assert pickle.loads(pickle.dumps(sentinel)) is sentinel


def test_zero_dlogp_temperature_dz_counted_switch():
    assert cake_backend.zero_dlogp_temperature_dz_counted() is True
    with _tau_dz_counted(False):
        assert cake_backend.zero_dlogp_temperature_dz_counted() is False
    assert cake_backend.zero_dlogp_temperature_dz_counted() is True
    with pytest.raises(ValueError):
        cake_backend.set_zero_dlogp_temperature_dz_counted(1)


def test_zero_dlogp_skip_eligibility(monkeypatch):
    """A backward is eligible when it reuses a compacted row index of at least the device's floor of rows on the cake
    backend, with the dz arm and, when it forms dX, the fused dX finalize (the only reader of a counted plan's dX
    accumulator), on a record with the partition kernel and the counted dz arm."""
    dev = torch.device("cuda", 0)
    sentinel = cake_backend.ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_MIN_ROWS", {"rows": sentinel})
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_ARCH", {0: "sm_100a"})
    monkeypatch.setattr(cake_backend, "_SKIP_RECORDS", {0: ("m", {}, True, True)})
    skip = cake_backend._zero_dlogp_skip
    at, below = torch.arange(1946), torch.arange(1945)
    assert skip(dev, at, "cake", True, True, True)
    assert not skip(dev, below, "cake", True, True, True)
    assert not skip(
        dev, None, "cake", True, True, True
    )  # every row valid: no row index
    assert not skip(dev, at, "reference", True, True, True)
    assert not skip(dev, at, "cake", False, True, True)  # logprob_dz=False
    assert not skip(dev, at, "cake", True, True, False)  # dX without the fused finalize
    assert skip(
        dev, at, "cake", True, False, False
    )  # no dX: the finalize does not matter
    assert not skip(torch.device("cpu"), at, "cake", True, True, True)
    with _zero_dlogp_floor(None):
        assert not skip(dev, torch.arange(100000), "cake", True, True, True)
    with _zero_dlogp_floor(0):
        assert skip(dev, torch.arange(1), "cake", True, True, True)
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_ARCH", {0: "sm_103a"})
    assert not skip(dev, torch.arange(3890), "cake", True, True, True)
    assert skip(dev, torch.arange(3891), "cake", True, True, True)
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_ARCH", {0: "sm_107a"})
    assert not skip(dev, torch.arange(4096), "cake", True, True, True)
    assert skip(dev, torch.arange(4097), "cake", True, True, True)
    for entry in (("m", {}, True, False), ("m", {}, False, False)):
        monkeypatch.setattr(cake_backend, "_SKIP_RECORDS", {0: entry})
        assert not skip(dev, torch.arange(8192), "cake", True, True, True)


def test_counted_stage_grammar():
    """The ``_dcnt`` forms: the dz arm (and its temperature form), the weight-gradient accumulate (1-D and blocked) and
    the bf16 fused cast (per thread and TMA); no counted dX GEMM, accumulate TMA epilogue or fp32 cast; every registry
    stage name parses back to itself."""
    sv, sf = cake_backend.stage_variant, cake_backend.stage_form
    assert cake_backend.DCNT_BASES == (
        "gemm_logits_dz",
        "gemm_dw_acc",
        "gemm_dw_cast_bf16",
    )
    assert sv("gemm_logits_dz", dcnt=True) == cake_backend.LOGPROB_DZ_DCNT_STAGE
    assert sv("gemm_dw_acc", dcnt=True) == "gemm_dw_acc_dcnt"
    assert sv("gemm_dw_acc", blocked=True, dcnt=True) == "gemm_dw_acc_gn_dcnt"
    assert sv("gemm_dw_cast_bf16", dcnt=True) == "gemm_dw_cast_bf16_dcnt"
    assert (
        sv("gemm_dw_cast_bf16", epi_store="tma", dcnt=True)
        == cake_backend.DW_CAST_TMA_DCNT_STAGE
    )
    assert (
        sf("gemm_logits_dz", dcnt=True, temperature=True)
        == cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE
    )
    for base, knobs in (
        ("gemm_dx", {}),
        ("gemm_dx", {"sliced": True}),
        ("gemm_dw_cast_f32", {}),
        ("gemm_dw_acc", {"epi_store": "tma"}),
        ("gemm_logits", {}),
        ("gemm_logits_lbl", {}),
    ):
        with pytest.raises(ValueError):
            sv(base, dcnt=True, **knobs)
    with pytest.raises(ValueError):
        sf("gemm_logits_lbl", dcnt=True, temperature=True)
    for stage in cake_jit.STAGES_WITH_TEMPERATURE:
        base, knobs = cake_backend.parse_stage(stage)
        assert sf(base, **knobs) == stage
        assert knobs["dcnt"] == stage.endswith(("_dcnt", "_dcnt_tau"))
    assert (
        cake_backend.parse_stage("gemm_dw_cast_bf16_tma_dcnt")[1]["epi_store"] == "tma"
    )
    assert (
        cake_backend.base_stage(cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE)
        == "gemm_logits_dz"
    )
    assert cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE in cake_backend.TEMPERATURE_STAGES
    assert {
        cake_backend.LOGPROB_DZ_DCNT_STAGE,
        cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE,
    } <= cake_backend.DZ_PRODUCER_STAGES


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("rows", [15 * 4096 + 339, 7711, 1946])
def test_make_plan_counts_the_dz_and_dw_gemms_only(arch, rows):
    """A counted plan (``dcnt=True``) keeps the plan without the skip's chunks, dX slice counts and dX forms (Phase 1a:
    the dX GEMM computes every row) and runs the counted forms of the dz arm and of every weight-gradient GEMM -- the
    last chunk's TMA cast in its counted form where ``dw_cast_tma_rule`` takes it, the counted per-thread cast with
    ``counted_tma_cast=False``.  A pinned weight-gradient TMA epilogue, a loss-entry plan, the hidden valid-row count, a
    plan without the dz arm and the reference path (no architecture) take no counted form."""
    problem = _problem(65031)
    kw = _plan_kw(arch, valid_rows=rows)
    plain = cake_backend.make_plan(problem, **kw)
    plan = cake_backend.make_plan(problem, dcnt=True, **kw)
    old = cake_backend.make_plan(problem, dcnt=True, counted_tma_cast=False, **kw)
    assert not plain.dcnt and plan.dcnt and old.dcnt
    assert plan.chunks == plain.chunks and plan.dx_slices == plain.dx_slices
    assert plan.variants == plain.variants
    for i in range(plan.num_chunks):
        assert plan.dx_stage_of(i) == plain.dx_stage_of(i)
        assert not plan.dx_stage_of(i).endswith("_dcnt")
        assert plan.recompute_stage(i) == "gemm_logits_dz_dcnt"
        if i < plan.num_chunks - 1:
            assert plan.dw_acc_stage(i) == plain.dw_acc_stage(i) + "_dcnt"
    assert plan.dw_cast_stage == plain.dw_cast_stage + "_dcnt"
    tma = cake_backend.dw_cast_tma_rule(plan.num_chunks, plan.chunks[-1][1], arch)
    assert (plan.dw_cast_stage == cake_backend.DW_CAST_TMA_DCNT_STAGE) == tma
    assert old.dw_cast_stage == "gemm_dw_cast_bf16_dcnt"
    assert old.variants[:-1] == plan.variants[:-1]
    assert {s.removesuffix("_dcnt") for s in plan.stages} == set(plain.stages)
    assert list(plan.stages) == [s for s in cake_jit.STAGES if s in plan.stages]
    keys = cake_backend.recompute_keys(plan)
    assert [k for k in keys if k[0].startswith("gemm_logits")] == [
        ("gemm_logits_dz_dcnt", i) for i in range(plan.num_chunks)
    ]
    assert all(k[0].endswith("_dcnt") for k in keys if k[0].startswith("gemm_dw"))
    assert cake_backend.recompute_keys(plain) == tuple(
        (k[0].removesuffix("_dcnt"), k[1]) for k in keys
    )
    pinned = dict(kw, gemm_tuning={"dw": {"epi_store": "tma"}})
    shipped = cake_backend.make_plan(problem, **pinned)
    counted = cake_backend.make_plan(problem, dcnt=True, **pinned)
    assert not counted.dcnt and counted == shipped
    off = [
        cake_backend.make_plan(
            _problem(65031, entry="loss"),
            dcnt=True,
            **dict(kw, logprob_dz=False, logprob_lbl=False),
        ),
        cake_backend.make_plan(problem, dcnt=True, **dict(kw, logprob_dz=False)),
        cake_backend.make_plan(problem, dcnt=True, **dict(kw, arch=None)),
        cake_backend.make_plan(problem, dcnt=True, hidden_count=True, **kw),
    ]
    assert not any(p.dcnt for p in off)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_make_plan_temperature_counted_and_mixed(arch):
    """With a temperature the counted plan runs ``gemm_logits_dz_dcnt_tau`` (its twin without the temperature forms:
    ``gemm_logits_dz_dcnt``), the mixed plan (``counted_dz=False``) the uncounted ``gemm_logits_dz_tau`` over every
    row (twin ``gemm_logits_dz``, the twin plan without the skip) beside the counted weight-gradient forms."""
    problem = cake_backend._with_temperature(_problem(8117))
    kw = _plan_kw(arch, valid_rows=7711)
    counted = cake_backend.make_plan_temperature(problem, dcnt=True, **kw)
    mixed = cake_backend.make_plan_temperature(
        problem, dcnt=True, counted_dz=False, **kw
    )
    plain = cake_backend.make_plan_temperature(problem, **kw)
    assert type(counted).__name__ == "_CountedTemperaturePlan"
    assert type(mixed).__name__ == "_TemperaturePlan" and mixed.dcnt
    assert not plain.dcnt
    for i in range(counted.num_chunks):
        assert counted.recompute_stage(i) == cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE
        assert mixed.recompute_stage(i) == cake_backend.LOGPROB_DZ_TAU_STAGE
        assert counted.without_temperature().recompute_stage(i) == "gemm_logits_dz_dcnt"
        assert mixed.without_temperature().recompute_stage(i) == "gemm_logits_dz"
    assert counted.without_temperature().dcnt and not mixed.without_temperature().dcnt
    for p in (counted, mixed):
        assert all(s.endswith("_dcnt") for s in p.stages if s.startswith("gemm_dw"))
        assert set(p.stages) <= set(cake_jit.STAGES_WITH_TEMPERATURE)
    assert cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE in counted.stages
    assert cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE not in mixed.stages


def test_backward_binding_key_holds_the_skip():
    X = torch.zeros((8, 256), dtype=torch.bfloat16)
    W = torch.zeros((512, 256), dtype=torch.bfloat16)
    labels = torch.zeros(8, dtype=torch.int64)
    lse, dlogp = torch.zeros(8), torch.zeros(8)

    def key(**kw):
        return cake_backend.logprob_backward_binding_key(
            X, W, labels, lse, dlogp, chunk_size=4096, need_dx=True, need_dw=True, **kw
        )

    assert key(zero_dlogp=True) != key() == key(zero_dlogp=False)


# ---------------------------------------------------------------------------
# Device (compute capability 10.0 / 10.3 / 10.7 with a registered program; GLM-class H = 6144, V = 154880)
# ---------------------------------------------------------------------------

# (T, chunk, zero rows of the partition kernel test)
PARTITION_CASES = [
    (1, 4096, "none"),
    (700, 4096, ("random", 0.3)),
    (4096, 4096, "all"),
    (8117, 4096, ("random", 0.166)),
    (16231, 4096, ("group", 0.5)),
    (65031, 4096, ("random", 0.166)),
    (262147, 128, ("group", 0.25)),
]


def _assert_partition_is(got, want):
    for name, g, w in zip(
        ("row_index", "lse", "scan", "counts"), got, want, strict=True
    ):
        assert g.dtype == w.dtype and g.shape == w.shape and g.device == w.device, name
        same = _same_bits(g, w) if g.is_floating_point() else torch.equal(g, w)
        assert same, name


@pytest.mark.parametrize("T, chunk, kind", PARTITION_CASES, ids=str)
def test_device_zero_dlogp_partition_kernel_is_the_reference(T, chunk, kind):
    """The partition kernel gives its reference's four outputs bit for bit (signed zeros, NaN gradients and NaN
    statistics among the inputs), on the first call (binds) and the second (remembered launch)."""
    _require_skip()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + T)
    labels = torch.randint(0, 1000, (T,), generator=gen)
    if T > 1:
        labels[torch.rand(T, generator=gen) < 0.1] = -100
    labels = labels.to(CUDA)
    dlogp = torch.randn(T, generator=gen).to(CUDA)
    dlogp[::97] = float("nan")
    zero = _zero_set(labels, kind)
    dlogp[zero] = 0.0
    dlogp[zero.nonzero().squeeze(1)[::2]] = -0.0
    index = (labels >= 0).nonzero().squeeze(1)
    lse = torch.randn(int(index.numel()), generator=gen).to(CUDA)
    lse[::131] = float("nan")
    cake_backend.BINDING_CACHE.clear()
    runs = [
        cake_backend.zero_dlogp_partition(labels, dlogp, index, lse, chunk)
        for _ in range(2)
    ]
    want = cake_backend.zero_dlogp_partition_reference(labels, dlogp, index, lse, chunk)
    torch.cuda.synchronize()
    for got in runs:
        _assert_partition_is(got, want)


def test_device_zero_dlogp_partition_cuda_graph():
    """The partition captured in a CUDA graph replays with other ``dlogp`` values (random, every and no row zero)
    written into the captured input and gives the reference's bits for them."""
    _require_skip()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + 3)
    T = 12289
    labels = torch.randint(0, 1000, (T,), generator=gen)
    labels[torch.rand(T, generator=gen) < 0.1] = -100
    labels = labels.to(CUDA)
    index = (labels >= 0).nonzero().squeeze(1)
    lse = torch.randn(int(index.numel()), generator=gen).to(CUDA)
    static = torch.randn(T, generator=gen).to(CUDA)
    cake_backend.zero_dlogp_partition(labels, static, index, lse, 4096)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = cake_backend.zero_dlogp_partition(labels, static, index, lse, 4096)
    for frac in (0.166, 1.0, 0.0):
        new = torch.randn(T, generator=gen)
        new[torch.rand(T, generator=gen) < frac] = 0.0
        static.copy_(new.to(CUDA))
        graph.replay()
        want = cake_backend.zero_dlogp_partition_reference(
            labels, static, index, lse, 4096
        )
        torch.cuda.synchronize()
        _assert_partition_is(got, want)


# (T, chunk, zero rows): 5 % ignored rows; 8117 -> 7711 valid rows (chunks 4096 + 3615), 6000 at chunk 2048 -> 5700
# (2048 + 2048 + a 1604-row last chunk: the TMA cast's counted form on sm_100a / sm_103a), 2048 -> one chunk of 1946
SKIP_CASES = [
    (8117, 4096, "none"),
    (8117, 4096, ("random", 0.166)),
    (8117, 4096, ("random", 0.5)),
    (8117, 4096, ("group", 0.5)),
    (8117, 4096, "all"),
    (6000, 2048, ("random", 0.166)),
    (6000, 2048, ("group", 0.166)),
    (2048, 4096, ("random", 0.5)),
]


def _skip_and_plain(inp, C, monkeypatch, runs=RUNS, **switches):
    """``runs`` calls with the skip at floor 0 (the first binds, the others replay the remembered binding) and one
    without it (floor ``None``); asserts that each skip call ran one partition and the call without it none.  Returns
    the runs, the call without the skip and the two backward plans."""
    cake_backend.BINDING_CACHE.clear()
    with _counting_partitions(monkeypatch) as calls:
        with _zero_dlogp_floor(0):
            got = [_logprob_call(inp, C, **switches) for _ in range(runs)]
        assert len(calls) == runs, calls
        skip_plans = [p for p in _backward_plans() if p.dcnt]
        with _zero_dlogp_floor(None):
            plain = _logprob_call(inp, C, **switches)
        assert len(calls) == runs, calls
    plain_plans = [p for p in _backward_plans() if not p.dcnt]
    assert len(skip_plans) == 1 and len(plain_plans) == 1
    return got, plain, skip_plans[0], plain_plans[0]


def _assert_skip_contract(inp, zero, C, got, plain, skip_plan, plain_plan):
    valid = inp.labels >= 0
    live = valid & ~zero
    for run in got[1:]:
        for key in ("logp", "dX", "dW"):
            assert _same_bits(run[key], got[0][key]), key  # the same bits on every run
    first = got[0]
    assert _same_bits(first["logp"], plain["logp"])  # the forward is the same
    assert not bool(_bits(first["dX"][zero | ~valid]).any())  # exact +0 rows
    rows = _live_rows_of_equal_dx_forms(inp.labels, zero, C, skip_plan, plain_plan)
    assert _same_bits(first["dX"][rows], plain["dX"][rows])
    if not bool(zero.any()):
        # no zero-gradient row: the row order, chunks and K steps of the call without the skip
        assert _same_bits(first["dX"], plain["dX"]) and _same_bits(
            first["dW"], plain["dW"]
        )
    if not bool(live.any()):
        assert not bool(first["dX"].float().any()) and not bool(
            first["dW"].float().any()
        )
        return
    # the deterministic suite's gates: the production ceiling is a temperature-1 figure and applies, as in
    # test_device_temperature_heterogeneous_matches_the_references, only where every temperature is at least 1
    tau = inp.temperature
    ceiling = tau is None or float(torch.as_tensor(tau).min()) >= 1.0
    _check_against_references(first, inp, entry="logprob", ceiling=ceiling)


@pytest.mark.parametrize("T, C, kind", SKIP_CASES, ids=str)
def test_device_zero_dlogp_skip_is_deterministic_and_accurate(
    glm_weight, monkeypatch, T, C, kind
):
    """The skip on the device: ``RUNS`` calls give the same ``logp`` / ``dX`` / ``dW`` bits; ``logp`` is the bits of
    the call without the skip; skipped and ignored rows get exact ``+0`` ``dX`` rows; a live row whose chunk keeps its
    dX slice count gets the bits of the call without the skip, and with no zero-gradient row ``dX`` and ``dW`` are that
    call's bits; the deterministic suite's accuracy gates against the FP64 oracle hold."""
    _require_skip()
    base = _device_inputs(T, W=glm_weight)
    zero = _zero_set(base.labels, kind)
    inp = _with_zero_dlogp(base, zero)
    got, plain, skip_plan, plain_plan = _skip_and_plain(inp, C, monkeypatch)
    assert skip_plan.chunks == plain_plan.chunks
    assert skip_plan.dx_slices == plain_plan.dx_slices
    _assert_skip_contract(inp, zero, C, got, plain, skip_plan, plain_plan)


@pytest.mark.parametrize("form", ["scalar", "hetero"])
@pytest.mark.parametrize("counted", [True, False], ids=("counted_dz", "mixed"))
@pytest.mark.parametrize("kind", ["none", ("random", 0.166), ("group", 0.5)], ids=str)
def test_device_temperature_zero_dlogp_skip(
    glm_weight, monkeypatch, form, counted, kind
):
    """The skip of a call with a temperature (the scalar form and a per-token ``[T]`` tensor; the counted temperature
    dz arm and the mixed plan): the same contract as without a temperature."""
    _require_skip()
    module_name, _ = cake_backend.record_for(CUDA)
    if counted and cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE not in set(
        cake_jit.registered_stages_with_temperature(module_name)
    ):
        pytest.skip("the record registers no counted temperature dz arm")
    T, C = 8117, 4096
    base = _device_inputs(T, W=glm_weight, temperature=form)
    zero = _zero_set(base.labels, kind)
    inp = _with_zero_dlogp(base, zero)
    with _tau_dz_counted(counted):
        got, plain, skip_plan, plain_plan = _skip_and_plain(inp, C, monkeypatch, runs=3)
    assert (
        skip_plan.temperature
        and (type(skip_plan).__name__ == "_CountedTemperaturePlan") == counted
    )
    _assert_skip_contract(inp, zero, C, got, plain, skip_plan, plain_plan)


def test_device_zero_dlogp_paths_without_the_skip(glm_weight, monkeypatch):
    """No partition and the bits of the call with the skip off: a call below the floor, ``dx_finalize=False``,
    ``logprob_dz=False``, a call whose labels are all valid (no row index) and a backward that recounts its rows."""
    _require_skip()
    base = _device_inputs(6000, W=glm_weight)
    zero = _zero_set(base.labels, ("random", 0.3))
    inp = _with_zero_dlogp(base, zero)
    with _zero_dlogp_floor(None):
        plain = _logprob_call(inp, 4096)
        plain_nofin = _logprob_call(inp, 4096, dx_finalize=False)
        plain_nodz = _logprob_call(inp, 4096, logprob_dz=False)
    with _counting_partitions(monkeypatch) as calls:
        with _zero_dlogp_floor(100000):
            below = _logprob_call(inp, 4096)
        with _zero_dlogp_floor(0):
            nofin = _logprob_call(inp, 4096, dx_finalize=False)
            nodz = _logprob_call(inp, 4096, logprob_dz=False)
            every = _device_inputs(6000, W=glm_weight, ignore_frac=0.0)
            every = _with_zero_dlogp(every, _zero_set(every.labels, ("random", 0.3)))
            _logprob_call(every, 4096)
            fr = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096)
            recount = cake_backend.backward_logprob(
                inp.X, inp.W, inp.labels, fr.lse, inp.dlogp, chunk_size=4096
            )
            torch.cuda.synchronize()
        assert calls == []
    for key in ("logp", "dX", "dW"):
        assert _same_bits(below[key], plain[key]), key
        assert _same_bits(nofin[key], plain_nofin[key]), key
        assert _same_bits(nodz[key], plain_nodz[key]), key
    assert _same_bits(recount[0], plain["dX"]) and _same_bits(recount[1], plain["dW"])


def test_device_zero_dlogp_skip_layouts(glm_weight, monkeypatch):
    """The skip with ``X`` read through a padded leading stride and a non-contiguous ``dlogp``: the bits of the
    contiguous call."""
    _require_skip()
    base = _device_inputs(8117, W=glm_weight)
    zero = _zero_set(base.labels, ("random", 0.166))
    inp = _with_zero_dlogp(base, zero)
    T, H = inp.X.shape
    pad = torch.zeros((T, H + 8), dtype=inp.X.dtype, device=CUDA)
    pad[:, :H] = inp.X
    strided = torch.zeros((T, 2), dtype=torch.float32, device=CUDA)
    strided[:, 1] = inp.dlogp
    with _counting_partitions(monkeypatch) as calls, _zero_dlogp_floor(0):
        want = _logprob_call(inp, 4096)
        padded = _logprob_call(inp, 4096, X=pad[:, :H])
        sparse = _logprob_call(inp, 4096, dlogp=strided[:, 1])
    assert len(calls) == 3
    for key in ("logp", "dX", "dW"):
        assert _same_bits(padded[key], want[key]), key
        assert _same_bits(sparse[key], want[key]), key


def test_device_zero_dlogp_backward_cuda_graph(glm_weight):
    """The backward with the skip captured in a CUDA graph (after a warm-up that binds it on the capture stream)
    replays with other ``dlogp`` values -- other zero rows and live counts per chunk, none and every row zero --
    written into the captured input and gives the eager backward's bits on those values."""
    _require_skip()
    base = _device_inputs(12289, W=glm_weight)
    kw = dict(chunk_size=4096)
    fr = cake_backend.forward_logprob(base.X, base.W, base.labels, **kw)
    torch.cuda.synchronize()
    assert fr.row_index is not None
    dlogp = base.dlogp.clone()

    def backward(d):
        return cake_backend.backward_logprob(
            base.X, base.W, base.labels, fr.lse, d, row_index=fr.row_index, **kw
        )

    with _zero_dlogp_floor(0):
        cake_backend.BINDING_CACHE.clear()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            backward(dlogp)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dx_g, dw_g = backward(dlogp)
        valid = base.labels >= 0
        for kind in (("random", 0.166), ("group", 0.5), "none", "all"):
            zero = _zero_set(base.labels, kind)
            new = _with_zero_dlogp(base, zero).dlogp
            dlogp.copy_(new)
            graph.replay()
            torch.cuda.synchronize()
            got_dx, got_dw = dx_g.clone(), dw_g.clone()
            want_dx, want_dw = backward(new.clone())
            torch.cuda.synchronize()
            assert not bool(_bits(got_dx[zero | ~valid]).any()), kind
            assert _same_bits(got_dx, want_dx) and _same_bits(got_dw, want_dw), kind


def test_device_zero_dlogp_rows_keep_a_nan_upstream_row_live(glm_weight):
    """A NaN ``dlogp`` row is a live row: NaN in its ``dX`` row and in ``dW``, as without the skip; the zero rows'
    ``dX`` rows stay exact ``+0``."""
    _require_skip()
    base = _device_inputs(4097, W=glm_weight)
    rows = (base.labels >= 0).nonzero().squeeze(1)
    zero = torch.zeros_like(base.labels, dtype=torch.bool)
    zero[rows[1:300]] = True
    inp = _with_zero_dlogp(base, zero)
    d = inp.dlogp.clone()
    d[rows[0]] = float("nan")
    with _zero_dlogp_floor(0):
        got = _logprob_call(replace(inp, dlogp=d))
    assert bool(torch.isnan(got["dX"][rows[0]].float()).all())
    assert bool(torch.isnan(got["dW"].float()).any())
    assert not bool(_bits(got["dX"][zero]).any())


def test_device_zero_dlogp_skip_adds_no_synchronization(glm_weight, monkeypatch):
    """With its binding remembered (the first call binds), the autograd backward with the skip -- the partition
    launch included -- runs under PyTorch's synchronization debug mode ``"error"``, as the det suite's backward."""
    _require_skip()
    base = _device_inputs(8117, W=glm_weight)
    zero = _zero_set(base.labels, ("random", 0.3))
    inp = _with_zero_dlogp(base, zero)
    cake_backend.BINDING_CACHE.clear()

    def backward(error_mode):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        logp = cake_backend.cake_lm_head_det_train_logprob(
            X, W, inp.labels, chunk_size=4096, backend="cake"
        )
        torch.cuda.synchronize()
        if error_mode:
            torch.cuda.set_sync_debug_mode("error")
        try:
            logp.backward(inp.dlogp)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        return X.grad

    with _counting_partitions(monkeypatch) as calls, _zero_dlogp_floor(0):
        for i in range(2):
            dx = backward(error_mode=bool(i))
            assert not bool(_bits(dx[zero]).any())
    assert len(calls) == 2


def test_device_zero_dlogp_backward_peak_allocated(glm_weight):
    """Memory rule: the remembered backward with the skip allocates at most the peak of the backward without it (the
    partition's outputs are offset by the row-statistics buffer the counted backward does not bind, the bool mask and
    the finalize's scan it does not form).  Reserved bytes are printed (the parent's measurement gates them)."""
    _require_skip()
    base = _device_inputs(8117, W=glm_weight)
    zero = _zero_set(base.labels, ("random", 0.166))
    inp = _with_zero_dlogp(base, zero)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)

    def peak(floor):
        with _zero_dlogp_floor(floor):
            for _ in range(2):  # the second backward replays the remembered binding
                X.grad = W.grad = None
                logp = cake_backend.cake_lm_head_det_train_logprob(
                    X, W, inp.labels, chunk_size=4096, backend="cake"
                )
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.memory_allocated()
                logp.backward(inp.dlogp)
                torch.cuda.synchronize()
                del logp
            return (
                torch.cuda.max_memory_allocated() - start,
                torch.cuda.max_memory_reserved(),
            )

    cake_backend.BINDING_CACHE.clear()
    off = peak(None)
    on = peak(0)
    print(
        f"\nzero-dlogp skip backward peak: allocated {on[0]} vs {off[0]} without the skip, "
        f"reserved {on[1]} vs {off[1]}"
    )
    assert on[0] <= off[0], (on, off)
