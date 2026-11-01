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

Per-head views and activation checkpointing of the Cake BF16 RMSNorm training
kernels with the non-deterministic weight gradient (``deterministic=False``;
issues #7045 items 2-3, #7039, #8066).

Per-head views: the H=256 query view ``[T, 24, 256]`` of a ``[T, 24, 512]``
buffer (strides ``(12288, 512, 1)``) and the contiguous key view ``[T, 4, 256]``
share one ``w[256]``, at irregular and large token counts (1 to 65536 tokens, up
to 1.5 M rows) and at token counts whose rows fall in every backward window of
the device's H=256 route (the window form -- single kernel or deterministic pair
-- is read from the registry per case).  Each case checks ``y`` / ``rstd`` /
``dx`` / ``dw`` against FP64, the launched records against the window's, ``dx``
bitwise equal to the 2-D call on a contiguous copy of the collapsed rows, a few
repeats (``dx`` stable, ``dw`` within the budget, the workspace at rest).

Checkpointing (``torch.utils.checkpoint``, ``use_reentrant=False``): the
replayed forward of the per-head views, H=5120 (2-D and an N-D ``[B, S, H]``
input) and the residual H=6144 route gives the same ``y`` and input gradients
bit for bit as the plain autograd call, with the kernel's FP32 ``dw`` (not
bitwise reproducible) at the deterministic budget; checkpointed micro-batches
sharing one weight accumulate into existing ``.grad`` tensors exactly as
autograd adds them; and a checkpointed forward keeps nothing on the device
beyond its outputs (measured with ``torch.cuda.memory_allocated``; the plain
forward also keeps the FP32 ``rstd``), while the backward leaves only the
gradients.

This suite imports only this package (through ``tests.test_helpers.cake_package``);
it runs unchanged against a vendored copy with or without the deterministic
sibling.
"""

from __future__ import annotations

import functools

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers import cake_rmsnorm_train_reference as ref
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")
forward = functools.partial(pkg.cake_rmsnorm_train_forward, deterministic=False)
backward = functools.partial(pkg.cake_rmsnorm_train_backward, deterministic=False)
workspace_for = functools.partial(
    pkg.cake_rmsnorm_train_backward_workspace, deterministic=False
)
rmsnorm = functools.partial(pkg.cake_rmsnorm, deterministic=False)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
W_DTYPES = ref.W_DTYPES
EPS = ref.EPS
SEED = 7045
VIEWS = {"query": 24, "key": 4}
# Irregular and large token counts of the per-head views (rows = tokens * heads).
VIEW_TOKENS = (1, 7, 1000, 4097, 16385, 65536)
VIEW_REPEATS = 3
# Retained device memory beyond the outputs: the caching allocator rounds every
# block up to 512 bytes.
ALLOCATOR_SLACK = 2048


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_rmsnorm_train_supported(_device(), deterministic=False)
    )


def _arch() -> str:
    return loader.arch_for_capability(_capability())


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train(deterministic=False) needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


# Module launches of the running test and the FP32 dw buffer of every backward
# module (the kernel's dw before the autograd wrapper casts it).
_LAUNCHED: list[str] = []
_DW_BUFFERS: list[torch.Tensor] = []


@pytest.fixture(autouse=True)
def record_launches(monkeypatch: pytest.MonkeyPatch):
    run_module = loader.run_module

    def recording(name, values, *, rows, device_index):
        _LAUNCHED.append(name)
        if loader.MODULES[name]["role"].startswith("bwd"):
            _DW_BUFFERS.append(values["dw"])
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", recording)
    _LAUNCHED.clear()
    _DW_BUFFERS.clear()
    yield
    _LAUNCHED.clear()
    _DW_BUFFERS.clear()


def _skip_unless_route(hidden: int, residual: bool = False, w: str = "bf16") -> None:
    for kind in ("fwd", "bwd"):
        if not loader.route_applies(
            device_capability=_capability(),
            hidden=hidden,
            kind=kind,
            residual=residual,
            w_dtype=w,
        ):
            pytest.skip(
                f"hidden={hidden} residual={residual} w={w} has no exported {kind} route here"
            )


def _records(hidden: int, residual: bool, w: str, rows: int, kind: str = "bwd"):
    return ref.window_records(loader, _arch(), hidden, kind, residual, w, rows)


def _generator(*keys) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def _randn(shape, generator, scale: float = 1.0) -> torch.Tensor:
    return (
        scale
        * torch.randn(shape, dtype=torch.float32, device=_device(), generator=generator)
    ).to(torch.bfloat16)


def _weight(hidden: int, generator, w: str) -> torch.Tensor:
    weight = 1.0 + 0.1 * torch.randn(
        hidden, dtype=torch.float32, device=_device(), generator=generator
    )
    return weight.to(W_DTYPES[w])


def _view(view: str, tokens: int, generator) -> torch.Tensor:
    """The per-head input: the query view of a ``[T, 24, 512]`` buffer or a contiguous key ``[T, 4, 256]``."""
    if view == "query":
        x = _randn((tokens, 24, 512), generator)[:, :, :256]
        assert x.stride() == (12288, 512, 1)
        return x
    return _randn((tokens, 4, 256), generator)


# --------------------------------------------------------------------------- per-head views


def _check_view(view: str, w: str, tokens: int, key: int) -> str:
    """One per-head case (module docstring); returns its report line."""
    heads = VIEWS[view]
    rows = tokens * heads
    generator = _generator(tokens, heads, w == "f32", key)
    x = _view(view, tokens, generator)
    weight = _weight(256, generator, w)
    g = _randn(x.shape, generator)
    eps = EPS[256]
    start = len(_LAUNCHED)
    y, rstd = forward(x, weight, eps)
    fwd_records = _LAUNCHED[start:]
    label = f"view={view} w={w} tokens={tokens} rows={rows}"
    assert y.shape == x.shape and y.is_contiguous(), label
    assert rstd.shape == x.shape[:-1] and rstd.dtype == torch.float32, label
    assert fwd_records == _records(256, False, w, rows, "fwd"), (label, fwd_records)
    y_ref, rstd_ref, _n = ref.reference_forward(x, weight, eps)
    ref.assert_bf16_close(y, y_ref, f"y {label}")
    del y_ref
    assert ref.rel_l2(rstd, rstd_ref) <= ref.RSTD_REL_L2, f"rstd {label}"
    expected = _records(256, False, w, rows)
    form = ref.form(loader, expected)
    dx_ref, dw_ref = ref.reference_backward(g, x, weight, rstd)
    ws = workspace_for(rows, 256, _device())
    dx0 = dw0 = None
    worst = worst_tol = spread = 0.0
    for _ in range(VIEW_REPEATS):
        start = len(_LAUNCHED)
        dx, dw = backward(g, x, weight, rstd, workspace=ws)
        assert _LAUNCHED[start:] == expected, (label, _LAUNCHED[start:], expected)
        assert dx.shape == x.shape and dx.is_contiguous() and dw.shape == (256,)
        if dx0 is None:
            dx0, dw0 = dx, dw
            ref.assert_bf16_close(dx, dx_ref, f"dx {label}")
        assert torch.equal(dx, dx0), f"{label}: dx changed between launches"
        error, tol = ref.assert_dw(dw, dw_ref, g, x, rstd, f"dw {label}", form)
        worst, worst_tol = max(worst, error), max(worst_tol, tol)
        spread = max(spread, (dw - dw0).abs().max().item())
        assert ref.workspace_at_rest(loader, ws), f"{label}: workspace not at rest"
    del dx_ref
    # the same rows as a contiguous 2-D call: the same program, the same dx bits
    dx2, _dw2 = backward(
        g.reshape(rows, 256),
        x.reshape(rows, 256).contiguous(),
        weight,
        rstd.reshape(rows),
    )
    assert torch.equal(dx0.reshape(rows, 256), dx2), (
        f"{label}: dx differs from the 2-D call"
    )
    program = loader.MODULES[expected[0]]["library"].rsplit("_", 1)[1][:8]
    return (
        f"VIEWS {_arch()} view={view} w={w} tokens={tokens} rows={rows} form={form} "
        f"program={program} repeats={VIEW_REPEATS} dw_rel_l2={worst:.3e} "
        f"dw_worst_tol_ratio={worst_tol:.3f} dw_max_spread={spread:.3e}"
    )


@pytest.mark.parametrize("tokens", VIEW_TOKENS)
@pytest.mark.parametrize("w", tuple(W_DTYPES))
@pytest.mark.parametrize("view", tuple(VIEWS))
def test_per_head_views_at_irregular_and_large_token_counts(
    view: str, w: str, tokens: int
) -> None:
    _skip_unless_route(256, w=w)
    print(_check_view(view, w, tokens, key=1))


def _window_tokens(heads: int, w: str) -> list[int]:
    """Token counts whose rows land on both ends of every backward window of the
    device's H=256 route (capped at 65536 tokens)."""
    tokens = set()
    for above, at_most, _library in ref.windows(loader, _arch(), 256, "bwd", False, w):
        first = above // heads + 1  # the smallest T with T * heads > above
        tokens.add(min(first, 65536))
        if at_most is not None:
            last = at_most // heads  # the largest T with T * heads <= at_most
            if last * heads > above:
                tokens.add(min(last, 65536))
    return sorted(t for t in tokens if t >= 1)


@pytest.mark.parametrize("w", tuple(W_DTYPES))
@pytest.mark.parametrize("view", tuple(VIEWS))
def test_per_head_views_in_every_backward_window(view: str, w: str) -> None:
    """Token counts at both ends of every H=256 backward window of the device
    (e.g. the sm_107a key view where the route switches between the
    deterministic pair and the single kernel)."""
    _skip_unless_route(256, w=w)
    forms = []
    for tokens in _window_tokens(VIEWS[view], w):
        line = _check_view(view, w, tokens, key=2)
        print(line)
        forms.append(line.split(" form=")[1].split()[0])
    assert forms, "no window token count"
    print(
        f"VIEWS_WINDOWS {_arch()} view={view} w={w} cases={len(forms)} "
        f"single={forms.count('single')} pair={forms.count('pair')}"
    )


# --------------------------------------------------------------------------- checkpointing


def _ckpt_cases():
    for w in W_DTYPES:
        for name in ("query", "key", "h5120", "h5120_nd", "res_h6144_nd"):
            for tokens in (1000, 4097):
                yield pytest.param(name, w, tokens, id=f"{name}-{w}-T{tokens}")


def _ckpt_inputs(name: str, w: str, tokens: int, key: int):
    """``(hidden, residual, x, u, g, g_h, weight)`` of a checkpoint case: the
    per-head views, H=5120 ``[T, H]`` and ``[4, T / 4, H]``-like N-D inputs, the
    residual H=6144 route on ``[2, ceil(T / 2), H]``."""
    generator = _generator(tokens, key, w == "f32", sum(map(ord, name)))
    if name in VIEWS:
        x = _view(name, tokens, generator)
        hidden, residual = 256, False
    elif name == "h5120":
        hidden, residual = 5120, False
        x = _randn((tokens, hidden), generator)
    elif name == "h5120_nd":
        hidden, residual = 5120, False
        x = _randn((4, (tokens + 3) // 4, hidden), generator)
    else:
        hidden, residual = 6144, True
        x = _randn((2, (tokens + 1) // 2, hidden), generator)
    u = _randn(x.shape, generator) if residual else None
    g = _randn(x.shape, generator)
    g_h = _randn(x.shape, generator) if residual else None
    return hidden, residual, x, u, g, g_h, _weight(hidden, generator, w)


def _autograd(xs, weight, eps, us, gs, g_hs, checkpointed: bool):
    """Forward and backward of one or more micro-batches sharing ``weight`` (a
    leaf); returns ``(outputs, input grads, residual grads)`` per micro-batch."""
    results = []
    for x, u, g, g_h in zip(xs, us, gs, g_hs, strict=True):
        xa = x.clone().requires_grad_(True)
        ua = u.clone().requires_grad_(True) if u is not None else None

        def step(xt, wt, ut):
            out = rmsnorm(xt, wt, eps, residual=ut)
            return out if ut is not None else (out,)

        if checkpointed:
            outs = torch.utils.checkpoint.checkpoint(
                step, xa, weight, ua, use_reentrant=False
            )
        else:
            outs = step(xa, weight, ua)
        torch.autograd.backward(outs, (g, g_h) if u is not None else (g,))
        results.append(
            (
                tuple(o.detach() for o in outs),
                xa.grad,
                ua.grad if ua is not None else None,
            )
        )
    return results


@pytest.mark.parametrize(("name", "w", "tokens"), list(_ckpt_cases()))
def test_checkpoint_replays_views_and_nd_inputs(name: str, w: str, tokens: int) -> None:
    """A checkpointed micro-batch (non-reentrant) gives the plain call's ``y``,
    ``h_new`` and input gradients bit for bit; both kernel ``dw`` are at the
    deterministic budget (and the autograd ``dw`` is the kernel's cast to the
    weight dtype)."""
    hidden, residual, x, u, g, g_h, weight = _ckpt_inputs(name, w, tokens, key=3)
    _skip_unless_route(hidden, residual, w)
    eps = EPS[hidden]
    rows = x.numel() // hidden
    outputs = forward(x, weight, eps, residual=u)
    normalized, rstd = (outputs[2] if residual else x), outputs[1]
    dx_ref, dw_ref = ref.reference_backward(g, normalized, weight, rstd, g_h=g_h)
    form = ref.form(loader, _records(hidden, residual, w, rows))
    label = f"ckpt {name} w={w} T={tokens} rows={rows}"
    dws = {}
    results = {}
    for checkpointed in (True, False):
        wa = weight.clone().requires_grad_(True)
        start = len(_DW_BUFFERS)
        results[checkpointed] = _autograd(
            [x], wa, eps, [u], [g], [g_h], checkpointed=checkpointed
        )[0]
        launches = len(_DW_BUFFERS) - start
        assert launches == len(_records(hidden, residual, w, rows)), (
            f"{label}: {launches} backward module launches"
        )
        dw32 = _DW_BUFFERS[-1].clone()
        ref.assert_dw(
            dw32, dw_ref, g, normalized, rstd, f"{label} ckpt={checkpointed}", form
        )
        assert wa.grad.dtype == weight.dtype and torch.equal(
            wa.grad, dw32.to(weight.dtype)
        )
        dws[checkpointed] = dw32
    (outs_c, dx_c, du_c), (outs_p, dx_p, du_p) = results[True], results[False]
    assert all(torch.equal(a, b) for a, b in zip(outs_c, outs_p, strict=True)), label
    assert dx_c.shape == x.shape and torch.equal(dx_c, dx_p), f"{label}: dx"
    if residual:
        assert torch.equal(du_c, du_p) and torch.equal(du_c, dx_c), f"{label}: du"
    ref.assert_bf16_close(dx_c, dx_ref, f"dx {label}")
    print(
        f"CKPT {_arch()} {name} w={w} T={tokens} rows={rows} form={form} "
        f"y_dx_bitwise=True dw_ckpt_vs_plain_max={(dws[True] - dws[False]).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("w", tuple(W_DTYPES))
@pytest.mark.parametrize("name", ("query", "h5120_nd", "res_h6144_nd"))
def test_checkpointed_micro_batches_accumulate_into_existing_grads(
    name: str, w: str
) -> None:
    """Three checkpointed micro-batches (different inputs, one shared weight leaf
    with a preset gradient) accumulate exactly as autograd adds: ``w.grad`` is the
    preset plus each kernel ``dw`` (cast to the weight dtype) in order, and every
    kernel ``dw`` is at the deterministic budget against its micro-batch's FP64
    reference."""
    cases = [_ckpt_inputs(name, w, 1000 + 37 * i, key=5 + i) for i in range(3)]
    hidden, residual = cases[0][0], cases[0][1]
    _skip_unless_route(hidden, residual, w)
    eps = EPS[hidden]
    weight = cases[0][6]
    preset = (_weight(hidden, _generator(hidden, 53), w) - 1.0).to(weight.dtype)
    wa = weight.clone().requires_grad_(True)
    wa.grad = preset.clone()
    expected = preset.clone()
    for index, (_h, _r, x, u, g, g_h, _w) in enumerate(cases):
        rows = x.numel() // hidden
        start = len(_DW_BUFFERS)
        (outs, dx, du) = _autograd([x], wa, eps, [u], [g], [g_h], checkpointed=True)[0]
        assert len(_DW_BUFFERS) - start == len(_records(hidden, residual, w, rows))
        dw32 = _DW_BUFFERS[-1].clone()
        outputs = forward(x, weight, eps, residual=u)
        normalized, rstd = (outputs[2] if residual else x), outputs[1]
        dx_ref, dw_ref = ref.reference_backward(g, normalized, weight, rstd, g_h=g_h)
        form = ref.form(loader, _records(hidden, residual, w, rows))
        ref.assert_dw(dw32, dw_ref, g, normalized, rstd, f"micro-batch {index}", form)
        ref.assert_bf16_close(dx, dx_ref, f"micro-batch {index} dx")
        expected = expected + dw32.to(weight.dtype)
        assert torch.equal(wa.grad, expected), (
            f"{name} w={w}: w.grad after micro-batch {index}"
        )
    print(f"CKPT_ACCUM {_arch()} {name} w={w} micro_batches={len(cases)} exact=True")


def _retained_cases():
    for w in W_DTYPES:
        for name in ("query", "key", "h5120", "h5120_nd", "res_h6144_nd"):
            yield pytest.param(name, w, id=f"{name}-{w}")


@pytest.mark.parametrize(("name", "w"), list(_retained_cases()))
def test_checkpointed_forward_retains_only_its_outputs(name: str, w: str) -> None:
    """Device memory held after a checkpointed forward is its outputs (``y`` and,
    with a residual, ``h_new``) to the allocator's rounding: 0 MiB beyond them
    (no ``rstd``, no workspace, no copy of an input).  The plain forward holds
    the outputs and the FP32 ``rstd``.  After the backward and once the outputs
    are dropped, only the gradients remain."""
    tokens = 4097
    hidden, residual, x, u, g, g_h, weight = _ckpt_inputs(name, w, tokens, key=7)
    _skip_unless_route(hidden, residual, w)
    eps = EPS[hidden]
    rows = x.numel() // hidden
    # load the modules and fill the allocator's pools first
    _autograd([x], weight.clone().requires_grad_(True), eps, [u], [g], [g_h], True)
    _DW_BUFFERS.clear()
    report = {}
    for checkpointed in (True, False):
        xa = x.clone().requires_grad_(True)
        ua = u.clone().requires_grad_(True) if residual else None
        wa = weight.clone().requires_grad_(True)

        def step(xt, wt, ut):
            out = rmsnorm(xt, wt, eps, residual=ut)
            return out if ut is not None else (out,)

        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        if checkpointed:
            outs = torch.utils.checkpoint.checkpoint(
                step, xa, wa, ua, use_reentrant=False
            )
        else:
            outs = step(xa, wa, ua)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - base
        outputs_bytes = sum(o.untyped_storage().nbytes() for o in outs)
        beyond = held - outputs_bytes
        torch.cuda.reset_peak_memory_stats()
        peak_base = torch.cuda.memory_allocated()
        torch.autograd.backward(outs, (g, g_h) if residual else (g,))
        torch.cuda.synchronize()
        backward_peak = torch.cuda.max_memory_allocated() - peak_base
        del outs
        _DW_BUFFERS.clear()
        torch.cuda.synchronize()
        grads = [xa.grad, wa.grad] + ([ua.grad] if residual else [])
        grads_bytes = sum(
            {
                t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in grads
            }.values()
        )
        left = torch.cuda.memory_allocated() - base - grads_bytes
        report[checkpointed] = (held, outputs_bytes, beyond, backward_peak, left)
        label = f"{name} w={w} rows={rows} checkpointed={checkpointed}"
        if checkpointed:
            assert beyond <= ALLOCATOR_SLACK, (
                f"{label}: a checkpointed forward holds {beyond} bytes beyond its "
                f"outputs ({outputs_bytes} bytes)"
            )
        else:
            assert beyond <= 4 * rows + ALLOCATOR_SLACK, (
                f"{label}: the forward holds {beyond} bytes beyond its outputs "
                f"(the FP32 rstd is {4 * rows})"
            )
        assert left <= ALLOCATOR_SLACK, f"{label}: {left} bytes left after the backward"
        del xa, ua, wa, grads
    mib = 1 / (1 << 20)
    (
        (_c_held, c_out, c_beyond, c_peak, c_left),
        (_p_held, _p_out, p_beyond, p_peak, p_left),
    ) = (
        report[True],
        report[False],
    )
    print(
        f"RETAINED {_arch()} {name} w={w} rows={rows} outputs_mib={c_out * mib:.3f} "
        f"ckpt_beyond_outputs_mib={c_beyond * mib:.4f} plain_beyond_outputs_mib={p_beyond * mib:.4f} "
        f"rstd_mib={4 * rows * mib:.4f} ckpt_bwd_peak_mib={c_peak * mib:.2f} "
        f"plain_bwd_peak_mib={p_peak * mib:.2f} left_after_bwd_bytes={c_left}/{p_left}"
    )
