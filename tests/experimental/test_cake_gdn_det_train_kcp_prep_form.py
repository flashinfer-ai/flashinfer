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

CPU tests of the forward form of #1_l2n (record ``fwd_prep_l2n_fwd``: the program without the backward-only ``akk`` /
``g_cum`` outputs): the registry's ``prep_form`` selection, and on the fake backend which #1 record every forward and
backward launches, that the forward form binds neither tape, and that every saved / replay / checkpoint / handle form
gives the bits of the full #1 (test knob ``prep_form`` 0).  The GPU rows are in ``test_cake_gdn_det_train_kcp_cp1.py``
(``test_prep_forward_form_is_bitwise_the_full_prep``).
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch

PACKAGE = "flashinfer.experimental.cake.cake_gdn_det_train_kcp"


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU tests run many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (the forward-form suite took about two hours on a shared CI host, against
    11 s on a GPU node), so this module runs with at most 8 threads and oneDNN off, as the non-deterministic KCP suite
    does."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _mod(sub: str | None = None):
    try:
        from tests.cake.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_det_train_kcp", sub)
    except (ImportError, KeyError):
        return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


kcp = _mod("cake_kcp")
plan = _mod("cake_kcp_plan")
jit = _mod("cake_jit")
CPU = torch.device("cpu")
BF16 = torch.bfloat16
GRADS = ("dq", "dk", "dv", "dg", "dbeta")
TENSORS = ("out",) + GRADS
CU = [0, 100, 740, 817, 1000]
FWD, FULL = "fwd_prep_l2n_fwd", "fwd_prep_l2n"


def _inputs(cu, H_qk: int = 2, HV: int = 6) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(7005)
    T = cu[-1]
    a_log = torch.rand(HV, generator=gen) * math.log(16.0)
    return {
        "q": torch.randn(T, H_qk, 128, generator=gen).to(BF16),
        "k": torch.randn(T, H_qk, 128, generator=gen).to(BF16),
        "v": torch.randn(T, HV, 128, generator=gen).to(BF16),
        "g": (
            -torch.exp(a_log)[None, :]
            * torch.nn.functional.softplus(torch.randn(T, HV, generator=gen))
        )
        .float()
        .contiguous(),
        "beta": torch.sigmoid(torch.randn(T, HV, generator=gen)).float().contiguous(),
        "dout": torch.randn(T, HV, 128, generator=gen).to(BF16),
    }


def _setup(cu=CU):
    part = plan.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    return plan.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU), _inputs(cu)


def _ops(ops):
    return tuple(ops[n] for n in ("q", "k", "v", "g", "beta"))


def _preps(kind: str) -> list[str]:
    return [r for r, _g in kcp.LAST_CALL[kind]["records"] if r.startswith("fwd_prep")]


def _saved_pass(ctx, ops, **o) -> dict[str, torch.Tensor]:
    o = {"backend": "fake", "use_qk_l2norm_in_kernel": True, **o}
    out, saved = kcp.cake_gdn_det_train_kcp_forward(*_ops(ops), ctx, **o)
    dq, dk, dv, dbeta, dg = kcp.cake_gdn_det_train_kcp_backward(
        *_ops(ops), ctx, ops["dout"], saved=saved, **o
    )
    return {"out": out, "dq": dq, "dk": dk, "dv": dv, "dbeta": dbeta, "dg": dg}


def _autograd(
    ctx, ops, *, ckpt: bool, handle: bool = False, **o
) -> dict[str, torch.Tensor]:
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    o = {"backend": "fake", "use_qk_l2norm_in_kernel": True, **o}
    hd = kcp.CakeGDNDetKcpRecomputeHandle() if handle else None

    def fn(q, k, v, g, beta, h):
        return kcp.cake_gdn_det_kcp(q, k, v, g, beta, ctx, recompute_handle=h, **o)

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, *leaves, hd, use_reentrant=False)
    else:
        out = fn(*leaves, hd)
    out.backward(ops["dout"])
    return {
        "out": out.detach(),
        **{n: lf.grad for n, lf in zip(GRADS, leaves, strict=True)},
    }


def _same(a: dict, b: dict, what: str) -> None:
    for n in TENSORS:
        assert torch.equal(a[n], b[n]), f"{what}: {n} differs"


def test_registry_selects_the_forward_form_by_backward_tapes() -> None:
    """``prep_form`` selection: ``backward_tapes`` True (the default) the full record, False the ``_fwd`` record; the
    forward form names every kernel parameter of the full one but ``akk_out`` / ``g_cum``, with the same launch and
    grid; the other programs keep their one record."""
    assert jit.select_record(FULL) == FULL
    assert jit.select_record(FULL, backward_tapes=True) == FULL
    assert jit.select_record(FULL, backward_tapes=False) == FWD
    assert jit.select_record("fwd_prep") == "fwd_prep"
    full, fwd = jit.MODULES[FULL], jit.MODULES[FWD]
    assert (fwd["program"], fwd["variant"]) == (FULL, "fwd")
    assert full["select_rule"] == {"kind": "prep_form", "backward_tapes": True}
    assert fwd["select_rule"] == {"kind": "prep_form", "backward_tapes": False}
    assert [p for p in full["arg_plan"] if p[1] not in ("akk_out", "g_cum")] == fwd[
        "arg_plan"
    ]
    assert set(full["logical"].values()) - set(fwd["logical"].values()) == {
        "akk",
        "g_cum",
    }
    assert fwd["launch"] == full["launch"] and fwd["grid_rule"] == full["grid_rule"]
    assert fwd["arches"] == full["arches"] and fwd["library"] != full["library"]
    assert jit.records_of_program(FULL) == (FULL, FWD)
    assert kcp.prep_form_shipped() and all(
        kcp.prep_form_shipped(a) for a in full["arches"]
    )
    assert kcp.TART_MODULE_OF[FWD] == "gdn_chunk_det_train_fwd_prep_l2n_v2"


def test_forward_form_routing() -> None:
    """Which #1 record each call launches (fake backend): a light forward (saving or not) the forward form and its
    backward the full #1 (the replay); test knob ``prep_form`` 0, ``save_mode="full"`` and the norm off never the
    forward form; the replay-mode backward's forward the full #1."""
    ctx, ops = _setup()
    o = {"backend": "fake", "use_qk_l2norm_in_kernel": True}
    out, saved = kcp.cake_gdn_det_train_kcp_forward(*_ops(ops), ctx, **o)
    assert _preps("fwd") == [FWD] and kcp.LAST_CALL["fwd"]["records"]
    kcp.cake_gdn_det_train_kcp_backward(*_ops(ops), ctx, ops["dout"], saved=saved, **o)
    assert _preps("bwd") == [FULL]
    kcp.cake_gdn_det_train_kcp_forward(*_ops(ops), ctx, save_for_backward=False, **o)
    assert _preps("fwd") == [FWD]
    kcp.cake_gdn_det_train_kcp_forward(
        *_ops(ops), ctx, test_knobs={"prep_form": 0}, **o
    )
    assert _preps("fwd") == [FULL]
    _, saved = kcp.cake_gdn_det_train_kcp_forward(
        *_ops(ops), ctx, save_mode="full", **o
    )
    assert _preps("fwd") == [FULL]
    kcp.cake_gdn_det_train_kcp_backward(
        *_ops(ops), ctx, ops["dout"], saved=saved, save_mode="full", **o
    )
    assert _preps("bwd") == []  # the full backward runs no #1
    _, saved = kcp.cake_gdn_det_train_kcp_forward(
        *_ops(ops), ctx, boundary_mode="replay", **o
    )
    assert _preps("fwd") == [FWD]
    kcp.cake_gdn_det_train_kcp_backward(
        *_ops(ops), ctx, ops["dout"], saved=saved, boundary_mode="replay", **o
    )
    assert _preps("bwd") == [FULL]
    kcp.cake_gdn_det_train_kcp_forward(*_ops(ops), ctx, backend="fake")
    assert _preps("fwd") == ["fwd_prep"]


def test_forward_form_binds_no_backward_tape() -> None:
    """The arena when #1 runs: the forward form binds neither ``akk`` nor ``g_cum`` (no plane is allocated for them);
    the full #1 (test knob ``prep_form`` 0) binds both."""
    ctx, ops = _setup()
    seen: dict[int, set] = {}

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if name == FULL and not call.backward:
                seen[int(call.opts.knob("prep_form"))] = set(call.ws._tapes) & {
                    "akk",
                    "g_cum",
                }
            super().run(name, args)

    previous = kcp.set_backend("fake", Recording())
    try:
        for pf in (1, 0):
            kcp.cake_gdn_det_train_kcp_forward(
                *_ops(ops),
                ctx,
                backend="fake",
                workspace=kcp.CakeGDNDetKcpWorkspace(),
                use_qk_l2norm_in_kernel=True,
                test_knobs={"prep_form": pf},
            )
    finally:
        kcp.set_backend("fake", previous)
    assert seen == {1: set(), 0: {"akk", "g_cum"}}


@pytest.mark.parametrize("segments", [1, 2])
def test_every_form_is_bitwise_the_full_prep(segments) -> None:
    """The light / full / replay-mode saved passes, autograd, checkpoint (replay form, kept tapes, ``"auto"``, full,
    replay mode) and the recompute handle with test knob ``prep_form`` 1 give the bits of ``prep_form`` 0."""
    ctx, ops = _setup()
    for o in (
        {},
        {"save_mode": "full"},
        {"boundary_mode": "replay"},
        {"save_transfer": False},
    ):
        ref = _saved_pass(ctx, ops, segments=segments, test_knobs={"prep_form": 0}, **o)
        _same(ref, _saved_pass(ctx, ops, segments=segments, **o), f"saved {o}")
    direct = _autograd(
        ctx, ops, ckpt=False, segments=segments, test_knobs={"prep_form": 0}
    )
    _same(direct, _autograd(ctx, ops, ckpt=False, segments=segments), "autograd")
    for o in (
        {"reuse_recompute_tapes": False},
        {"reuse_recompute_tapes": True},
        {"reuse_recompute_tapes": "auto"},
        {"save_mode": "full"},
        {"boundary_mode": "replay"},
    ):
        got = _autograd(ctx, ops, ckpt=True, segments=segments, **o)
        _same(direct, got, f"checkpoint {o}")
        if o.get("reuse_recompute_tapes") is True:
            # the recompute forward hands its tapes over: it runs the full #1, the backward none
            assert kcp.LAST_CALL["bwd"]["tapes_source"] == "kept"
            assert _preps("bwd") == []
    _same(
        direct,
        _autograd(
            ctx, ops, ckpt=True, handle=True, save_transfer=False, segments=segments
        ),
        "handle",
    )
