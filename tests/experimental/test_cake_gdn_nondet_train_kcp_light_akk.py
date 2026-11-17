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

CPU tests of ``save_mode="light_akk"`` of the non-deterministic KCP GDN package
(``flashinfer.experimental.cake.cake_gdn_nondet_train_kcp``, ``use_qk_l2norm_in_kernel=True``): the forward keeps #1's BF16
``Akk`` tape and the light replay runs ``fwd_prep_replay_l2n`` (#1_l2n reading that tape instead of forming it).

* the default: a call with the in-kernel norm that leaves ``save_mode`` unset runs ``"light_akk"`` (the options, the
  entries, autograd, the checkpoint form and the byte queries); without the norm it runs ``"light"``; explicit values
  keep their meaning;
* the fake program: every output but ``akk`` bitwise #1_l2n's, ``akk`` read and never written, and a changed tape
  changes ``w`` / ``u`` (it is read);
* the host on the fake backend: ``"light_akk"`` is bitwise ``"light"`` in every form (saved pass, ``save_transfer``
  off, segments, autograd, checkpoint, replay boundary); the launch census (the backward's #1_l2n becomes the
  replay, nothing else moves); the saved set (the ``Akk`` tape, the round trip and the refusals); a corrupted saved
  tape reaches the gradients; under ``torch.utils.checkpoint(use_reentrant=False)`` the mode saves as ``"light"``;
* the byte model: the saved set gains ``128 R HV`` bytes, the forward and backward arenas lose ``Akk``, a caller-held
  workspace binds what ``cake_gdn_nondet_train_kcp_workspace_layout`` reports for both modes, and under
  ``torch.utils.checkpoint(use_reentrant=False)`` a ``"light_akk"`` call binds the ``save_mode="light"`` figure.

The bitwise checks here hold on the deterministic fake backend (host plumbing), which is not bitwise the device
programs.  The device rows (``"light_akk"`` within the budgets of ``"light"`` on NaN-filled allocations with the launch
census and the saved bytes, ``graph="replay"`` steps, the multi-rank rows; bits recorded as data, never gated) are in
``test_cake_gdn_nondet_train_kcp_cp1.py`` and ``test_cake_gdn_nondet_train_kcp_dist.py``; the two-rank fake-backend rows in
``test_cake_gdn_nondet_train_kcp.py``.
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import pytest
import torch


def _mod(sub: str | None = None):
    """The package (or one of its modules) through the test-helper registry: a stale entry fails here, never skips."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_nondet_train_kcp", sub)


kcp = _mod("cake_kcp")
plan = _mod("cake_kcp_plan")
gbytes = _mod("cake_bytes")
opt = _mod("cake_options")
jit = _mod("cake_jit")
CPU = torch.device("cpu")
GRADS = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd order of the leaves (q, k, v, g, beta)
TENSORS = ("out",) + GRADS
BF16 = torch.bfloat16
REPLAY = "fwd_prep_replay_l2n"
L2N = dict(use_qk_l2norm_in_kernel=True)
CU = [0, 100, 740, 817, 1000]
PREP_OUTS = ("w", "u", "aqk", "g_cum", "rowscale", "qscale", "gamma")


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (see ``test_cake_gdn_nondet_train_kcp.py``), so this module runs with at most 8
    threads and oneDNN off, as that suite does."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _raw_inputs(
    cu: list[int], H_qk: int, HV: int, seed: int = 20261104
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    T = cu[-1]
    a_log = torch.rand(HV, generator=gen) * math.log(16.0)

    def raw(H):
        scale = torch.exp2(torch.randint(-8, 9, (T, H, 1), generator=gen).float())
        return (torch.randn(T, H, 128, generator=gen) * scale).to(BF16)

    return {
        "q": raw(H_qk),
        "k": raw(H_qk),
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


def _setup(H_qk: int = 2, HV: int = 6, cu=CU):
    part = plan.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    return ctx, _raw_inputs(cu, H_qk, HV)


def _ops(ops):
    return tuple(ops[n] for n in ("q", "k", "v", "g", "beta"))


def _fwd(ctx, ops, ws=None, **o):
    return kcp.cake_gdn_nondet_train_kcp_forward(
        *_ops(ops), ctx, workspace=ws, backend="fake", **L2N, **o
    )


def _bwd(ctx, ops, saved, ws=None, **o):
    return kcp.cake_gdn_nondet_train_kcp_backward(
        *_ops(ops),
        ctx,
        ops["dout"],
        saved=saved,
        workspace=ws,
        backend="fake",
        **L2N,
        **o,
    )


def _saved_pass(ctx, ops, ws=None, **o) -> dict:
    out, saved = _fwd(ctx, ops, ws, **o)
    fwd = dict(kcp.LAST_CALL["fwd"])
    dq, dk, dv, dbeta, dg = _bwd(ctx, ops, saved, ws, **o)
    return {
        "out": out,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dbeta": dbeta,
        "dg": dg,
        "saved": saved,
        "fwd": fwd,
        "bwd": dict(kcp.LAST_CALL["bwd"]),
    }


def _autograd(ctx, ops, *, ckpt: bool, **o) -> dict:
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def fn(q, k, v, g, beta):
        return kcp.cake_gdn_nondet_kcp(
            q, k, v, g, beta, ctx, backend="fake", **L2N, **o
        )

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, *leaves, use_reentrant=False)
    else:
        out = fn(*leaves)
    out.backward(ops["dout"])
    return {
        "out": out.detach(),
        **{n: lf.grad for n, lf in zip(GRADS, leaves, strict=True)},
    }


def _same(a: dict, b: dict, what: str, names=TENSORS) -> None:
    for n in names:
        assert torch.equal(a[n], b[n]), f"{what}: {n} differs"


def _local_call(ctx, ops, *, backward: bool):
    opts = kcp.make_options(backend="fake", **L2N)
    call, _ = kcp._prepare(
        *_ops(ops),
        ctx,
        scale=None,
        opts=opts,
        workspace=kcp.CakeGDNNondetKcpWorkspace(),
        backward=backward,
        dout=ops["dout"] if backward else None,
    )
    return call


# ----------------------------------------------------------------------------------------------------------------
# the vocabulary


def test_the_tier_and_the_program_are_registered() -> None:
    assert (
        opt.SAVE_MODES == ("light", "light_akk", "full")
        and opt.SAVE_MODE_DEFAULT == "light"
        and opt.SAVE_MODE_DEFAULT_L2N == "light_akk"
    )
    assert (
        opt.SAVE_MODES_AKK == ("light_akk",)
        and kcp.AKK_SAVE_MODES == opt.SAVE_MODES_AKK
    )
    assert (
        gbytes.SAVE_MODES == opt.SAVE_MODES
        and gbytes.AKK_SAVE_MODES == kcp.AKK_SAVE_MODES
    )
    assert kcp.saved_tapes("light_akk") == kcp.AKK_TAPES == ("akk",)
    assert kcp.saved_tapes("light") == () and kcp.saved_tapes("full") == kcp.FWD_TAPES
    assert (
        kcp.LIGHT_FORWARD_SAVE_MODES
        == ("light", "light_akk")
        == gbytes.LIGHT_FORWARD_SAVE_MODES
    )
    assert (
        tuple(gbytes.TAPE_NAME_OF.get(n, n) for n in gbytes.LIGHT_TAPE_ORDER)
        == kcp.FWD_TAPES
    )
    assert (
        REPLAY in kcp.PROGRAMS
        and REPLAY in kcp.LOCAL_PROGRAMS
        and REPLAY not in kcp.L2N_PROGRAMS
    )
    assert kcp.AKK_REPLAY_PROGRAMS == (REPLAY,) and kcp.AKK_REPLAY_OF == {
        "fwd_prep_l2n": REPLAY
    }
    assert kcp.PROGRAM_ARGS[REPLAY] == kcp.PROGRAM_ARGS["fwd_prep_l2n"]
    assert kcp.TART_MODULE_OF[REPLAY] == "gdn_chunk_train_fwd_prep_replay_l2n_v1"
    # the registry record: #1_l2n's launch with akk read; a local program (no _kcp library)
    assert REPLAY in jit.PROGRAMS and REPLAY in jit.LOCAL_PROGRAMS
    assert jit.select_record(REPLAY) == REPLAY
    rec, base = jit.MODULES[REPLAY], jit.MODULES["fwd_prep_l2n"]
    assert rec["program"] == REPLAY and not rec["library"].endswith("_kcp")
    assert rec["logical"]["akk"] == "akk" and set(rec["logical"].values()) <= set(
        kcp.PROGRAM_ARGS[REPLAY]
    )
    assert rec["launch"] == base["launch"] and rec["grid_rule"] == base["grid_rule"]
    for name in ("light", "light_akk", "full"):
        assert name not in (
            "prep",
            "walk",
            "exchange",
            "end",
            "graph",
            "replay_forward",
            "recompute_forward",
        )


def test_light_akk_needs_the_norm() -> None:
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        kcp.make_options(save_mode="light_akk")
    assert kcp.make_options(save_mode="light_akk", **L2N).save_mode == "light_akk"
    ctx, ops = _setup()
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        kcp.cake_gdn_nondet_train_kcp_forward(
            *_ops(ops), ctx, backend="fake", save_mode="light_akk"
        )
    for query in (
        gbytes.cake_gdn_nondet_train_kcp_saved_layout,
        gbytes.cake_gdn_nondet_train_kcp_workspace_layout,
    ):
        with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
            query(ctx, H_qk=2, HV=6, save_mode="light_akk")
        query(ctx, H_qk=2, HV=6, save_mode="light_akk", **L2N)


def test_light_akk_is_the_default_with_the_norm() -> None:
    """A call with the in-kernel norm that leaves ``save_mode`` unset runs ``"light_akk"``, a call without the norm runs
    ``"light"`` (``None`` is the unset value of the options, of the entries' ``save_mode`` keyword and of the byte
    queries); an explicit value keeps its meaning; an options instance that left the mode unset resolves it again
    when a call merges ``use_qk_l2norm_in_kernel`` into it."""
    mo = kcp.make_options
    assert opt.default_save_mode(True) == "light_akk"
    assert opt.default_save_mode(False) == "light"
    assert mo(**L2N).save_mode == "light_akk" and mo().save_mode == "light"
    assert mo(save_mode=None, **L2N).save_mode == "light_akk"
    assert mo(save_mode=None).save_mode == "light"
    assert opt.CakeGDNNondetKcpOptions(**L2N).save_mode == "light_akk"
    assert opt.CakeGDNNondetKcpOptions().save_mode == "light"
    for mode in opt.SAVE_MODES:
        assert mo(save_mode=mode, **L2N).save_mode == mode
    assert mo(save_mode="light").save_mode == "light"
    assert mo(save_mode="full").save_mode == "full"
    assert mo(**L2N).key() == mo(save_mode="light_akk", **L2N).key()
    assert mo(**L2N).key() != mo(save_mode="light", **L2N).key()
    unset = opt.CakeGDNNondetKcpOptions(backend="fake")
    assert mo(unset, **L2N).save_mode == "light_akk"
    assert mo(mo(unset, **L2N), use_qk_l2norm_in_kernel=False).save_mode == "light"
    explicit = opt.CakeGDNNondetKcpOptions(backend="fake", save_mode="light")
    assert mo(explicit, **L2N).save_mode == "light"
    # the entries
    ctx, ops = _setup()
    out, saved = _fwd(ctx, ops)
    assert saved.save_mode == "light_akk" and len(saved.tapes) == 1
    _bwd(ctx, ops, saved)
    bwd = kcp.LAST_CALL["bwd"]
    assert bwd["save_mode"] == "light_akk" and bwd["akk_source"] == "saved"
    assert REPLAY in bwd["sequence"]
    _o, light = _fwd(ctx, ops, save_mode="light")
    assert saved.bytes - light.bytes == int(ctx.R) * ops["v"].shape[1] * 128
    _o, off = kcp.cake_gdn_nondet_train_kcp_forward(*_ops(ops), ctx, backend="fake")
    assert off.save_mode == "light"
    # autograd: the direct node runs the mode, a non-reentrant checkpoint saves as "light"; the same bits
    direct = _autograd(ctx, ops, ckpt=False)
    assert kcp.LAST_CALL["bwd"]["save_mode"] == "light_akk"
    assert kcp.LAST_CALL["bwd"]["akk_source"] == "saved"
    a_light = _autograd(ctx, ops, ckpt=False, save_mode="light")
    _same(direct, a_light, "default vs light")
    ckpt = _autograd(ctx, ops, ckpt=True)
    assert kcp.LAST_CALL["bwd"]["save_mode"] == "light"
    assert kcp.LAST_CALL["bwd"]["akk_source"] is None
    _same(direct, ckpt, "checkpointed default")
    # the byte queries: unset is the call's default
    H_qk, HV = ops["q"].shape[1], ops["v"].shape[1]
    for query in (
        gbytes.cake_gdn_nondet_train_kcp_saved_layout,
        gbytes.cake_gdn_nondet_train_kcp_workspace_layout,
        gbytes.cake_gdn_nondet_train_kcp_saved_bytes,
        gbytes.cake_gdn_nondet_train_kcp_workspace_bytes,
    ):
        want = query(ctx, H_qk=H_qk, HV=HV, save_mode="light_akk", **L2N)
        assert query(ctx, H_qk=H_qk, HV=HV, **L2N) == want, query.__name__
        assert query(ctx, H_qk=H_qk, HV=HV, save_mode=None, **L2N) == want
        assert query(ctx, H_qk=H_qk, HV=HV) == query(
            ctx, H_qk=H_qk, HV=HV, save_mode="light"
        ), query.__name__
    for query in (
        kcp.cake_gdn_nondet_kcp,
        kcp.cake_gdn_nondet_train_kcp_forward,
        kcp.cake_gdn_nondet_train_kcp_backward,
        gbytes.cake_gdn_nondet_train_kcp_saved_bytes,
        gbytes.cake_gdn_nondet_train_kcp_workspace_bytes,
    ):
        assert inspect.signature(query).parameters["save_mode"].default is None


# ----------------------------------------------------------------------------------------------------------------
# the fake program


def test_fake_replay_is_fwd_prep_l2n_reading_the_akk_tape() -> None:
    """The replay's outputs are #1_l2n's bit for bit (every one, the norm's rows and r included) from #1_l2n's ``Akk``;
    ``akk`` is read, never written; it is a launch of its own record; a changed tape changes ``w`` / ``u`` only."""
    ctx, ops = _setup()
    fwd = _local_call(ctx, ops, backward=False)
    for name in (*kcp.FWD_TAPES, *kcp.QK_TAPES, kcp.QK_RSTD):
        fwd.tapes[name] = fwd.tape(name).zero_()
    fwd.local("fwd_prep")
    assert fwd.sequence == ["fwd_prep_l2n"]
    want = {
        n: fwd.tape(n).clone() for n in (*PREP_OUTS, "akk", *kcp.QK_TAPES, kcp.QK_RSTD)
    }

    def replay(akk: torch.Tensor):
        call = _local_call(ctx, ops, backward=True)
        for name in (*kcp.FWD_TAPES, *kcp.QK_TAPES, kcp.QK_RSTD):
            call.tapes[name] = call.tape(name).zero_()
        call.tapes["akk"] = akk
        call.scratch["akk_source"] = "saved"
        call.local("fwd_prep")
        assert call.sequence == [REPLAY] and [r for r, _ in call.records] == [REPLAY]
        return call

    akk = want["akk"].clone()
    call = replay(akk)
    for n in (*PREP_OUTS, *kcp.QK_TAPES, kcp.QK_RSTD):
        assert torch.equal(call.tape(n), want[n]), n
    assert torch.equal(akk, want["akk"])  # read only
    # the program is the record's: launched directly it is refused as a local name
    with pytest.raises(ValueError, match="not a local program"):
        call.local(REPLAY)
    # negative control: a changed tape changes w / u (it is read), nothing else
    bad = want["akk"].clone()
    nz = bad.view(-1).nonzero()[0, 0]
    bad.view(-1)[nz] = bad.view(-1)[nz] * 2 + 1
    call = replay(bad)
    assert not torch.equal(call.tape("w"), want["w"]) and not torch.equal(
        call.tape("u"), want["u"]
    )
    for n in ("aqk", "g_cum", "rowscale", "qscale", "gamma", *kcp.QK_TAPES):
        assert torch.equal(call.tape(n), want[n]), n


# ----------------------------------------------------------------------------------------------------------------
# the host: bitwise, census, saved set


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("save_transfer", [True, False])
def test_light_akk_is_bitwise_light(segments, save_transfer, monkeypatch) -> None:
    # the kernels write every element of a per-call buffer; the fake backend writes each Akk block's valid rows and
    # columns only, so zero-filled buffers keep the unwritten entries equal across the three passes
    monkeypatch.setattr(
        kcp,
        "_alloc",
        lambda shape, dtype, device: torch.zeros(
            tuple(shape), dtype=dtype, device=device
        ),
    )
    ctx, ops = _setup()
    o = dict(segments=segments, save_transfer=save_transfer)
    ref = _saved_pass(ctx, ops, save_mode="light", **o)
    got = _saved_pass(ctx, ops, save_mode="light_akk", **o)
    _same(ref, got, f"light_akk segments={segments} save_transfer={save_transfer}")
    assert got["bwd"]["akk_source"] == "saved" and ref["bwd"]["akk_source"] is None
    assert got["bwd"]["tapes_source"] == ref["bwd"]["tapes_source"] == "replay"
    assert got["bwd"]["qk_source"] == "replay" and got["bwd"]["rstd_source"] == "saved"
    # the launch census: the forward is light's; the backward's #1_l2n is the replay, nothing else moves
    assert got["fwd"]["sequence"] == ref["fwd"]["sequence"]
    assert got["fwd"]["fwd_o"] == ref["fwd"]["fwd_o"]
    swap = [REPLAY if p == "fwd_prep_l2n" else p for p in ref["bwd"]["sequence"]]
    assert (
        got["bwd"]["sequence"] == swap
        and ref["bwd"]["sequence"].count("fwd_prep_l2n") == 1
    )
    assert [g for _r, g in got["bwd"]["records"]] == [
        g for _r, g in ref["bwd"]["records"]
    ]
    # the saved tape is the forward's Akk: the full set's Akk bit for bit
    full = _saved_pass(ctx, ops, save_mode="full", **o)
    _same(ref, full, "full")
    assert torch.equal(got["saved"].tape("akk"), full["saved"].tape("akk"))
    assert got["saved"].save_mode == "light_akk" and len(got["saved"].tapes) == 1


def test_light_akk_autograd_checkpoint_and_replay_boundary() -> None:
    """Autograd: the replay runs; under a non-reentrant checkpoint the mode saves as "light" (the recompute's set,
    the kept hand-off of ``reuse_recompute_tapes=True`` applies); ``boundary_mode="replay"`` keeps nothing."""
    ctx, ops = _setup()
    direct = _autograd(ctx, ops, ckpt=False, save_mode="light")
    got = _autograd(ctx, ops, ckpt=False, save_mode="light_akk")
    _same(direct, got, "autograd")
    assert (
        kcp.LAST_CALL["bwd"]["akk_source"] == "saved"
        and REPLAY in kcp.LAST_CALL["bwd"]["sequence"]
    )
    for reuse in (False, True, "auto"):
        got = _autograd(
            ctx, ops, ckpt=True, save_mode="light_akk", reuse_recompute_tapes=reuse
        )
        _same(direct, got, f"checkpoint reuse={reuse}")
        assert (
            kcp.LAST_CALL["bwd"]["save_mode"] == "light"
            and kcp.LAST_CALL["bwd"]["akk_source"] is None
        )
        assert REPLAY not in kcp.LAST_CALL["bwd"]["sequence"]
        if reuse is True:
            assert kcp.LAST_CALL["bwd"]["tapes_source"] == "kept"
    assert (
        kcp._checkpoint_opts(kcp.make_options(save_mode="light_akk", **L2N)).save_mode
        == "light_akk"
    )  # outside
    got = _saved_pass(ctx, ops, save_mode="light_akk", boundary_mode="replay")
    _same(direct, got, "replay boundary")
    assert (
        got["saved"] is None
        and got["bwd"]["akk_source"] is None
        and REPLAY not in got["bwd"]["sequence"]
    )


def test_checkpoint_detection_reads_the_marker_or_the_hook_names() -> None:
    """The checkpoint form of the mode reads torch's pack hooks (:func:`_in_checkpoint`): torch 2.14 marks them
    ``_checkpoint_internal``, torch 2.13 does not, so they are also recognised by module and qualified name.  Both
    forwards of a non-reentrant checkpoint (the original and the recompute) see the checkpoint; no hooks and a user's
    hooks do not.  The running torch's two hooks carry the pinned names, and a marker-free stand-in is recognised by
    those names alone (the torch 2.13 path), never by the name in another module."""
    seen = []

    def fn(x):
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
        seen.append((kcp._in_checkpoint(), top[0]))
        return x * x

    from torch.utils.checkpoint import checkpoint

    x = torch.ones(4, requires_grad=True)
    checkpoint(fn, x, use_reentrant=False).sum().backward()
    assert [inside for inside, _pack in seen] == [True, True]
    names = {
        (inspect.unwrap(pack).__module__, inspect.unwrap(pack).__qualname__)
        for _inside, pack in seen
    }
    assert names == {
        (kcp._CHECKPOINT_HOOK_MODULE, n) for n in kcp._CHECKPOINT_PACK_HOOKS
    }

    def pack_hook(t):
        return t

    for name in kcp._CHECKPOINT_PACK_HOOKS:
        pack_hook.__module__, pack_hook.__qualname__ = kcp._CHECKPOINT_HOOK_MODULE, name
        assert kcp._checkpoint_pack(pack_hook), name
    pack_hook.__module__ = __name__
    assert not kcp._checkpoint_pack(pack_hook)
    assert not kcp._in_checkpoint()
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        assert not kcp._in_checkpoint()


def test_saved_tape_reaches_the_gradients() -> None:
    """Mutation check: the backward reads the saved tape (a changed ``Akk`` changes the gradients; the output of the
    forward is untouched)."""
    ctx, ops = _setup()
    out, saved = _fwd(ctx, ops, save_mode="light_akk")
    ref = _bwd(ctx, ops, saved, save_mode="light_akk")
    akk = saved.tape("akk")
    nz = akk.view(-1).nonzero()[:64, 0]
    akk.view(-1)[nz] = akk.view(-1)[nz] * 2
    got = _bwd(ctx, ops, saved, save_mode="light_akk")
    assert any(not torch.equal(a, b) for a, b in zip(ref, got, strict=True))


def test_saved_set_round_trip_and_refusals() -> None:
    ctx, ops = _setup()
    _o, saved = _fwd(ctx, ops, save_mode="light_akk")
    _o, light = _fwd(ctx, ops, save_mode="light")
    assert saved.flags() == light.flags() == (True, False, True)
    assert len(saved.tensors()) == len(light.tensors()) + 1
    back = kcp.CakeGDNNondetKcpSaved.from_tensors(
        saved.tensors(), saved.flags(), saved.meta(), ctx
    )
    assert back.save_mode == "light_akk" and back.tape("akk") is saved.tape("akk")
    with pytest.raises(KeyError):
        light.tape("akk")
    # a set of the other mode, a missing tape and a tape of another shape are refused
    with pytest.raises(ValueError, match="save_mode"):
        _bwd(ctx, ops, saved, save_mode="light")
    with pytest.raises(ValueError, match="save_mode"):
        _bwd(ctx, ops, light, save_mode="light_akk")
    # the default with the norm is "light_akk"
    with pytest.raises(ValueError, match="save_mode"):
        _bwd(ctx, ops, light)
    with pytest.raises(ValueError, match="saved tapes"):
        _bwd(ctx, ops, dataclasses.replace(saved, tapes=()), save_mode="light_akk")
    with pytest.raises(ValueError, match="akk"):
        _bwd(
            ctx,
            ops,
            dataclasses.replace(saved, tapes=(saved.tape("akk")[:-64],)),
            save_mode="light_akk",
        )
    with pytest.raises(ValueError, match="akk"):
        _bwd(
            ctx,
            ops,
            dataclasses.replace(saved, tapes=(saved.tape("akk").float(),)),
            save_mode="light_akk",
        )


# ----------------------------------------------------------------------------------------------------------------
# the byte model


def test_saved_bytes_gain_the_akk_tape() -> None:
    ctx, ops = _setup()
    H_qk, HV = ops["q"].shape[1], ops["v"].shape[1]
    akk = int(ctx.R) * HV * 64 * 2
    for o in ({}, {"save_transfer": False}, {"segments": 2}):
        _x, saved = _fwd(ctx, ops, save_mode="light_akk", **o)
        _x, light = _fwd(ctx, ops, save_mode="light", **o)
        assert saved.bytes - light.bytes == akk == saved.tape("akk").numel() * 2
        kw = {k: v for k, v in o.items() if k != "segments"}
        if "segments" in o:
            kw["P"] = o["segments"]
        lay = gbytes.cake_gdn_nondet_train_kcp_saved_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="light_akk", with_conv=False, **L2N, **kw
        )
        base = gbytes.cake_gdn_nondet_train_kcp_saved_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="light", with_conv=False, **L2N, **kw
        )
        assert lay["akk"] == akk and {k: v for k, v in lay.items() if k != "akk"} == {
            k: v for k, v in base.items() if k != "akk"
        }
        assert sum(lay.values()) - sum(base.values()) == saved.bytes - light.bytes, o
        full = gbytes.cake_gdn_nondet_train_kcp_saved_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="full", with_conv=False, **L2N, **kw
        )
        assert full["akk"] == akk
    # the replay boundary keeps nothing in either mode
    assert gbytes.cake_gdn_nondet_train_kcp_saved_bytes(
        ctx, H_qk=H_qk, HV=HV, save_mode="light_akk", boundary_mode="replay", **L2N
    ) == gbytes.cake_gdn_nondet_train_kcp_saved_bytes(
        ctx, H_qk=H_qk, HV=HV, save_mode="light", boundary_mode="replay", **L2N
    )


def test_workspace_layout_loses_the_akk_arena_tape() -> None:
    ctx, ops = _setup()
    H_qk, HV = ops["q"].shape[1], ops["v"].shape[1]
    akk = int(ctx.R) * HV * 64 * 2
    for kw in (
        {},
        {"save_transfer": False},
        {"graph": "replay"},
        {"reuse_recompute_tapes": False},
    ):
        light = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="light", **L2N, **kw
        )
        got = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="light_akk", **L2N, **kw
        )
        assert "fwd.akk" not in got and "bwd.akk" not in got, kw
        assert (
            light["bwd.akk"] == akk and light["bwd.qk_rstd"] == got["bwd.qk_rstd"] > 0
        ), kw
        assert got["bwd.kept_tapes"] == 0, kw
        assert light["fwd.phase.prep"] - got["fwd.phase.prep"] == akk, kw
        phases = [k for k in light if k.startswith("bwd.phase.")]
        assert phases and phases == [k for k in got if k.startswith("bwd.phase.")], kw
        for k in phases:  # every backward phase holds the replayed tapes: Akk is the saved set's in "light_akk"
            assert light[k] - got[k] == akk, (kw, k)
        assert light["backward"] - got["backward"] == akk, kw
    # without a saved set the mode is the light form
    a = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
        ctx, H_qk=H_qk, HV=HV, save_for_backward=False, save_mode="light", **L2N
    )
    b = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
        ctx, H_qk=H_qk, HV=HV, save_for_backward=False, save_mode="light_akk", **L2N
    )
    assert a == b


class _PeakWorkspace:
    def __new__(cls):
        class Tracked(kcp.CakeGDNNondetKcpWorkspace):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.peak = 0

            def _upd(self):
                self.peak = max(self.peak, self.bytes)

            def tape(self, name):
                t = super().tape(name)
                self._upd()
                return t

            def pool(self, name, n=None):
                t = super().pool(name, n)
                self._upd()
                return t

            def ensure(self, *a, **k):
                tb = super().ensure(*a, **k)
                self._upd()
                return tb

            def adopt_tape(self, name, t):
                r = super().adopt_tape(name, t)
                self._upd()
                return r

        return Tracked()


def test_caller_held_workspace_binds_the_layout() -> None:
    """A caller-held workspace's forward and backward peaks are the layout's ``forward`` / ``backward`` for
    ``"light"`` and ``"light_akk"`` (the replay form, eager, saved set), and differ by the Akk tape."""
    ctx, ops = _setup()
    H_qk, HV = ops["q"].shape[1], ops["v"].shape[1]
    peaks = {}
    for mode in ("light", "light_akk"):
        ws = _PeakWorkspace()
        out, saved = _fwd(ctx, ops, ws, save_mode=mode, reuse_recompute_tapes=False)
        fwd_peak, ws.peak = ws.peak, 0
        _bwd(ctx, ops, saved, ws, save_mode=mode, reuse_recompute_tapes=False)
        lay = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx,
            H_qk=H_qk,
            HV=HV,
            save_mode=mode,
            reuse_recompute_tapes=False,
            with_conv=False,
            **L2N,
        )
        peaks[mode] = (fwd_peak, ws.peak, lay["forward"], lay["backward"])
    (fl, bl, lfl, lbl), (fa, ba, lfa, lba) = peaks["light"], peaks["light_akk"]
    assert (fl - fa, bl - ba) == (lfl - lfa, lbl - lba), peaks
    assert (fl, bl) == (lfl, lbl) and (fa, ba) == (lfa, lba), peaks


@pytest.mark.parametrize("reuse", [True, False], ids=["kept", "replay-form"])
def test_checkpointed_light_akk_binds_the_light_figure(reuse) -> None:
    """Under ``torch.utils.checkpoint(use_reentrant=False)`` a ``"light_akk"`` call runs as ``"light"`` (the original
    forward and the recompute): a caller-held workspace peaks at the ``save_mode="light"`` figures of
    ``cake_gdn_nondet_train_kcp_workspace_layout`` (forward, and the recompute + backward), the same bytes as a checkpointed
    ``"light"`` call, and not at the ``"light_akk"`` figures (which keep ``Akk`` out of the arena), so a checkpointed
    ``"light_akk"`` layer is sized with ``save_mode="light"`` (cake_bytes, README, facade)."""
    from torch.utils.checkpoint import checkpoint

    ctx, ops = _setup()
    H_qk, HV = ops["q"].shape[1], ops["v"].shape[1]
    peaks = {}
    for mode in ("light", "light_akk"):
        ws = _PeakWorkspace()
        leaves = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]

        def fn(q, k, v, g, beta, _ws=ws, _mode=mode):
            return kcp.cake_gdn_nondet_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                workspace=_ws,
                backend="fake",
                save_mode=_mode,
                reuse_recompute_tapes=reuse,
                **L2N,
            )

        out = checkpoint(fn, *leaves, use_reentrant=False)
        fwd_peak, ws.peak = ws.peak, 0
        out.backward(ops["dout"])
        assert kcp.LAST_CALL["bwd"]["save_mode"] == "light", mode
        assert REPLAY not in kcp.LAST_CALL["bwd"]["sequence"], mode
        peaks[mode] = (fwd_peak, ws.peak)
    q = dict(H_qk=H_qk, HV=HV, reuse_recompute_tapes=reuse, with_conv=False, **L2N)
    light = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
        ctx, save_mode="light", **q
    )
    akk = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
        ctx, save_mode="light_akk", **q
    )
    assert peaks["light_akk"] == peaks["light"], peaks
    assert peaks["light_akk"] == (light["forward"], light["backward"]), (peaks, light)
    assert peaks["light_akk"] != (akk["forward"], akk["backward"]), (peaks, akk)
    assert (
        gbytes.cake_gdn_nondet_train_kcp_workspace_bytes(ctx, save_mode="light", **q)
        == light["workspace"]
    )
