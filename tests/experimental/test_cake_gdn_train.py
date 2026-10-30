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

The CP = 1 local path of the Cake packed GDN training package
(``flashinfer.gdn_train`` with a one-rank context) against the exact FP64
reference of ``tests/test_helpers/cake_gdn_train_reference.py``.

Covered here: the six outputs ``out, dq, dk, dv, dbeta, dg`` on ragged packs
of the normal gate regime (natural-log gates down to the -42 class; zero
elementwise mismatches), the adversarial weak-erasure regimes (contract,
finiteness and a relative-RMS sanity bound; mismatch counts recorded as
evidence), the six head pairs (16, 48), (8, 24), (4, 12),
(2, 6), (1, 3), (4, 4), the degenerate one-token / zero-token packs, the packed
QKV strided view against the contiguous input, and the explicit forward /
backward entries against autograd (output order ``dq, dk, dv, dbeta, dg``,
saved-context against replay).  The multi-rank rows, the bitwise contracts and
the runtime contracts live in ``test_cake_gdn_train_kcp_dist.py``; the CPU plan,
table and byte-query tests in ``test_cake_gdn_train_kcp.py``.
"""

from __future__ import annotations

import pytest
import torch

from tests.test_helpers.cake_gdn_train_reference import (
    BLOCK,
    D,
    GRAD_NAMES,
    STORAGE_OF,
    TENSORS,
    elementwise_mismatch,
    error_metrics,
    head_params,
    make_inputs,
    reference_pack,
)

gdn_api = pytest.importorskip(
    "flashinfer.gdn_train",
    reason="the cake_gdn_train facade is not part of this tree",
)

# Accuracy gate, declared before the first run: every output elementwise within
# ``|cand - ref| <= ATOL + RTOL * |ref|`` of the exact FP64 reference of the same BF16 / FP32 inputs with ZERO
# mismatches on all six tensors; ``out`` / ``dq`` / ``dk`` / ``dv`` are compared as stored BF16, ``dg`` / ``dbeta`` as
# FP32.  The rel-RMS / max-abs of every tensor is recorded beside the verdict (the budget tables of the multi-rank
# rows are derived from the same metrics).
ATOL = RTOL = 1e-2
# The zero-mismatch gate is the contract of the normal gate regime (the correctness rows).  The adversarial weak-erasure
# regimes (slow isotropic / anisotropic decay over one long document) are judged by the multi-rank budget tables with
# their negative control; here they keep the contract, finiteness and a sanity bound on the relative RMS error, and
# their mismatch counts are recorded as evidence.  Bound declared before the first run of these shapes.
GATED_REGIMES = ("normal", "normal+decay4")
ADVERSARIAL_REL_RMS_MAX = 1e-2
OPERANDS = ("q", "k", "v", "g", "beta")
EXPLICIT_GRAD_ORDER = (
    "dq",
    "dk",
    "dv",
    "dbeta",
    "dg",
)  # the backward entries' output order
STORAGE_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}

# The pinned-decay GEN row: the normal regime's per-head decay rate e^A is raised to at least PINNED_DECAY_MIN, so every
# full 64-token chunk's gate sum |G_L| exceeds 100: the class where e^{G_L} overflows FP32 unless the kernel rescales
# within the chunk.
PINNED_DECAY_MIN = 4.0
PINNED_CHUNK_GATE_SUM_MIN = 100.0

# id, document lengths (none a multiple of 64 unless the case says so), (H_qk, HV), operand regime
SHAPES = (
    ("c1-2x6-ragged", (100, 640, 77, 1000, 129), (2, 6), "normal"),
    ("c1-2x6-gen-decay-pinned", (100, 640, 77, 1000, 129), (2, 6), "normal+decay4"),
    ("c2-4x12-one-doc", (3000,), (4, 12), "normal"),
    ("c3-1x3-tiny-docs", (2048, 5, 3000, 1), (1, 3), "normal"),
    ("c4-8x24-chunk-interior", (1500, 1500), (8, 24), "normal"),
    ("c5-4x4-63-64-65", (63 + 64 + 65 + 500,), (4, 4), "normal"),
    ("c8s-16x48", (2048 + 37, 1000), (16, 48), "normal"),
    ("a1s-2x6-slow-decay", (1201,), (2, 6), "A1"),
    ("a2s-2x6-anisotropic", (1301,), (2, 6), "A2"),
    ("one-token", (1,), (2, 6), "normal"),
    ("zero-tokens", (0,), (2, 6), "normal"),
)


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return cu


def _inputs(cu: list[int], heads: tuple[int, int], regime: str) -> dict:
    """The row's operands; ``"normal+decay4"`` is the normal regime with every value head's decay rate e^A raised
    to at least PINNED_DECAY_MIN (g = -max(e^A, 4) * softplus(x), i.e. g scaled by max(1, 4 / e^A) per head)."""
    if regime != "normal+decay4":
        return make_inputs(cu, *heads, regime=regime)
    inputs = make_inputs(cu, *heads, regime="normal")
    rate = torch.exp(
        head_params(*heads)["a_log"]
    )  # [HV] decay rates e^A of the normal regime
    inputs["g"] = (
        (inputs["g"] * torch.clamp(PINNED_DECAY_MIN / rate, min=1.0))
        .float()
        .contiguous()
    )
    for d in range(len(cu) - 1):
        for c0 in range(cu[d], cu[d + 1] - BLOCK + 1, BLOCK):
            gate_sum = inputs["g"][0, c0 : c0 + BLOCK].sum(
                0
            )  # [HV] per full in-document chunk
            assert bool((gate_sum.abs() > PINNED_CHUNK_GATE_SUM_MIN).all()), (
                d,
                c0,
                gate_sum,
            )
    return inputs


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    supported = getattr(gdn_api, "is_gdn_train_supported", None)
    if supported is not None and not supported(index):
        pytest.skip("the package does not support this device")
    return torch.device("cuda", index)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _local_context(cu: list[int], device: torch.device):
    """The one-rank context of the local path (the public entry names; group None = single process)."""
    part = gdn_api.kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, layout="contiguous"
    )
    return gdn_api.build_kcp_context(part, None, 0, device)


def _run_fwd_bwd(inputs: dict, ctx, device: torch.device) -> dict:
    """Autograd forward + backward of the local path; the results keyed by tensor name (no batch axis)."""
    leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
    out = gdn_api.chunk_gdn_kcp(*leaves, ctx)
    grads = torch.autograd.grad(out, leaves, inputs["dout"].to(device))
    res = {"out": out.detach()[0]}
    res.update({n: gr.detach()[0] for n, gr in zip(GRAD_NAMES, grads, strict=True)})
    return res


def _check_contract(res: dict, inputs: dict) -> None:
    T, H, HV = inputs["q"].shape[1], inputs["q"].shape[2], inputs["v"].shape[2]
    expect = {
        "out": (T, HV, D),
        "dq": (T, H, D),
        "dk": (T, H, D),
        "dv": (T, HV, D),
        "dg": (T, HV),
        "dbeta": (T, HV),
    }
    for n in TENSORS:
        assert tuple(res[n].shape) == expect[n], (n, tuple(res[n].shape))
        assert res[n].dtype == STORAGE_DTYPE[STORAGE_OF[n]], (n, res[n].dtype)
        assert bool(torch.isfinite(res[n]).all()), f"{n} has non-finite values"


@pytest.mark.parametrize(
    ("lengths", "heads", "regime"),
    [s[1:] for s in SHAPES],
    ids=[s[0] for s in SHAPES],
)
def test_local_path_matches_the_exact_reference(
    device, record_property, lengths, heads, regime
) -> None:
    cu = _cu(lengths)
    inputs = _inputs(cu, heads, regime)
    ref, _ = reference_pack(inputs, device)
    res = _run_fwd_bwd(inputs, _local_context(cu, device), device)
    _check_contract(res, inputs)
    rows, failures = [], []
    for n in TENSORS:
        r = ref[n].to(device)
        m = error_metrics(res[n], r)
        e = elementwise_mismatch(res[n], r, ATOL, RTOL)
        rows.append(
            f"| {heads} | {regime} | {cu[-1]} | {n} | {STORAGE_OF[n]} | {m['rel_rms']:.3e} | {m['max_abs']:.3e} | {e['mismatch']} / {e['numel']} |"
        )
        record_property(f"{n}_rel_rms", m["rel_rms"])
        record_property(f"{n}_max_abs", m["max_abs"])
        record_property(f"{n}_mismatch", e["mismatch"])
        if regime in GATED_REGIMES and not e["pass"]:
            failures.append(
                f"{n}: {e['mismatch']} of {e['numel']} elements outside |cand - ref| <= {ATOL} + {RTOL} |ref| (worst excess {e['worst_excess']:.3e}), rel-RMS {m['rel_rms']:.3e}, max-abs {m['max_abs']:.3e}, worst {m['worst']}"
            )
        if regime not in GATED_REGIMES and not m["rel_rms"] <= ADVERSARIAL_REL_RMS_MAX:
            failures.append(
                f"{n}: adversarial regime {regime} rel-RMS {m['rel_rms']:.3e} > {ADVERSARIAL_REL_RMS_MAX} (max-abs {m['max_abs']:.3e}, {e['mismatch']} of {e['numel']} elements outside the elementwise band, worst {m['worst']})"
            )
    print(
        "\n| heads | regime | tokens | tensor | storage | rel-RMS | max-abs | mismatches |\n|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
    )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    ("lengths", "heads"),
    [((300, 777), (2, 6)), ((1000, 37), (16, 48))],
    ids=["2x6", "16x48"],
)
def test_packed_qkv_strided_view_equals_the_contiguous_input(
    device, lengths, heads
) -> None:
    H, HV = heads
    cu = _cu(lengths)
    inputs = make_inputs(cu, H, HV)
    T = cu[-1]
    # the packed QKV row: q at head 0, k at head H, v at head 2H; token stride (2H + HV) * 128 elements
    packed = torch.empty(1, T, 2 * H + HV, D, dtype=torch.bfloat16, device=device)
    packed[:, :, :H] = inputs["q"].to(device)
    packed[:, :, H : 2 * H] = inputs["k"].to(device)
    packed[:, :, 2 * H :] = inputs["v"].to(device)
    views = {
        "q": packed[:, :, :H],
        "k": packed[:, :, H : 2 * H],
        "v": packed[:, :, 2 * H :],
    }
    for n, t in views.items():
        assert not t.is_contiguous(), n
        assert t.stride(1) == (2 * H + HV) * D, (n, t.stride())
        assert torch.equal(t.cpu(), inputs[n]), n
    ctx = _local_context(cu, device)
    contiguous = _run_fwd_bwd(inputs, ctx, device)
    strided = _run_fwd_bwd({**inputs, **views}, ctx, device)
    for n in TENSORS:
        assert torch.equal(strided[n], contiguous[n]), (
            f"{n}: strided view differs from the contiguous input"
        )


def test_explicit_entries_match_autograd_and_keep_the_output_order(device) -> None:
    cu = _cu((100, 640, 77))
    inputs = make_inputs(cu, 2, 6)
    ctx = _local_context(cu, device)
    ops = [inputs[n].to(device) for n in OPERANDS]
    dout = inputs["dout"].to(device)
    out, saved = gdn_api.chunk_gdn_kcp_fwd(*ops, ctx, save_for_backward=True)
    grads = gdn_api.chunk_gdn_kcp_bwd(*ops, ctx, dout, saved=saved)
    assert len(grads) == len(EXPLICIT_GRAD_ORDER)
    auto = _run_fwd_bwd(inputs, ctx, device)
    assert torch.equal(out[0], auto["out"])
    for n, t in zip(EXPLICIT_GRAD_ORDER, grads, strict=True):
        assert t.dtype == STORAGE_DTYPE[STORAGE_OF[n]], (n, t.dtype)
        assert torch.equal(t[0], auto[n]), (
            f"{n} differs between the explicit entry and autograd"
        )
    replay = gdn_api.chunk_gdn_kcp_bwd(*ops, ctx, dout, saved=None)
    for n, t_saved, t_replay in zip(EXPLICIT_GRAD_ORDER, grads, replay, strict=True):
        assert torch.equal(t_saved, t_replay), (
            f"{n}: saved-context and replay backward differ"
        )


# --------------------------------------------------------------------------- issue #7074 (1): sync-free context preparation (one GPU)
ISSUE_CU = [
    0,
    131071,
    218602,
    261987,
]  # the issue's layout: 261,987 tokens in 3 documents, (H_qk, HV) = (2, 6)


def test_context_preparation_does_not_synchronize(device) -> None:
    """``kcp_partition`` + ``build_kcp_context`` + ``prepare`` of a new layout make no synchronizing CUDA call (the
    tables travel in pinned, asynchronous copies), and every device table is bitwise the table built on the CPU."""
    plan = pytest.importorskip("flashinfer.experimental.cake_gdn_train.cake_kcp_plan")
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        part = gdn_api.kcp_partition(
            torch.tensor(ISSUE_CU, dtype=torch.int64), 1, layout="contiguous"
        )
        ctx = gdn_api.build_kcp_context(part, None, 0, device)
        prepared = ctx.prepare(2, 6)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize(device)
    ref = plan.build_kcp_context(part, None, 0, "cpu", sm_count=ctx.sm_count)
    ref_prepared = ref.prepare(2, 6)
    pairs = (
        [(f"tables.{k}", ctx.tables[k], ref.tables[k]) for k in ctx.tables]
        + [
            (f"halo.{k}", ctx.halo_tables[k], ref.halo_tables[k])
            for k in ctx.halo_tables
        ]
        + [
            ("frag_table", ctx.frag_table, ref.frag_table),
            ("cu_seqlens_local", ctx.cu_seqlens_local, ref.cu_seqlens_local),
        ]
        + [
            (f"prepared.{k}", prepared.device[k], ref_prepared.device[k])
            for k in prepared.device
        ]
    )
    assert len(pairs) > 20
    for name, dev_t, cpu_t in pairs:
        assert dev_t.device.type == "cuda", name
        assert dev_t.dtype == cpu_t.dtype and tuple(dev_t.shape) == tuple(
            cpu_t.shape
        ), name
        assert dev_t.data_ptr() % plan.UPLOAD_ALIGN_BYTES == 0, name
        assert torch.equal(dev_t.cpu(), cpu_t), name
    assert prepared.P == ref_prepared.P


# --------------------------------------------------------------------------- issue #7074 (4): the workspace bytes the backward binds (one GPU)


class _PeakWorkspace:
    """A ``KcpWorkspace`` recording the most bytes it binds (every allocation / release path)."""

    def __new__(cls, kcp):
        class Tracked(kcp.KcpWorkspace):
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


@pytest.mark.parametrize(
    ("cu", "heads"),
    [(ISSUE_CU, (2, 6)), ([0, 11000, 23000, 32768], (16, 48))],
    ids=["issue-2x6", "32k-16x48"],
)
def test_workspace_bytes_match_what_the_backward_binds(device, cu, heads) -> None:
    """``gdn_kcp_workspace_layout`` is what a caller-held workspace binds (tolerance 0): the forward's tape-walk peak,
    and the backward's phase maximum of each tape form -- the kept-tape form (the ten light tapes are the recompute's
    hand-off tensors, four of them re-adopted after #8) and the replay form (the ten light tapes are workspace)."""
    import gc

    from torch.utils.checkpoint import checkpoint

    kcp = pytest.importorskip("flashinfer.experimental.cake_gdn_train.cake_kcp")
    gb = pytest.importorskip("flashinfer.experimental.cake_gdn_train.cake_bytes")
    H, HV = heads
    inputs = make_inputs(cu, H, HV)
    ctx = _local_context(cu, device)
    ctx.prepare(H, HV)
    leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
    dout = inputs["dout"].to(device)
    for form in (True, False):
        layout = gb.gdn_kcp_workspace_layout(
            ctx, H_qk=H, HV=HV, reuse_recompute_tapes=form
        )
        ws = _PeakWorkspace(kcp)

        def fn(*a, _ws=ws, _form=form):
            return gdn_api.chunk_gdn_kcp(
                *a, ctx, workspace=_ws, reuse_recompute_tapes=_form
            )

        out = checkpoint(fn, *leaves, use_reentrant=False)
        torch.cuda.synchronize(device)
        assert ws.peak == layout["forward"], (form, ws.peak, layout["forward"])
        ws.peak = 0
        out.backward(dout)
        torch.cuda.synchronize(device)
        assert ws.peak == layout["backward"], (form, ws.peak, layout["backward"])
        assert ws.bytes == layout["bwd.phase.end"], (
            form,
            ws.bytes,
            layout["bwd.phase.end"],
        )
        assert gdn_api.gdn_kcp_workspace_bytes(
            ctx, H_qk=H, HV=HV, reuse_recompute_tapes=form
        ) == max(layout["forward"], layout["backward"])
        for lf in leaves:
            lf.grad = None
        del out, ws
        gc.collect()
        torch.cuda.empty_cache()
    assert kcp.kept_registry_size() == 0


# --------------------------------------------------------------------------- issue #7074 (2): reuse_recompute_tapes="auto" (one GPU)


@pytest.mark.parametrize(
    "later_saver", [False, True], ids=["gdn-closes-the-region", "later-saver"]
)
def test_reuse_recompute_tapes_auto_follows_the_region_shape(
    device, later_saver
) -> None:
    """``"auto"`` hands the recompute's tapes to the backward when the GDN node is the region's last saver (frame
    proof ``dead``, kept form) and takes the replay form when a later operation of the region saves a tensor
    (``holders_remain``); ``True`` always hands off (``always``), ``False`` never (``off``).  Every form is bitwise the
    direct autograd result, the registry is empty afterwards, and the ``"auto"`` backward -- the recompute forward,
    the frame proof over the autograd graph, the take -- makes no synchronizing CUDA call."""
    kcp = pytest.importorskip("flashinfer.experimental.cake_gdn_train.cake_kcp")
    H, HV = 2, 6
    cu = [0, 1500, 4200, 6144]
    inputs = make_inputs(cu, H, HV)
    ctx = _local_context(cu, device)
    leaves = [inputs[n].to(device).detach().requires_grad_(True) for n in OPERANDS]
    proj = None
    if later_saver:
        gen = torch.Generator(device="cpu").manual_seed(7074)
        proj = (
            (torch.randn(HV * D, HV * D, generator=gen) * 0.02)
            .to(torch.bfloat16)
            .to(device)
        )
        dout = (
            inputs["dout"].to(device).reshape(1, -1, HV * D)
            if proj is not None
            else None
        )
        proj_in = proj
        proj = None  # the projection acts on the flattened heads: wrap below

        def flat_proj(out, w=proj_in):
            return torch.nn.functional.linear(
                out.reshape(out.shape[0], out.shape[1], -1), w
            )

        proj = flat_proj
    else:
        dout = inputs["dout"].to(device)

    def region(ckpt, form, sync_error=False):
        from torch.utils.checkpoint import checkpoint

        def fn(*a):
            out = gdn_api.chunk_gdn_kcp(
                *a, ctx, **({} if form is None else {"reuse_recompute_tapes": form})
            )
            return out if proj is None else proj(out)

        for lf in leaves:
            lf.grad = None
        out = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
        torch.cuda.synchronize(device)
        if sync_error:
            torch.cuda.set_sync_debug_mode("error")
        try:
            out.backward(dout)
        finally:
            if sync_error:
                torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize(device)
        return {
            "out": out.detach(),
            **{
                n: lf.grad.detach().clone()
                for n, lf in zip(OPERANDS, leaves, strict=True)
            },
        }

    direct = region(False, None)
    verified = kcp.torch_release_verified()
    for form, reason, source in (
        (True, "always", "kept"),
        (False, "off", "replay"),
        (
            "auto",
            ("holders_remain" if later_saver else "dead") if verified else None,
            "replay" if (later_saver or not verified) else "kept",
        ),
    ):
        res = region(True, form, sync_error=(form == "auto"))
        for n in direct:
            assert torch.equal(res[n], direct[n]), (later_saver, form, n)
        got_reason = kcp.LAST_CALL["fwd"]["hand_off_reason"]
        if reason is not None:
            assert got_reason == reason, (later_saver, form, got_reason)
        assert kcp.LAST_CALL["bwd"]["tapes_source"] == source, (
            later_saver,
            form,
            kcp.LAST_CALL["bwd"]["tapes_source"],
        )
        assert kcp.kept_registry_size() == 0, (later_saver, form)
