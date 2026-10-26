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
