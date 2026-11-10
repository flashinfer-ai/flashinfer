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

Backend suite of the Cake non-deterministic GDN chunked-training package
(``flashinfer.experimental.cake.cake_gdn_nondet_train``).

The suite mirrors the deterministic package's suite (``test_cake_gdn_det_train.py``) with every bitwise contract
replaced by the accuracy budgets of that suite (the same constants, declared before the first run, never derived from a
non-deterministic output): two runs, the two table routes, the longest-first order, two streams, the two preps, the
CUDA graph replay and the equal-length batch agree within the budgets.  It adds the non-deterministic contract: the
mode in the names only (the exact package names, each with ``nondet``; no ``deterministic`` parameter; the old names
gone) and the backward's refusal of a tape without this package's tag, the interaction with PyTorch's deterministic
algorithms (no alert, no fallback), the launches served by this package's records, the five source head pairs, strided
packed QKV views, non-reentrant checkpointing and its zero retained bytes, gradient accumulation into a nonzero
``.grad``, changed shapes over poisoned reused workspaces, the worst case over repeated runs, two concurrent caller
streams, cached device tables, workspaces and backward plans across streams and CUDA graph captures, a resident
side-stream kernel and the same-input comparison with the deterministic package.  The tail's own
contract: it serves every value-head group, leaves its counters zero after every launch, launches in segments that
give the bits of one launch, gives the bits of the cluster tails of a value-head group of two and three (``dq`` /
``dk`` rounded once after the FP32 group sum), keeps the deterministic bits of ``o``, ``dv``, ``dg`` and ``dbeta`` where
the adjoint rounds ``dv2`` as the deterministic package does and its own bits run to run, and protocol controls (stale
counters must break ``dq`` / ``dk``, stale workspace slots must not).  The packed row: q / k / v as head views of one
row that requires grad give the bits of the views route through one packed gradient row (a leaf or a non-leaf row,
other consumers of the row, a row without a gradient, checkpoint replay).

The suite names its packages through the shared alias helper and imports nothing else from flashinfer.  The tests of
the facade ``flashinfer.cake_gdn_nondet_train`` and of the old module path live in
``test_cake_gdn_nondet_train_api.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import inspect
import json
import math
import re
import time
import types
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake_gdn_nondet_train", "cake_backend")
cake_chain = cake_package("cake_gdn_nondet_train", "cake_chain")
cake_jit = cake_package("cake_gdn_nondet_train", "cake_jit")
cake_plan = cake_package("cake_gdn_nondet_train", "cake_plan")
cake_gdn_nondet_train = cake_package("cake_gdn_nondet_train")
cake_gdn_nondet = cake_gdn_nondet_train.cake_gdn_nondet
# The deterministic package: the same-input comparison point (never a source of budgets).
cake_gdn_det_train = cake_package("cake_gdn_det_train")
det_backend = cake_package("cake_gdn_det_train", "cake_backend")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
ARCHES = tuple(cake_jit.ARCHES)
HEAD_DIM = 128
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
OUTPUT_NAMES = ("o",) + GRAD_NAMES

# Accuracy gate against the FP64 recurrence of the same BF16 / FP32 inputs, declared before the first run: per output
# tensor, the relative RMS error at most REL_RMS_MAX and the largest absolute error at most MAX_ABS_REL times the
# largest reference magnitude.  The chain stores o / dq / dk / dv in BF16 (a 2^-8 relative rounding step); its
# campaign rows measured at most 8.3e-3 of the largest reference magnitude on every output.  The same bounds gate the
# agreement of two runs of this package and of this package against the deterministic one (the second as reference).
REL_RMS_MAX = 1e-2
MAX_ABS_REL = 2e-2
L2NORM_EPS = 1e-6  # the prep stage's q / k normalization: x / sqrt(sum x^2 + 1e-6)


# --------------------------------------------------------------------------- helpers


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {ARCHES}"
        )
    return torch.device("cuda", index)


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _cu(seq_lens) -> list[int]:
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + int(n))
    return cu


def _inputs(
    seq_lens, heads, beta_dtype, device, *, seed=0, batch=1, normalized_qk=False
):
    """Packed ``[1, T, ...]`` operands (or ``[batch, T, ...]`` for an equal-length batch) drawn on the host."""
    num_heads, num_v_heads = heads
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, total)
    q = torch.randn(*shape, num_heads, HEAD_DIM, generator=gen)
    k = torch.randn(*shape, num_heads, HEAD_DIM, generator=gen)
    if normalized_qk:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(*shape, num_v_heads, HEAD_DIM, generator=gen)
    g = torch.rand(*shape, num_v_heads, generator=gen) * -0.5
    beta = torch.rand(*shape, num_v_heads, generator=gen)
    do = torch.randn(*shape, num_v_heads, HEAD_DIM, generator=gen)
    out = {
        "q": q.to(torch.bfloat16),
        "k": k.to(torch.bfloat16),
        "v": v.to(torch.bfloat16),
        "g": g.float(),
        "beta": beta.to(beta_dtype),
        "do": do.to(torch.bfloat16),
    }
    return {name: t.to(device).contiguous() for name, t in out.items()}


# The forward route sets that take a call off the prep's forward (prep, wy_tc2, fwd_ho), with their off values: a test
# that holds this package's prep forward to the deterministic package's bits turns them off in both packages.  The
# deterministic package's round-once tail and FP32 dv_local adjoint switches (F32X_TAIL_ARCHS, DVL32_ARCHS) are off
# there too, so its value-head group of three runs tail_g3 on dhu_tc2, the bits this package's fixed-order tails
# reproduce.
PREP_FORWARD_OFF = (
    ("FWD_A_ARCHS", ()),
    ("FWD_SPLIT_ARCHS", ()),
    ("FWD_A_SERIAL_KEEP", {}),
    ("F32X_TAIL_ARCHS", ()),
    ("DVL32_ARCHS", ()),
)


def _prep_forward_on_both(monkeypatch) -> None:
    """Keep both packages on the prep's forward: FWD_A_ARCHS and FWD_SPLIT_ARCHS empty in this package, and each
    setting of :data:`PREP_FORWARD_OFF` off in the deterministic package where that package has it (a deterministic
    package that runs the A or the split forward, or its round-once tail / FP32 dv_local adjoint, would otherwise
    compare other bits)."""
    monkeypatch.setattr(cake_chain, "FWD_A_ARCHS", ())
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_ARCHS", ())
    det_chain = cake_package("cake_gdn_det_train", "cake_chain")
    for name, off in PREP_FORWARD_OFF:
        if getattr(det_chain, name, None) is not None:
            monkeypatch.setattr(det_chain, name, off)


def _run(inputs, seq_lens, *, route="host", initial_state=None, output_final_state=False, dht=None,
         use_qk_l2norm_in_kernel=True, leaves=None, det=False, sync=True, **pins):  # fmt: skip
    """Forward and backward through the package (``det``: through the deterministic package); returns the outputs and
    gradients (and the final state / its initial-state gradient when asked).  ``leaves`` replaces the fresh leaf
    copies of the inputs it names; ``sync=False`` returns without synchronizing the device."""
    leaves = {
        **{
            name: inputs[name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        },
        **(leaves or {}),
    }
    h0 = None
    if initial_state is not None:
        h0 = initial_state.detach().clone().requires_grad_(True)
    kwargs = {}
    if seq_lens is not None:
        cu = _cu(seq_lens)
        kwargs["cu_seqlens"] = torch.tensor(
            cu, dtype=torch.int64, device=inputs["q"].device
        )
        kwargs["cu_seqlens_cpu"] = (
            torch.tensor(cu, dtype=torch.int64) if route == "host" else None
        )
    entry = cake_gdn_det_train.cake_gdn_det if det else cake_gdn_nondet
    o, final = entry(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=1.0 / math.sqrt(HEAD_DIM),
        initial_state=h0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **kwargs,
        **pins,
    )
    loss = (o.float() * inputs["do"].float()).sum()
    if output_final_state and dht is not None:
        loss = loss + (final * dht).sum()
    loss.backward()
    if sync:
        torch.cuda.synchronize()
    res = {
        "o": o.detach(),
        "dq": leaves["q"].grad,
        "dk": leaves["k"].grad,
        "dv": leaves["v"].grad,
        "dg": leaves["g"].grad,
        "dbeta": leaves["beta"].grad,
    }
    if output_final_state:
        res["final_state"] = final.detach()
    if h0 is not None:
        res["dh0"] = h0.grad
    return res


def _reference(inputs, seq_lens, *, initial_state=None, dht=None, normalize=True):
    """The FP64 token recurrence (FLA's gated delta rule) and its autograd gradients, per sequence and value head
    ``hv`` (key head ``hv // (HV / H)``) with the state ``S`` ``[K, V]``::

        S = S * exp(g_t);  u_t = beta_t (v_t - S^T k_t);  S = S + k_t u_t^T;  o_t = scale S^T q_t
    """
    dt = torch.float64
    leaves = {
        name: inputs[name].detach().to(dt).requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    h0 = None
    if initial_state is not None:
        h0 = initial_state.detach().to(dt).requires_grad_(True)
    q, k, v, g, beta = (leaves[n].flatten(0, 1) for n in ("q", "k", "v", "g", "beta"))
    num_heads, num_v_heads = q.shape[1], v.shape[1]
    group = num_v_heads // num_heads
    if normalize:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2NORM_EPS)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2NORM_EPS)
    q = q.repeat_interleave(group, dim=1) * (1.0 / math.sqrt(HEAD_DIM))
    k = k.repeat_interleave(group, dim=1)
    outs, finals = [], []
    start = 0
    for i, n in enumerate(seq_lens):
        S = (
            h0[i]
            if h0 is not None
            else torch.zeros(num_v_heads, HEAD_DIM, HEAD_DIM, dtype=dt, device=q.device)
        )
        for t in range(start, start + int(n)):
            S = S * torch.exp(g[t])[:, None, None]
            u = beta[t][:, None] * (v[t] - torch.einsum("hk,hkv->hv", k[t], S))
            S = S + k[t][:, :, None] * u[:, None, :]
            outs.append(torch.einsum("hk,hkv->hv", q[t], S))
        finals.append(S)
        start += int(n)
    o = torch.stack(outs, 0).reshape(inputs["v"].shape)
    final = torch.stack(finals, 0)
    loss = (o * inputs["do"].to(dt)).sum()
    if dht is not None:
        loss = loss + (final * dht.to(dt)).sum()
    loss.backward()
    res = {
        "o": o.detach(),
        "dq": leaves["q"].grad,
        "dk": leaves["k"].grad,
        "dv": leaves["v"].grad,
        "dg": leaves["g"].grad,
        "dbeta": leaves["beta"].grad,
        "final_state": final.detach(),
    }
    if h0 is not None:
        res["dh0"] = h0.grad
    return res


def _metrics(a: torch.Tensor, ref: torch.Tensor) -> dict:
    a64 = a.detach().to(torch.float64)
    r64 = ref.detach().to(torch.float64)
    diff = (a64 - r64).abs()
    max_ref = float(r64.abs().max())
    rms_ref = float(r64.square().mean().sqrt())
    return {
        "max_abs": float(diff.max()),
        "max_ref": max_ref,
        "rel_rms": float(diff.square().mean().sqrt()) / rms_ref if rms_ref > 0 else 0.0,
        "finite": bool(torch.isfinite(a64).all()),
    }


def _sizing(label: str, got: torch.Tensor, ref: torch.Tensor) -> str:
    """One SIZING line (``pytest -s``): the relative RMS error and the largest error over the largest reference value."""
    m = _metrics(got, ref)
    rel = m["max_abs"] / m["max_ref"] if m["max_ref"] > 0 else 0.0
    line = f"SIZING {label} rel_rms={m['rel_rms']:.3e} max_abs_rel={rel:.3e}"
    print(line)
    return line


def _assert_close_to_reference(got: dict, ref: dict, names) -> None:
    report = {}
    failed = []
    for name in names:
        assert got[name].shape == ref[name].shape, (
            name,
            got[name].shape,
            ref[name].shape,
        )
        m = _metrics(got[name], ref[name])
        report[name] = m
        if not (
            m["finite"]
            and m["rel_rms"] <= REL_RMS_MAX
            and m["max_abs"] <= MAX_ABS_REL * m["max_ref"]
        ):
            failed.append(name)
    assert not failed, f"outside the accuracy gate: {failed}; metrics {report}"


def _assert_within_budget(got: dict, ref: dict, names) -> None:
    """Two results agree within the accuracy budgets above, ``ref`` as the reference (the non-deterministic package
    replaces the deterministic suite's bitwise contracts with this check)."""
    _assert_close_to_reference(got, ref, names)


def _record_launches(monkeypatch) -> list[str]:
    """The stages the package launches from here on, in order."""
    stages: list[str] = []
    launch = cake_backend.launch_stage

    def recording(stage, *args, **kwargs):
        stages.append(stage)
        return launch(stage, *args, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return stages


def _preps(stages) -> set[str]:
    return {stage for stage in stages if stage.startswith("prep")}


# --------------------------------------------------------------------------- registry and manifest (no device)


# the fused cluster tails' cluster sizes: one CTA per value head of a key head
FUSED_TAIL_STAGES = {
    "tail_g2": 2,
    "tail_g2_bf16": 2,
    "tail_g3": 3,
    "tail_g3_bf16": 3,
    "tail_g2_f32x": 2,
    "tail_g2_f32x_bf16": 2,
    "tail_g3_f32x": 3,
    "tail_g3_f32x_bf16": 3,
}


# The stages the package launches with programmatic dependent launch (their kernels wait on the producer grid first
# and trigger the dependent launch last): the backward chain of the one-wave adjoint split and of the FP32 dv_local
# adjoint pair, and the stitch in every use.
PDL_STAGES = frozenset(
    {
        "stitch",
        "dhu_tc2_dvl32",
        "dhu_tc2_dvl32_modes",
        "dhu_tc2_dvl32_gm",
        "stitch_mt",
        "piece_chain",
        "tail_g2_f32x",
        "tail_g2_f32x_bf16",
        "tail_g3_f32x",
        "tail_g3_f32x_bf16",
    }
)


def test_registry_records_are_well_formed() -> None:
    # one stage per call-site argument list (the count follows the chain's table)
    stages = cake_jit.STAGES
    assert len(stages) == len(set(stages)) == len(cake_chain.LAUNCH_ARGS)
    assert set(cake_chain.LAUNCH_ARGS) == set(stages)
    for arch in ARCHES:
        for stage in cake_jit.STAGES:
            serving = [
                name
                for name, rec in cake_jit.MODULES.items()
                if rec["stage"] == stage and arch in rec["arches"]
            ]
            assert len(serving) == 1, (stage, arch, serving)
    libraries = set()
    for name, rec in cake_jit.MODULES.items():
        assert set(rec["arches"]) <= set(ARCHES), name
        assert rec["ffi_entry"] == "run", name
        assert re.fullmatch(r"cake_gdn_train_[0-9a-f]{20}", rec["library"]), name
        assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        launch = rec["launch"]
        assert launch["cooperative"] is False, name
        assert launch["use_pdl"] is (rec["stage"] in PDL_STAGES), name
        assert launch["cluster"] == (FUSED_TAIL_STAGES.get(rec["stage"], 1), 1, 1), name
        assert len(launch["block"]) == 3 and launch["block"][0] % 32 == 0, name
        assert launch["dynamic_smem_bytes"] >= 0, name
        # the record's kernel parameters are the call site's, possibly reordered (the chain reorders them)
        params = cake_backend.parameters(name)
        assert sorted(params) == sorted(cake_chain.LAUNCH_ARGS[rec["stage"]]), name
        libraries.add(rec["library"])
    # one program per record: a kernel swap renames exactly the records it touches; the two shared libraries are the
    # sm_107a wy_tc2, which is the sm_100a / sm_103a wy_tc2 program, and the sm_100a / sm_103a fwd_ho_vh2, which is the
    # fwd_ho program
    wy, wy2, wy2_107 = (
        cake_jit.MODULES[name] for name in ("wy_tc", "wy_tc2", "wy_tc2_sm_107a")
    )
    assert wy2["arches"] == ["sm_100a", "sm_103a"] and wy2_107["arches"] == ["sm_107a"]
    assert (
        wy2_107["library"] == wy2["library"] != wy["library"]
        and "sm_107a" in wy["arches"]
    )
    assert (
        cake_jit.MODULES["fwd_ho_vh2"]["library"]
        == cake_jit.MODULES["fwd_ho"]["library"]
    )
    assert len(libraries) == len(cake_jit.MODULES) - 2


def test_the_wy_stage_follows_the_architecture() -> None:
    assert cake_chain.WY_TC2_ARCHS == ("sm_100a", "sm_103a", "sm_107a")
    assert {arch: cake_chain.wy_stage(arch) for arch in ARCHES} == {
        "sm_100a": "wy_tc2",
        "sm_103a": "wy_tc2",
        "sm_107a": "wy_tc2",
    }
    # wy_tc2 reads v in place: wy_tc's arguments and v's token pitch in heads
    assert cake_chain.LAUNCH_ARGS["wy_tc2"] == (
        *cake_chain.LAUNCH_ARGS["wy_tc"],
        "v_row_heads",
    )
    plan, plan2 = (cake_jit.MODULES[name]["arg_plan"] for name in ("wy_tc", "wy_tc2"))
    assert [arg for arg in plan2 if arg != ["parameter", "v_row_heads"]] == plan


def test_the_stages_that_read_v_by_rows_take_its_pitch() -> None:
    # wy_tc2, wy_bwd, the package tail and the fixed-order cluster tails read v in place through its token pitch in
    # heads (their last parameter); the FP32-partial cluster tails take v's token-row stride in elements
    in_place = ("wy_tc2", "wy_bwd", "wy_bwd_bf16", "tail", "tail_bf16", "tail_g2", "tail_g2_bf16", "tail_g3",
                "tail_g3_bf16")  # fmt: skip
    for stage in cake_jit.STAGES:
        args = cake_chain.LAUNCH_ARGS[stage]
        assert ("v_row_heads" in args) == (stage in in_place), stage
        if stage in in_place:
            assert args[-1] == "v_row_heads" and "v" in args, stage
    for stage in cake_chain.F32X_TAILS:
        assert "v_row_stride" in cake_chain.LAUNCH_ARGS[stage]


def test_v_rows_reads_whole_head_rows_in_place() -> None:
    # the stages that read v take its token pitch in 128-element heads: the value heads of a packed
    # [T, (2 H + HV) * 128] row, contiguous rows, and leading axes that fold into the token axis; other layouts
    # are copied (None)
    tokens, heads, v_heads = 8, 2, 6
    width = (2 * heads + v_heads) * HEAD_DIM
    row = torch.empty(2, tokens, width, dtype=torch.bfloat16)
    packed = row[..., 2 * heads * HEAD_DIM :].unflatten(-1, (v_heads, HEAD_DIM))
    assert cake_chain._v_rows(packed) == 2 * heads + v_heads
    assert cake_chain._v_rows(packed[:1, :5]) == 2 * heads + v_heads
    dense = torch.empty(2, tokens, v_heads, HEAD_DIM, dtype=torch.bfloat16)
    assert cake_chain._v_rows(dense) == v_heads
    head_major = torch.empty(1, v_heads, tokens, HEAD_DIM, dtype=torch.bfloat16)
    assert cake_chain._v_rows(head_major.transpose(1, 2)) is None
    half_head = torch.empty(1, tokens, v_heads * HEAD_DIM + 64, dtype=torch.bfloat16)
    half_head = half_head[..., : v_heads * HEAD_DIM].unflatten(-1, (v_heads, HEAD_DIM))
    assert cake_chain._v_rows(half_head) is None
    batch_minor = torch.empty(tokens, 2, v_heads, HEAD_DIM, dtype=torch.bfloat16)
    assert cake_chain._v_rows(batch_minor.transpose(0, 1)) is None
    flat = torch.empty(tokens * v_heads * HEAD_DIM + 1, dtype=torch.bfloat16)
    assert cake_chain._v_rows(flat[1:].view(1, tokens, v_heads, HEAD_DIM)) is None


def test_the_value_half_split_rule() -> None:
    assert cake_chain.FWD_HO_VH2_ARCHS == ("sm_107a",)
    assert 0.0 < cake_chain.FWD_HO_VH2_STEP < 1.0
    assert cake_chain.LAUNCH_ARGS["fwd_ho_vh2"] == cake_chain.LAUNCH_ARGS["fwd_ho"]
    split, fwd_ho = cake_jit.MODULES["fwd_ho_vh2_sm_107a"], cake_jit.MODULES["fwd_ho"]
    assert split["stage"] == "fwd_ho_vh2" and split["arches"] == ["sm_107a"]
    assert split["arg_plan"] == fwd_ho["arg_plan"]
    assert split["library"] != fwd_ho["library"]
    assert split["launch"] == fwd_ho["launch"]
    # sm_100a / sm_103a never run the split: the stage's record there is the fwd_ho program
    other = cake_jit.MODULES["fwd_ho_vh2"]
    assert other["stage"] == "fwd_ho_vh2" and other["arches"] == ["sm_100a", "sm_103a"]
    assert {
        key: value for key, value in other.items() if key not in ("arches", "stage")
    } == {key: value for key, value in fwd_ho.items() if key not in ("arches", "stage")}
    assert [cake_jit.stage_record("fwd_ho_vh2", arch) for arch in ARCHES] == [
        "fwd_ho_vh2",
        "fwd_ho_vh2",
        "fwd_ho_vh2_sm_107a",
    ]
    # the doubled grid only where the record is the split; elsewhere fwd_ho's grid (the program's own)
    assert [cake_chain.fwd_ho_vh2_splits(arch) for arch in ARCHES] == [
        False,
        False,
        True,
    ]
    engaged = cake_chain.fwd_ho_vh2_engaged
    # twice the CTAs in one round of one CTA per SM
    assert engaged(6, 16, 192) and not engaged(6, 16, 191)
    assert engaged(1, 1, 2) and not engaged(1, 1, 1)
    assert not engaged(6, 16, 148) and not engaged(6, 16, 152)


def test_scan_records_carry_the_planner_constants() -> None:
    for kind in cake_chain.SCAN_KINDS:
        for vb, ks in cake_chain.REG_SCAN_CONFIGS:
            rec = cake_jit.MODULES[f"{kind}_r{vb}k{ks}"]
            assert (rec["scan"]["v_block"], rec["scan"]["k_split"]) == (vb, ks)
            assert rec["scan"]["select_smem_bytes"] > 0
        for vb, ks in cake_chain.PRESCAN_CONFIGS:
            rec = cake_jit.MODULES[f"{kind}_pre_r{vb}k{ks}"]
            assert (rec["scan"]["v_block"], rec["scan"]["k_split"]) == (vb, ks)
            assert rec["scan"]["max_segment_chunks"] == 128
    for arch in ARCHES:
        sset = cake_chain.stage_set(arch)
        for kind in cake_chain.SCAN_KINDS:
            assert sset.reg_of(kind) == cake_chain.REG_SCAN_CONFIGS, (arch, kind)
            assert sset.split_of(kind) == cake_chain.PRESCAN_CONFIGS, (arch, kind)
        const = dict(cake_chain._dhu_tc_constants(arch))
        assert set(const) == set(cake_chain.DHU_TC_CONSTANTS), arch
        assert (
            0 < const["t_lat"] <= const["t_full"]
            and const["t_unit"] > 0
            and const["c_dvl"] > 0
        ), arch
        gain, min_saving = cake_chain._sdt_constants(arch)
        assert 0 < gain <= 1 and min_saving >= 0, arch
        ctas_per_sm, window = cake_chain._prep_tma_constants(arch)
        if arch == "sm_107a":  # window 1.0: no plain-prep band
            assert ctas_per_sm > 0 and window == 1.0, arch
            continue
        assert ctas_per_sm > 0 and window > 1, arch


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {
        f"{rec['library']}_{arch}"
        for rec in cake_jit.MODULES.values()
        for arch in rec["arches"]
    }
    assert set(entries) == wanted
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert entry["path"] == f"aot/{name}.so", name
        for rel, digest in entry["sha256"].items():
            data = (PACKAGE_DIR / rel).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


# --------------------------------------------------------------------------- launch entries (no device)


def _fake_libraries(monkeypatch, arches) -> tuple[list, list]:
    """``cake_backend`` on fake devices: device ``i`` of architecture ``arches[i]``, every library a namespace whose
    FFI entry records ``(record, arch, args)``, and no resolved entry yet.  Returns the calls and the loads."""
    calls: list = []
    loads: list = []

    def load(name, arch):
        loads.append((name, arch))

        def entry(*args):
            calls.append((name, arch, args))

        return types.SimpleNamespace(**{cake_backend.record(name)["ffi_entry"]: entry})

    monkeypatch.setattr(cake_backend, "device_arch", lambda index: arches[index])
    monkeypatch.setattr(cake_backend, "load", load)
    monkeypatch.setattr(cake_backend, "_ENTRIES", {})
    return calls, loads


def test_launch_stage_resolves_each_stage_once_per_device(monkeypatch) -> None:
    # three devices, two of one architecture: each (stage, device) resolves on its first launch, then launches from
    # its cached entry with the values and the grid as ints
    arches = (ARCHES[0], ARCHES[-1], ARCHES[0])
    calls, loads = _fake_libraries(monkeypatch, arches)
    names = [cake_backend.stage_record("wy_tc", arch) for arch in arches]
    values = tuple(range(len(cake_backend.parameters(names[0]))))
    for _ in range(3):
        for index in range(len(arches)):
            cake_backend.launch_stage("wy_tc", (5, 2, 1), values, device_index=index)
    assert loads == list(zip(names, arches, strict=True))
    assert sorted(cake_backend._ENTRIES) == [("wy_tc", i) for i in range(len(arches))]
    per_pass = [
        (n, arch, values + (5, 2, 1)) for n, arch in zip(names, arches, strict=True)
    ]
    assert calls == per_pass * 3


def test_launch_stage_keeps_its_errors(monkeypatch) -> None:
    calls, loads = _fake_libraries(monkeypatch, (ARCHES[0],))
    name = cake_backend.stage_record("wy_tc", ARCHES[0])
    expected = cake_backend.parameters(name)
    count = len(expected)
    # a stage the registry does not serve
    with pytest.raises(KeyError):
        cake_backend.launch_stage("no_such_stage", (1, 1, 1), (), device_index=0)
    # the wrong argument count raises before the library loads (first launch) and on a cached entry
    for got in (count - 1, count + 1):
        with pytest.raises(ValueError) as error:
            cake_backend.launch_stage(
                "wy_tc", (1, 1, 1), tuple(range(got)), device_index=0
            )
        assert str(error.value) == (
            f"stage wy_tc ({name}) takes {count} arguments {expected}; got {got}"
        )
        assert loads == [] and cake_backend._ENTRIES == {}
    cake_backend.launch_stage("wy_tc", (1, 1, 1), tuple(range(count)), device_index=0)
    with pytest.raises(ValueError) as error:
        cake_backend.launch_stage(
            "wy_tc", (1, 1, 1), tuple(range(count + 1)), device_index=0
        )
    assert str(error.value) == (
        f"stage wy_tc ({name}) takes {count} arguments {expected}; got {count + 1}"
    )
    assert len(loads) == 1 and len(calls) == 1

    # a device that is not an exported target: device_arch's error, nothing cached for it
    def unexported(index):
        raise RuntimeError(f"device {index} is not an exported target")

    monkeypatch.setattr(cake_backend, "device_arch", unexported)
    with pytest.raises(RuntimeError, match="not an exported target"):
        cake_backend.launch_stage(
            "wy_tc", (1, 1, 1), tuple(range(count)), device_index=1
        )
    assert sorted(cake_backend._ENTRIES) == [("wy_tc", 0)]


# --------------------------------------------------------------------------- plan pins and host tables (no device)


def test_options_validate_their_pins() -> None:
    opts = cake_chain.make_options(
        scan_config=[64, 1], segment_chunks=0, finalize="finalize"
    )
    assert opts.scan_config == (64, 1) and opts.segment_chunks == 0
    assert opts.save_state is True and opts.host_cache is True and opts.dhu_scan is None
    assert opts.prep is None
    assert cake_chain.make_options(dhu_scan="tc").dhu_scan == "tc"
    assert cake_chain.make_options(prep="tma").prep == "tma"
    assert opts.lpt and opts.streams == 2
    assert cake_chain.make_options(lpt=False).lpt is False
    assert cake_chain.make_options(streams=1).streams == 1
    # a call without pins shares these options (the entry skips the validation)
    assert cake_chain.make_options() == cake_chain._DEFAULT_OPTIONS
    for bad in (
        {"scan_config": (32,)},
        {"scan_config": (32, 0)},
        {"scan_config": (32.0, 2)},
        {"segment_chunks": -1},
        {"segment_chunks": True},
        {"finalize": "finalize_g2"},
        {"warmup_log2": float("nan")},
        {"dhu_scan": "auto"},
        {"prep": "auto"},
        {"streams": 0},
        {"streams": 3},
        {"streams": True},
        {"streams": 2.0},
        {"streams": "2"},
    ):
        with pytest.raises(ValueError):
            cake_chain.make_options(**bad)


def test_finalize_rule() -> None:
    rule = cake_chain.finalize_stage
    # the three-head MLP finalize while its grid stays within 4 CTAs per SM, else the generic one
    assert rule(100, 4, 12, 148) == "finalize_mlp"
    assert rule(4 * 148 // 4 + 1, 4, 12, 148) == "finalize"
    assert rule(10, 4, 8, 148) == "finalize"
    assert rule(10, 4, 12, 148, forced="finalize") == "finalize"
    assert rule(10_000, 4, 12, 148, forced="finalize_mlp") == "finalize_mlp"
    with pytest.raises(ValueError):
        rule(10, 4, 8, 148, forced="finalize_mlp")
    with pytest.raises(ValueError):
        rule(10, 4, 10, 148)


# The campaign rows the selection rule decides on (serial adjoint plans on SM100 / SM103; on the 216 SMs of R200 the
# planner splits the adjoint scans of r01, r03 and r08 into segments), and the decision recorded with the rule's
# constants: dhu_tc2 on r01, r03, r05, r06, r08, r09 (r01: T_tc = 64 steps x t(64 CTAs), B200 154.9 us, GB300
# 141.9 us, R200 141.6 us).
_REAL_15 = (
    2756,
    4479,
    4053,
    4052,
    5896,
    4229,
    5238,
    3013,
    4429,
    4175,
    3743,
    4029,
    5404,
    3129,
    5875,
)
RULE_ROWS = (
    ("r01", (4096,) * 4, 16, True),
    ("r03", (4096,) * 8, 16, True),
    ("r05", (4096,) * 16, 16, True),
    ("r06", _REAL_15, 16, True),
    ("r08", (4096,) * 4, 32, True),
    ("r09", (4096,) * 16, 32, True),
)
RULE_DEVICES = {
    "sm_100a": ((148, 232448, 10), 154.9),
    "sm_103a": ((152, 232448, 10), 141.9),
    "sm_107a": ((216, 232448, 10), 113.3),
}
# The rows whose adjoint plan the planner splits into segments on a device (the others run serial plans).
SPLIT_ADJOINT_ROWS = {"sm_107a": ("r01", "r03", "r08")}


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_dhu_tc_rule_on_the_campaign_rows(arch, monkeypatch) -> None:
    props, r01_tc = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    for row, seq_lens, num_v_heads, selected in RULE_ROWS:
        _fwd, bwd = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        chunks = tuple(-(-n // 64) for n in seq_lens)
        serial = bwd.segment_len == 0 or max(chunks) <= bwd.segment_len
        assert serial is (row not in SPLIT_ADJOINT_ROWS.get(arch, ())), (row, bwd)
        rec = cake_chain.dhu_tc_pricing(chunks, num_v_heads, 0)
        assert rec["select"] is selected, (row, rec)
        if row == "r01":
            assert abs(rec["T_tc"] - r01_tc) < 0.1, rec


# The split (64, 1) adjoint plans of the campaign rows and the #7074 packs on each device, and the ones whose pass 2
# the split rule runs on dhu_tc2 (measured faster there; the other split plans keep the register pass 2).
_REAL_3 = (5406, 3448, 3896)
_REAL_7 = (4199, 2781, 4282, 4448, 4723, 5804, 3863)
SDT_ROWS = (
    ("r01", (4096,) * 4, 16),
    ("r02", _REAL_3, 16),
    ("r03", (4096,) * 8, 16),
    ("r04", _REAL_7, 16),
    ("r07", (16384,), 16),
    ("r08", (4096,) * 4, 32),
    ("r10", (16384,), 32),
    ("issue", (131071, 87531, 43385), 6),
    ("P3", (128147, 131027), 6),
    ("P1", (126565, 32714, 33696, 15081, 22979, 10782), 6),
    ("A16", (11000, 12000, 9768), 48),
)
_SDT_SPLIT_SM100 = {"r02", "r04", "r07", "r10", "issue", "P3", "P1"}
SDT_SPLIT = {
    "sm_100a": _SDT_SPLIT_SM100,
    "sm_103a": _SDT_SPLIT_SM100,
    "sm_107a": _SDT_SPLIT_SM100 | {"r01", "r03", "r08", "A16"},
}
# with the dv_local stage the register form needs priced in, every split (64, 1) row of these layouts runs pass 2 on
# dhu_tc2 (sm_100a r07: a 41 us priced saving, above the 40 us floor)
SDT_SELECTED = SDT_SPLIT


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_sdt_rule_on_the_split_rows(arch, monkeypatch) -> None:
    props, _r01_tc = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    split, selected = set(), set()
    for row, seq_lens, num_v_heads in SDT_ROWS:
        _fwd, bwd = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        chunks = tuple(-(-n // 64) for n in seq_lens)
        if bwd.segment_len == 0 or max(chunks) <= bwd.segment_len:
            continue
        if tuple(bwd.config) != cake_chain.SDT_CONFIG:
            continue
        split.add(row)
        rec = cake_chain.sdt_pricing(chunks, bwd.segment_len, num_v_heads, 0)
        assert rec["select"] is (
            rec["T_tc2"] <= rec["gain"] * rec["T_reg2"]
            and rec["saving"] >= rec["min_saving"]
        ), (row, rec)
        if rec["select"]:
            selected.add(row)
    assert split == SDT_SPLIT[arch]
    assert selected == SDT_SELECTED[arch]


def test_two_stream_split_rule() -> None:
    split = cake_chain.two_stream_split
    assert split(1, 32, 148) is None  # one sequence
    assert split(4, 32, 148) is None  # one round of 128 CTAs
    assert split(74, 4, 148) is None  # two full rounds: no idle slot
    assert split(4, 6, 5) is None  # 24 CTAs leave 1 idle slot; every group leaves more
    # 288 of the 512 CTAs leave 8 idle slots instead of 80
    assert split(16, 32, 148) == 9
    assert split(16, 32, 152) == 14  # 448 CTAs leave 8 idle slots instead of 96
    assert split(5, 4, 8) == 4  # 8 and 16 CTAs both fill their rounds: the larger group


# Whether a serial dhu_tc2 adjoint scan beside tail_g3 (a value-head group of three) runs as the R form, per
# architecture: on B200 / B300 / GB300 (the deterministic package's rule), not on R200.
SERIAL_G3_R_FORM = {"sm_100a": True, "sm_103a": True, "sm_107a": False}
# The (2, 6) head pair over 64 x 4096 tokens: a serial table whose adjoint scan groups as (group A, group B) sequences,
# per SM count (the deterministic package's groups)
SERIAL_G3_GROUPS = {148: (49, 15), 152: (25, 39), 212: (35, 29), 216: (36, 28)}


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_adjoint_r_form_rule(arch) -> None:
    rule = cake_chain.adjoint_r_form
    # (group, tail, serial dhu_tc2 scan, split pass 2 on dhu_tc2) -> runs the two groups
    cases = {
        (2, "tail_g2", True, False): True,
        (2, "tail", True, False): True,
        (2, "unfused", True, False): False,
        (3, "tail_g3", True, False): SERIAL_G3_R_FORM[arch],
        (3, "tail_g3_f32x", True, False): SERIAL_G3_R_FORM[arch],
        (2, "tail_g2_f32x", True, False): True,
        (3, "tail", True, False): False,
        (3, "unfused", True, False): False,
        (1, "tail", True, False): False,
        (4, "tail", True, False): False,
        (3, "tail_g3", False, True): True,
        (3, "tail_g3_f32x", False, True): True,
        (2, "tail_g2_f32x", False, True): False,
        (3, "tail", False, True): False,
        (2, "tail_g2", False, True): False,
        (3, "tail_g3", False, False): False,
        (2, "tail_g2", False, False): False,
    }
    for (group, form, serial, split), want in cases.items():
        got = rule(arch, group, form, serial_tc=serial, split_tc=split)
        assert got is want, (arch, group, form, serial, split)


@pytest.mark.parametrize(
    "arch, sm_count",
    [
        ("sm_100a", 148),
        ("sm_103a", 148),
        ("sm_103a", 152),
        ("sm_107a", 212),
        ("sm_107a", 216),
    ],
    ids=["B200", "B300", "GB300", "R200-212", "R200-216"],
)
def test_serial_r_form_of_a_group_of_three(arch, sm_count, monkeypatch) -> None:
    # the serial table of the (2, 6) head pair over 64 x 4096 tokens and its adjoint groups; the backward runs them
    # beside the cluster tail of a group of three where SERIAL_G3_R_FORM says so (one dhu_tc2 launch elsewhere)
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, 232448, 10)
    )
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    seq_lens, heads = (4096,) * 64, (2, 6)
    plans, _bwd_tc = cake_chain.host_plans(seq_lens, heads[1], 0)
    assert all(plan.segment_len <= 0 for plan in plans), plans
    meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans, lpt=True)
    _fwd, bwd = cake_chain.sequence_groups(meta, heads[1], sm_count)
    assert bwd is not None
    assert (bwd.split, len(seq_lens) - bwd.split) == SERIAL_G3_GROUPS[sm_count]
    form = cake_chain.tail_form(*heads, arch=arch, num_chunks=meta.num_chunks)
    assert form == "tail_g3_f32x"
    assert (
        cake_chain.adjoint_r_form(arch, 3, form, serial_tc=True, split_tc=False)
        is SERIAL_G3_R_FORM[arch]
    )


# The two-stream groups of the campaign rows, (forward group A, adjoint group A) in sequences: the forward's group A
# is the partial round, the adjoint's the full rounds.  The other rows run one stream (one sequence or one round).
STREAM_ROWS = {
    "sm_100a": {"r05": (7, 9), "r06": (6, 9), "r09": (7, 9)},
    "sm_103a": {"r05": (7, 9), "r06": (6, 9), "r09": (2, 14)},
    "sm_107a": {"r05": (3, 13), "r06": (2, 13), "r09": (3, 13)},
}


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_two_stream_groups_on_the_campaign_rows(arch, monkeypatch) -> None:
    props, _r01 = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    for row, seq_lens, num_v_heads, _form in RULE_ROWS:
        plans = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        expected = STREAM_ROWS[arch].get(row)
        for lpt in (True, False):
            meta = cake_chain.build_chunk_meta(
                seq_lens, "cpu", scan_plans=plans, lpt=lpt
            )
            fwd, bwd = cake_chain.sequence_groups(meta, num_v_heads, props[0])
            if expected is None:
                assert fwd is None and bwd is None, (row, fwd, bwd)
                continue
            assert fwd is not None and bwd is not None, (row, plans)
            assert (fwd.split, bwd.split) == expected, (row, lpt, fwd, bwd)
            starts = meta.seq_chunk_start.tolist()
            assert (
                fwd.split_chunk == starts[fwd.split]
                and bwd.split_chunk == starts[bwd.split]
            ), (row, lpt)


def test_sequence_groups_partition_a_serial_table() -> None:
    seq_lens, num_v_heads = (300, 77, 640, 1000, 0, 129), 4
    n = len(seq_lens)
    serial = (cake_chain.ScanPlan((32, 2), 0), cake_chain.ScanPlan((64, 1), 0))
    for lpt in (False, True):
        meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=serial, lpt=lpt)
        starts = meta.seq_chunk_start.tolist()
        for sm_count in range(2, n * num_v_heads + 1):
            k = cake_chain.two_stream_split(n, num_v_heads, sm_count)
            fwd, bwd = cake_chain.sequence_groups(meta, num_v_heads, sm_count)
            if k is None:
                assert fwd is None and bwd is None, sm_count
                continue
            # group A: the table's first sequences, at a chunk boundary; group B: the rest (neither empty of chunks)
            for groups, split in ((fwd, n - k), (bwd, k)):
                if groups is None:
                    assert starts[split] in (0, meta.num_chunks), (sm_count, split)
                    continue
                assert groups.split == split and groups.split_chunk == starts[split], (
                    sm_count,
                    groups,
                )
                assert 0 < groups.split_chunk < meta.num_chunks
    # a split plan, a forward walk and the device route run one stream
    split_plans = (cake_chain.ScanPlan((32, 2), 4), cake_chain.ScanPlan((64, 1), 4))
    meta = cake_chain.build_chunk_meta(
        (2000, 300), "cpu", scan_plans=split_plans, lpt=True
    )
    assert meta.fwd_segments.walk is not None
    assert cake_chain.sequence_groups(meta, num_v_heads, 5) == (None, None)
    serial_meta = cake_chain.build_chunk_meta((2000, 300), "cpu", scan_plans=serial)
    # 8 CTAs on 5 SMs split; without a layout on the host they do not
    assert cake_chain.sequence_groups(serial_meta, num_v_heads, 5) != (None, None)
    device_meta = dataclasses.replace(serial_meta, seq_lens=None)
    assert cake_chain.sequence_groups(device_meta, num_v_heads, 5) == (None, None)


def test_one_round_split_rule() -> None:
    # one_round_split: a serial adjoint scan of one round that fills more than half of the SMs, its sequences longest
    # first: group B is every sequence of the fewest chunks, kept when the SMs idle from its end to the longest walk's
    # end (the spare ones and those of the sequences that have ended) reach the floor, in walk steps per SM
    rule = cake_chain.one_round_split
    a16 = tuple(-(-n // 64) for n in sorted(_PAIR_LAYOUT, reverse=True))
    assert a16 == (188, 172, 153)
    # B200: 4 x 35 + 48 x (35 + 16) = 2588 idle steps on 148 SMs, 17.5 per SM
    assert rule(a16, 48, 148, 12.0) == 2
    assert rule(a16, 48, 148, 17.4) == 2
    assert rule(a16, 48, 148, 17.6) is None
    assert rule(a16, 48, 152, 0.0) == 2
    assert (
        rule((188, 153, 153), 48, 148, 0.0) == 1
    )  # every sequence of the fewest chunks
    for chunks, hv, sms in (
        (a16, 48, 143),  # more than one round
        (a16, 24, 148),  # at most half the SMs
        ((172, 172, 172), 48, 148),  # one length
        ((188,), 100, 148),  # one sequence
        ((153, 172, 188), 48, 148),  # not longest first
    ):
        assert rule(chunks, hv, sms, 0.0) is None, (chunks, hv, sms)


# The adjoint groups of the bench layouts on a serial table per device: B200 (ONE_ROUND_TAIL_DEVICES) forks the
# (16, 48) head pair over the pair layout before both walks, group B its shortest sequence; the R form of 64 x 4096
# stays; the other devices, the (8, 24) head pair and a table in the caller's order run one walk and one tail.
@pytest.mark.parametrize(
    "arch, sm_count",
    [("sm_100a", 148), ("sm_100a", 152), ("sm_103a", 152), ("sm_107a", 208)],
    ids=["B200", "GB200", "GB300", "R200"],
)
def test_one_round_groups_on_the_bench_layouts(arch, sm_count, monkeypatch) -> None:
    assert cake_chain.ONE_ROUND_TAIL_DEVICES == {("sm_100a", 148): 12.0}
    floor = cake_chain.ONE_ROUND_TAIL_DEVICES.get((arch, sm_count))
    serial = (cake_chain.ScanPlan((64, 1), 0), cake_chain.ScanPlan((64, 1), 0))
    for seq_lens, num_v_heads, lpt, want in (
        (_PAIR_LAYOUT, 48, True, (2, 188 + 172) if floor else None),
        (_PAIR_LAYOUT, 48, False, None),
        (_PAIR_LAYOUT, 24, True, None),
    ):
        meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=serial, lpt=lpt)
        fwd, bwd = cake_chain.sequence_groups(
            meta, num_v_heads, sm_count, one_round=floor
        )
        assert fwd is None, (seq_lens, num_v_heads, fwd)
        if want is None:
            assert bwd is None, (seq_lens, num_v_heads, lpt, bwd)
        else:
            assert bwd == cake_chain.StreamGroups(*want, early=True), bwd
            assert meta.seq_chunk_start.tolist()[bwd.split] == bwd.split_chunk
    meta = cake_chain.build_chunk_meta((4096,) * 64, "cpu", scan_plans=serial, lpt=True)
    k = cake_chain.two_stream_split(64, 6, sm_count)
    _fwd, bwd = cake_chain.sequence_groups(meta, 6, sm_count, one_round=floor)
    assert bwd is not None and (bwd.split, bwd.early) == (k, False), bwd
    # B200 plans the (16, 48) head pair over the pair layout on serial tables, so the backward takes the groups
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, 232448, 10)
    )
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    if (arch, sm_count) == ("sm_100a", 148):
        plans = cake_chain.host_plans(_PAIR_LAYOUT, 48, 0, group=3)[0]
        assert all(plan.segment_len <= 0 for plan in plans), plans


# The prep dispatch window on each architecture's SM count (slots = ctas_per_sm x SMs, the record's select
# constants): the plain prep inside (slots, window x slots], prep_tma outside; 4 x 4096 tokens at H = 8 (2048 CTAs)
# lie inside on sm_100a and sm_103a.  sm_107a has window 1.0, no band: prep_tma for every grid.
@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_prep_rule(arch, monkeypatch) -> None:
    props, _r01 = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    rule = cake_chain.use_prep_tma
    ctas_per_sm, window = cake_chain._prep_tma_constants(arch)
    slots = ctas_per_sm * props[0]
    if arch == "sm_107a":
        assert window == 1.0
        assert all(rule(n, 1, 0) for n in (1, slots, slots + 1, 4 * slots, 32 * slots))
        assert rule(4 * 4096 // 64, 8, 0) and rule(16 * 4096 // 64, 8, 0)
        assert rule(2, 2, 0)
        return
    assert rule(slots, 1, 0) and not rule(slots + 1, 1, 0)
    assert not rule(int(window * slots), 1, 0) and rule(int(window * slots) + 1, 1, 0)
    assert not rule(4 * 4096 // 64, 8, 0)
    assert rule(16 * 4096 // 64, 8, 0) and rule(2, 2, 0)


def test_host_tables_follow_the_layout() -> None:
    plans = (cake_chain.ScanPlan((32, 2), 0), cake_chain.ScanPlan((64, 1), 4))
    seq_lens = (100, 640, 1, 64)
    meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans)
    chunks = [2, 10, 1, 1]
    assert meta.num_chunks == sum(chunks) and meta.num_seqs == 4
    assert meta.seq_chunk_start.tolist() == [0, 2, 12, 13, 14]
    assert meta.chunk_start.tolist()[:3] == [0, 64, 100]
    assert meta.chunk_len.tolist()[:2] == [64, 36]
    assert meta.chunk_len.tolist()[-2:] == [1, 64]
    fwd, bwd = meta.fwd_segments, meta.bwd_segments
    # the serial plan: one segment per sequence
    assert fwd.num_segments == 4 and fwd.seq_seg_start.tolist() == [0, 1, 2, 3, 4]
    # segments of 4 chunks: the 10-chunk sequence splits into 3 balanced segments
    assert bwd.seg_chunk_start.tolist() == [0, 2, 5, 8, 12, 13, 14]
    assert bwd.seg_seq.tolist() == [0, 1, 1, 1, 2, 3]
    assert bwd.seq_seg_start.tolist() == [0, 1, 4, 5, 6]
    # the caller's order: no sequence order and the identity walk
    assert meta.order is None and meta.order_index is None
    assert fwd.walk is None and bwd.walk is None


def _segment_lens(seg_chunk_start) -> list[int]:
    starts = seg_chunk_start.tolist()
    return [b - a for a, b in zip(starts[:-1], starts[1:], strict=True)]


@pytest.mark.parametrize("fwd_segment_len", [0, 4], ids=["serial", "split"])
def test_lpt_tables_reorder_the_sequences_and_the_walk(fwd_segment_len) -> None:
    seq_lens = (100, 640, 1, 64, 2000, 640)
    chunks = [-(-n // 64) for n in seq_lens]
    offsets = _cu(seq_lens)
    plans = (
        cake_chain.ScanPlan((32, 2), fwd_segment_len),
        cake_chain.ScanPlan((64, 1), 0),
    )
    meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans, lpt=True)
    plain = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans)
    order = meta.order
    # the sequences longest first by chunk count, ties in the caller's order: a permutation
    assert sorted(order) == list(range(len(seq_lens)))
    assert order == cake_chain.lpt_sequence_order(seq_lens) == (4, 1, 5, 0, 2, 3)
    assert [chunks[s] for s in order] == sorted(chunks, reverse=True)
    assert meta.order_index.tolist() == [
        list(order),
        [order.index(s) for s in range(len(order))],
    ]
    # every chunk keeps its token offset and length; sequence vs of the table is the caller's order[vs]
    seq_chunk_start = meta.seq_chunk_start.tolist()
    for vs, seq in enumerate(order):
        c0, c1 = seq_chunk_start[vs], seq_chunk_start[vs + 1]
        assert c1 - c0 == chunks[seq]
        if c1 > c0:
            assert meta.chunk_start.tolist()[c0] == offsets[seq]
    pairs = sorted(zip(meta.chunk_start.tolist(), meta.chunk_len.tolist(), strict=True))
    assert pairs == sorted(
        zip(plain.chunk_start.tolist(), plain.chunk_len.tolist(), strict=True)
    )
    assert (
        meta.num_chunks == plain.num_chunks
        and meta.seq_lens == plain.seq_lens == seq_lens
    )
    # fwd_ho's walk: the forward segments longest first, ties in table order (None: already in that order)
    fwd = meta.fwd_segments
    lens = _segment_lens(fwd.seg_chunk_start)
    walk = fwd.walk.tolist() if fwd.walk is not None else list(range(fwd.num_segments))
    assert sorted(walk) == list(range(fwd.num_segments))
    assert walk == sorted(range(len(lens)), key=lambda seg: (-lens[seg], seg))
    if fwd_segment_len == 0:
        # the table order of a serial plan is already longest first
        assert fwd.walk is None
    else:
        assert fwd.walk is not None
    assert meta.bwd_segments.walk is None
    # an already longest-first layout keeps the caller's tables
    equal = cake_chain.build_chunk_meta((4096,) * 4, "cpu", scan_plans=plans, lpt=True)
    assert equal.order is None and equal.order_index is None


# --------------------------------------------------------------------------- against the FP64 recurrence


# id, sequence lengths, (H, HV), beta dtype, table route, keyword pins, l2norm in kernel
CASES = (
    ("pack-2x6-host-fp32beta", (100, 640, 77, 300), (2, 6), torch.float32, "host", {}, True),
    ("pack-2x6-device-bf16beta", (100, 640, 77, 300), (2, 6), torch.bfloat16, "device", {}, True),
    ("pack-4x8-host-bf16beta", (63, 64, 65, 500), (4, 8), torch.bfloat16, "host", {}, True),
    ("one-token", (1,), (1, 3), torch.float32, "host", {}, True),
    ("scan-32x2-serial", (300, 1000), (2, 4), torch.bfloat16, "host", {"scan_config": (32, 2), "segment_chunks": 0}, True),
    ("scan-32x1-serial", (300, 1000), (2, 4), torch.bfloat16, "host", {"scan_config": (32, 1), "segment_chunks": 0}, True),
    ("scan-64x1-serial", (300, 1000), (2, 4), torch.float32, "device", {"scan_config": (64, 1), "segment_chunks": 0}, True),
    ("split-32x2-host", (2000, 300), (1, 2), torch.bfloat16, "host", {"scan_config": (32, 2), "segment_chunks": 4}, True),
    ("split-64x1-device", (2000, 300), (1, 2), torch.float32, "device", {"scan_config": (64, 1), "segment_chunks": 6}, True),
    ("split-warmup", (2000,), (1, 3), torch.bfloat16, "host", {"scan_config": (32, 2), "segment_chunks": 8, "warmup_log2": -30.0}, True),
    ("finalize-forced", (100, 640, 77, 300), (2, 6), torch.bfloat16, "host", {"tail": "unfused", "finalize": "finalize"}, True),
    ("finalize-mlp-forced", (100, 640, 77, 300), (2, 6), torch.bfloat16, "host", {"tail": "unfused", "finalize": "finalize_mlp"}, True),
    ("tail-g2-pin", (100, 640, 77, 300), (2, 4), torch.float32, "host", {"tail": "tail_g2"}, True),
    ("unfused-pin-g2-device", (300, 1000), (2, 4), torch.bfloat16, "device", {"tail": "unfused"}, True),
    ("tail-g4-device", (100, 640, 77, 300), (1, 4), torch.bfloat16, "device", {}, True),
    ("tail-red-g2-pin", (100, 640, 77, 300), (2, 4), torch.bfloat16, "host", {"tail": "tail"}, True),
    ("tail-red-g3-h1-pin", (100, 640, 77, 300), (1, 3), torch.float32, "device", {"tail": "tail"}, True),
    ("recompute", (100, 640, 77, 300), (2, 6), torch.bfloat16, "host", {"save_state": False}, True),
    ("uncached", (100, 640, 77, 300), (2, 6), torch.float32, "device", {"host_cache": False}, True),
    ("no-l2norm", (200, 513), (2, 6), torch.bfloat16, "host", {}, False),
    ("recompute-split", (2000, 300), (1, 2), torch.bfloat16, "host", {"save_state": False, "scan_config": (32, 2), "segment_chunks": 4}, True),
    ("dhu-tc-g2-host", (100, 640, 77, 300), (2, 4), torch.bfloat16, "host", {"dhu_scan": "tc"}, True),
    ("dhu-tc-g3-device", (100, 640, 77, 300), (2, 6), torch.float32, "device", {"dhu_scan": "tc"}, True),
    ("dhu-tc-g1-recompute", (300, 1000), (2, 2), torch.bfloat16, "host", {"dhu_scan": "tc", "save_state": False}, True),
    ("dhu-reg-g2", (300, 1000), (4, 8), torch.float32, "host", {"dhu_scan": "reg", "scan_config": (64, 1)}, True),
    ("prep-tma-g3-device", (100, 640, 77, 300), (2, 6), torch.bfloat16, "device", {"prep": "tma"}, True),
    ("prep-plain-g2-host", (63, 64, 65, 500), (4, 8), torch.float32, "host", {"prep": "plain"}, True),
)  # fmt: skip


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_matches_the_fp64_recurrence(case, device) -> None:
    _id, seq_lens, heads, beta_dtype, route, pins, l2norm = case
    inputs = _inputs(seq_lens, heads, beta_dtype, device, normalized_qk=not l2norm)
    got = _run(inputs, seq_lens, route=route, use_qk_l2norm_in_kernel=l2norm, **pins)
    assert got["dbeta"].dtype == beta_dtype and got["dg"].dtype == torch.float32
    assert got["o"].dtype == torch.bfloat16
    ref = _reference(inputs, seq_lens, normalize=l2norm)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES)


@pytest.mark.parametrize(
    "pins",
    [{}, {"dhu_scan": "tc"}, {"scan_config": (32, 2), "segment_chunks": 4}],
    ids=["rule", "dhu-tc", "split"],
)
def test_initial_and_final_state(pins, device) -> None:
    seq_lens, heads = (200, 333), (2, 4)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=3)
    gen = torch.Generator().manual_seed(7)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    got = _run(
        inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, **pins
    )
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


@pytest.mark.parametrize("route", ["host", "device"])
def test_empty_sequences_carry_their_state(route, device) -> None:
    # the offsets may repeat (FLA's contract): an empty sequence passes its initial state through and its final-state
    # gradient back to dh0
    seq_lens, heads = (200, 0, 333, 0), (2, 4)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=37)
    gen = torch.Generator().manual_seed(41)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    got = _run(
        inputs,
        seq_lens,
        route=route,
        initial_state=h0,
        output_final_state=True,
        dht=dht,
    )
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


# --------------------------------------------------------------------------- long layouts and the pre-scan capacity

# Layouts whose device-route bound layout (one sequence of ``T // 64 + N - (N - 1)`` chunks) prices a split longer
# than one pre-scan CTA covers unless the search rejects it.  The two routes plan different layouts, so their outputs
# are compared with tolerance; each one carries the error of its own plan (twice the FP64 gate's max-abs bound).
LONG_LAYOUTS = (
    ("16x4096-h8", (4096,) * 16, (8, 16)),
    ("15-real-h8", _REAL_15, (8, 16)),
    ("1x16384-h16", (16384,), (16, 32)),
)  # fmt: skip


@pytest.mark.parametrize("layout", LONG_LAYOUTS, ids=[c[0] for c in LONG_LAYOUTS])
def test_split_plans_fit_the_prescan_capacity(layout, device) -> None:
    _id, seq_lens, heads = layout
    index = device.index
    for lens in (
        seq_lens,
        None,
    ):  # host route: real lengths; device route: the bound layout
        fwd, bwd = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), heads[1], index, lens
        )
        for kind, plan in (("fwd_h", fwd), ("dhu", bwd)):
            if plan.segment_len:
                capacity = cake_chain._prescan_capacity(kind, tuple(plan.config))
                assert 0 < plan.segment_len <= capacity, (
                    kind,
                    plan,
                    capacity,
                    lens is None,
                )


@pytest.mark.parametrize("layout", LONG_LAYOUTS, ids=[c[0] for c in LONG_LAYOUTS])
def test_long_device_route_runs_and_repeats_within_budget(layout, device) -> None:
    _id, seq_lens, heads = layout
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=13)
    first = _run(inputs, seq_lens, route="device")
    _assert_within_budget(first, _run(inputs, seq_lens, route="device"), OUTPUT_NAMES)
    host = _run(inputs, seq_lens, route="host")
    report = {name: _metrics(first[name], host[name]) for name in OUTPUT_NAMES}
    bad = [
        name
        for name, m in report.items()
        if not (
            m["finite"]
            and m["rel_rms"] <= REL_RMS_MAX
            and m["max_abs"] <= 2 * MAX_ABS_REL * m["max_ref"]
        )
    ]
    assert not bad, f"device vs host route outside the cross-plan gate: {bad}; {report}"


def test_segments_beyond_the_prescan_capacity_raise(device) -> None:
    seq_lens = (
        140 * 64,
    )  # one sequence of 140 chunks: a forced 129-chunk segment length splits it
    inputs = _inputs(seq_lens, (1, 2), torch.bfloat16, device, seed=17)
    with pytest.raises(ValueError, match="exceeds"):
        _run(inputs, seq_lens, scan_config=(64, 1), segment_chunks=129)


# --------------------------------------------------------------------------- agreement within the budgets


@pytest.mark.parametrize(
    "pins",
    [
        {"scan_config": (32, 2), "segment_chunks": 0, "dhu_scan": "reg"},
        {"scan_config": (64, 1), "segment_chunks": 0, "dhu_scan": "tc"},
        {"scan_config": (32, 2), "segment_chunks": 4, "dhu_scan": "reg"},
        {"scan_config": (64, 1), "segment_chunks": 6, "dhu_scan": "reg"},
        {"scan_config": (64, 1), "segment_chunks": 6, "dhu_scan": "tc"},
    ],
    ids=["serial-32x2", "serial-dhu-tc", "split-32x2", "split-64x1", "split-64x1-tc"],
)
def test_table_routes_agree(pins, device) -> None:
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=11)
    host = _run(inputs, seq_lens, route="host", **pins)
    dev = _run(inputs, seq_lens, route="device", **pins)
    _assert_within_budget(host, dev, OUTPUT_NAMES)


@pytest.mark.parametrize("route", ["host", "device"])
def test_run_to_run_within_budget(route, device) -> None:
    seq_lens = (2048, 100, 513)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=5)
    first = _run(inputs, seq_lens, route=route)
    for _ in range(2):
        _assert_within_budget(first, _run(inputs, seq_lens, route=route), OUTPUT_NAMES)
    uncached = _run(inputs, seq_lens, route=route, host_cache=False)
    _assert_within_budget(first, uncached, OUTPUT_NAMES)


@pytest.mark.parametrize(
    "pins",
    [{}, {"save_state": False}, {"scan_config": (32, 2), "segment_chunks": 4}],
    ids=["rule", "recompute", "fwd-split"],
)
def test_lpt_order_matches_the_caller_order(pins, device) -> None:
    # the host route's longest-first sequence order and fwd_ho walk against the caller's order, with an initial state
    # and a final-state gradient (state rows gathered and scattered): every output within the budgets
    seq_lens, heads = (300, 1000, 77, 640), (2, 4)
    assert cake_chain.lpt_sequence_order(seq_lens) == (1, 3, 0, 2)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=53)
    gen = torch.Generator().manual_seed(59)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    runs = {
        lpt: _run(
            inputs,
            seq_lens,
            initial_state=h0,
            output_final_state=True,
            dht=dht,
            lpt=lpt,
            **pins,
        )
        for lpt in (True, False)
    }
    _assert_within_budget(
        runs[True], runs[False], OUTPUT_NAMES + ("final_state", "dh0")
    )


def test_device_route_keeps_the_caller_order(device) -> None:
    seq_lens = (300, 1000, 77, 640)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int32, device=device)
    options = cake_chain.make_options(scan_config=(32, 2), segment_chunks=4)
    index = cake_chain._device_index(device)
    meta = cake_chain._chunk_meta(cu, None, 1, sum(seq_lens), index, 4, options)
    assert meta.seq_lens is None and meta.order is None and meta.order_index is None
    assert meta.fwd_segments.walk is None and meta.bwd_segments.walk is None
    walk = cake_chain._walk(device, meta.fwd_segments, options)
    assert walk.tolist() == list(range(meta.fwd_segments.num_segments))


def _small_device(monkeypatch, device, sm_count: int) -> None:
    """Let the planner and the two-stream rule see ``sm_count`` SMs (small layouts then group), with fresh layout
    caches so no table planned here outlives the test."""
    _sm, smem, major = cake_chain._device_props(cake_chain._device_index(device))
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, smem, major)
    )
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    monkeypatch.setattr(cake_chain, "_DHU_AUTO", {})
    monkeypatch.setattr(cake_chain, "_SDT_AUTO", {})
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})


def _record_streams(monkeypatch) -> list[tuple[str, int]]:
    """The (stage, stream) of every launch from here on, in order."""
    launches: list[tuple[str, int]] = []
    launch = cake_backend.launch_stage

    def recording(stage, grid, values, **kwargs):
        launches.append(
            (stage, torch.cuda.current_stream(kwargs["device_index"]).cuda_stream)
        )
        return launch(stage, grid, values, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return launches


# (H, HV) -> the SMs the two-stream tests pretend: 4 sequences then split (HV 4: 16 CTAs on 6 SMs, adjoint A = 3;
# HV 6: 24 CTAs on 7 SMs, adjoint A = 1), and the forward's group A is the rest
STREAM_CASES = {(2, 4): 6, (2, 6): 7}
STREAM_LAYOUT = (300, 77, 640, 1000)  # every group boundary falls after a partial chunk
# serial plans on dhu_tc2: both directions can group
STREAM_PINS = {"segment_chunks": 0, "dhu_scan": "tc"}


def _on_device(stages: list[str]) -> list[str]:
    """``stages`` with ``wy_tc`` as this device's forward WY stage (:func:`cake_chain.wy_stage`) and ``dhu_tc2`` as its
    tcgen05 adjoint scan (:func:`cake_chain.dhu_tc2_stage`)."""
    arch = cake_backend.device_arch(torch.cuda.current_device())
    named = {
        "wy_tc": cake_chain.wy_stage(arch),
        "dhu_tc2": cake_chain.dhu_tc2_stage(arch),
    }
    return [named.get(stage, stage) for stage in stages]


@pytest.mark.parametrize("save_state", [True, False], ids=["saved", "recompute"])
@pytest.mark.parametrize("lpt", [True, False], ids=["lpt", "caller-order"])
@pytest.mark.parametrize("heads", sorted(STREAM_CASES), ids=["g2", "g3"])
def test_two_streams_match_one_stream(
    heads, lpt, save_state, device, monkeypatch
) -> None:
    seq_lens = STREAM_LAYOUT
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    g3_r_form = heads[1] == 3 * heads[0] and SERIAL_G3_R_FORM.get(arch, False)
    _small_device(monkeypatch, device, STREAM_CASES[heads])
    assert (
        cake_chain.two_stream_split(len(seq_lens), heads[1], STREAM_CASES[heads])
        is not None
    )
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=61)
    gen = torch.Generator().manual_seed(67)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    runs = []
    for streams in (2, 1, 2):
        start = len(launches)
        runs.append(_run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, streams=streams,
                         lpt=lpt, save_state=save_state, **STREAM_PINS))  # fmt: skip
        side = [stage for stage, stream in launches[start:] if stream != main]
        # the side stream: wy_tc and fwd_ho of the forward's group A, dhu_tc2 of the adjoint's group B (beside the
        # fused tail of a value-head group of two, and beside the group of three's cluster tail on the architectures
        # of SERIAL_G3_R_FORM)
        expected = _on_device(
            ["wy_tc", "fwd_ho"]
            + (["dhu_tc2"] if heads[1] == 2 * heads[0] or g3_r_form else [])
        )
        assert side == (expected if streams == 2 else []), (streams, launches[start:])
    _assert_within_budget(runs[0], runs[1], names)
    _assert_within_budget(runs[0], runs[2], names)
    if lpt and save_state:
        ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
        _assert_close_to_reference(runs[0], ref, names)


# a split adjoint table on dhu_tc2 beside tail_g3_f32x (pins: (64, 1) tables of 4-chunk segments, pass 2 on dhu_tc2): 10
# segments of HV 6 on 7 SMs group as 7 + 3, so on an architecture of SPLIT_R_FORM the adjoint runs the R form over the
# segments (dhu_tc2 of the last 3 on the side stream); the same bits as one stream and as one dhu_tc2 launch
SPLIT_R_PINS = {
    "scan_config": (64, 1),
    "segment_chunks": 4,
    "dhu_scan": "tc",
    "tail": "tail_g3_f32x",
}


@pytest.mark.parametrize("lpt", [True, False], ids=["lpt", "caller-order"])
def test_split_r_form_matches_one_stream(lpt, device, monkeypatch) -> None:
    seq_lens, heads = STREAM_LAYOUT, (2, 6)
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    _small_device(monkeypatch, device, STREAM_CASES[heads])
    monkeypatch.setattr(cake_chain, "SPLIT_R_FORM", (arch,))
    options = cake_chain.make_options(lpt=lpt, **SPLIT_R_PINS)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
    meta = cake_chain._chunk_meta(
        cu.to(device), cu, 1, sum(seq_lens), index, heads[1], options, 3
    )
    assert meta.bwd_segments.num_segments == 10 and meta.fwd_groups is None
    assert meta.bwd_groups is not None and meta.bwd_groups.split == 7, meta.bwd_groups
    kept = cake_chain._host_plans(seq_lens, heads[1], index, options, group=3)[2]
    assert not kept  # a pinned table is not kept for the form, it takes it
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=71)
    gen = torch.Generator().manual_seed(73)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    runs = []
    for streams in (2, 1, 2):
        start = len(launches)
        runs.append(_run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, streams=streams,
                         lpt=lpt, **SPLIT_R_PINS))  # fmt: skip
        side = [stage for stage, stream in launches[start:] if stream != main]
        assert side == (_on_device(["dhu_tc2"]) if streams == 2 else []), (
            streams,
            launches[start:],
        )
    monkeypatch.setattr(cake_chain, "SPLIT_R_FORM", ())
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    start = len(launches)
    one = _run(
        inputs,
        seq_lens,
        initial_state=h0,
        output_final_state=True,
        dht=dht,
        lpt=lpt,
        **SPLIT_R_PINS,
    )
    assert [stage for stage, _ in launches[start:]].count(
        _on_device(["dhu_tc2"])[0]
    ) == 1
    for run in runs[1:] + [one]:
        _assert_same_bits(run, runs[0], names)


# The bench rows of a value-head group of three that run the R form beside the group's cluster tail on B200 / B300 /
# GB300 (the (2, 6) head pair): 64 x 4096 tokens on its serial table, and six packed sequences on the planner's split
# adjoint table that SPLIT_R_FORM keeps.  The deterministic package runs the same tables and groups.
G3_R_FORM_ROWS = {
    "64x4096": (4096,) * 64,
    "P1": (126565, 32714, 33696, 15081, 22979, 10782),
}


@pytest.mark.parametrize("plan", ["planner", "stage1"])
@pytest.mark.parametrize("row", sorted(G3_R_FORM_ROWS))
def test_group_of_three_r_form_keeps_its_bits(row, plan, device, monkeypatch) -> None:
    # every route the package ships for the row (the forward's A / split routes, the FP32 dv_local adjoint, the
    # cluster tail): within the budgets of the deterministic package.  Then on the prep's forward in both packages
    # (prep, wy_tc2, fwd_ho; fwd_a_route's forward: test_fwd_a_within_the_budgets_of_the_prep_forward): the R form
    # beside tail_g3_f32x gives the bits of one dhu_tc2 launch (one stream); at the same tables the package tail's bits
    # (the planner's split table that SPLIT_R_FORM keeps serves a group of three's cluster tail only, so that
    # comparison runs without it), within the budgets of the deterministic package; without the FP32 dv_local adjoint
    # the deterministic package's o, dv, dg and dbeta (dq / dk within its budgets), and with tail_g3 pinned all of its
    # bits
    # plan "planner": the planner's tables (ADJOINT_SPLIT_ARCHS empty); plan "stage1": the shipped plan, where a row the
    # one-wave split prices runs the split beside the group's cluster tail instead of the R form
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    seq_lens, heads = G3_R_FORM_ROWS[row], (2, 6)
    if plan == "planner":
        monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=331)
    _assert_within_budget(
        _run(inputs, seq_lens), _run(inputs, seq_lens, det=True), OUTPUT_NAMES
    )
    _prep_forward_on_both(monkeypatch)
    if (
        plan == "stage1"
        and cake_chain._adjoint_split_length(
            seq_lens, heads[1], index, cake_chain.make_options()
        )
        is not None
    ):
        _check_group_of_three_split(inputs, seq_lens, device, monkeypatch)
        return
    if not SERIAL_G3_R_FORM.get(arch, False):
        pytest.skip(f"{arch} runs one dhu_tc2 launch on these rows")
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    got = _run(inputs, seq_lens)
    side = [stage for stage, stream in launches if stream != main]
    tc2 = _on_device(["dhu_tc2"])[0]
    # the adjoint's group B on the side stream (the forward's two groups, if any, run wy_tc / fwd_ho there)
    assert side.count(tc2) == 1, launches
    assert [stage for stage, _ in launches].count(tc2) == 2, launches
    assert "tail_g3_f32x" in [stage for stage, _ in launches], launches
    _assert_same_bits(_run(inputs, seq_lens), got, OUTPUT_NAMES)
    _assert_same_bits(_run(inputs, seq_lens, streams=1), got, OUTPUT_NAMES)
    split_r_form = cake_chain.SPLIT_R_FORM
    monkeypatch.setattr(cake_chain, "SPLIT_R_FORM", ())
    _assert_same_bits(
        _run(inputs, seq_lens, tail="tail", host_cache=False),
        _run(inputs, seq_lens, host_cache=False),
        OUTPUT_NAMES,
    )
    monkeypatch.setattr(cake_chain, "SPLIT_R_FORM", split_r_form)
    det = _run(inputs, seq_lens, det=True)
    _assert_within_budget(got, det, OUTPUT_NAMES)
    monkeypatch.setattr(cake_chain, "DVL32_ARCHS", ())
    _assert_same_bits(_run(inputs, seq_lens, tail="tail_g3"), det, OUTPUT_NAMES)
    plain = _run(inputs, seq_lens)
    for name in ("o", "dv", "dg", "dbeta"):
        assert torch.equal(plain[name], det[name]), name
    _assert_within_budget(plain, det, ("dq", "dk"))


def _check_group_of_three_split(inputs, seq_lens, device, monkeypatch) -> None:
    """A group of three on a row the one-wave split prices: the split's forms and its composition beside tail_g3_f32x,
    no R form and no other adjoint scan; the bits repeat, on one stream too, the package tail gives the cluster
    tail's bits at the same tables, and every output stays within the budgets of the deterministic package."""
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    launches = _record_streams(monkeypatch)
    got = _run(inputs, seq_lens)
    stages = [stage for stage, _ in launches]
    composer = cake_chain._adjoint_chain_stage(arch)
    assert stages.count("dhu_tc2_dvl32_modes") >= 1 and composer in stages, stages
    assert "tail_g3_f32x" in stages, stages
    assert not [
        s
        for s in stages
        if s in ("dhu_tc2", "dhu_tc2_dvl32", "dv_local")
        or s.startswith(("dhu_pre", "dhu_r"))
    ], stages
    _assert_same_bits(_run(inputs, seq_lens), got, OUTPUT_NAMES)
    _assert_same_bits(_run(inputs, seq_lens, streams=1), got, OUTPUT_NAMES)
    _assert_same_bits(_run(inputs, seq_lens, tail="tail", host_cache=False), _run(inputs, seq_lens, host_cache=False),
                      OUTPUT_NAMES)  # fmt: skip
    _assert_within_budget(got, _run(inputs, seq_lens, det=True), OUTPUT_NAMES)


def test_one_round_groups_keep_the_bits_of_one_walk(device, monkeypatch) -> None:
    # the one-round groups (one_round_split) of the (16, 48) head pair over the pair layout: both walks forked
    # together, group A's walk and tail on the caller's stream, group B's (the shortest sequence) on the side stream;
    # the bits of one walk and one tail (one stream), run to run, through the packed row and the head views, and the
    # recorded backward plan replays the general path's launches and bits.  The device's floor is lifted to 0 so that
    # every device whose SMs hold the 144 CTAs in one round more than half full runs the groups.
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    sm_count = cake_chain._device_props(index)[0]
    seq_lens, heads = _PAIR_LAYOUT, (16, 48)
    if not 3 * 48 <= sm_count < 2 * 3 * 48:
        pytest.skip(
            f"{sm_count} SMs do not hold the adjoint scan in one round more than half full"
        )
    plans = cake_chain.host_plans(seq_lens, heads[1], index, group=3)[0]
    if any(plan.segment_len > 0 for plan in plans):
        pytest.skip(f"{arch} plans the pair layout on split tables")
    monkeypatch.setattr(cake_chain, "ONE_ROUND_TAIL_DEVICES", {(arch, sm_count): 0.0})
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=347)
    main = torch.cuda.current_stream(device).cuda_stream
    launches = _record_streams(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "BWD_PLAN", False)
        general = _run(inputs, seq_lens)
    tc2 = cake_chain.dhu_tc2_stage(arch)
    form = cake_chain.tail_form(*heads, arch=arch)
    assert form in cake_chain.GROUP_TAILS, form
    tails = (form, form + "_bf16")  # the BF16 form for a BF16 beta
    stages = [stage for stage, _ in launches]
    side = [stage for stage, stream in launches if stream != main]
    side_tails = [i for i, stage in enumerate(side) if stage in tails]
    assert stages.count(tc2) == 2 and side.count(tc2) == 1, launches
    assert len([s for s in stages if s in tails]) == 2 and len(side_tails) == 1, (
        launches
    )
    assert side.index(tc2) < side_tails[0], side
    log = _logged_launches(monkeypatch)
    replays = _counted_replays(monkeypatch)
    recorded = _run(inputs, seq_lens)
    first = list(log)
    # the fork and both walks come first, back to back, before the host work that makes the tail's outputs: no
    # gradient is made before the fork (the walks run while the host makes them)
    (plan,) = cake_chain._BWD_PLANS.values()
    ops = plan.ops
    forks = [i for i, op in enumerate(ops) if op[0] == cake_chain._OP_FORK]
    assert len(forks) == 1, ops
    f = forks[0]
    assert [op[:2] for op in ops[f + 1 : f + 5]] == [
        (cake_chain._OP_LAUNCH, tc2),
        (cake_chain._OP_STREAM, 1),
        (cake_chain._OP_LAUNCH, tc2),
        (cake_chain._OP_STREAM, 0),
    ], ops[f : f + 5]
    made = {op[1] for op in ops[:f] if op[0] == cake_chain._OP_EMPTY}
    made |= {
        op[1] + j for op in ops[:f] if op[0] == cake_chain._OP_QKV for j in range(3)
    }
    assert not made & set(plan.grads[:5]), (made, plan.grads)
    log.clear()
    runs = [_run(inputs, seq_lens) for _ in range(4)]
    assert log == first * 4 and len(replays) == 4
    for got in (recorded, *runs, _run_packed(inputs, seq_lens, heads)):
        _assert_same_bits(got, general, OUTPUT_NAMES)
    _assert_same_bits(_run(inputs, seq_lens, streams=1), general, OUTPUT_NAMES)
    monkeypatch.setattr(cake_chain, "ONE_ROUND_TAIL_DEVICES", {})
    launches.clear()
    one = _run(inputs, seq_lens, host_cache=False)
    assert [stage for stage, _ in launches].count(tc2) == 1, launches
    _assert_same_bits(one, general, OUTPUT_NAMES)


def test_two_streams_survive_interleaved_allocations(device, monkeypatch) -> None:
    # the caller's stream allocates, writes and frees between two-stream calls (pooled and fresh workspaces) and,
    # after every side-stream launch, while the side stream works: memory released before the join would be
    # overwritten with NaN on the caller's stream
    seq_lens, heads = STREAM_LAYOUT, (2, 4)
    _small_device(monkeypatch, device, STREAM_CASES[heads])
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=73)
    one = _run(inputs, seq_lens, streams=1, **STREAM_PINS)
    gen = torch.Generator().manual_seed(79)
    main = torch.cuda.current_stream(device)
    launch = cake_backend.launch_stage
    side_launches = []

    def churning(stage, grid, values, **kwargs):
        out = launch(stage, grid, values, **kwargs)
        if torch.cuda.current_stream(kwargs["device_index"]) != main:
            side_launches.append(stage)
            with torch.cuda.stream(main):
                for size in (1 << 12, 1 << 16, 1 << 20):
                    torch.full((size,), float("nan"), device=device)
        return out

    monkeypatch.setattr(cake_backend, "launch_stage", churning)
    for i in range(16):
        junk = [
            torch.full(
                (int(torch.randint(1, 1 << 20, (1,), generator=gen)),),
                float("nan"),
                device=device,
            )
            for _ in range(3)
        ]
        got = _run(inputs, seq_lens, streams=2, host_cache=i % 2 == 0, **STREAM_PINS)
        del junk
        _assert_within_budget(got, one, OUTPUT_NAMES)
    assert side_launches == _on_device(["wy_tc", "fwd_ho", "dhu_tc2"]) * 16, (
        side_launches
    )


def test_two_streams_under_cuda_graph_capture(device, monkeypatch) -> None:
    seq_lens, heads = STREAM_LAYOUT, (2, 4)
    _small_device(monkeypatch, device, STREAM_CASES[heads])
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=83)
    eager = _run(inputs, seq_lens, streams=1, **STREAM_PINS)
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def step():
        for leaf in leaves.values():
            leaf.grad = None
        o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                      scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                                      cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True, streams=2,
                                      **STREAM_PINS)  # fmt: skip
        (o.float() * inputs["do"].float()).sum().backward()
        return o

    # warm up on a side stream (the tables, the workspaces and the libraries), then capture forward and backward
    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup):
        for _ in range(2):
            step()
    torch.cuda.current_stream(device).wait_stream(warmup)
    launches = _record_streams(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = step()
    captured = list(launches)
    # the capture forked the side stream: two streams, the side one for wy_tc, fwd_ho and dhu_tc2
    streams = {stream for _stage, stream in captured}
    assert len(streams) == 2, captured
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    got = {
        "o": o,
        **{"d" + name: leaves[name].grad for name in ("q", "k", "v", "g", "beta")},
    }
    _assert_within_budget(got, eager, OUTPUT_NAMES)
    assert len(launches) == len(captured)  # replays launch nothing from the host


def test_launches_resolve_once_and_reuse_their_entries(device, monkeypatch) -> None:
    # the first call resolves every stage it launches once; the second resolves nothing and launches the same stages
    # from the cached entries, within the budgets
    monkeypatch.setattr(cake_backend, "_ENTRIES", {})
    resolved: list[str] = []
    resolve = cake_backend._stage_entry

    def counting(stage, device_index, count):
        resolved.append(stage)
        return resolve(stage, device_index, count)

    monkeypatch.setattr(cake_backend, "_stage_entry", counting)
    stages = _record_launches(monkeypatch)
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 4), torch.bfloat16, device, seed=101)
    first = _run(inputs, seq_lens)
    n = len(stages)
    assert sorted(resolved) == sorted(set(stages))
    second = _run(inputs, seq_lens)
    assert len(resolved) == len(set(stages)) and stages[n:] == stages[:n]
    _assert_within_budget(first, second, OUTPUT_NAMES)
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    for stage in set(stages):
        entry, arity, name = cake_backend._ENTRIES[(stage, index)]
        assert name == cake_backend.stage_record(stage, arch)
        assert arity == len(cake_backend.parameters(name))
        assert entry is getattr(
            cake_backend.load(name, arch), cake_backend.record(name)["ffi_entry"]
        )


@pytest.mark.parametrize("streams", [1, 2])
def test_launches_run_on_the_callers_stream(streams, device, monkeypatch) -> None:
    # The caller's stream sleeps, then rewrites q (before the forward) and do (before the backward).  A launch on any
    # other stream would run during the sleep and read the old q / do; the outputs must be those of the new ones.
    seq_lens, heads = STREAM_LAYOUT, (2, 4)
    _small_device(monkeypatch, device, STREAM_CASES[heads])
    old = _inputs(seq_lens, heads, torch.bfloat16, device, seed=103)
    new = _inputs(seq_lens, heads, torch.bfloat16, device, seed=107)
    want = _run(
        {**old, "q": new["q"], "do": new["do"]},
        seq_lens,
        streams=streams,
        **STREAM_PINS,
    )
    cu = _cu(seq_lens)
    caller = torch.cuda.Stream(device=device)
    caller.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(caller):
        leaves = {
            name: old[name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        }
        do = old["do"].clone()
        torch.cuda._sleep(200_000_000)
        with torch.no_grad():
            leaves["q"].copy_(new["q"])
        o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                      scale=1.0 / math.sqrt(HEAD_DIM),
                                      cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                                      cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64), use_qk_l2norm_in_kernel=True,
                                      streams=streams, **STREAM_PINS)  # fmt: skip
        torch.cuda._sleep(200_000_000)
        do.copy_(new["do"])
        o.backward(do)
    torch.cuda.synchronize()
    got = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_within_budget(got, want, OUTPUT_NAMES)


def test_low_level_entries_reject_host_tensors(device) -> None:
    # the low-level entries skip the API checks: a host tensor's launch runs on the current device (as
    # torch.device("cpu").index maps there) and the binding rejects it with its ValueError
    cu = torch.tensor([0, 100, 164], dtype=torch.int32)
    assert cake_chain._tensor_index(cu) == torch.cuda.current_device()
    assert cake_chain._tensor_index(cu.to("meta")) == torch.cuda.current_device()
    index = cake_chain._device_index(device)
    assert cake_chain._tensor_index(cu.to(device)) == index
    plans = cake_chain.scan_plans(164, 2, 4, index, (100, 64))
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        cake_chain.build_chunk_meta_from_cu_seqlens(cu, 164, scan_plans=plans)


def test_each_caller_stream_forks_its_own_side_stream(device) -> None:
    index = cake_chain._device_index(device)
    a, b = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    side = cake_chain._side_stream(index, a)
    assert cake_chain._side_stream(index, a) is side
    assert cake_chain._side_stream(index, b) not in (side, a, b)
    assert side not in (a, b)
    assert side.priority == torch.cuda.Stream.priority_range()[1]


def test_workspaces_are_keyed_by_the_current_stream(device) -> None:
    index = cake_chain._device_index(device)
    for stream in (torch.cuda.current_stream(device), torch.cuda.Stream(device=device)):
        with torch.cuda.stream(stream):
            key = (index, stream.cuda_stream)
            assert cake_chain._stream_key(index) == key
            assert cake_chain._POOLED.table(index) is cake_chain._POOLED.tables[key]


@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
def test_prep_forms_agree(route, beta_dtype, device, monkeypatch) -> None:
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, (2, 6), beta_dtype, device, seed=19)
    stages = _record_launches(monkeypatch)
    tma = _run(inputs, seq_lens, route=route, prep="tma")
    tma_preps = _preps(stages)
    stages.clear()
    plain = _run(inputs, seq_lens, route=route, prep="plain")
    suffix = "_bf16beta" if beta_dtype == torch.bfloat16 else ""
    assert tma_preps == {"prep_tma" + suffix} and _preps(stages) == {"prep" + suffix}
    _assert_within_budget(tma, plain, OUTPUT_NAMES)


@pytest.mark.parametrize("save_state", [True, False], ids=["saved", "recompute"])
def test_the_wy_stage_of_the_device_gives_the_wy_tc_bits(
    save_state, device, monkeypatch
) -> None:
    # every forward WY launch (and the backward's recompute) runs the device's stage, wy_tc2; wy_tc in its place
    # gives the same bits
    seq_lens = (300, 77, 640, 17)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=29)
    wy = cake_chain.wy_stage(cake_backend.device_arch(torch.cuda.current_device()))
    stages = _record_launches(monkeypatch)
    ruled = _run(inputs, seq_lens, save_state=save_state)
    launched = [stage for stage in stages if stage.startswith("wy_tc")]
    assert launched and set(launched) == {wy}, stages
    stages.clear()
    monkeypatch.setattr(cake_chain, "wy_stage", lambda arch: "wy_tc")
    plain = _run(inputs, seq_lens, save_state=save_state)
    replaced = [stage for stage in stages if stage.startswith("wy_tc")]
    assert replaced == ["wy_tc"] * len(launched), stages
    _assert_same_bits(ruled, plain, OUTPUT_NAMES)


def _saved_list(o) -> tuple:
    """The saved tensors of the package's autograd node behind ``o``."""
    node = o.grad_fn
    while node is not None and not type(node).__name__.startswith("CakeGDNNondet"):
        node = node.next_functions[0][0]
    assert node is not None, o.grad_fn
    return node.saved_tensors


def test_the_saved_list_keeps_its_arity(device, monkeypatch) -> None:
    # the saved list has the same entries, shapes, dtypes and devices whatever the tape keeps: the state of the
    # device's tape level (T_full: w, h and v_new, never u) under save_state, none without it, and a storage-free
    # placeholder (every stride 0) for the rest; the backward of the T_full tape gives the bits of the whole state's
    seq_lens = (300, 77, 640, 17)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=31)
    arch = cake_backend.device_arch(torch.cuda.current_device())
    cu = _cu(seq_lens)
    lists = {}
    for save_state in (True, False):
        leaves = {
            name: inputs[name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        }
        o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                               scale=1.0 / math.sqrt(HEAD_DIM), use_qk_l2norm_in_kernel=True,
                               cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                               cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64), save_state=save_state)  # fmt: skip
        saved = _saved_list(o)
        # the last entry: the M-share segment state, a placeholder on this layout (no split forward)
        assert len(saved) == 10 + len(cake_plan.STATE_NAMES)
        assert (
            all(s == 0 for s in saved[-1].stride()) and saved[-1].dtype == torch.float32
        )
        kept = cake_plan.tape_state(arch, save_state)
        for name, t in zip(cake_plan.STATE_NAMES, saved[9:-1], strict=True):
            placeholder = (
                all(s == 0 for s in t.stride())
                and t.untyped_storage().nbytes() == t.element_size()
            )
            assert placeholder is (name not in kept), (save_state, name, t.stride())
        lists[save_state] = [
            None if t is None else (t.shape, t.dtype, t.device) for t in saved
        ]
    assert "u" not in cake_plan.tape_state(arch, True)
    assert lists[True] == lists[False]
    t_full = _run(inputs, seq_lens)
    monkeypatch.setattr(
        cake_plan,
        "LEVEL_STATE",
        types.MappingProxyType({"T_full": cake_plan.STATE_NAMES}),
    )
    whole = _run(inputs, seq_lens)
    _assert_same_bits(t_full, whole, OUTPUT_NAMES)


def test_unaligned_qk_take_the_plain_prep(device, monkeypatch) -> None:
    # q / k as contiguous views 8 bytes past a 16-byte boundary: the tensor maps of prep_tma need 16-byte aligned
    # addresses, the plain prep's 4-element loads 8; a prep_tma pin still runs the plain prep, within the budgets.
    seq_lens = (300, 77)
    inputs = _inputs(seq_lens, (2, 4), torch.float32, device, seed=23)
    leaves = {}
    for name in ("q", "k"):
        t = inputs[name]
        buf = torch.empty(t.numel() + 4, dtype=t.dtype, device=device)
        leaves[name] = buf[4:].view(t.shape).copy_(t).requires_grad_(True)
        assert leaves[name].is_contiguous() and leaves[name].data_ptr() % 16 == 8
    stages = _record_launches(monkeypatch)
    shifted = _run(inputs, seq_lens, leaves=leaves, prep="tma")
    assert _preps(stages) == {"prep"}
    aligned = _run(inputs, seq_lens, prep="plain")
    _assert_within_budget(shifted, aligned, OUTPUT_NAMES)


@pytest.mark.parametrize(
    "preps", [(None, None), ("tma", "plain")], ids=["rule", "tma-vs-plain"]
)
def test_equal_length_batch_matches_the_packed_layout(preps, device) -> None:
    batch, seq_len, heads = 3, 150, (2, 6)
    inputs = _inputs((seq_len,), heads, torch.float32, device, seed=9, batch=batch)
    batched = _run(inputs, None, prep=preps[0])
    packed_inputs = {
        name: t.reshape(1, batch * seq_len, *t.shape[2:]).contiguous()
        for name, t in inputs.items()
    }
    packed = _run(packed_inputs, (seq_len,) * batch, route="host", prep=preps[1])
    for name in OUTPUT_NAMES:
        packed[name] = packed[name].reshape(batched[name].shape)
    _assert_within_budget(batched, packed, OUTPUT_NAMES)


# --------------------------------------------------------------------------- input contract


def test_rejects_unsupported_inputs(device) -> None:
    inputs = _inputs((64,), (1, 2), torch.float32, device)
    with pytest.raises(NotImplementedError):
        cake_gdn_nondet(
            inputs["q"].half(), inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        )
    cu = torch.tensor([0, 64], dtype=torch.int64, device=device)
    two = {name: torch.cat([t, t], 0) for name, t in inputs.items()}
    with pytest.raises(ValueError):
        cake_gdn_nondet(
            two["q"], two["k"], two["v"], two["g"], two["beta"], cu_seqlens=cu
        )
    with pytest.raises(ValueError):
        cake_gdn_nondet(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
            segment_chunks=-1,
        )  # fmt: skip


# One packed call of three sequences (100, 30, 34 tokens; H = 2, HV = 4) with an initial state, and the argument each
# rejected case replaces; every one raises ValueError before the first launch.
def _invalid_calls(device):
    inputs = _inputs((100, 30, 34), (2, 4), torch.float32, device, seed=29)
    q, k, v, g, beta = (inputs[n] for n in ("q", "k", "v", "g", "beta"))
    cu = torch.tensor([0, 100, 130, 164], dtype=torch.int64)
    h0 = torch.zeros(3, 4, HEAD_DIM, HEAD_DIM, device=device)
    base = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": h0, "cu_seqlens": cu.to(device),
            "cu_seqlens_cpu": cu}  # fmt: skip
    cases = {
        "q-on-host": {"q": q.cpu()},
        "g-on-host": {"g": g.cpu()},
        "initial-state-on-host": {"initial_state": h0.cpu()},
        "cu-seqlens-on-host": {"cu_seqlens": cu},
        "k-heads": {"k": k[:, :, :1]},
        "head-dim-64": {"q": q[..., :64], "k": k[..., :64]},
        "v-tokens": {"v": v[:, :100]},
        "value-heads-not-a-multiple": {
            "v": v[:, :, :3],
            "g": g[:, :, :3],
            "beta": beta[:, :, :3],
        },
        "no-value-heads": {
            "v": v[:, :, :0],
            "g": g[:, :, :0],
            "beta": beta[:, :, :0],
            "initial_state": h0[:, :0],
        },
        "g-shape": {"g": g[:, :, :2]},
        "beta-shape": {"beta": beta[:, :100]},
        "initial-state-rows": {"initial_state": h0[:2]},
        "initial-state-dims": {
            "initial_state": torch.zeros(3, 4, 64, HEAD_DIM, device=device)
        },
        "cu-seqlens-2d": {"cu_seqlens": cu.to(device).view(1, -1)},
        "cu-seqlens-float": {"cu_seqlens": cu.to(device).float()},
        "cu-seqlens-one-offset": {
            "cu_seqlens": cu[:1].to(device),
            "cu_seqlens_cpu": cu[:1],
        },
        "host-offsets-count": {"cu_seqlens_cpu": torch.tensor([0, 130, 164])},
        "host-offsets-float": {"cu_seqlens_cpu": cu.double()},
        "host-offsets-start": {"cu_seqlens_cpu": torch.tensor([4, 100, 130, 164])},
        "host-offsets-decreasing": {"cu_seqlens_cpu": torch.tensor([0, 130, 100, 164])},
        "host-offsets-end": {"cu_seqlens_cpu": torch.tensor([0, 100, 130, 160])},
        "host-offsets-list-end": {"cu_seqlens_cpu": [0, 100, 130, 200]},
        "g-not-a-tensor": {"g": None},
        "beta-not-a-tensor": {"beta": 0.5},
        "k-on-meta": {"k": k.to("meta")},
        "cu-seqlens-list": {"cu_seqlens": [0, 100, 130, 164]},
    }
    return base, cases


_INVALID_IDS = (
    "q-on-host", "g-on-host", "initial-state-on-host", "cu-seqlens-on-host", "k-heads", "head-dim-64", "v-tokens",
    "value-heads-not-a-multiple", "no-value-heads", "g-shape", "beta-shape", "initial-state-rows",
    "initial-state-dims", "cu-seqlens-2d", "cu-seqlens-float", "cu-seqlens-one-offset", "host-offsets-count",
    "host-offsets-float", "host-offsets-start", "host-offsets-decreasing", "host-offsets-end", "host-offsets-list-end",
    "g-not-a-tensor", "beta-not-a-tensor", "k-on-meta", "cu-seqlens-list",
)  # fmt: skip

# The exact message of each rejected case (the checks' contract: type and text).
_OFFSETS = "cu_seqlens must be a 1-D int32 / int64 tensor of at least two offsets; got "
_HOST_OFFSETS = (
    "cu_seqlens_cpu must start at 0, not decrease and end at the 164 tokens of q; got "
)
_INVALID_MESSAGES = {
    "q-on-host": "q must be a CUDA tensor",
    "g-on-host": "g must be a CUDA tensor",
    "initial-state-on-host": "initial_state must be a CUDA tensor",
    "cu-seqlens-on-host": "cu_seqlens must be a CUDA tensor",
    "k-heads": "k must have q's shape (1, 164, 2, 128); got (1, 164, 1, 128)",
    "head-dim-64": "q must be [B, T, H, 128]; got (1, 164, 2, 64)",
    "v-tokens": "v must be [1, 164, HV, 128]; got (1, 100, 4, 128)",
    "value-heads-not-a-multiple": "the value heads (HV=3) must be a positive multiple of the q / k heads (H=2)",
    "no-value-heads": "the value heads (HV=0) must be a positive multiple of the q / k heads (H=2)",
    "g-shape": "g must be (1, 164, 4); got (1, 164, 2)",
    "beta-shape": "beta must be (1, 164, 4); got (1, 100, 4)",
    "initial-state-rows": "initial_state must be (3, 4, 128, 128) (one state per sequence); got (2, 4, 128, 128)",
    "initial-state-dims": "initial_state must be (3, 4, 128, 128) (one state per sequence); got (3, 4, 64, 128)",
    "cu-seqlens-2d": _OFFSETS + "torch.int64 (1, 4)",
    "cu-seqlens-float": _OFFSETS + "torch.float32 (4,)",
    "cu-seqlens-one-offset": _OFFSETS + "torch.int64 (1,)",
    "host-offsets-count": "cu_seqlens_cpu has 3 offsets, cu_seqlens 4",
    "host-offsets-float": "cu_seqlens_cpu must be a 1-D int32 / int64 tensor; got torch.float64 (4,)",
    "host-offsets-start": _HOST_OFFSETS + "[4, 100, 130, 164] ending at 164",
    "host-offsets-decreasing": _HOST_OFFSETS + "[0, 130, 100, 164] ending at 164",
    "host-offsets-end": _HOST_OFFSETS + "[0, 100, 130, 160] ending at 160",
    "host-offsets-list-end": _HOST_OFFSETS + "[0, 100, 130, 200] ending at 200",
    "g-not-a-tensor": "g must be a CUDA tensor",
    "beta-not-a-tensor": "beta must be a CUDA tensor",
    "k-on-meta": "k must be a CUDA tensor",
    "cu-seqlens-list": "cu_seqlens must be a CUDA tensor",
}  # fmt: skip


def _call_rejected(call) -> ValueError:
    with pytest.raises(ValueError) as error:
        cake_gdn_nondet(
            call["q"], call["k"], call["v"], call["g"], call["beta"], initial_state=call["initial_state"],
            output_final_state=True, cu_seqlens=call["cu_seqlens"], cu_seqlens_cpu=call["cu_seqlens_cpu"],
            use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
    return error.value


@pytest.mark.parametrize("case", _INVALID_IDS)
def test_rejects_invalid_inputs_before_any_launch(case, device, monkeypatch) -> None:
    base, cases = _invalid_calls(device)
    assert set(cases) == set(_INVALID_IDS) == set(_INVALID_MESSAGES)
    call = {**base, **cases[case]}
    stages = _record_launches(monkeypatch)
    assert str(_call_rejected(call)) == _INVALID_MESSAGES[case]
    assert stages == [], stages
    # the unmodified call is accepted
    o, final = cake_gdn_nondet(
        base["q"], base["k"], base["v"], base["g"], base["beta"], initial_state=base["initial_state"],
        output_final_state=True, cu_seqlens=base["cu_seqlens"], cu_seqlens_cpu=base["cu_seqlens_cpu"],
        use_qk_l2norm_in_kernel=True,
    )  # fmt: skip
    assert o.shape == base["v"].shape and final.shape == base["initial_state"].shape


@pytest.mark.parametrize("name", ["k", "beta", "initial_state", "cu_seqlens"])
def test_rejects_inputs_on_another_device(name, device, monkeypatch) -> None:
    # every tensor must be on q's device: one input moved to a second device raises before the first launch
    if torch.cuda.device_count() < 2:
        pytest.skip("a second CUDA device required")
    other = torch.device("cuda", (device.index + 1) % torch.cuda.device_count())
    base, _ = _invalid_calls(device)
    call = {**base, name: base[name].to(other)}
    stages = _record_launches(monkeypatch)
    assert str(_call_rejected(call)) == f"{name} is on {other}, q on {device}"
    assert stages == [], stages


def test_device_route_does_not_synchronize(device) -> None:
    # the checks read tensor metadata only: a device-route forward and backward runs under the error sync-debug mode
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 4), torch.bfloat16, device, seed=31)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int32, device=device)
    h0 = torch.zeros(
        len(seq_lens), 4, HEAD_DIM, HEAD_DIM, device=device, requires_grad=True
    )

    def call():
        leaves = {
            n: inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        }
        o, final = cake_gdn_nondet(
            leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"], initial_state=h0,
            output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        ((o.float() * inputs["do"].float()).sum() + final.sum()).backward()

    call()  # loads the libraries and fills the caches
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        call()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


# --------------------------------------------------------------------------- the names and the mode (no device)

# The public names of the package: the package root's ``__all__`` (the facade's are tested in
# ``test_cake_gdn_nondet_train_api.py``).  Every one carries the mode word ``nondet`` as a word of its own (see
# ``_name_tokens``: ``det`` is a substring of ``nondet``, so a substring check cannot tell the modes apart).
PACKAGE_NAMES = ("CakeGDNNondetFunction", "cake_gdn_nondet", "cake_gdn_nondet_train_backward",
                 "cake_gdn_nondet_train_forward")  # fmt: skip
# The package's entries and, in the same order, the deterministic package's entries (its det names).
ENTRY_NAMES = ("cake_gdn_nondet", "CakeGDNNondetFunction", "cake_gdn_nondet_train_forward",
               "cake_gdn_nondet_train_backward")  # fmt: skip
DET_NAMES = (
    "cake_gdn_det",
    "CakeGDNDetFunction",
    "cake_gdn_det_train_forward",
    "cake_gdn_det_train_backward",
)
# The names before the rename (FlashInfer #8928): none of them resolves any more.
OLD_NAMES = (
    "chunk_gated_delta_rule",
    "ChunkGatedDeltaRuleFunction",
    "gdn_chunk_forward",
    "gdn_chunk_backward",
)
_ALERT = "does not have a deterministic implementation"
# One word of a name part: an acronym run before a capitalized word, a capitalized or lower-case word, the rest of an
# acronym run, digits.
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _name_tokens(name: str) -> list[str]:
    """The lower-cased words of ``name``, split on ``_`` and on CamelCase boundaries, an acronym run as one word
    (``CakeGDNNondetFunction``: ``cake``, ``gdn``, ``nondet``, ``function``)."""
    return [word.lower() for part in name.split("_") for word in _WORD.findall(part)]


def test_the_public_names_are_the_nondet_names() -> None:
    # the split itself: an acronym run is one word, and ``det`` is a word of a det name only
    for name, words in (
        ("CakeGDNNondetFunction", "cake gdn nondet function"),
        ("cake_gdn_nondet_train_forward", "cake gdn nondet train forward"),
        ("CakeGDNDetFunction", "cake gdn det function"),
    ):
        assert _name_tokens(name) == words.split(), name
    assert sorted(cake_gdn_nondet_train.__all__) == sorted(PACKAGE_NAMES)
    for name in cake_gdn_nondet_train.__all__:
        tokens = _name_tokens(name)
        assert "nondet" in tokens and "det" not in tokens, (name, tokens)
        assert getattr(cake_gdn_nondet_train, name) is not None, name
    # one object per entry: the package root re-exports cake_chain's
    for name in PACKAGE_NAMES:
        assert name in cake_chain.__all__, name
        assert getattr(cake_gdn_nondet_train, name) is getattr(cake_chain, name), name
    assert cake_gdn_nondet.__name__ == "cake_gdn_nondet"
    cls = cake_chain.CakeGDNNondetFunction
    assert cls.__name__ == cls.__qualname__ == "CakeGDNNondetFunction"
    assert cls.__module__ == cake_chain.__name__
    # the package path is the module name; the library identity is unchanged
    assert cake_jit.PACKAGE == "flashinfer.experimental.cake.cake_gdn_nondet_train"
    assert cake_jit.LIBRARY_PREFIX == "cake_gdn_train"


def test_the_old_names_are_gone_and_the_det_names_are_the_det_package() -> None:
    for module in (cake_gdn_nondet_train, cake_chain):
        exported = set(getattr(module, "__all__", ()))
        for name in OLD_NAMES + DET_NAMES:
            assert not hasattr(module, name), (module.__name__, name)
            assert name not in exported, (module.__name__, name)
    # the deterministic package's entries (its det names) are distinct objects from this package's entries
    for name, det_name in zip(ENTRY_NAMES, DET_NAMES, strict=True):
        assert getattr(cake_gdn_nondet_train, name) is not getattr(
            cake_gdn_det_train, det_name
        ), name


def _callables(module) -> dict:
    """The callables of ``module.__all__``, an autograd Function by its ``forward`` and ``backward`` (and by its
    ``apply`` when that signature names its parameters)."""
    var_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    found = {}
    for name in module.__all__:
        obj = getattr(module, name)
        if isinstance(obj, type):
            found[f"{name}.forward"] = obj.forward
            found[f"{name}.backward"] = obj.backward
            # ``torch.autograd.Function.apply`` takes ``*args, **kwargs``: its signature binds any keyword, so the
            # check reads ``forward``, the parameters ``apply`` passes on, and ``apply`` only if it names them.  Some
            # torch builds expose ``apply`` as a builtin without a signature (``ValueError``): it names nothing then
            try:
                params = list(inspect.signature(obj.apply).parameters.values())
            except ValueError:
                params = None
            if params is not None and all(p.kind not in var_kinds for p in params):
                found[f"{name}.apply"] = obj.apply
        else:
            found[name] = obj
    return found


def test_no_entry_takes_a_deterministic_parameter() -> None:
    # every callable of the package root's ``__all__``: binding the keyword fails at the signature
    callables = _callables(cake_gdn_nondet_train)
    assert {"cake_gdn_nondet", "CakeGDNNondetFunction.forward", "cake_gdn_nondet_train_forward",
            "cake_gdn_nondet_train_backward"} <= set(callables)  # fmt: skip
    for name, fn in callables.items():
        try:
            inspect.signature(fn).bind_partial(deterministic=False)
        except TypeError as exc:
            assert "deterministic" in str(exc), (name, exc)
        else:
            pytest.fail(f"{name} binds deterministic=")


def test_a_deterministic_keyword_is_a_type_error() -> None:
    # host operands: the call fails at its signature, before any check or launch
    q = torch.zeros(1, 64, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros(1, 64, 2, HEAD_DIM, dtype=torch.bfloat16)
    g, beta = torch.zeros(1, 64, 2), torch.zeros(1, 64, 2)
    for entry in (cake_gdn_nondet_train.cake_gdn_nondet, cake_chain.cake_gdn_nondet):
        for value in (True, False):
            with pytest.raises(TypeError, match="deterministic"):
                entry(q, q, v, g, beta, deterministic=value)
    with pytest.raises(TypeError, match="deterministic"):
        cake_chain.cake_gdn_nondet_train_forward(
            q, q, v, g, beta, meta=None, arch=None, deterministic=False
        )
    with pytest.raises(TypeError, match="deterministic"):
        cake_chain.cake_gdn_nondet_train_backward({}, None, deterministic=False)
    # the Function takes its twelve inputs only: a thirteenth (the former mode argument) is refused
    options = cake_chain.make_options()
    apply_args = (q, q, v, g, beta, 1.0, None, False, None, None, True, options)
    with pytest.raises(TypeError):
        cake_chain.CakeGDNNondetFunction.apply(*apply_args, False)


def _entry_calls() -> dict:
    """One call per forward entry on host operands: each fails at its operand checks."""
    q = torch.zeros(1, 64, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros(1, 64, 2, HEAD_DIM, dtype=torch.bfloat16)
    g, beta = torch.zeros(1, 64, 2), torch.zeros(1, 64, 2)
    options = cake_chain.make_options()
    apply_args = (q, q, v, g, beta, 1.0, None, False, None, None, True, options)
    pkg, forward = cake_gdn_nondet_train, cake_chain.cake_gdn_nondet_train_forward
    return {
        "cake_gdn_nondet": lambda: pkg.cake_gdn_nondet(q, q, v, g, beta),
        "CakeGDNNondetFunction": lambda: pkg.CakeGDNNondetFunction.apply(*apply_args),
        "cake_gdn_nondet_train_forward": lambda: forward(
            q, q, v, g, beta, meta=None, arch=None
        ),
    }


@contextlib.contextmanager
def _deterministic_algorithms(mode: bool, *, warn_only: bool = False):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def test_torch_deterministic_algorithms_do_not_alert_the_forward_entries() -> None:
    # the forward entries run fixed-order programs: under PyTorch's deterministic algorithms (an alert as an error,
    # a warning as an error) each proceeds to its operand checks, which refuse the host operands here
    for warn_only in (False, True):
        with (
            _deterministic_algorithms(True, warn_only=warn_only),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            for name, call in _entry_calls().items():
                with pytest.raises(Exception) as info:  # noqa: PT011 - any operand refusal, never the alert
                    call()
                assert _ALERT not in str(info.value), (name, warn_only, info.value)


@pytest.mark.parametrize(
    "tape", [{}, {"mode": "det"}, {"mode": None}], ids=["det-tape", "other", "none"]
)
def test_backward_refuses_a_tape_without_the_nondet_tag(tape) -> None:
    with pytest.raises(ValueError, match="this tape does not carry that tag"):
        cake_chain.cake_gdn_nondet_train_backward({"arch": ARCHES[0], **tape}, None)


# --------------------------------------------------------------------------- the non-deterministic contract (device)


def test_low_level_entries_round_trip_and_refuse_a_det_tape(device) -> None:
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=263)
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    det_chain = cake_package("cake_gdn_det_train", "cake_chain")
    q, k, v, g, beta, do = (inputs[n] for n in ("q", "k", "v", "g", "beta", "do"))

    def meta_of(chain):
        plans = chain.scan_plans(
            sum(seq_lens), len(seq_lens), heads[1], index, seq_lens
        )
        return chain.build_chunk_meta(seq_lens, device, scan_plans=plans)

    tape = cake_chain.cake_gdn_nondet_train_forward(
        q, k, v, g, beta, meta=meta_of(cake_chain), arch=arch
    )
    assert tape["mode"] == "nondet"
    grads = cake_chain.cake_gdn_nondet_train_backward(tape, do)
    assert all(bool(torch.isfinite(grads[n]).all()) for n in GRAD_NAMES)
    det_tape = det_chain.cake_gdn_det_train_forward(
        q, k, v, g, beta, meta=meta_of(det_chain), arch=arch
    )
    assert det_tape.get("mode") != "nondet"
    with pytest.raises(ValueError, match="this tape does not carry that tag"):
        cake_chain.cake_gdn_nondet_train_backward(det_tape, do)


# (H, HV) and tail pins: every tail the rule dispatches or a pin forces
FIXED_ORDER_CASES = (
    ((2, 4), {}),
    ((2, 6), {}),
    ((1, 3), {}),
    ((1, 4), {}),
    ((2, 2), {}),
    ((2, 2), {"tail": "tail"}),
    ((2, 6), {"tail": "tail"}),
    ((2, 4), {"tail": "tail_g2"}),
    ((2, 6), {"tail": "tail_g3"}),
    ((2, 6), {"tail": "unfused"}),
    ((2, 4), {"tail": "tail_g2_f32x"}),
    ((1, 3), {"tail": "tail_g3_f32x"}),
)


@pytest.mark.parametrize(
    "heads, pins",
    FIXED_ORDER_CASES,
    ids=["rule-g2", "rule-g3", "rule-g3-h1", "rule-g4", "rule-g1", "g1-tail", "g3-tail", "g2-tail-g2", "g3-tail-g3",
         "g3-unfused", "g2-f32x", "g3-h1-f32x"],
)  # fmt: skip
def test_torch_deterministic_algorithms_change_nothing(
    heads, pins, device, monkeypatch
) -> None:
    # every launch of the package is fixed-order: under PyTorch's deterministic algorithms (an alert as an error,
    # or as a warning) the autograd route and the low-level route run without an alert, to the same bits
    seq_lens = (300, 77)
    pins = dict(pins)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=251)
    want = _run(inputs, seq_lens, **pins)
    for warn_only in (False, True):
        with (
            _deterministic_algorithms(True, warn_only=warn_only),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            got = _run(inputs, seq_lens, **pins)
        assert not [w for w in caught if _ALERT in str(w.message)], warn_only
        _assert_same_bits(got, want, OUTPUT_NAMES)
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    options = cake_chain.make_options(**pins)
    plans = cake_chain.scan_plans(
        sum(seq_lens), len(seq_lens), heads[1], index, seq_lens
    )
    meta = cake_chain.build_chunk_meta(seq_lens, device, scan_plans=plans)
    q, k, v, g, beta, do = (inputs[n] for n in ("q", "k", "v", "g", "beta", "do"))
    with _deterministic_algorithms(True):
        tape = cake_chain.cake_gdn_nondet_train_forward(q, k, v, g, beta, meta=meta, arch=arch,
                                                        options=options)  # fmt: skip
        cake_chain.cake_gdn_nondet_train_backward(tape, do)


def test_launches_are_this_packages_records(device, monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the deterministic package launched a kernel")

    monkeypatch.setattr(det_backend, "launch_stage", refuse)
    stages = _record_launches(monkeypatch)
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    seq_lens = (300, 77, 640)
    for heads in ((2, 4), (2, 6), (1, 1)):
        stages.clear()
        _run(_inputs(seq_lens, heads, torch.bfloat16, device, seed=257), seq_lens)
        libraries = {
            cake_jit.MODULES[cake_backend.stage_record(stage, arch)]["library"]
            for stage in stages
        }
        assert stages and all(lib.startswith("cake_gdn_train_") for lib in libraries), (
            heads,
            libraries,
        )


TAIL_STAGES = ("tail", "tail_bf16")
DET_TAIL_STAGES = (
    "tail_g2",
    "tail_g2_bf16",
    "tail_g3",
    "tail_g3_bf16",
    "dqkwg",
    "wy_bwd",
    "wy_bwd_bf16",
    "finalize",
    "finalize_mlp",
)
# the cluster tails that sum FP32 partials (a value-head group of two / three)
F32X_TAIL_STAGES = (
    "tail_g2_f32x",
    "tail_g2_f32x_bf16",
    "tail_g3_f32x",
    "tail_g3_f32x_bf16",
)
TAIL_LAYOUT = (300, 77, 640)  # 5 + 2 + 10 chunks


def _bits_zero(t: torch.Tensor) -> bool:
    return not bool(t.view(torch.int32).any())


def _assert_same_bits(got: dict, want: dict, names) -> None:
    for name in names:
        assert torch.equal(got[name], want[name]), name


def _assert_tail_counters_zero() -> None:
    """Every pooled tail counter buffer of the package is zero, bit for bit (the workspace slots are written before
    they are read)."""
    torch.cuda.synchronize()
    for key, buffers in cake_chain._POOLED.zeroed.items():
        for role, t in buffers.items():
            if role != "tail_ws":
                assert _bits_zero(t), (key, role)


def _recorded_tail_counters(monkeypatch, workspaces) -> list[tuple[str, torch.Tensor]]:
    """The (role, buffer) of every self-cleaning tail counter buffer that ``workspaces`` (the pooled ones of eager
    calls, the fresh ones of a CUDA graph capture) hands out from here on (the ``tail_ws`` slots are written before
    they are read)."""
    counters: list[tuple[str, torch.Tensor]] = []
    take_zeroed = workspaces.take_zeroed

    def recording(device_index, role, rows, row_shape, dtype, device):
        buffer = take_zeroed(device_index, role, rows, row_shape, dtype, device)
        if role != "tail_ws":
            counters.append((role, buffer))
        return buffer

    monkeypatch.setattr(workspaces, "take_zeroed", recording)
    return counters


def _record_tail_launches(monkeypatch, device) -> list[tuple[str, tuple]]:
    """The (stage, grid) of every launch from here on; after each launch of the package tail, the counters it was
    given are zero (the launch's self-cleaning contract) and its workspace holds a slot per value head of its tiles."""
    launches: list[tuple[str, tuple]] = []
    launch = cake_backend.launch_stage
    arch = cake_backend.device_arch(cake_chain._device_index(device))

    def checking(stage, grid, values, **kwargs):
        out = launch(stage, grid, values, **kwargs)
        launches.append((stage, tuple(grid)))
        if stage in TAIL_STAGES:
            names = cake_backend.parameters(cake_backend.stage_record(stage, arch))
            ws, count = values[names.index("ws")], values[names.index("ws_count")]
            group = (
                values[names.index("num_v_heads")] // values[names.index("num_heads")]
            )
            torch.cuda.synchronize()
            assert _bits_zero(count), (stage, grid)
            assert (
                ws.shape[1:] == (cake_chain.TAIL_TILE_FLOATS,)
                and count.dtype == torch.uint32
            )
            assert values[names.index("ws_tiles")] <= count.shape[0]
            assert ws.shape[0] == count.shape[0] * group, (
                stage,
                ws.shape,
                count.shape,
            )
        return out

    monkeypatch.setattr(cake_backend, "launch_stage", checking)
    return launches


def _num_chunks(seq_lens) -> int:
    return sum(-(-n // cake_chain.CHUNK) for n in seq_lens)


@pytest.mark.parametrize(
    "beta_dtype", [torch.float32, torch.bfloat16], ids=["fp32beta", "bf16beta"]
)
@pytest.mark.parametrize(
    "heads",
    [(2, 2), (2, 4), (2, 6), (1, 4), (4, 8)],
    ids=["g1", "g2", "g3", "g4", "h4-g2"],
)
def test_the_tail_serves_every_group_and_leaves_its_counters_zero(
    heads, beta_dtype, device, monkeypatch
) -> None:
    form = "tail"
    seq_lens = TAIL_LAYOUT
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=271)
    launches = _record_tail_launches(monkeypatch, device)
    for host_cache in (True, False):
        launches.clear()
        got = _run(inputs, seq_lens, host_cache=host_cache, tail=form)
        stages = [stage for stage, _grid in launches]
        tail = form + ("_bf16" if beta_dtype == torch.bfloat16 else "")
        assert tail in stages and not (
            set(DET_TAIL_STAGES) | set(F32X_TAIL_STAGES) | set(TAIL_STAGES) - {tail}
        ) & set(stages), stages
        grids = [grid for stage, grid in launches if stage == tail]
        assert sum(grid[0] for grid in grids) == _num_chunks(seq_lens) * heads[1], grids
        _assert_close_to_reference(got, _reference(inputs, seq_lens), OUTPUT_NAMES)
    _assert_tail_counters_zero()


@pytest.mark.parametrize("slots", [1, 5, 16], ids=["slots1", "slots5", "slots16"])
def test_segmented_tail_launches_match_one_launch(slots, device, monkeypatch) -> None:
    # a workspace of `slots` slots: the chunk table in launches of max(1, slots // HV) chunks, in stream order, to the
    # same bits (each group sums its slots in value-head order whatever the segments)
    seq_lens, heads, form = TAIL_LAYOUT, (2, 6), "tail"
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=277)
    whole = _run(inputs, seq_lens, tail=form)
    monkeypatch.setattr(cake_chain, "TAIL_WORKSPACE_TILES", slots)
    launches = _record_tail_launches(monkeypatch, device)
    got = _run(inputs, seq_lens, tail=form)
    ranges, ws_tiles = cake_chain.tail_segments(_num_chunks(seq_lens), *heads)
    assert ws_tiles == max(1, slots // heads[1]) * heads[0] and len(ranges) > 1
    assert [grid for stage, grid in launches if stage == form + "_bf16"] == [
        ((b - a) * heads[1], 1, 1) for a, b in ranges
    ]
    _assert_same_bits(got, whole, OUTPUT_NAMES)
    _assert_close_to_reference(got, _reference(inputs, seq_lens), OUTPUT_NAMES)
    _assert_tail_counters_zero()


# (H, HV) -> the tail the rule dispatches on a large call, per architecture: the cluster tails that round dq / dk once,
# tail_g2_f32x at G = 2 and tail_g3_f32x at G = 3, else the package tail
TAIL_RULE = {
    arch: {
        (2, 4): "tail_g2_f32x",
        (8, 16): "tail_g2_f32x",
        (16, 32): "tail_g2_f32x",
        (1, 3): "tail_g3_f32x",
        (2, 6): "tail_g3_f32x",
        (16, 48): "tail_g3_f32x",
        (2, 2): "tail",
        (16, 16): "tail",
        (1, 4): "tail",
        (4, 16): "tail",
    }
    for arch in ("sm_100a", "sm_103a", "sm_107a")
}
# the value-head groups without a cluster tail (G = 1 and G >= 4) and the sizes in chunks (None: a large call) the
# rule is checked at
NO_CLUSTER_HEADS = ((4, 4), (2, 2), (1, 1), (1, 4), (2, 16), (1, 8))
RULE_CHUNKS = (None, 1, 2, 7, 255, 256, 1024, 4096)


def test_the_rule_rounds_dq_dk_once_at_every_size() -> None:
    # the rule takes no fixed-order tail (those round each value head's dq / dk partials before their sum): the
    # cluster tail of G = 2 / 3 and the package tail of every other group, at every size
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for h, hv in NO_CLUSTER_HEADS:
            for chunks in RULE_CHUNKS:
                assert (
                    cake_chain.tail_form(h, hv, arch=arch, num_chunks=chunks)
                    == cake_chain.TAIL_DEFAULT
                ), (arch, h, hv, chunks)
        for (h, hv), form in (
            ((2, 4), "tail_g2_f32x"),
            ((1, 3), "tail_g3_f32x"),
            ((2, 6), "tail_g3_f32x"),
        ):
            for chunks in RULE_CHUNKS:
                assert cake_chain.tail_form(h, hv, arch=arch, num_chunks=chunks) == form, (arch, h, hv, chunks)  # fmt: skip


# the stages a backward launches per tail (beyond the shared scans), and the ones it never launches
_TAIL_SETS = {
    f: {f, f + "_bf16"}
    for f in ("tail", "tail_g2", "tail_g3", "tail_g2_f32x", "tail_g3_f32x")
}
_TAIL_SETS["unfused"] = {"dqkwg", "finalize", "finalize_mlp"}
TAIL_LAUNCHES = {
    form: (want, set().union(*(other for f, other in _TAIL_SETS.items() if f != form)))
    for form, want in _TAIL_SETS.items()
}


@pytest.mark.parametrize(
    "heads",
    [(2, 4), (1, 3), (2, 6), (2, 2), (1, 4)],
    ids=["g2", "g3-h1", "g3", "g1", "g4"],
)
def test_the_dispatched_tail_runs(heads, device, monkeypatch) -> None:
    # without a pin the backward launches the tail the rule names, and only that one (the package tail for G = 1 and
    # G = 4 on this small layout); the pins launch theirs
    seq_lens = TAIL_LAYOUT
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=289)
    launches = _record_tail_launches(monkeypatch, device)
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    group = heads[1] // heads[0]
    if group not in (2, 3):
        assert (
            cake_chain.tail_form(*heads, arch=arch, num_chunks=_num_chunks(seq_lens))
            == "tail"
        )
    forms = [None, "tail", "unfused"] + (
        [f"tail_g{group}", f"tail_g{group}_f32x"] if group in (2, 3) else []
    )
    for form in forms:
        launches.clear()
        got = _run(inputs, seq_lens, **({} if form is None else {"tail": form}))
        stages = {stage for stage, _grid in launches}
        want, never = TAIL_LAUNCHES[
            form
            or cake_chain.tail_form(*heads, arch=arch, num_chunks=_num_chunks(seq_lens))
        ]
        assert stages & want and not stages & never, (form, sorted(stages))
        _assert_close_to_reference(got, _reference(inputs, seq_lens), OUTPUT_NAMES)


# small calls of the groups without a cluster tail (G = 1 and G = 4), one sequence of one chunk and a single token
# among them
SMALL_TAIL_CASES = {
    "g1": ((2, 2), (300, 77)),
    "g1-h4": ((4, 4), (64,)),
    "g4": ((1, 4), (300, 77)),
    "g4-h2": ((2, 8), (130, 1, 64)),
}


@pytest.mark.parametrize("case", sorted(SMALL_TAIL_CASES))
def test_small_calls_round_dq_dk_once(case, device, monkeypatch) -> None:
    # without a pin a small call of a group without a cluster tail runs the package tail (the group's FP32 dq / dk
    # partials summed in value-head order and rounded once), not the fixed-order unfused tail: the pinned package
    # tail's bits, o / dv / dg / dbeta of the unfused pin, within the accuracy gate of the FP64 recurrence
    heads, seq_lens = SMALL_TAIL_CASES[case]
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    assert (
        cake_chain.tail_form(*heads, arch=arch, num_chunks=_num_chunks(seq_lens))
        == "tail"
    )
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=293)
    launches = _record_tail_launches(monkeypatch, device)
    got = _run(inputs, seq_lens)
    stages = {stage for stage, _grid in launches}
    want, never = TAIL_LAUNCHES["tail"]
    assert stages & want and not stages & never, sorted(stages)
    _assert_same_bits(_run(inputs, seq_lens, tail="tail"), got, OUTPUT_NAMES)
    unfused = _run(inputs, seq_lens, tail="unfused")
    for name in ("o", "dv", "dg", "dbeta"):
        assert torch.equal(got[name], unfused[name]), name
    _assert_close_to_reference(got, _reference(inputs, seq_lens), OUTPUT_NAMES)


def test_tail_segments_rule(monkeypatch) -> None:
    rule = cake_chain.tail_segments
    assert rule(17, 2, 2) == (((0, 17),), 0)  # a group of one: one launch, no workspace
    assert rule(17, 2, 4, 1024) == (((0, 17),), 34)
    assert rule(17, 2, 6, 30) == (((0, 5), (5, 10), (10, 15), (15, 17)), 10)
    assert rule(3, 16, 48, 8) == (
        ((0, 1), (1, 2), (2, 3)),
        16,
    )  # at least one chunk per launch
    assert rule(0, 2, 4) == ((), 0)
    # a tile is the group's G slots: the slots bound G x the tiles
    assert rule(17, 2, 6, 12) == (
        tuple((c, min(c + 2, 17)) for c in range(0, 17, 2)),
        4,
    )
    # the dispatch rule: the fastest tail measured per architecture, value-head group and size (B200 and GB300);
    # large calls here
    for arch, want in TAIL_RULE.items():
        assert {
            heads: cake_chain.tail_form(*heads, arch=arch) for heads in want
        } == want, arch
        for h, hv in ((2, 2), (4, 16), (1, 4)):
            for chunks in (1, 2, 4096):
                assert (
                    cake_chain.tail_form(h, hv, arch=arch, num_chunks=chunks)
                    == cake_chain.TAIL_DEFAULT
                ), (arch, h, hv, chunks)
        # a cluster tail serves its group at any size
        assert (
            cake_chain.tail_form(2, 4, arch=arch, num_chunks=1)
            == cake_chain.tail_form(2, 4, arch=arch)
            == "tail_g2_f32x"
        )
        assert (
            cake_chain.tail_form(2, 6, arch=arch, num_chunks=1)
            == cake_chain.tail_form(2, 6, arch=arch)
            == "tail_g3_f32x"
        )
    # an architecture without a cluster tail for a group of three: the package tail at every size
    monkeypatch.setitem(cake_chain.CLUSTER_TAILS, "sm_107a", {2: "tail_g2"})
    for chunks in (1, None):
        assert (
            cake_chain.tail_form(2, 6, arch="sm_107a", num_chunks=chunks)
            == cake_chain.TAIL_DEFAULT
        )
    assert (
        cake_chain.tail_form(2, 4, "tail_g2", arch="sm_100a") == "tail_g2"
        and cake_chain.tail_form(2, 6, "unfused", arch="sm_103a") == "unfused"
        and cake_chain.tail_form(2, 6, "tail_g3", arch="sm_100a") == "tail_g3"
        and cake_chain.tail_form(2, 4, "tail_g2_f32x", arch="sm_107a") == "tail_g2_f32x"
        and cake_chain.tail_form(1, 3, "tail_g3_f32x", arch="sm_103a") == "tail_g3_f32x"
    )
    with pytest.raises(
        ValueError, match="tail_g2 form serves a value-head group of two"
    ):
        cake_chain.tail_form(2, 6, "tail_g2", arch="sm_103a")
    with pytest.raises(
        ValueError, match="tail_g3 form serves a value-head group of three"
    ):
        cake_chain.tail_form(2, 4, "tail_g3", arch="sm_103a")
    with pytest.raises(
        ValueError, match="tail_g2_f32x form serves a value-head group of two"
    ):
        cake_chain.tail_form(2, 6, "tail_g2_f32x", arch="sm_100a")
    with pytest.raises(
        ValueError, match="tail_g3_f32x form serves a value-head group of three"
    ):
        cake_chain.tail_form(2, 4, "tail_g3_f32x", arch="sm_100a")
    with pytest.raises(ValueError, match="tail must be one of"):
        cake_chain.make_options(tail="tail_g4")
    for gone in ("tail_bulk", "tail_ordered"):
        with pytest.raises(ValueError, match="tail must be one of"):
            cake_chain.make_options(tail=gone)


@pytest.mark.parametrize(
    "heads", [(2, 4), (2, 6), (2, 2), (1, 4)], ids=["g2", "g3", "g1", "g4"]
)
def test_the_tail_keeps_the_deterministic_bits_but_dq_dk(
    heads, device, monkeypatch
) -> None:
    # o, dv, dg and dbeta have one writer each: the same bits as the deterministic package and as this package's
    # fixed-order tails; dq / dk round once after the FP32 group sum in value-head order, within the budgets of both
    # and the same bits run to run (the planner's tables, as the deterministic package plans them); the cluster tail
    # of G = 2 / 3 gives the package tail's bits.  The adjoint rounds dv2 as the deterministic package does
    # (DVL32_ARCHS empty; the FP32 dv_local adjoint: test_the_tails_on_the_fp32_dv_local_adjoint_stay_within_the_budgets)
    # and both packages run the prep's forward (prep, wy_tc2, fwd_ho; fwd_a_route's forward:
    # test_fwd_a_within_the_budgets_of_the_prep_forward)
    _prep_forward_on_both(monkeypatch)
    monkeypatch.setattr(cake_chain, "DVL32_ARCHS", ())
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    seq_lens = TAIL_LAYOUT
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=281)
    got = _run(inputs, seq_lens, tail="tail")
    _assert_same_bits(_run(inputs, seq_lens, tail="tail"), got, OUTPUT_NAMES)
    det = _run(inputs, seq_lens, det=True)
    pinned = [_run(inputs, seq_lens, tail="unfused")]
    group = heads[1] // heads[0]
    if group in (2, 3):
        pinned.append(_run(inputs, seq_lens, tail=f"tail_g{group}"))
        _assert_same_bits(
            _run(inputs, seq_lens, tail=f"tail_g{group}_f32x"), got, OUTPUT_NAMES
        )
    for name in ("o", "dv", "dg", "dbeta"):
        assert torch.equal(got[name], det[name]), name
        assert all(torch.equal(got[name], run[name]) for run in pinned), name
    for run in pinned:  # the fixed-order tails keep the deterministic package's bits
        assert all(torch.equal(run[name], det[name]) for name in OUTPUT_NAMES)
    _assert_within_budget(got, det, ("dq", "dk"))


@pytest.mark.parametrize(
    "heads", [(2, 4), (2, 6), (2, 2), (1, 4)], ids=["g2", "g3", "g1", "g4"]
)
def test_the_tails_on_the_fp32_dv_local_adjoint_stay_within_the_budgets(
    heads, device, monkeypatch
) -> None:
    # with the FP32 dv_local adjoint (dhu_tc2_dvl32 rounds dv2 once) the tail forms still share o, dv, dg and dbeta,
    # the package tail and the cluster tail of G = 2 / 3 give the same bits (run to run too), and every output stays
    # within the budgets of the deterministic package
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if arch not in cake_chain.DVL32_ARCHS:
        pytest.skip(f"{arch} runs the adjoint without the FP32 dv_local values")
    _prep_forward_on_both(monkeypatch)
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    seq_lens = TAIL_LAYOUT
    group = heads[1] // heads[0]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=281)
    got = _run(inputs, seq_lens, tail="tail")
    _assert_same_bits(_run(inputs, seq_lens, tail="tail"), got, OUTPUT_NAMES)
    pinned = [_run(inputs, seq_lens, tail="unfused")]
    if group in (2, 3):
        pinned.append(_run(inputs, seq_lens, tail=f"tail_g{group}"))
        _assert_same_bits(
            _run(inputs, seq_lens, tail=f"tail_g{group}_f32x"), got, OUTPUT_NAMES
        )
    for name in ("o", "dv", "dg", "dbeta"):
        assert all(torch.equal(got[name], run[name]) for run in pinned), name
    _assert_within_budget(got, _run(inputs, seq_lens, det=True), OUTPUT_NAMES)


@pytest.mark.parametrize("poison", ["counter", "workspace"])
def test_a_poisoned_tail_counter_breaks_dq_dk_a_poisoned_slot_does_not(
    poison, device
) -> None:
    # protocol control: the tail relies on zero counters (created zero, left zero) and writes every workspace slot
    # before the group reads it.  Counters left at G: no CTA of a group reads G - 1, so no group finalizes (its dq /
    # dk rows keep the NaN of the poisoned allocator), dq / dk leave the budgets and the counters stay nonzero.  Slots
    # left at 64: every CTA overwrites its slot, so all outputs keep their bits.
    seq_lens, heads = TAIL_LAYOUT, (2, 4)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=283)
    cake_chain._POOLED.zeroed.clear()
    try:
        good = _run(inputs, seq_lens, streams=1, tail="tail")
        key = (
            cake_chain._device_index(device),
            torch.cuda.current_stream(device).cuda_stream,
        )
        buffers = cake_chain._POOLED.zeroed[key]
        if poison == "counter":
            buffers["tail_count"].view(torch.int32).fill_(heads[1] // heads[0])
        else:
            buffers["tail_ws"].fill_(64.0)
        _poison_workspaces(device)
        bad = _run(inputs, seq_lens, streams=1, tail="tail")
        if poison == "counter":
            torch.cuda.synchronize()
            # the counters stay nonzero: the check after every launch would catch it
            assert not _bits_zero(buffers["tail_count"])
            with pytest.raises(AssertionError):
                _assert_within_budget(bad, good, ("dq", "dk"))
            _assert_same_bits(bad, good, ("o", "dv", "dg", "dbeta"))
        else:
            _assert_same_bits(bad, good, OUTPUT_NAMES)
    finally:
        cake_chain._POOLED.zeroed.clear()  # the next call creates zeroed buffers again
    _assert_within_budget(
        _run(inputs, seq_lens, streams=1, tail="tail"), good, OUTPUT_NAMES
    )


@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
@pytest.mark.parametrize("heads", [(2, 4), (2, 6)], ids=["g2", "g3"])
def test_split_pass2_on_dhu_tc2_matches_the_register_pass2(
    heads, beta_dtype, device, monkeypatch
) -> None:
    # a split (64, 1) plan's pass 2 on dhu_tc2 from the stitched segment states against the register pass 2 at the
    # same segment table: the same bits on every output, with an initial state and a final-state gradient.  The
    # adjoint rounds dv2 as the register pass 2 does (DVL32_ARCHS empty; the FP32 dv_local adjoint:
    # test_split_pass2_on_dhu_tc2_dvl32_within_the_budgets_of_the_register_pass2)
    monkeypatch.setattr(cake_chain, "DVL32_ARCHS", ())
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=83)
    gen = torch.Generator().manual_seed(89)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    launches = _record_streams(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    runs = {}
    for scan in ("tc", "reg"):
        start = len(launches)
        runs[scan] = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, scan_config=(64, 1),
                          segment_chunks=6, dhu_scan=scan)  # fmt: skip
        stages = [stage for stage, _stream in launches[start:]]
        assert "stitch" in stages, stages
        # the tc form: the fused pre-scan computes the dv_local tiles it reads, no dv_local stage
        want = (
            ("dhu_pre_dvl_r64k1", "dhu_tc2")
            if scan == "tc"
            else ("dv_local", "dhu_pre_r64k1", "dhu_r64k1")
        )
        other = (
            ("dv_local", "dhu_pre_r64k1", "dhu_r64k1")
            if scan == "tc"
            else ("dhu_pre_dvl_r64k1", "dhu_tc2")
        )
        assert all(x in stages for x in want) and not any(x in stages for x in other), (
            scan,
            stages,
        )
    _assert_same_bits(runs["tc"], runs["reg"], names)
    _assert_close_to_reference(
        runs["tc"], _reference(inputs, seq_lens, initial_state=h0, dht=dht), names
    )


@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
@pytest.mark.parametrize("heads", [(2, 4), (2, 6)], ids=["g2", "g3"])
def test_split_pass2_on_dhu_tc2_dvl32_within_the_budgets_of_the_register_pass2(
    heads, beta_dtype, device, monkeypatch
) -> None:
    # the same plan with the FP32 dv_local adjoint (dhu_tc2_dvl32 rounds dv2 once): o and the final state keep the
    # register pass 2's bits; every output within the budgets of the register pass 2 and of the FP64 recurrence.  The
    # pre-scan keeps its BF16 dv_local tile (dhu_pre_dvl_r64k1)
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if arch not in cake_chain.DVL32_ARCHS:
        pytest.skip(f"{arch} runs the adjoint without the FP32 dv_local values")
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=83)
    gen = torch.Generator().manual_seed(89)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    launches = _record_streams(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    assert cake_chain.dhu_tc2_stage(arch) == "dhu_tc2_dvl32"
    assert cake_chain.sdt_prescan(arch) == cake_chain.SDT_PRESCAN
    tc_stages = (cake_chain.SDT_PRESCAN, "dhu_tc2_dvl32")
    reg_stages = ("dv_local", "dhu_pre_r64k1", "dhu_r64k1")
    runs = {}
    for scan in ("tc", "reg"):
        start = len(launches)
        runs[scan] = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, scan_config=(64, 1),
                          segment_chunks=6, dhu_scan=scan)  # fmt: skip
        stages = [stage for stage, _stream in launches[start:]]
        assert "stitch" in stages, stages
        want, other = (
            (tc_stages, reg_stages) if scan == "tc" else (reg_stages, tc_stages)
        )
        assert all(x in stages for x in want) and not any(x in stages for x in other), (scan, stages)  # fmt: skip
    _assert_same_bits(runs["tc"], runs["reg"], ("o", "final_state"))
    _assert_within_budget(runs["tc"], runs["reg"], names)
    _assert_close_to_reference(
        runs["tc"], _reference(inputs, seq_lens, initial_state=h0, dht=dht), names
    )


@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
@pytest.mark.parametrize("route", ["host", "device"])
def test_group_of_three_tail_matches_the_unfused_tail(
    route, beta_dtype, device, monkeypatch
) -> None:
    # HV = 3 H: the fused tail_g3 (pinned; the rule's tail at G = 3 on sm_103a and sm_107a) against dqkwg + the WY
    # pullback + each finalize (the "unfused" pin): the same bits on every output, with an initial state and a
    # final-state gradient
    seq_lens, heads = (1500, 77, 640, 1), (2, 6)
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=97)
    gen = torch.Generator().manual_seed(101)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    stages = _record_launches(monkeypatch)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    tail = "tail_g3_bf16" if beta_dtype == torch.bfloat16 else "tail_g3"
    runs = {}
    for fin in ("", "finalize", "finalize_mlp"):
        start = len(stages)
        pins = {"tail": "unfused", "finalize": fin} if fin else {"tail": "tail_g3"}
        runs[fin] = _run(inputs, seq_lens, route=route, initial_state=h0, output_final_state=True, dht=dht,
                         **pins)  # fmt: skip
        got = stages[start:]
        if fin:
            assert fin in got and "dqkwg" in got and tail not in got, (fin, got)
        else:
            assert tail in got and "dqkwg" not in got, got
            assert not any(stage.startswith(("wy_bwd", "finalize")) for stage in got), (
                got
            )
    _assert_same_bits(runs[""], runs["finalize"], names)
    _assert_same_bits(runs[""], runs["finalize_mlp"], names)


# the tc-priced adjoint table (TC_PLAN): (row, seq_lens, HV, the architectures that replace the planner's table
# there: B200 148 SMs, GB300 152 SMs; measured faster where replaced)
_TC_BOTH = ("sm_100a", "sm_103a")
TC_PLAN_ROWS = (
    ("r01", (4096,) * 4, 16, _TC_BOTH),
    ("r02", _REAL_3, 16, _TC_BOTH),
    ("r03", (4096,) * 8, 16, ()),
    ("r04", _REAL_7, 16, ("sm_100a",)),
    ("r07", (16384,), 16, _TC_BOTH),
    ("r08", (4096,) * 4, 32, ()),
    ("r10", (16384,), 32, _TC_BOTH),
    ("issue", (131071, 87531, 43385), 6, _TC_BOTH),
    ("P3", (128147, 131027), 6, _TC_BOTH),
    ("P1", (126565, 32714, 33696, 15081, 22979, 10782), 6, _TC_BOTH),
    ("A16", (11000, 12000, 9768), 48, ()),
    ("64x4096", (4096,) * 64, 6, ()),
    ("h1_hv3", (11000, 12000, 9768), 3, ("sm_100a",)),
)


@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_tc_plan_rule(arch, monkeypatch) -> None:
    props, _r01_tc = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    cap = cake_chain._prescan_capacity("dhu", cake_chain.SDT_CONFIG)
    for row, seq_lens, num_v_heads, replaced_on in TC_PLAN_ROWS:
        chunks = tuple(-(-n // 64) for n in seq_lens)
        _fwd, bwd = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )
        rec = cake_chain.tc_plan_pricing(chunks, num_v_heads, 0, bwd)
        gain, min_saving = cake_chain.tc_plan_thresholds(arch)
        if rec["select"]:
            assert rec["saving"] >= min_saving, (row, rec)
            assert rec["best"]["T"] <= gain * rec["plan_T"], (
                row,
                rec,
            )
            assert (
                0 <= rec["segment_len"] <= cap
                and tuple(bwd.config) == cake_chain.SDT_CONFIG
            ), (row, rec)
        if arch in _TC_BOTH:
            assert rec["select"] is (arch in replaced_on), (row, rec)


@pytest.mark.parametrize("plan", ["planner", "stage1"])
@pytest.mark.parametrize("heads", [(8, 16), (8, 24)], ids=["g2", "g3"])
def test_tc_plan_changes_only_the_adjoint_table(
    heads, plan, device, monkeypatch
) -> None:
    # one 16384-token sequence (the r07 form; gates kept above the floors with TC_PLAN_GAIN = 1, no saving floor; no
    # CHAIN_PLAN tables): the tc-priced adjoint table leaves the forward's bits, runs on dhu_tc2 (no dv_local stage), repeats bitwise
    # (fixed-order cluster tails) and stays within the budgets of the planner's table; a plan pin keeps the planner
    # plan "planner": the planner's tables (ADJOINT_SPLIT_ARCHS empty); plan "stage1": the shipped plan, where the
    # one-wave split takes the adjoint table on the architectures of ADJOINT_SPLIT_ARCHS (_check_tc_plan_split)
    seq_lens = (16384,)
    index = cake_chain._device_index(device)
    if plan == "planner":
        monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
    elif (
        cake_chain._adjoint_split_length(
            seq_lens, heads[1], index, cake_chain.make_options()
        )
        is not None
    ):
        _check_tc_plan_split(seq_lens, heads, device, monkeypatch)
        return
    monkeypatch.setattr(cake_chain, "TC_PLAN_MIN_SAVING", 0.0)
    monkeypatch.setattr(cake_chain, "TC_PLAN_GAIN", 1.0)
    monkeypatch.setattr(cake_chain, "TC_PLAN_SELECT", {})
    monkeypatch.setattr(cake_chain, "CHAIN_PLAN", {})
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    _fwd, bwd = cake_chain.scan_plans(16384, 1, heads[1], index, seq_lens)
    rec = cake_chain.tc_plan_pricing((256,), heads[1], index, bwd)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=307)
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    base = _run(inputs, seq_lens)
    monkeypatch.setattr(cake_chain, "TC_PLAN", True)
    launches = _record_streams(monkeypatch)
    runs = [_run(inputs, seq_lens) for _ in range(2)]
    stages = [stage for stage, _stream in launches]
    assert torch.equal(runs[0]["o"], base["o"])
    _assert_same_bits(runs[1], runs[0], OUTPUT_NAMES)
    _assert_within_budget(runs[0], base, OUTPUT_NAMES)
    if rec["select"]:
        arch = cake_backend.device_arch(index)
        tc2 = cake_chain.dhu_tc2_stage(arch)
        want = (cake_chain.sdt_prescan(arch), tc2) if rec["segment_len"] else (tc2,)
        assert all(x in stages for x in want) and "dv_local" not in stages, (
            rec,
            stages,
        )
    else:  # the pricing keeps the planner's table: the same bits
        _assert_same_bits(runs[0], base, OUTPUT_NAMES)
    pins = {"scan_config": tuple(bwd.config), "segment_chunks": bwd.segment_len}
    on = _run(inputs, seq_lens, **pins)
    monkeypatch.setattr(cake_chain, "TC_PLAN", False)
    _assert_same_bits(on, _run(inputs, seq_lens, **pins), OUTPUT_NAMES)


def _check_tc_plan_split(seq_lens, heads, device, monkeypatch) -> None:
    """The one-wave split in place of the tc plan: the forward's bits as on the planner's tables, the split's forms and
    composition (no dv_local, no other adjoint scan), bits that repeat and do not depend on TC_PLAN, within the budgets
    of the planner's table; a plan pin keeps the planner's tables."""
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    _fwd, bwd = cake_chain.scan_plans(
        sum(seq_lens), len(seq_lens), heads[1], index, seq_lens
    )
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=307)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
        m.setattr(cake_chain, "TC_PLAN", False)
        base = _run(inputs, seq_lens)
    launches = _record_streams(monkeypatch)
    runs = [_run(inputs, seq_lens) for _ in range(2)]
    stages = [stage for stage, _stream in launches]
    assert torch.equal(runs[0]["o"], base["o"])
    _assert_same_bits(runs[1], runs[0], OUTPUT_NAMES)
    _assert_within_budget(runs[0], base, OUTPUT_NAMES)
    composer = cake_chain._adjoint_chain_stage(arch)
    assert "dhu_tc2_dvl32_modes" in stages and composer in stages, stages
    assert not [
        s
        for s in stages
        if s in ("dv_local", "dhu_tc2", "dhu_tc2_dvl32")
        or s.startswith(("dhu_pre", "dhu_r"))
    ], stages
    monkeypatch.setattr(cake_chain, "TC_PLAN", not cake_chain.TC_PLAN)
    _assert_same_bits(_run(inputs, seq_lens, host_cache=False), runs[0], OUTPUT_NAMES)
    pins = {"scan_config": tuple(bwd.config), "segment_chunks": bwd.segment_len}
    on = _run(inputs, seq_lens, **pins)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
        _assert_same_bits(on, _run(inputs, seq_lens, **pins), OUTPUT_NAMES)


# --------------------------------------------------------------------------- deeper split tables (CHAIN_PLAN)

# The layouts of the bench's rows, packs, head pairs and group shapes (benchmarks/bench_cake_gdn_nondet_train.py): (sequence
# lengths, HV); the tables of host_plans do not depend on H, so the group shapes that repeat a row's layout and HV are
# left out (the value-head group enters only through the split R form, SPLIT_R_FORM, tested with its group below).
_PAIR_LAYOUT = (11000, 12000, 9768)
BENCH_LAYOUTS = {
    "r01": ((4096,) * 4, 16),
    "r02": (_REAL_3, 16),
    "r03": ((4096,) * 8, 16),
    "r04": (_REAL_7, 16),
    "r05": ((4096,) * 16, 16),
    "r06": (_REAL_15, 16),
    "r07": ((16384,), 16),
    "r08": ((4096,) * 4, 32),
    "r09": ((4096,) * 16, 32),
    "r10": ((16384,), 32),
    "64x4096": ((4096,) * 64, 6),
    "A16": (_PAIR_LAYOUT, 48),
    "P1": ((126565, 32714, 33696, 15081, 22979, 10782), 6),
    "P3": ((128147, 131027), 6),
    "issue": ((131071, 87531, 43385), 6),
    "h8_hv24": (_PAIR_LAYOUT, 24),
    "h4_hv12": (_PAIR_LAYOUT, 12),
    "h2_hv6": (_PAIR_LAYOUT, 6),
    "h1_hv3": (_PAIR_LAYOUT, 3),
    "g1_16x4096_h8": ((4096,) * 16, 8),
    "g3_4x4096_h8": ((4096,) * 4, 24),
    "g3_16x4096_h1": ((4096,) * 16, 3),
}
# The layouts whose tables CHAIN_PLAN deepens on R200 (212 and 216 SMs): row -> (forward, adjoint) segment length of the
# deeper table (None: the table without CHAIN_PLAN).  Every other layout keeps its tables.
CHAIN_PLAN_ROWS = {
    212: {
        "r07": (None, 20),
        "A16": (43, None),
        "h2_hv6": (None, 16),
        "h1_hv3": (16, 16),
        "g3_16x4096_h1": (None, 16),
    },
    216: {
        "r07": (None, 20),
        "h2_hv6": (None, 15),
        "h1_hv3": (16, 15),
        "g3_16x4096_h1": (None, 16),
    },
}


def _segments(chunks, plan) -> int:
    if plan.segment_len <= 0 or max(chunks) <= plan.segment_len:
        return len(chunks)
    return sum(-(-n // plan.segment_len) for n in chunks)


@pytest.mark.parametrize(
    "arch, sm_count",
    [("sm_100a", 148), ("sm_103a", 148), ("sm_103a", 152)],
    ids=["B200", "B300", "GB300"],
)
def test_chain_plan_keeps_the_b200_b300_gb300_tables(
    arch, sm_count, monkeypatch
) -> None:
    # CHAIN_PLAN lists sm_107a only: on B200, B300 and GB300 every bench layout keeps the tables (and the dhu_tc2
    # pricing of its adjoint table) that it plans without CHAIN_PLAN, also with a saving floor of 0
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, 232448, 10)
    )
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    assert arch not in cake_chain.CHAIN_PLAN
    options = cake_chain.make_options()
    for row, (seq_lens, num_v_heads) in BENCH_LAYOUTS.items():
        chunks = tuple(-(-n // 64) for n in seq_lens)
        with monkeypatch.context() as m:
            m.setattr(cake_chain, "CHAIN_PLAN", {})
            want = cake_chain.host_plans(seq_lens, num_v_heads, 0, options)
        for table in (
            cake_chain.CHAIN_PLAN,
            {
                name: {**const, "min_saving": 0.0}
                for name, const in cake_chain.CHAIN_PLAN.items()
            },
        ):
            with monkeypatch.context() as m:
                m.setattr(cake_chain, "CHAIN_PLAN", table)
                assert (
                    cake_chain.host_plans(seq_lens, num_v_heads, 0, options) == want
                ), row
                for kind, plan in zip(("fwd", "bwd"), want[0], strict=True):
                    rec = cake_chain.chain_plan_pricing(
                        kind, chunks, num_v_heads, 0, plan
                    )
                    assert rec["select"] is False, (row, kind, rec)


@pytest.mark.parametrize("plan", ["planner", "stage1"])
@pytest.mark.parametrize("sm_count", sorted(CHAIN_PLAN_ROWS))
def test_chain_plan_rule_on_r200(sm_count, plan, monkeypatch) -> None:
    # plan "planner": the planner's tables (ADJOINT_SPLIT_ARCHS empty), CHAIN_PLAN_ROWS in both directions; plan
    # "stage1": the shipped plan, where on sm_107a in ADJOINT_SPLIT_ARCHS the adjoint table is the one-wave split or a
    # serial table whatever CHAIN_PLAN prices, and CHAIN_PLAN_ROWS holds for the forward tables
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, 232448, 10)
    )
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: "sm_107a")
    # CHAIN_PLAN's own tables: the serial forward table of FWD_A_SERIAL_KEEP (test_fwd_a_serial_keep_rule) is off
    monkeypatch.setattr(cake_chain, "FWD_A_SERIAL_KEEP", {})
    if plan == "planner":
        monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
    split_arch = "sm_107a" in cake_chain.ADJOINT_SPLIT_ARCHS
    const = cake_chain.CHAIN_PLAN["sm_107a"]
    cap = {
        "fwd": cake_chain._prescan_capacity("fwd_h", cake_chain.SDT_CONFIG),
        "bwd": cake_chain._prescan_capacity("dhu", cake_chain.SDT_CONFIG),
    }
    options = cake_chain.make_options()
    deepened = {}
    for row, (seq_lens, num_v_heads) in BENCH_LAYOUTS.items():
        chunks = tuple(-(-n // 64) for n in seq_lens)
        got, bwd_tc = cake_chain.host_plans(seq_lens, num_v_heads, 0, options)
        with monkeypatch.context() as m:
            m.setattr(cake_chain, "CHAIN_PLAN", {})
            want, want_tc = cake_chain.host_plans(seq_lens, num_v_heads, 0, options)
        lens = tuple(
            None if new == old else new.segment_len
            for new, old in zip(got, want, strict=True)
        )
        if lens != (None, None):
            deepened[row] = lens
        split = cake_chain.two_stream_split(len(chunks), num_v_heads, sm_count)
        priced = cake_chain.adjoint_split_pricing(
            chunks, num_v_heads, sm_count, "sm_107a"
        )
        for kind, new, old in zip(("fwd", "bwd"), got, want, strict=True):
            if kind == "bwd" and split_arch:
                # stage 1: the one-wave split where it prices cheaper, a serial table otherwise, with CHAIN_PLAN or not
                if priced["select"]:
                    split_plan = cake_chain.ScanPlan(
                        cake_chain.SDT_CONFIG, priced["segment_len"]
                    )
                    assert new == old == split_plan, (row, new, old)
                    assert not bwd_tc and not want_tc, row
                else:
                    assert _segments(chunks, new) == len(chunks), (row, new)
                    assert _segments(chunks, old) == len(chunks), (row, old)
                continue
            # a serial table that the two-stream rule groups keeps its form, whatever the pricing
            grouped = _segments(chunks, old) == len(chunks) and split is not None
            rec = cake_chain.chain_plan_pricing(kind, chunks, num_v_heads, 0, old)
            assert (new != old) is (rec["select"] and not grouped), (row, kind, rec)
            # a changed table: a deeper SDT_CONFIG split table within the pre-scan capacity, priced cheaper
            if new != old:
                assert tuple(new.config) == cake_chain.SDT_CONFIG, (row, kind, new)
                assert 0 < new.segment_len <= cap[kind], (row, kind, new)
                assert _segments(chunks, new) > _segments(chunks, old), (
                    row,
                    kind,
                    new,
                    old,
                )
                assert rec["saving"] >= const["min_saving"], (row, kind, rec)
                assert rec["best"]["T"] <= const["gain"] * rec["plan_T"], (
                    row,
                    kind,
                    rec,
                )
        if not split_arch:
            assert bwd_tc is (want_tc or got[1] != want[1]), row
    if split_arch:
        forward = {row: lens[0] for row, lens in deepened.items() if lens[0]}
        assert forward == {
            row: lens[0] for row, lens in CHAIN_PLAN_ROWS[sm_count].items() if lens[0]
        }
    else:
        assert deepened == CHAIN_PLAN_ROWS[sm_count]
    # plan pins (and a dhu_scan pin) keep the planner's tables
    seq_lens = (16384,)
    for pins in (
        {"scan_config": (64, 1), "segment_chunks": 0},
        {"segment_chunks": 64},
        {"dhu_scan": "tc"},
    ):
        pinned = cake_chain.make_options(**pins)
        planner = cake_chain.scan_plans(16384, 1, 16, 0, seq_lens, scan_config=pinned.scan_config,
                                        segment_chunks=pinned.segment_chunks)  # fmt: skip
        assert cake_chain.host_plans(seq_lens, 16, 0, pinned) == (
            tuple(planner),
            False,
        ), pins


# The forward table rule of FWD_A_SERIAL_KEEP on the gating rows, per SM count of R200 (sm_107a): row -> whether its
# split forward table gives way to the planner's cheapest serial table.  The 3 sequences of 11000 / 12000 / 9768 tokens
# with HV 48 (144 serial CTAs, fill 0.68, the serial table 1.06 x the split table's price) take it; with HV 24 (72 CTAs,
# 1.51 x) and fewer value heads, and the long rows, the split table stays (64 x 4096 is serial without the rule); so does
# g3_4x4096_h8 (96 CTAs, fill 0.45, 1.08 x: below the fill).  B200 and GB300 keep every table.
FWD_A_SERIAL_ROWS = {
    212: {
        "A16": True,
        "h8_hv24": False,
        "h4_hv12": False,
        "h2_hv6": False,
        "h1_hv3": False,
        "issue": False,
        "P3": False,
        "P1": False,
        "64x4096": False,
        "g3_4x4096_h8": False,
    }
}


@pytest.mark.parametrize(
    "arch, sm_count",
    [("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 212), ("sm_107a", 216)],
    ids=["B200", "GB300", "R200-212", "R200-216"],
)
def test_fwd_a_serial_keep_rule(arch, sm_count, monkeypatch) -> None:
    # a split forward table priced within FWD_A_SERIAL_KEEP's factor of the cheapest serial table becomes that serial
    # table (whatever CHAIN_PLAN prices), so fwd_a_route's table condition holds; the adjoint table does not change;
    # plan pins keep the planner's tables
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, 232448, 10)
    )
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    options = cake_chain.make_options()
    kept, ratios, tables = set(), {}, {}
    for row, (seq_lens, num_v_heads) in BENCH_LAYOUTS.items():
        chunks = tuple(-(-n // 64) for n in seq_lens)
        (fwd, bwd), bwd_tc = cake_chain.host_plans(seq_lens, num_v_heads, 0, options)
        with monkeypatch.context() as m:
            m.setattr(cake_chain, "FWD_A_SERIAL_KEEP", {})
            (fwd0, bwd0), tc0 = cake_chain.host_plans(seq_lens, num_v_heads, 0, options)
        assert (bwd, bwd_tc) == (bwd0, tc0), row
        planner = cake_chain.scan_plans(
            sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
        )[0]
        rec = cake_chain.fwd_serial_keep_pricing(seq_lens, num_v_heads, 0, planner)
        ratios[row], tables[row] = rec["ratio"], fwd
        if fwd == fwd0:
            assert not rec["select"], (row, rec)
            continue
        kept.add(row)
        assert rec["select"] and rec["ratio"] <= rec["limit"], (row, rec)
        assert rec["fill"] >= rec["min_fill"], (row, rec)
        assert fwd == rec["serial"] and fwd.segment_len == 0, (row, fwd)
        assert _segments(chunks, fwd0) > len(chunks), (row, fwd0)
        meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=(fwd, bwd))
        assert meta.fwd_segments.num_segments == len(seq_lens), row
    if arch not in cake_chain.FWD_A_SERIAL_KEEP:
        assert kept == set()
    for row, want in FWD_A_SERIAL_ROWS.get(sm_count, {}).items():
        assert (row in kept) is want, (row, sorted(kept), ratios)
    if sm_count in FWD_A_SERIAL_ROWS:
        assert 1.0 < ratios["A16"] <= 1.10 < ratios["h8_hv24"], ratios
        assert ratios["g3_4x4096_h8"] <= 1.10, ratios  # kept split by the fill
        chunks = tuple(-(-n // 64) for n in _PAIR_LAYOUT)
        assert _segments(chunks, tables["h8_hv24"]) > len(chunks), tables["h8_hv24"]
    pinned = cake_chain.make_options(scan_config=(64, 1), segment_chunks=64)
    planner = cake_chain.scan_plans(sum(_PAIR_LAYOUT), 3, 48, 0, _PAIR_LAYOUT, scan_config=(64, 1),
                                    segment_chunks=64)  # fmt: skip
    assert planner[0].segment_len == 64, planner
    assert cake_chain.host_plans(_PAIR_LAYOUT, 48, 0, pinned) == (tuple(planner), False)


# The split R form (SPLIT_R_FORM): the bench layouts of a value-head group of three whose planner adjoint table it
# keeps, per (architecture, SM count) -> (segment length, segments, adjoint group A in segments): the deterministic
# package's table and groups.  Every other layout and group keeps the tables of host_plans.
_SPLIT_R_148 = {
    "P1": (110, 37, 24),
    "P3": (114, 36, 24),
    "issue": (114, 36, 24),
    "g3_4x4096_h8": (22, 12, 6),
}
SPLIT_R_ROWS = {
    ("sm_100a", 148): _SPLIT_R_148,
    ("sm_103a", 148): _SPLIT_R_148,
    ("sm_103a", 152): {
        "P1": (110, 37, 25),
        "P3": (108, 38, 25),
        "issue": (114, 36, 25),
        "g3_4x4096_h8": (22, 12, 6),
    },
    ("sm_107a", 212): {"A16": (94, 6, 4)},
    ("sm_107a", 216): {},
}


@pytest.mark.parametrize(
    "arch, sm_count",
    [
        ("sm_100a", 148),
        ("sm_103a", 148),
        ("sm_103a", 152),
        ("sm_107a", 212),
        ("sm_107a", 216),
    ],
    ids=["B200", "B300", "GB300", "R200-212", "R200-216"],
)
@pytest.mark.parametrize("plan", ["planner", "stage1"])
def test_split_r_form_rule(arch, sm_count, plan, monkeypatch) -> None:
    # plan "planner": the planner's tables (ADJOINT_SPLIT_ARCHS empty), SPLIT_R_ROWS; plan "stage1": the shipped plan,
    # where on the architectures of ADJOINT_SPLIT_ARCHS every adjoint table is the one-wave split or a serial table and
    # the R form keeps none
    split_arch = plan == "stage1" and arch in cake_chain.ADJOINT_SPLIT_ARCHS
    if plan == "planner":
        monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, 232448, 10)
    )
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    options = cake_chain.make_options()
    kept = {}
    for row, (seq_lens, num_v_heads) in BENCH_LAYOUTS.items():
        want = cake_chain.host_plans(seq_lens, num_v_heads, 0, options)
        if split_arch:
            length = cake_chain._adjoint_split_length(seq_lens, num_v_heads, 0, options)
            chunks = tuple(-(-n // 64) for n in seq_lens)
            if length is not None:
                assert (
                    want[0][1] == cake_chain.ScanPlan(cake_chain.SDT_CONFIG, length)
                    and not want[1]
                ), row
            else:
                assert (
                    want[0][1].segment_len <= 0 or max(chunks) <= want[0][1].segment_len
                ), (row, want)
        for group in (1, 2, 3):
            if num_v_heads % group:
                continue
            plans, bwd_tc, keep = cake_chain._host_plans(
                seq_lens, num_v_heads, 0, options, group=group
            )
            if not keep:
                assert (plans, bwd_tc) == want, (row, group)
                continue
            assert group == 3 and arch in cake_chain.SPLIT_R_FORM, (row, group)
            planner = cake_chain.scan_plans(
                sum(seq_lens), len(seq_lens), num_v_heads, 0, seq_lens
            )
            # the forward's table as without the form, the planner's adjoint table, priced for dhu_tc2 by sdt_pricing
            assert plans == (want[0][0], planner[1]) and bwd_tc is False, row
            chunks = tuple(-(-n // 64) for n in seq_lens)
            assert cake_chain.sdt_pricing(
                chunks, planner[1].segment_len, num_v_heads, 0
            )["select"], row
            meta = cake_chain.build_chunk_meta(
                seq_lens, "cpu", scan_plans=plans, lpt=True
            )
            fwd_groups, bwd_groups = cake_chain.sequence_groups(
                meta, num_v_heads, sm_count, split_adjoint=True
            )
            assert fwd_groups is None and bwd_groups is not None, row
            kept[row] = (
                plans[1].segment_len,
                meta.bwd_segments.num_segments,
                bwd_groups.split,
            )
            # the groups only beside the split R form's tail (a pin to another tail keeps the tables of host_plans)
            assert not cake_chain._host_plans(seq_lens, num_v_heads, 0, cake_chain.make_options(tail="tail"),
                                              group=3)[2]  # fmt: skip
    assert kept == ({} if split_arch else SPLIT_R_ROWS[(arch, sm_count)])
    # plan pins (and a dhu_scan pin) keep the planner's tables
    for pins in ({"scan_config": (64, 1), "segment_chunks": 94}, {"dhu_scan": "tc"}):
        assert not cake_chain._host_plans(
            _PAIR_LAYOUT, 48, 0, cake_chain.make_options(**pins), group=3
        )[2], pins


@pytest.mark.parametrize(
    "seq_lens, heads",
    [((16384,), (8, 16)), (_PAIR_LAYOUT, (1, 3))],
    ids=["r07", "h1-hv3"],
)
def test_chain_plan_tables_stay_within_the_budgets(
    seq_lens, heads, device, monkeypatch
) -> None:
    # the deeper tables of CHAIN_PLAN (this architecture's constants, else sm_107a's; no saving floor): the forward runs
    # the pre-scan, the stitch and fwd_ho on a deeper forward table, the backward the fused pre-scan and dhu_tc2 on a
    # deeper adjoint table; the bits repeat run to run and stay within the budgets of the tables without CHAIN_PLAN;
    # plan pins keep the planner's tables (the same bits with and without CHAIN_PLAN).  The tables are the call's: its
    # value-head group selects a form of cake_plan.ADJOINT_FORMS (GB200's H 1 / HV 3 keeps the planner's tables)
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    const = cake_chain.CHAIN_PLAN.get(arch, cake_chain.CHAIN_PLAN["sm_107a"])
    forced = {arch: {**const, "min_saving": 0.0}}
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=313)
    monkeypatch.setattr(cake_chain, "CHAIN_PLAN", {})
    group = heads[1] // heads[0]
    (fwd0, bwd0), _tc = cake_chain.host_plans(seq_lens, heads[1], index, group=group)
    base = _run(inputs, seq_lens)
    pins = {"scan_config": tuple(fwd0.config), "segment_chunks": fwd0.segment_len}
    base_pinned = _run(inputs, seq_lens, **pins)
    monkeypatch.setattr(cake_chain, "CHAIN_PLAN", forced)
    (fwd, bwd), _tc = cake_chain.host_plans(seq_lens, heads[1], index, group=group)
    stages = _record_launches(monkeypatch)
    runs = [_run(inputs, seq_lens) for _ in range(2)]
    _assert_same_bits(runs[1], runs[0], OUTPUT_NAMES)
    _assert_within_budget(runs[0], base, OUTPUT_NAMES)
    if (fwd, bwd) == (fwd0, bwd0):
        _assert_same_bits(runs[0], base, OUTPUT_NAMES)
    if fwd != fwd0:
        assert _segments(tuple(-(-n // 64) for n in seq_lens), fwd) > _segments(
            tuple(-(-n // 64) for n in seq_lens), fwd0
        )
        _stage, _vb, pre, _ = cake_chain._scan_stages(
            "fwd_h", cake_chain.SDT_CONFIG, arch
        )
        assert all(x in stages for x in (pre, "stitch")), stages
        assert "fwd_ho" in stages or "fwd_ho_vh2" in stages, stages
    if bwd != bwd0:
        assert all(
            x in stages
            for x in (cake_chain.sdt_prescan(arch), cake_chain.dhu_tc2_stage(arch))
        ), stages
        assert "dv_local" not in stages, stages
    _assert_same_bits(_run(inputs, seq_lens, **pins), base_pinned, OUTPUT_NAMES)


# The bench layouts whose forward runs fwd_ho_vh2 per device (SMs, architecture), on the host route's tables
# (host_plans: the planner's tables, TC_PLAN and CHAIN_PLAN, which prices the split): on R200 the one-launch forwards of
# r07 and h2_hv6 on the planner's tables (96 and 102 CTAs; without the split's price CHAIN_PLAN deepens them to 208 and
# 198 CTAs) and of h1_hv3 on CHAIN_PLAN's deeper table (99 CTAs); no layout on sm_100a / sm_103a.
_VH2_R200 = {"r07", "h2_hv6", "h1_hv3"}
VH2_BENCH_ROWS = {
    ("sm_100a", 148): set(),
    ("sm_103a", 148): set(),
    ("sm_103a", 152): set(),
    ("sm_107a", 208): _VH2_R200,
    ("sm_107a", 212): _VH2_R200,
    ("sm_107a", 216): _VH2_R200,
}


@pytest.mark.parametrize("arch, sm_count", sorted(VH2_BENCH_ROWS))
def test_value_half_split_on_the_bench_layouts(arch, sm_count, monkeypatch) -> None:
    monkeypatch.setattr(
        cake_chain, "_device_props", lambda _index: (sm_count, 232448, 10)
    )
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    options = cake_chain.make_options()
    engaged = set()
    for row, (seq_lens, num_v_heads) in BENCH_LAYOUTS.items():
        plans, bwd_tc = cake_chain.host_plans(seq_lens, num_v_heads, 0, options)
        meta = cake_chain.build_chunk_meta(seq_lens, "cpu", scan_plans=plans, lpt=True)
        fwd, _bwd = cake_chain.sequence_groups(meta, num_v_heads, sm_count)
        segments = meta.fwd_segments.num_segments
        if (
            fwd is None
            and arch in cake_chain.FWD_HO_VH2_ARCHS
            and cake_chain.fwd_ho_vh2_engaged(segments, num_v_heads, sm_count)
        ):
            engaged.add(row)
            # the forward table the split runs is not one CHAIN_PLAN deepened past the split
            chunks = tuple(-(-n // 64) for n in seq_lens)
            rec = cake_chain.chain_plan_pricing("fwd", chunks, num_v_heads, 0, plans[0])
            assert not rec.get("select", False), (row, rec)
    assert engaged == VH2_BENCH_ROWS[(arch, sm_count)]
    # without the split's price (FWD_HO_VH2_ARCHS empty) r07 takes CHAIN_PLAN's deeper fwd_ho table on R200
    monkeypatch.setattr(cake_chain, "FWD_HO_VH2_ARCHS", ())
    (fwd_plan, _), _tc = cake_chain.host_plans((16384,), 16, 0, options)
    assert (fwd_plan.segment_len == 20) is (arch == "sm_107a"), fwd_plan


# (layout, heads, route, pins): a serial plan with grouped value heads and partial chunks, a split plan (segment start
# states from the pre-scan and the stitch) and the device route (the table's segment bound, padding segments
# included); each forward scan is one launch of at most half the SMs of any of the devices
VH2_FORMS = {
    "serial": ((300, 77, 640, 17), (2, 4), "host", {}),
    "split": (
        (2000, 300),
        (1, 2),
        "host",
        {"scan_config": (32, 2), "segment_chunks": 4},
    ),
    "device": ((100, 640, 77, 300), (2, 6), "device", {}),
}


@pytest.mark.parametrize("form", sorted(VH2_FORMS))
def test_the_value_half_split_stays_within_the_budgets(
    form, device, monkeypatch
) -> None:
    # with the rule on for the device's architecture (fwd_ho_vh2 is delivered for every architecture) the forward scan
    # runs fwd_ho_vh2, else fwd_ho on the same table (the planner's tables: CHAIN_PLAN off); both runs stay within the
    # budgets against the FP64 recurrence, the final state and the initial-state gradient included
    seq_lens, heads, route, pins = VH2_FORMS[form]
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    monkeypatch.setattr(cake_chain, "CHAIN_PLAN", {})
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=83)
    gen = torch.Generator().manual_seed(89)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    ref = _reference(inputs, seq_lens, initial_state=h0)
    stages = _record_launches(monkeypatch)
    runs = []
    for archs, stage in (((arch,), "fwd_ho_vh2"), ((), "fwd_ho")):
        monkeypatch.setattr(cake_chain, "FWD_HO_VH2_ARCHS", archs)
        stages.clear()
        runs.append(_run(inputs, seq_lens, route=route, initial_state=h0, output_final_state=True, **pins))  # fmt: skip
        assert [name for name in stages if name.startswith("fwd_ho")] == [stage], stages
        _assert_close_to_reference(runs[-1], ref, OUTPUT_NAMES + ("final_state", "dh0"))
    _assert_within_budget(runs[0], runs[1], OUTPUT_NAMES + ("final_state", "dh0"))


# #7038 / #7548 form: a one-thread sleep kernel resident on a side stream for the whole call; the gate is declared
# before the run (a stage that waited for every SM, or a whole-device launch, would stretch the call to the sleep).
# The caller's stream sleeps first, so the host has enqueued the whole call before its first launch runs: the events
# time the device work, not the launch overhead, which varies with the host's load on a call this short.
SIDE_STREAM_RATIO_MAX = 1.3
SIDE_STREAM_ITERS = 9


@pytest.mark.parametrize("heads", [(2, 4), (2, 6)], ids=["g2", "g3"])
def test_forward_backward_beside_a_resident_side_stream_kernel(heads, device) -> None:
    seq_lens = (4096,) * 4
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=287)
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }

    def step() -> None:
        for leaf in leaves.values():
            leaf.grad = None
        o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                         scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                                         cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip
        torch.autograd.backward(o, inputs["do"])

    clock_khz = getattr(
        torch.cuda.get_device_properties(device), "clock_rate", 1_900_000
    )
    side = torch.cuda.Stream(device=device)

    def timed(lead_cycles: int, side_cycles: int = 0) -> float:
        torch.cuda.synchronize()
        if side_cycles:
            with torch.cuda.stream(side):
                torch.cuda._sleep(side_cycles)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        if lead_cycles:
            torch.cuda._sleep(lead_cycles)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    for _ in range(2):
        step()
    wall = sorted(timed(0) for _ in range(3))[1]
    # four times the call's wall time, in SM cycles (ms x kHz)
    lead = int(4.0 * wall * clock_khz)
    median = SIDE_STREAM_ITERS // 2
    alone = sorted(timed(lead) for _ in range(SIDE_STREAM_ITERS))[median]
    # the lead and about three times the call
    side_cycles = lead + int(3.0 * alone * clock_khz)
    beside = [timed(lead, side_cycles) for _ in range(SIDE_STREAM_ITERS)]
    ratio = sorted(beside)[median] / alone
    assert ratio <= SIDE_STREAM_RATIO_MAX, (ratio, alone, beside)


# The five head pairs of the source issue (#7074), on one layout of partial, full and multi-chunk sequences.
HEAD_PAIRS = ((16, 48), (8, 24), (4, 12), (2, 6), (1, 3))
PAIR_LAYOUT = (63, 130, 257)


@pytest.mark.parametrize(
    "heads", HEAD_PAIRS, ids=[f"h{h}-hv{hv}" for h, hv in HEAD_PAIRS]
)
def test_source_head_pairs_match_the_fp64_recurrence(heads, device) -> None:
    inputs = _inputs(PAIR_LAYOUT, heads, torch.float32, device, seed=211)
    got = _run(inputs, PAIR_LAYOUT)
    _assert_close_to_reference(got, _reference(inputs, PAIR_LAYOUT), OUTPUT_NAMES)
    _assert_within_budget(got, _run(inputs, PAIR_LAYOUT, det=True), OUTPUT_NAMES)


DET_CASES = tuple(c for c in CASES if c[0] in ("pack-2x6-host-fp32beta", "pack-4x8-host-bf16beta", "one-token",
                                               "split-32x2-host", "dhu-tc-g3-device", "recompute", "no-l2norm",
                                               "finalize-forced"))  # fmt: skip


@pytest.mark.parametrize("case", DET_CASES, ids=[c[0] for c in DET_CASES])
def test_matches_the_deterministic_package(case, device) -> None:
    # same inputs, one process, the bounds declared above; both packages also against the FP64 recurrence
    _id, seq_lens, heads, beta_dtype, route, pins, l2norm = case
    inputs = _inputs(seq_lens, heads, beta_dtype, device, normalized_qk=not l2norm)
    det_pins = {name: value for name, value in pins.items() if name != "tail"}
    det = _run(
        inputs,
        seq_lens,
        route=route,
        use_qk_l2norm_in_kernel=l2norm,
        det=True,
        **det_pins,
    )
    got = _run(inputs, seq_lens, route=route, use_qk_l2norm_in_kernel=l2norm, **pins)
    _assert_within_budget(got, det, OUTPUT_NAMES)
    ref = _reference(inputs, seq_lens, normalize=l2norm)
    _assert_close_to_reference(det, ref, OUTPUT_NAMES)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES)


REPEATS = 8


@pytest.mark.parametrize("pins", [{}, {"tail": "tail"}], ids=["rule", "tail"])
@pytest.mark.parametrize(
    "heads", [(2, 4), (2, 6), (2, 2), (1, 4)], ids=["g2", "g3", "g1", "g4"]
)
def test_worst_case_over_repeated_runs(heads, pins, device) -> None:
    # the gate holds for the worst of REPEATS runs, per output tensor (the dispatched tail and the package tail)
    seq_lens = (2048, 100, 513)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=223)
    ref = _reference(inputs, seq_lens)
    runs = [_run(inputs, seq_lens, **pins) for _ in range(REPEATS)]
    worst = {}
    for name in OUTPUT_NAMES:
        metrics = [_metrics(run[name], ref[name]) for run in runs]
        worst[name] = {
            "rel_rms": max(m["rel_rms"] for m in metrics),
            "max_abs": max(m["max_abs"] for m in metrics),
            "max_ref": metrics[0]["max_ref"],
            "finite": all(m["finite"] for m in metrics),
            "run_spread": max(
                float((run[name].double() - runs[0][name].double()).abs().max())
                for run in runs
            ),
        }
    failed = [name for name, m in worst.items()
              if not (m["finite"] and m["rel_rms"] <= REL_RMS_MAX and m["max_abs"] <= MAX_ABS_REL * m["max_ref"])]  # fmt: skip
    assert not failed, (
        f"worst of {REPEATS} runs outside the accuracy gate: {failed}; {worst}"
    )


def test_strided_packed_qkv_views(device) -> None:
    # q / k / v as views of one packed [1, T, (2 H + HV) * 128] projection (token stride (2 H + HV) * 128)
    seq_lens, (h, hv) = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, (h, hv), torch.bfloat16, device, seed=227)
    total = sum(seq_lens)
    flat = [inputs[n].reshape(1, total, -1) for n in ("q", "k", "v")]
    packed = torch.cat(flat, -1).detach().clone().requires_grad_(True)
    widths = (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)
    q, k, v = (part.unflatten(-1, (-1, HEAD_DIM)) for part in packed.split(widths, -1))
    assert not q.is_contiguous() and q.stride(1) == (2 * h + hv) * HEAD_DIM
    g, beta = (inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    cu = _cu(seq_lens)
    o, _ = cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM),
                                     cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                                     cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64),
                                     use_qk_l2norm_in_kernel=True)  # fmt: skip
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    dq, dk, dv = (
        part.unflatten(-1, (-1, HEAD_DIM)) for part in packed.grad.split(widths, -1)
    )
    got = {
        "o": o.detach(),
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": g.grad,
        "dbeta": beta.grad,
    }
    _assert_within_budget(got, _run(inputs, seq_lens), OUTPUT_NAMES)
    _assert_close_to_reference(got, _reference(inputs, seq_lens), OUTPUT_NAMES)


def _run_packed(inputs, seq_lens, heads, **pins):
    """:func:`_run` (host route) with q / k / v as head views of one packed ``[1, T, (2 H + HV) * 128]`` leaf; ``dq``
    / ``dk`` / ``dv`` are the views of the leaf's gradient."""
    h, hv = heads
    total = inputs["q"].shape[1]
    widths = (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)
    flat = [inputs[n].reshape(1, total, -1) for n in ("q", "k", "v")]
    packed = torch.cat(flat, -1).detach().clone().requires_grad_(True)
    q, k, v = (part.unflatten(-1, (-1, HEAD_DIM)) for part in packed.split(widths, -1))
    assert (
        not q.is_contiguous()
        and not v.is_contiguous()
        and q.stride(1) == (2 * h + hv) * HEAD_DIM
    )
    g, beta = (inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    cu = _cu(seq_lens)
    o, _ = cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM),
                           cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=inputs["q"].device),
                           cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64), use_qk_l2norm_in_kernel=True,
                           **pins)  # fmt: skip
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    dq, dk, dv = (
        part.unflatten(-1, (-1, HEAD_DIM)) for part in packed.grad.split(widths, -1)
    )
    return {
        "o": o.detach(),
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": g.grad,
        "dbeta": beta.grad,
    }


@pytest.mark.parametrize("band", ["rule", "plain-band", "plain-pin"])
@pytest.mark.parametrize("heads", [(2, 6), (1, 3), (2, 4)], ids=["g3", "g3-h1", "g2"])
def test_packed_views_give_the_contiguous_bits(
    band, heads, device, monkeypatch
) -> None:
    # q / k / v as head views of one packed row give the bits of contiguous copies, every output and gradient:
    # prep_tma reads q / k in place (also where the rule's band would run the plain prep on contiguous rows: the two
    # preps give the same bits), v is copied byte for byte; a plain pin copies q / k as well.  The prep's forward:
    # fwd_a_route's forward reads the packed row in place too (test_fwd_a_packed_views_give_the_contiguous_bits)
    monkeypatch.setattr(cake_chain, "FWD_A_ARCHS", ())
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_ARCHS", ())
    seq_lens = (300, 77, 640, 1)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=233)
    pins = {"prep": "plain"} if band == "plain-pin" else {}
    if band == "plain-band":
        monkeypatch.setattr(cake_chain, "use_prep_tma", lambda *_args: False)
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens, **pins)
    want_preps = _preps(stages)
    stages.clear()
    got = _run_packed(inputs, seq_lens, heads, **pins)
    got_preps = _preps(stages)
    _assert_same_bits(got, want, OUTPUT_NAMES)
    expected = {"rule": ({"prep_tma"}, {"prep_tma"}), "plain-band": ({"prep"}, {"prep_tma"}),
                "plain-pin": ({"prep"}, {"prep"})}[band]  # fmt: skip
    assert (want_preps, got_preps) == expected, (want_preps, got_preps)


def test_dense_copies_the_bytes(device) -> None:
    # the word copies of a strided view keep every BF16 bit pattern (NaN payloads included): head views at element
    # offsets 0 and 512 take 16-byte words, at offset 4 8-byte words, at offset 3 the element copy
    words = torch.randint(
        -(2**15), 2**15, (257, 1280), dtype=torch.int32, device=device
    ).to(torch.int16)
    row = words.view(torch.bfloat16)
    for start, width in ((0, 256), (512, 768), (4, 256), (3, 256)):
        view = row[:, start : start + width].unflatten(-1, (-1, HEAD_DIM))
        dense = cake_chain._dense(view)
        assert dense.is_contiguous() and dense.shape == view.shape
        assert torch.equal(dense.view(torch.int16), view.contiguous().view(torch.int16))
    contiguous = row[:5]
    assert cake_chain._dense(contiguous) is contiguous


def test_tma_rows_takes_row_major_head_views() -> None:
    # prep_tma's in-place reading: head views of a packed row; not a transposed, unaligned or overlapping view
    row = torch.empty(64, (2 * 2 + 6) * HEAD_DIM, dtype=torch.bfloat16)
    q = row[:, : 2 * HEAD_DIM].view(64, 2, HEAD_DIM)
    assert cake_chain._tma_rows(q) and cake_chain._tma_rows(q[None])
    assert not cake_chain._tma_rows(q.transpose(0, 1))
    assert not cake_chain._tma_rows(row[:, 4 : 4 + 2 * HEAD_DIM].view(64, 2, HEAD_DIM))
    assert not cake_chain._tma_rows(
        torch.empty(64, HEAD_DIM, dtype=torch.bfloat16)[:, None].expand(64, 2, HEAD_DIM)
    )
    odd = torch.empty(64, 2 * HEAD_DIM + 4, dtype=torch.bfloat16)[
        :, : 2 * HEAD_DIM
    ].view(64, 2, HEAD_DIM)
    assert not cake_chain._tma_rows(
        odd
    )  # token stride of 520 bytes: not a 16-byte multiple
    batched = torch.empty(3, 64, 2, HEAD_DIM, dtype=torch.bfloat16)
    assert cake_chain._tma_rows(batched) and not cake_chain._tma_rows(batched[::2])


def _storage_bytes(t: torch.Tensor) -> int:
    """The caching allocator's figure for one tensor's storage (512-byte rounding)."""
    return (t.untyped_storage().nbytes() + 511) // 512 * 512


def _allocator_bytes(device) -> tuple[int, int]:
    """(requested, block) bytes of the caching allocator's live allocations on ``device``."""
    stats = torch.cuda.memory_stats(device)
    return int(stats["requested_bytes.all.current"]), int(
        stats["allocated_bytes.all.current"]
    )


def test_checkpoint_matches_and_retains_no_bytes(device) -> None:
    # non-reentrant torch.utils.checkpoint: the gradients within the budgets of the plain call, and a checkpointed
    # forward keeps nothing on the device beyond its output (#7039)
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=229)
    eager = _run(inputs, seq_lens)
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }

    def layer(q, k, v, g, beta):
        o, _ = cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                                         cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip
        return o

    for _ in range(2):  # the libraries, the tables and the workspaces of this layout
        for leaf in leaves.values():
            leaf.grad = None
        o = checkpoint(layer, *leaves.values(), use_reentrant=False)
        (o.float() * inputs["do"].float()).sum().backward()
    del o
    for leaf in leaves.values():
        leaf.grad = None
    gc.collect()
    torch.cuda.synchronize()
    requested, blocks = _allocator_bytes(device)
    o = checkpoint(layer, *leaves.values(), use_reentrant=False)
    torch.cuda.synchronize()
    requested_after, blocks_after = _allocator_bytes(device)
    # requested bytes: the block count (memory_allocated) may exceed the output without a retained tensor, since the
    # caching allocator hands out an unsplit cached block when its remainder is below the split threshold (1 MiB)
    retained = requested_after - requested - o.untyped_storage().nbytes()
    unsplit = blocks_after - blocks - _storage_bytes(o)
    assert retained == 0, (
        f"a checkpointed forward retains {retained} requested bytes beyond its output (blocks +{unsplit})"
    )
    assert unsplit < 2**20, unsplit
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    got = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_within_budget(got, eager, OUTPUT_NAMES)


def test_gradients_accumulate_into_nonzero_grad(device) -> None:
    # two backward passes into the same leaves: every gradient doubles (a lost or duplicated update would show)
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=231)
    ref = _reference(inputs, seq_lens)
    cu = _cu(seq_lens)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    for _ in range(2):
        o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                         scale=1.0 / math.sqrt(HEAD_DIM),
                                         cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                                         cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64),
                                         use_qk_l2norm_in_kernel=True)  # fmt: skip
        (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    got = {"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")}
    doubled = {name: 2 * ref[name] for name in GRAD_NAMES}
    _assert_close_to_reference(got, doubled, GRAD_NAMES)


def _poison_workspaces(device) -> None:
    """NaN (or -1) in every pooled workspace of the package, and NaN in the blocks the caching allocator hands out next."""
    for table in cake_chain._POOLED.tables.values():
        for buffers in table.values():
            for t in buffers.values():
                t.fill_(float("nan") if t.is_floating_point() else -1)
    junk = [
        torch.full((size,), float("nan"), device=device)
        for size in (1 << 26, 1 << 22, 1 << 18)
    ]
    del junk


# The layouts of one sequence of calls: changed tokens, sequences and heads, then the first again.
POISON_SHAPES = (
    ((300, 77, 640), (2, 6)),
    ((2048, 100), (2, 4)),
    ((63, 64, 65, 500), (4, 8)),
    ((300, 77, 640), (2, 6)),
)


@pytest.mark.parametrize("host_cache", [True, False], ids=["pooled", "fresh"])
def test_changed_shapes_over_poisoned_workspaces(host_cache, device) -> None:
    for i, (seq_lens, heads) in enumerate(POISON_SHAPES):
        inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=233 + i)
        _poison_workspaces(device)
        got = _run(inputs, seq_lens, host_cache=host_cache)
        _assert_close_to_reference(got, _reference(inputs, seq_lens), OUTPUT_NAMES)
        _assert_tail_counters_zero()  # the self-cleaning tail buffers are not poisoned: they stay zero


def test_concurrent_caller_streams(device) -> None:
    # two caller streams run two problems at once (each with its own pooled workspaces), three times
    problems = (((300, 77, 640), (2, 6), 239), ((1000, 64), (2, 4), 241))
    data = [
        (seq_lens, _inputs(seq_lens, heads, torch.bfloat16, device, seed=seed))
        for seq_lens, heads, seed in problems
    ]
    want = [_run(inputs, seq_lens, tail="tail") for seq_lens, inputs in data]
    streams = [torch.cuda.Stream(device=device) for _ in data]
    for _ in range(3):
        got = []
        for stream, (seq_lens, inputs) in zip(streams, data, strict=True):
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                got.append(_run(inputs, seq_lens, sync=False, tail="tail"))
        torch.cuda.synchronize()
        for result, reference in zip(got, want, strict=True):
            _assert_within_budget(result, reference, OUTPUT_NAMES)
    # each caller stream took its own tail workspace (chosen at the launch), and left it zero
    index = cake_chain._device_index(device)
    keys = [(index, stream.cuda_stream) for stream in streams]
    assert len(
        {cake_chain._POOLED.zeroed[key]["tail_ws"].data_ptr() for key in keys}
    ) == len(keys)
    _assert_tail_counters_zero()


# (H, HV) and the call's keywords: the default tails (one cluster-tail launch, no counters) and the pinned stage
# "tail", whose launches count their tiles in self-cleaning counters taken from the call's workspaces
@pytest.mark.parametrize(
    ("heads", "pins"),
    [
        ((2, 4), {}),
        ((2, 6), {}),
        ((2, 4), {"tail": "tail"}),
        ((2, 6), {"tail": "tail"}),
    ],
    ids=["g2", "g3", "g2-tail", "g3-tail"],
)
def test_cuda_graph_replay_matches_eager(heads, pins, device, monkeypatch) -> None:
    seq_lens = STREAM_LAYOUT
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=269)
    eager = _run(inputs, seq_lens, streams=1, **pins)
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def step():
        for leaf in leaves.values():
            leaf.grad = None
        o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                         scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                                         cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True,
                                         streams=1, **pins)  # fmt: skip
        (o.float() * inputs["do"].float()).sum().backward()
        return o

    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream(device))
    # no backward plan of an earlier call: the warm-up's first backward takes its tails' workspaces itself
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
    warm = _recorded_tail_counters(monkeypatch, cake_chain._POOLED)
    with torch.cuda.stream(warmup):
        for _ in range(2):
            step()
    torch.cuda.current_stream(device).wait_stream(warmup)
    # the capture takes fresh workspaces: the tail counters its kernels use are its own, on the eager calls' route
    captured = _recorded_tail_counters(monkeypatch, cake_chain._FRESH)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = step()
    assert bool(captured) == bool(warm), (len(warm), len(captured))
    if pins.get("tail") == "tail":
        assert captured, "the captured tail took no counters of its own"
    for _ in range(
        3
    ):  # every replay alone: the captured kernels leave their scratch as they found it
        graph.replay()
        torch.cuda.synchronize()
        got = {
            "o": o,
            **{"d" + name: leaves[name].grad for name in ("q", "k", "v", "g", "beta")},
        }
        _assert_within_budget(got, eager, OUTPUT_NAMES)
        _assert_tail_counters_zero()
        for role, counter in captured:
            assert _bits_zero(counter), role


# --------------------------------------------------------------------------- round-once cluster tails, one packed row

# (H, HV) of the cluster tails: tail_g2_f32x (a value-head group of two) and tail_g3_f32x (three)
F32X_HEADS = {"g2": (2, 4), "g2-h4": (4, 8), "g3": (2, 6), "g3-h1": (1, 3)}


@pytest.mark.parametrize(
    "beta_dtype", [torch.float32, torch.bfloat16], ids=["fp32beta", "bf16beta"]
)
@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize("heads", list(F32X_HEADS.values()), ids=list(F32X_HEADS))
def test_cluster_tails_round_dq_dk_once(
    heads, route, beta_dtype, device, monkeypatch
) -> None:
    # the rule's cluster tail exchanges the group's FP32 dq / dk partials over distributed shared memory and rounds
    # dq / dk to BF16 once, after the FP32 sum in value-head order: the package tail's bits (the FP32 workspace slots
    # summed in value-head order) on every output, with an initial state and a final-state gradient; the
    # deterministic package's tail_g2 / tail_g3, which round each value head's partials first, share every other
    # output and differ in dq / dk; all within the budgets of the FP64 recurrence
    seq_lens = (1500, 77, 640, 1)
    group = heads[1] // heads[0]
    inputs = _inputs(seq_lens, heads, beta_dtype, device, seed=401)
    gen = torch.Generator().manual_seed(409)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    names = OUTPUT_NAMES + ("final_state", "dh0")
    kwargs = {
        "route": route,
        "initial_state": h0,
        "output_final_state": True,
        "dht": dht,
    }
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, **kwargs)
    form = f"tail_g{group}_f32x" + ("_bf16" if beta_dtype == torch.bfloat16 else "")
    assert form in stages and not set(DET_TAIL_STAGES) & set(stages), stages
    stages.clear()
    tail = _run(inputs, seq_lens, tail="tail", **kwargs)
    assert not set(F32X_TAIL_STAGES) & set(stages), stages
    _assert_same_bits(got, tail, names)
    per_head = _run(inputs, seq_lens, tail=f"tail_g{group}", **kwargs)
    _assert_same_bits(per_head, got, ("o", "dv", "dg", "dbeta", "final_state", "dh0"))
    assert not (
        torch.equal(per_head["dq"], got["dq"])
        and torch.equal(per_head["dk"], got["dk"])
    )
    _assert_close_to_reference(
        got, _reference(inputs, seq_lens, initial_state=h0, dht=dht), names
    )


@pytest.mark.parametrize("heads", [(2, 4), (2, 6)], ids=["g2", "g3"])
def test_cluster_tails_repeat_their_bits(heads, device) -> None:
    # five runs of the rule's route on the packed row, the same bits on every output
    seq_lens = (2000, 300, 77)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=419)
    first = _run_packed(inputs, seq_lens, heads)
    for _ in range(4):
        _assert_same_bits(_run_packed(inputs, seq_lens, heads), first, OUTPUT_NAMES)


def _packed_inputs(inputs, heads):
    """One contiguous ``[1, T, (2 H + HV) * 128]`` BF16 tensor of q / k / v (no grad) and the row widths."""
    h, hv = heads
    total = inputs["q"].shape[1]
    flat = [inputs[n].reshape(1, total, -1) for n in ("q", "k", "v")]
    return torch.cat(flat, -1).detach(), (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)


def _views(row, widths):
    return tuple(part.unflatten(-1, (-1, HEAD_DIM)) for part in row.split(widths, -1))


def _call(views, g, beta, seq_lens, **pins):
    cu = _cu(seq_lens)
    o, _ = cake_gdn_nondet(*views, g, beta, scale=1.0 / math.sqrt(HEAD_DIM),
                           cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=g.device),
                           cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64), use_qk_l2norm_in_kernel=True,
                           **pins)  # fmt: skip
    return o


def _row_results(o, row_grad, widths, g, beta) -> dict:
    dq, dk, dv = _views(row_grad, widths)
    return {
        "o": o.detach(),
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": g.grad,
        "dbeta": beta.grad,
    }


@pytest.mark.parametrize("pins", [{}, {"tail": "tail"}], ids=["rule", "tail-pin"])
@pytest.mark.parametrize(
    "heads", [(2, 6), (1, 3), (2, 4), (1, 4)], ids=["g3", "g3-h1", "g2", "g4"]
)
def test_packed_row_gives_the_view_bits(heads, pins, device, monkeypatch) -> None:
    # q / k / v as head views of one packed row that requires grad: under a cluster tail the row is the autograd input
    # (CakeGDNNondetPackedFunction) and the backward returns one packed gradient row the tail wrote, without the
    # split's concatenation; the same bits as the views route (PACKED_ROW off) on every output.  Another tail form or
    # group takes the views route
    seq_lens = (300, 77, 640, 1)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=421)
    data, widths = _packed_inputs(inputs, heads)
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    packed_route = (
        cake_chain.tail_form(*heads, pins.get("tail"), arch=arch)
        in cake_chain.F32X_TAILS
    )
    results, nodes = {}, {}
    for packed in (True, False):
        monkeypatch.setattr(cake_chain, "PACKED_ROW", packed)
        row = data.clone().requires_grad_(True)
        g, beta = (
            inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta")
        )
        o = _call(_views(row, widths), g, beta, seq_lens, **pins)
        nodes[packed] = type(o.grad_fn).__name__
        (o.float() * inputs["do"].float()).sum().backward()
        torch.cuda.synchronize()
        results[packed] = _row_results(o, row.grad, widths, g, beta)
    want_node = (
        "CakeGDNNondetPackedFunctionBackward"
        if packed_route
        else "CakeGDNNondetFunctionBackward"
    )
    assert (
        nodes[True] == want_node
        and nodes[False] != "CakeGDNNondetPackedFunctionBackward"
    ), nodes
    _assert_same_bits(results[True], results[False], OUTPUT_NAMES)
    _assert_same_bits(results[True], _run(inputs, seq_lens, **pins), OUTPUT_NAMES)


def test_packed_row_of_a_non_leaf_with_other_consumers(device, monkeypatch) -> None:
    # the row as a non-leaf (a projection's output) that another consumer also reads, two backward passes into the
    # projection's leaf: autograd sums the packed gradient row with the other consumer's gradient and accumulates both
    # passes; the same bits as the views route
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=431)
    data, widths = _packed_inputs(inputs, heads)
    weight = torch.randn(data.shape, generator=torch.Generator().manual_seed(433)).to(
        device
    )
    grads = {}
    for packed in (True, False):
        monkeypatch.setattr(cake_chain, "PACKED_ROW", packed)
        leaf = data.float().clone().requires_grad_(True)
        g, beta = (
            inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta")
        )
        for _ in range(2):
            row = leaf.to(torch.bfloat16)  # a non-leaf packed row
            o = _call(_views(row, widths), g, beta, seq_lens)
            assert (
                type(o.grad_fn).__name__ == "CakeGDNNondetPackedFunctionBackward"
            ) is packed
            loss = (o.float() * inputs["do"].float()).sum() + (
                row.float() * weight
            ).sum()
            loss.backward()
        torch.cuda.synchronize()
        grads[packed] = {"row": leaf.grad, "dg": g.grad, "dbeta": beta.grad}
    _assert_same_bits(grads[True], grads[False], ("row", "dg", "dbeta"))


def test_packed_row_without_a_row_gradient(device, monkeypatch) -> None:
    # only g and beta require grad: no packed route (the row records no gradient); their gradients have the bits of
    # the full backward's
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=439)
    data, widths = _packed_inputs(inputs, heads)
    full = _run_packed(inputs, seq_lens, heads)
    g, beta = (inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    o = _call(_views(data.clone(), widths), g, beta, seq_lens)
    assert type(o.grad_fn).__name__ == "CakeGDNNondetFunctionBackward"
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    _assert_same_bits(
        {"o": o.detach(), "dg": g.grad, "dbeta": beta.grad}, full, ("o", "dg", "dbeta")
    )


def test_packed_row_under_checkpoint_replay(device) -> None:
    # non-reentrant torch.utils.checkpoint over the packed row: the replayed forward takes the packed route again, the
    # gradients have the bits of the eager call, and the checkpointed forward keeps nothing beyond its output
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=443)
    data, widths = _packed_inputs(inputs, heads)
    row = data.clone().requires_grad_(True)
    g, beta = (inputs[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    nodes = []

    def layer(row, g, beta):
        o = _call(_views(row, widths), g, beta, seq_lens)
        nodes.append(type(o.grad_fn).__name__)
        return o

    def step(checkpointed):
        for leaf in (row, g, beta):
            leaf.grad = None
        o = (
            checkpoint(layer, row, g, beta, use_reentrant=False)
            if checkpointed
            else layer(row, g, beta)
        )
        (o.float() * inputs["do"].float()).sum().backward()
        torch.cuda.synchronize()
        return _row_results(o, row.grad, widths, g, beta)

    eager = step(False)
    for _ in range(2):  # the libraries, the tables and the workspaces of this layout
        replayed = step(True)
        _assert_same_bits(replayed, eager, OUTPUT_NAMES)
    assert set(nodes) == {"CakeGDNNondetPackedFunctionBackward"}, nodes
    for leaf in (row, g, beta):
        leaf.grad = None
    gc.collect()
    torch.cuda.synchronize()
    requested, _blocks = _allocator_bytes(device)
    o = checkpoint(layer, row, g, beta, use_reentrant=False)
    torch.cuda.synchronize()
    retained = _allocator_bytes(device)[0] - requested - o.untyped_storage().nbytes()
    assert retained == 0, (
        f"a checkpointed forward retains {retained} requested bytes beyond its output"
    )
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    _assert_same_bits(_row_results(o, row.grad, widths, g, beta), eager, OUTPUT_NAMES)


def test_packed_row_recognizes_only_its_layout() -> None:
    # the head views of one contiguous BF16 row that requires grad, at offsets 0, H x 128 and 2 H x 128 with its row
    # stride; not copies, other offsets, other widths, another dtype, a row without grad or views of a larger tensor
    h, hv, t = 2, 6, 5
    widths = (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)
    row = torch.zeros(1, t, sum(widths), dtype=torch.bfloat16, requires_grad=True)
    q, k, v = _views(row, widths)
    assert cake_chain._packed_row(q, k, v) is row
    batched = torch.zeros(3, t, sum(widths), dtype=torch.bfloat16, requires_grad=True)
    assert cake_chain._packed_row(*_views(batched, widths)) is batched
    assert cake_chain._packed_row(q.detach().clone().requires_grad_(), k, v) is None
    assert cake_chain._packed_row(k, q, v) is None  # q / k swapped: wrong offsets
    wide = torch.zeros(
        1, t, sum(widths) + HEAD_DIM, dtype=torch.bfloat16, requires_grad=True
    )
    assert cake_chain._packed_row(*_views(wide[..., : sum(widths)], widths)) is None
    assert (
        cake_chain._packed_row(*_views(row.detach().float().requires_grad_(), widths))
        is None
    )
    assert cake_chain._packed_row(*_views(row.detach(), widths)) is None
    larger = torch.zeros(2, t, sum(widths), dtype=torch.bfloat16, requires_grad=True)
    assert cake_chain._packed_row(*_views(larger[:1], widths)) is None


def test_row_stride_of_head_views() -> None:
    # the token-row stride the cluster tails take: heads x 128 for contiguous rows, the packed row's width for its head
    # views; a view whose heads or channels are not dense, or whose leading axes do not fold, is refused
    h, hv, t = 2, 6, 7
    widths = (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM)
    row = torch.zeros(3, t, sum(widths), dtype=torch.bfloat16)
    q, _k, v = _views(row, widths)
    assert cake_chain._row_stride(q) == cake_chain._row_stride(v) == sum(widths)
    assert cake_chain._row_stride(torch.zeros(1, t, hv, HEAD_DIM)) == hv * HEAD_DIM
    with pytest.raises(ValueError):
        cake_chain._row_stride(v.transpose(-1, -2))
    with pytest.raises(ValueError):
        cake_chain._row_stride(v[::2])


# --------------------------------------------------------------------------- the one-wave split of the adjoint walk

# The one-wave piece tables of the bench rows on a 148-SM device: (tokens per sequence, HV) -> (L, pieces per sequence)
PIECE_ROWS = {
    "issue": (((131071, 87531, 43385), 6), (158, (13, 9, 5))),
    "P3": (((128147, 131027), 6), (158, (13, 13))),
    "P1": (
        ((126565, 32714, 33696, 15081, 22979, 10782), 6),
        (132, (15, 4, 4, 2, 3, 2)),
    ),
    "h8": (((11000, 12000, 9768), 24), (63, (3, 3, 3))),
    "h4": (((11000, 12000, 9768), 12), (38, (5, 5, 5))),
    "h2": (((11000, 12000, 9768), 6), (20, (9, 10, 8))),
    "h1": (((11000, 12000, 9768), 3), (11, (16, 18, 14))),
    "A16": (((11000, 12000, 9768), 48), (94, (2, 2, 2))),
}


def _pieces(chunks, length) -> tuple[int, ...]:
    return tuple(len(cake_chain.segment_bounds(n, length)) - 1 for n in chunks)


@pytest.mark.parametrize("row", sorted(PIECE_ROWS))
def test_bwd_piece_table_on_the_bench_rows(row) -> None:
    (seq_lens, num_v_heads), (length, pieces) = PIECE_ROWS[row]
    chunks = tuple(-(-n // 64) for n in seq_lens)
    assert cake_chain.bwd_piece_table(chunks, num_v_heads, 148) == length
    assert _pieces(chunks, length) == pieces
    # one wave each: the first wave (last and middle pieces, one-piece sequences), the second (all but the last)
    assert sum(max(p - 1, 1) for p in pieces) * num_v_heads <= 148
    assert sum(p - 1 for p in pieces) * num_v_heads <= 148
    # the smallest such length
    assert sum(max(p - 1, 1) for p in _pieces(chunks, length - 1)) * num_v_heads > 148


def test_bwd_piece_table_rule() -> None:
    gen = torch.Generator().manual_seed(7007)
    for _ in range(200):
        num = int(torch.randint(1, 9, (), generator=gen))
        chunks = tuple(int(x) for x in torch.randint(0, 400, (num,), generator=gen))
        num_v_heads = int(torch.randint(1, 13, (), generator=gen))
        sm_count = int(torch.randint(8, 200, (), generator=gen))
        length = cake_chain.bwd_piece_table(chunks, num_v_heads, sm_count)

        def ctas(n_len: int, chunks=chunks, num_v_heads=num_v_heads) -> int:
            return sum(max(-(-n // n_len) - 1, 1) for n in chunks) * num_v_heads

        if len(chunks) * num_v_heads > sm_count or max(chunks) < 2:
            assert length is None
            continue
        smallest = next(n for n in range(1, max(chunks) + 1) if ctas(n) <= sm_count)
        assert length == (smallest if max(chunks) > smallest else None), (
            chunks,
            num_v_heads,
            sm_count,
        )
    # the rule reads the shapes, HV and the SM count only
    assert cake_chain.bwd_piece_table((), 4, 148) is None
    assert cake_chain.bwd_piece_table((2048,), 200, 148) is None
    assert cake_chain.bwd_piece_table((1, 1, 1), 4, 148) is None


def test_adjoint_split_entry_lists() -> None:
    # three sequences: one piece, three pieces, two pieces (segments [0, 1), [1, 4), [4, 6))
    seed, seq, tape, warm = (cake_chain.SPLIT_SEED, cake_chain.SPLIT_SEED_SEQ, cake_chain.SPLIT_TAPE,
                             cake_chain.SPLIT_WARM)  # fmt: skip
    walk_i, walk_m, summary_i, summary_m, front_i, front_m = (
        cake_chain.adjoint_split_lists([0, 1, 4, 6])
    )
    # the last pieces from dht, no start adjoint stored (the summaries carry it)
    assert walk_i == [0, 3, 5] and walk_m == [seed | seq | tape] * 3
    # a summary of every piece but each sequence's first, last pieces included, under the warm-up bound
    assert summary_i == [2, 3, 5] and summary_m == [warm] * 3
    assert front_i == [1, 2, 4] and front_m == [seed | tape] * 3
    # every segment's tapes are written once: by a last-piece walk or another piece's walk
    assert sorted(walk_i + front_i) == list(range(6))


@pytest.mark.parametrize("row", sorted(PIECE_ROWS))
def test_adjoint_split_pricing_on_the_bench_rows(row) -> None:
    (seq_lens, num_v_heads), (length, _) = PIECE_ROWS[row]
    chunks = tuple(-(-n // 64) for n in seq_lens)
    for arch in cake_chain.ADJOINT_SPLIT_COSTS:
        rec = cake_chain.adjoint_split_pricing(chunks, num_v_heads, 148, arch)
        assert (
            rec["segment_len"] == length
            and rec["serial_us"] > 0
            and rec["split_us"] > 0
        ), (row, rec)
        assert rec["select"] is (rec["split_us"] < rec["serial_us"]), (row, rec)
    # no table, no split: 64 sequences x 6 value heads exceed 148 SMs; an architecture without costs never splits
    rec = cake_chain.adjoint_split_pricing((64,) * 64, 6, 148, "sm_100a")
    assert rec["select"] is False and rec["segment_len"] == 0
    assert (
        cake_chain.adjoint_split_pricing(chunks, num_v_heads, 148, "sm_90a")["select"]
        is False
    )


def test_adjoint_split_one_stream_rule(monkeypatch) -> None:
    # wave 1 runs on one stream where the shorter of its two launches costs at most the host's fork and join (``fork``):
    # with GB200's costs and a fork cost of 65 us on 152 SMs the pair rows (1, 3) and (2, 6) (pieces of 10 and 20
    # chunks) take one stream and (4, 12) (38 chunks) two; the delivered entries without a fork cost keep two streams
    # on every bench row, sm_103a's (GB300) takes one stream at the h1 and h2 rows only (on 148 and on 152 SMs), and a
    # row without summaries (no sequence of three pieces) never forks
    chunks = tuple(-(-n // 64) for n in (11000, 12000, 9768))
    for arch, arch_costs in cake_chain.ADJOINT_SPLIT_COSTS.items():
        one_rows = {"h1", "h2"} if arch == "sm_103a" else set()
        assert (arch_costs["fork"] > 0.0) is bool(one_rows), (arch, arch_costs)
        for name, ((seq_lens, num_v_heads), _) in PIECE_ROWS.items():
            row = tuple(-(-n // 64) for n in seq_lens)
            rec = cake_chain.adjoint_split_pricing(row, num_v_heads, 148, arch)
            assert rec["one_stream"] is (name in one_rows), (arch, name, rec)
    for num_v_heads, one in {3: True, 6: True, 12: False, 24: False}.items():
        rec = cake_chain.adjoint_split_pricing(chunks, num_v_heads, 152, "sm_103a")
        assert rec["segment_len"] > 0 and rec["one_stream"] is one, (num_v_heads, rec)
    costs = {"walk": 2.38, "summary": 2.79, "serial_low": 2.18, "serial_full": 2.42, "stitch": 9.3,
             "stitch_piece": 3.1, "fixed": 15.0}  # fmt: skip
    # the delivered device entry of GB200 (sm_100a, 152 SMs) is these costs with a fork cost of 65 us
    assert cake_chain.adjoint_split_costs("sm_100a", 152) == {**costs, "fork": 65.0}
    assert (
        cake_chain.adjoint_split_costs("sm_100a", 148)
        is cake_chain.ADJOINT_SPLIT_COSTS["sm_100a"]
    )
    for num_v_heads, one in {3: True, 6: True, 12: False}.items():
        rec = cake_chain.adjoint_split_pricing(chunks, num_v_heads, 152, "sm_100a")
        assert rec["segment_len"] > 0 and rec["one_stream"] is one, (num_v_heads, rec)
    monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_DEVICE_COSTS", {})
    for fork, want in (
        (65.0, {3: True, 6: True, 12: False}),
        (0.0, {3: False, 6: False, 12: False}),
    ):
        monkeypatch.setattr(
            cake_chain, "ADJOINT_SPLIT_COSTS", {"sm_100a": {**costs, "fork": fork}}
        )
        for num_v_heads, one in want.items():
            rec = cake_chain.adjoint_split_pricing(chunks, num_v_heads, 152, "sm_100a")
            assert rec["segment_len"] > 0 and rec["one_stream"] is one, (
                fork,
                num_v_heads,
                rec,
            )
    monkeypatch.setattr(
        cake_chain, "ADJOINT_SPLIT_COSTS", {"sm_100a": {**costs, "fork": 1e9}}
    )
    rec = cake_chain.adjoint_split_pricing(chunks, 48, 152, "sm_100a")
    assert (
        _pieces(chunks, rec["segment_len"]) == (2, 2, 2) and rec["one_stream"] is False
    ), rec


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_host_plans_take_the_split_or_the_serial_table(arch, monkeypatch) -> None:
    # on an architecture of ADJOINT_SPLIT_ARCHS a host layout's adjoint table is the one-wave split or a serial table,
    # never a pre-scan's split table; plan pins keep the planner's tables
    if arch not in cake_chain.ADJOINT_SPLIT_ARCHS:
        pytest.skip(f"{arch} is not an architecture of ADJOINT_SPLIT_ARCHS")
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: (148, 232448, 10))
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    options = cake_chain.make_options()
    for row, ((seq_lens, num_v_heads), (length, _)) in PIECE_ROWS.items():
        chunks = tuple(-(-n // 64) for n in seq_lens)
        (_fwd, bwd), bwd_tc = cake_chain.host_plans(seq_lens, num_v_heads, 0, options)
        rec = cake_chain.adjoint_split_pricing(chunks, num_v_heads, 148, arch)
        if rec["select"]:
            assert (
                bwd == cake_chain.ScanPlan(cake_chain.SDT_CONFIG, length) and not bwd_tc
            ), row
        else:
            assert bwd.segment_len <= 0 or max(chunks) <= bwd.segment_len, (row, bwd)
    pinned = cake_chain.make_options(segment_chunks=64)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
        want = cake_chain.host_plans((131071, 87531, 43385), 6, 0, pinned)
    assert cake_chain.host_plans((131071, 87531, 43385), 6, 0, pinned) == want


# A layout the split cuts on a device of SPLIT_SMS SMs with HV 4: sequences of 11, 24 and 4 chunks take pieces of 4
# chunks (3, 6 and 1 pieces: wave 1 walks 3 pieces beside 5 summaries, wave 2 walks 7 pieces); partial last chunks.
SPLIT_LAYOUT = (700, 1500, 230)
SPLIT_HEADS = (2, 4)
SPLIT_SMS = 40
_SPLIT_STAGES = ("dhu_tc2_dvl32_modes", "dhu_tc2_dvl32_gm")


def _split_device(monkeypatch, device, *, split: bool, chain: str | None = None) -> str:
    """The small device of :func:`_small_device` with SPLIT_SMS SMs, the device's architecture in
    ADJOINT_SPLIT_ARCHS and costs that select the split (``split``) or the serial table; ``chain`` pins the stage that
    composes the end adjoints (``ADJOINT_CHAIN_STAGES``' preference otherwise)."""
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if not all(cake_chain._available(stage, arch) for stage in _SPLIT_STAGES):
        pytest.skip(f"the split's stages are not delivered for {arch}")
    if chain is not None:
        if not cake_chain._available(chain, arch):
            pytest.skip(f"{chain} is not delivered for {arch}")
        monkeypatch.setattr(cake_chain, "ADJOINT_CHAIN_STAGES", (chain,))
    if cake_chain._adjoint_chain_stage(arch) is None:
        pytest.skip(f"no chain stage is delivered for {arch}")
    _small_device(monkeypatch, device, SPLIT_SMS)
    monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset({arch}))
    costs = {"walk": 1.0, "summary": 1.0, "serial_low": 1.0, "serial_full": 1.0, "stitch": 0.0, "stitch_piece": 0.0,
             "fixed": 0.0 if split else 1e9}  # fmt: skip
    monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_COSTS", {arch: costs})
    monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_DEVICE_COSTS", {})
    monkeypatch.setattr(cake_plan, "ADJOINT_FORMS", types.MappingProxyType({}))
    # the two-stream route (its 10 pieces x 4 value heads would fit the 40 SMs in one wave)
    monkeypatch.setattr(cake_chain, "ONE_WAVE_SPLIT_ARCHS", frozenset())
    return arch


@pytest.mark.parametrize(
    "lpt, states, chain",
    [
        (True, True, None),
        (True, False, None),
        (False, True, None),
        (False, False, None),
        (True, True, "piece_chain"),
    ],
    ids=[
        "lpt-h0-dht",
        "lpt-no-states",
        "caller-order-h0-dht",
        "caller-order-no-states",
        "lpt-h0-dht-piece-chain",
    ],
)
def test_adjoint_split_matches_the_fp64_recurrence_and_the_serial_walk(
    lpt, states, chain, device, monkeypatch
) -> None:
    seq_lens, heads = SPLIT_LAYOUT, SPLIT_HEADS
    index = cake_chain._device_index(device)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7007)
    gen = torch.Generator().manual_seed(7008)
    h0 = dht = None
    if states:
        h0 = (
            0.1
            * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
        ).to(device)
        dht = torch.randn(
            len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen
        ).to(device)
    names = OUTPUT_NAMES + (("final_state", "dh0") if states else ())
    runs = {}
    for split in (True, False):
        with monkeypatch.context() as m:
            arch = _split_device(m, device, split=split, chain=chain)
            composer = cake_chain._adjoint_chain_stage(arch)
            options = cake_chain.make_options(lpt=lpt)
            cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
            meta = cake_chain._chunk_meta(cu.to(device), cu, 1, sum(seq_lens), index, heads[1], options,
                                          heads[1] // heads[0])  # fmt: skip
            if split:
                assert meta.bwd_split is not None and meta.bwd_segments.segment_len == 4
                assert meta.bwd_segments.num_segments == 10 and meta.bwd_groups is None
                assert (
                    meta.bwd_split.num_walk,
                    meta.bwd_split.num_summary,
                    meta.bwd_split.num_front,
                ) == (3, 7, 7)
            else:
                assert meta.bwd_split is None and meta.bwd_segments.num_segments == len(
                    seq_lens
                )
            launches = _record_streams(m)
            got = [_run(inputs, seq_lens, initial_state=h0, output_final_state=states, dht=dht, lpt=lpt)
                   for _ in range(3)]  # fmt: skip
            stages = [stage for stage, _ in launches]
            if split:
                main = torch.cuda.current_stream(device).cuda_stream
                assert (
                    stages.count("dhu_tc2_dvl32_modes") == 6
                    and stages.count("dhu_tc2_dvl32_gm") == 3
                ), stages
                # the backward's composition, once per run (the forward's plan may stitch too)
                assert stages.count(composer) >= 3, (composer, stages)
                if composer == "piece_chain":
                    assert stages.count("piece_chain") == 3, stages
                assert not [
                    s for s in stages if s.startswith(("dhu_pre", "dv_local", "dhu_r"))
                ], stages
                assert {
                    stream for stage, stream in launches if stage == "dhu_tc2_dvl32_gm"
                } != {main}
            else:
                assert not set(stages) & {"dhu_tc2_dvl32_modes", "dhu_tc2_dvl32_gm"}, (
                    stages
                )
            for run in got[1:]:
                _assert_same_bits(run, got[0], names)  # the same launches, run to run
            runs[split] = got[0]
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(runs[True], ref, names)
    _assert_within_budget(runs[True], runs[False], names)


def test_adjoint_split_dh0_sizing(device, monkeypatch) -> None:
    # the end adjoints of the split's pieces composed by the stitch (M x + B with hi / lo BF16 operand pairs: about 16
    # bits per piece boundary, from the third piece on) against piece_chain (the same maps and flags in FP32 FFMA), and
    # the stitch without the summaries' warm-up bound (warmup_log2 -1e30): dh0 per sequence (3, 6 and 1 pieces of 4
    # chunks) and over all, each against the FP64 recurrence; the SIZING lines (pytest -s) report the errors and the
    # accuracy gate holds every output
    seq_lens, heads = SPLIT_LAYOUT, SPLIT_HEADS
    index = cake_chain._device_index(device)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7021)
    gen = torch.Generator().manual_seed(7022)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    names = OUTPUT_NAMES + ("final_state", "dh0")
    runs = {}
    for label, chain, pins in (("stitch", "stitch", {}), ("stitch_nocut", "stitch", {"warmup_log2": -1e30}),
                               ("piece_chain", "piece_chain", {})):  # fmt: skip
        with monkeypatch.context() as m:
            arch = _split_device(m, device, split=True, chain=chain)
            cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
            meta = cake_chain._chunk_meta(cu.to(device), cu, 1, sum(seq_lens), index, heads[1],
                                          cake_chain.make_options(), heads[1] // heads[0])  # fmt: skip
            assert meta.bwd_split is not None and meta.bwd_segments.segment_len == 4
            assert meta.bwd_segments.num_segments == 10
            launches = _record_streams(m)
            runs[label] = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, **pins)  # fmt: skip
            assert chain in [stage for stage, _ in launches], (chain, launches)
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    pieces = [((n + 63) // 64 + 3) // 4 for n in seq_lens]  # (3, 6, 1)
    assert max(pieces) >= 3, pieces
    tag = f"bwd_dh0 {arch} piece_len=4"
    for i, n in enumerate(seq_lens):
        for chain, run in runs.items():
            _sizing(
                f"{tag} seq={i} tokens={n} pieces={pieces[i]} {chain}_vs_fp64",
                run["dh0"][i],
                ref["dh0"][i],
            )
        _sizing(f"{tag} seq={i} tokens={n} pieces={pieces[i]} stitch_vs_piece_chain", runs["stitch"]["dh0"][i],
                runs["piece_chain"]["dh0"][i])  # fmt: skip
    for chain, run in runs.items():
        _sizing(f"{tag} all {chain}_vs_fp64", run["dh0"], ref["dh0"])
    _sizing(
        f"{tag} all stitch_vs_piece_chain",
        runs["stitch"]["dh0"],
        runs["piece_chain"]["dh0"],
    )
    for run in runs.values():
        _assert_close_to_reference(run, ref, names)
    _assert_within_budget(runs["stitch"], runs["piece_chain"], names)


@pytest.mark.parametrize("how", ["fork_cost", "streams_1"])
def test_adjoint_split_one_stream_gives_the_two_stream_bits(
    how, device, monkeypatch
) -> None:
    # wave 1 on the caller's stream, chosen by the pricing's fork cost or by a caller that keeps every launch on its
    # stream, runs the summaries then the walks there: the same bits as the summaries on the side stream
    seq_lens, heads = SPLIT_LAYOUT, SPLIT_HEADS
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7011)
    main = torch.cuda.current_stream(device).cuda_stream
    runs, gm_streams = {}, {}
    for one in (False, True):
        with monkeypatch.context() as m:
            arch = _split_device(m, device, split=True)
            pins = {}
            if one and how == "fork_cost":
                m.setitem(cake_chain.ADJOINT_SPLIT_COSTS[arch], "fork", 1e9)
            if one and how == "streams_1":
                pins["streams"] = 1
            launches = _record_streams(m)
            runs[one] = _run(inputs, seq_lens, **pins)
            stages = [stage for stage, _ in launches]
            assert (
                stages.count("dhu_tc2_dvl32_gm") == 1
                and stages.count("dhu_tc2_dvl32_modes") == 2
            ), stages
            gm_streams[one] = {
                stream for stage, stream in launches if stage == "dhu_tc2_dvl32_gm"
            }
            # wave 1's walks follow its summaries on one stream; on two streams they are launched after them too
            assert stages.index("dhu_tc2_dvl32_gm") < stages.index(
                "dhu_tc2_dvl32_modes"
            ), stages
    assert gm_streams[True] == {main} and main not in gm_streams[False], gm_streams
    _assert_same_bits(runs[True], runs[False], OUTPUT_NAMES)


def test_adjoint_split_one_wave_tables() -> None:
    # the one-wave route's table: every segment, seeded from its own end adjoint with tapes, after the other parts
    seed, tape = cake_chain.SPLIT_SEED, cake_chain.SPLIT_TAPE
    split = cake_chain.adjoint_split_tables([0, 1, 4, 6], 4, "cpu", one_wave=True)
    assert split.one_wave and split.num_every == 6
    assert split.every.tolist() == [list(range(6)), [seed | tape] * 6]
    assert (split.num_walk, split.num_summary, split.num_front) == (3, 3, 3)
    plain = cake_chain.adjoint_split_tables([0, 1, 4, 6], 4, "cpu")
    assert not plain.one_wave and plain.every is None and plain.num_every == 0


@pytest.mark.parametrize(
    "lpt, states",
    [(True, True), (False, False)],
    ids=["lpt-h0-dht", "caller-order-no-states"],
)
def test_adjoint_split_one_wave_gives_the_two_stream_bits(
    lpt, states, device, monkeypatch
) -> None:
    # every piece of the split layout fits the small device in one wave (10 pieces x 4 value heads = 40 SMs): with
    # the one-wave route the summaries, the stitch and one walk of every piece run on the caller's stream, and the
    # gradients are the two-stream route's bits (the last pieces start from the stitch's copy of dht, or zeros)
    seq_lens, heads = SPLIT_LAYOUT, SPLIT_HEADS
    index = cake_chain._device_index(device)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7031)
    gen = torch.Generator().manual_seed(7032)
    h0 = dht = None
    if states:
        h0 = (
            0.1
            * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
        ).to(device)
        dht = torch.randn(
            len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen
        ).to(device)
    names = OUTPUT_NAMES + (("final_state", "dh0") if states else ())
    main = torch.cuda.current_stream(device).cuda_stream
    runs = {}
    for one_wave in (False, True):
        with monkeypatch.context() as m:
            arch = _split_device(m, device, split=True)
            if cake_chain._adjoint_chain_stage(arch) != "stitch":
                pytest.skip(f"the one-wave route needs the stitch on {arch}")
            if one_wave:
                m.setattr(cake_chain, "ONE_WAVE_SPLIT_ARCHS", frozenset({arch}))
            options = cake_chain.make_options(lpt=lpt)
            cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
            meta = cake_chain._chunk_meta(cu.to(device), cu, 1, sum(seq_lens), index, heads[1], options,
                                          heads[1] // heads[0])  # fmt: skip
            split = meta.bwd_split
            assert split is not None and split.one_wave is one_wave, split
            assert split.num_every == (10 if one_wave else 0), split
            launches = _record_streams(m)
            got = [_run(inputs, seq_lens, initial_state=h0, output_final_state=states, dht=dht, lpt=lpt)
                   for _ in range(2)]  # fmt: skip
            stages = [stage for stage, _ in launches]
            assert stages.count("dhu_tc2_dvl32_gm") == 2, stages
            assert stages.count("dhu_tc2_dvl32_modes") == (2 if one_wave else 4), stages
            split_streams = {
                stream for stage, stream in launches if stage in _SPLIT_STAGES
            }
            assert (split_streams == {main}) is one_wave, split_streams
            _assert_same_bits(got[1], got[0], names)
            runs[one_wave] = got[0]
    _assert_same_bits(runs[True], runs[False], names)


def _record_gm_flags(monkeypatch) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(seg_index, seg_flag) of every ``dhu_tc2_dvl32_gm`` launch from here on (its last argument and the fourth from
    last in the record's order)."""
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    launch = cake_backend.launch_stage

    def recording(stage, grid, values, **kwargs):
        if stage == "dhu_tc2_dvl32_gm":
            seen.append((values[-4], values[-1]))
        return launch(stage, grid, values, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return seen


@pytest.mark.parametrize("gate_scale", [1.0, 0.002], ids=["strong", "weak"])
def test_adjoint_split_warm_bound(gate_scale, device, monkeypatch) -> None:
    # the summaries' warm-up bound (SPLIT_WARM, the threshold of warmup_log2): under strong gates every summary stops
    # early (seg_flag 0) and the gradients stay within the budgets of the unbounded split (warmup_log2 -1e30, every
    # flag 1) and of the FP64 recurrence; under weak gates (about e^-0.06 per chunk) no piece is cut, so the bound
    # changes no bit; run to run the same bits either way
    seq_lens, heads = SPLIT_LAYOUT, SPLIT_HEADS
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7021)
    inputs["g"] = (inputs["g"] * (gate_scale / 0.5)).contiguous()
    runs, flags = {}, {}
    for bounded in (True, False):
        with monkeypatch.context() as m:
            _split_device(m, device, split=True)
            seen = _record_gm_flags(m)
            pins = {} if bounded else {"warmup_log2": -1.0e30}
            got = [_run(inputs, seq_lens, **pins) for _ in range(2)]
            assert seen, "the split ran no summary"
            rows, flag = seen[-1]
            flags[bounded] = flag[rows.long()].cpu()
        _assert_same_bits(got[1], got[0], OUTPUT_NAMES)
        runs[bounded] = got[0]
    assert bool((flags[False] == 1).all()), flags[False]
    if gate_scale >= 1.0:
        assert bool((flags[True] == 0).all()), flags[True]
        _assert_within_budget(runs[True], runs[False], OUTPUT_NAMES)
        _assert_close_to_reference(
            runs[True], _reference(inputs, seq_lens), OUTPUT_NAMES
        )
    else:
        assert bool((flags[True] == 1).all()), flags[True]
        _assert_same_bits(runs[True], runs[False], OUTPUT_NAMES)


def test_adjoint_forms_keep_gb200_h1_hv3_off_the_split(monkeypatch) -> None:
    # cake_plan.ADJOINT_FORMS, frozen: GB200 (sm_100a, 152 SMs) at H 1 / HV 3 keeps the planner's tables (no one-wave
    # split, as on an architecture outside ADJOINT_SPLIT_ARCHS; the warm-up cut kept) although the pricing takes the
    # split there (on one stream); H 2 / HV 6, H 4 / HV 12, other SM counts and architectures, a call without its group
    # and a pinned call keep the pricing
    assert dict(cake_plan.ADJOINT_FORMS) == {("sm_100a", 152, 1, 3): "nosplit"}
    for key in (
        ("sm_100a", 148, 1, 3),
        ("sm_100a", 152, 2, 6),
        ("sm_100a", 152, 4, 12),
        ("sm_100a", 152, 8, 24),
        ("sm_103a", 152, 1, 3),
        ("sm_100a", 152, 1, 6),
    ):
        assert cake_plan.adjoint_form(*key) is None, key
    chunks = tuple(-(-n // 64) for n in (11000, 12000, 9768))
    rec = cake_chain.adjoint_split_pricing(chunks, 3, 152, "sm_100a")
    assert rec["segment_len"] > 0 and rec["one_stream"] is True, rec
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: (152, 232448, 10))
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: "sm_100a")
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    options = cake_chain.make_options()
    seq_lens = (11000, 12000, 9768)
    got = cake_chain._host_plans(seq_lens, 3, 0, options, group=3)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
        want = cake_chain._host_plans(seq_lens, 3, 0, options, group=3)
    assert got == want
    assert cake_chain._adjoint_form(0, 3, 3, options) == "nosplit"
    assert cake_chain._adjoint_split_length(seq_lens, 3, 0, options, 3) is None
    if all(cake_chain._available(stage, "sm_100a") for stage in _SPLIT_STAGES):
        assert cake_chain._adjoint_split_length(seq_lens, 3, 0, options) is not None
        for num_v_heads in (6, 12):
            assert (
                cake_chain._adjoint_split_length(seq_lens, num_v_heads, 0, options, 3)
                is not None
            ), num_v_heads
    for num_v_heads in (6, 12):
        assert cake_chain._adjoint_form(0, num_v_heads, 3, options) is None
    assert cake_chain._adjoint_form(0, 3, None, options) is None
    assert (
        cake_chain._adjoint_form(0, 3, 3, cake_chain.make_options(segment_chunks=64))
        is None
    )
    assert cake_chain._adjoint_form(0, 6, 2, options) is None


def test_adjoint_form_gives_the_bits_of_no_split(device, monkeypatch) -> None:
    # a call that the plan table's "nosplit" form serves makes the launches (stages, grids, arguments) and the bits of
    # the planner without the split (ADJOINT_SPLIT_ARCHS empty), forward and backward: wave 1's summaries never run
    seq_lens, heads = (700, 1500, 230), (1, 3)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7013)
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    runs, logs = {}, {}
    for form in (True, False):
        with monkeypatch.context() as m:
            _split_device(m, device, split=True)
            if form:
                m.setattr(
                    cake_plan,
                    "ADJOINT_FORMS",
                    types.MappingProxyType({(arch, SPLIT_SMS, 1, 3): "nosplit"}),
                )
            else:
                m.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset())
            logs[form] = _logged_launches(m)
            runs[form] = _run(inputs, seq_lens)
            stages = [entry[0] for entry in logs[form]]
            assert "dhu_tc2_dvl32_gm" not in stages, stages
    assert logs[True] == logs[False]
    _assert_same_bits(runs[True], runs[False], OUTPUT_NAMES)
    # without the form the pricing splits this layout (wave 1's summaries run)
    with monkeypatch.context() as m:
        _split_device(m, device, split=True)
        launches = _logged_launches(m)
        _run(inputs, seq_lens)
    assert "dhu_tc2_dvl32_gm" in [entry[0] for entry in launches]


def test_adjoint_split_packed_views_give_the_contiguous_bits(
    device, monkeypatch
) -> None:
    seq_lens, heads = SPLIT_LAYOUT, SPLIT_HEADS
    _split_device(monkeypatch, device, split=True)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7009)
    want = _run(inputs, seq_lens)
    got = _run_packed(inputs, seq_lens, heads)
    _assert_same_bits(got, want, OUTPUT_NAMES)


# --------------------------------------------------------------------------- the backward plan


def _logged_launches(monkeypatch) -> list:
    """Every launch from here on as (stage, grid, stream, the values: a tensor by its shape, strides and dtype, a
    scalar itself)."""
    log = []
    launch = cake_backend.launch_stage

    def logged(stage, grid, values, **kwargs):
        stream = torch.cuda.current_stream(kwargs["device_index"]).cuda_stream
        args = tuple(
            (tuple(v.shape), v.stride(), v.dtype) if isinstance(v, torch.Tensor) else v
            for v in values
        )
        log.append((stage, tuple(grid), stream, args))
        return launch(stage, grid, values, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", logged)
    return log


def _counted_replays(monkeypatch) -> list:
    """The plan of every backward that replays one (:func:`cake_chain._replay`), in order."""
    replays = []
    replay = cake_chain._replay

    def counted(plan, *args):
        replays.append(plan)
        return replay(plan, *args)

    monkeypatch.setattr(cake_chain, "_replay", counted)
    return replays


# case -> (sequence lengths, or None for an equal-length batch of two 640-token sequences; (H, HV); "packed" for the
# head views of one packed row, else contiguous q / k / v; the call's keywords, "h0" for an initial state)
BWD_PLAN_CASES = {
    "packed-g3": ((300, 77, 640, 17), (2, 6), "packed", {}),
    "packed-g2": ((300, 77, 640, 17), (2, 4), "packed", {}),
    "g3-h1-h0": ((64, 1, 2049), (1, 3), "views", {"h0": True}),
    "tail-one-stream": (TAIL_LAYOUT, (2, 6), "views", {"tail": "tail", "streams": 1}),
    "tail_g2": (TAIL_LAYOUT, (2, 4), "views", {"tail": "tail_g2"}),
    "unfused": (TAIL_LAYOUT, (2, 4), "views", {"tail": "unfused"}),
    "equal-length": (None, (2, 6), "views", {}),
}


def _plan_case(case, device):
    """The inputs and the forward + backward of a :data:`BWD_PLAN_CASES` case, and the names it returns."""
    seq_lens, heads, form, pins = BWD_PLAN_CASES[case]
    pins = dict(pins)
    layout, batch = (seq_lens, 1) if seq_lens is not None else ((640,), 2)
    inputs = _inputs(layout, heads, torch.bfloat16, device, seed=7101, batch=batch)
    h0 = None
    if pins.pop("h0", False):
        gen = torch.Generator().manual_seed(7102)
        h0 = (
            0.1 * torch.randn(len(layout), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
        ).to(device)

    def run():
        if form == "packed":
            return _run_packed(inputs, seq_lens, heads, **pins)
        return _run(inputs, seq_lens, initial_state=h0, **pins)

    return run, OUTPUT_NAMES + (("dh0",) if h0 is not None else ())


@pytest.mark.parametrize("case", list(BWD_PLAN_CASES), ids=list(BWD_PLAN_CASES))
def test_the_backward_plan_replays_the_general_path(case, device, monkeypatch) -> None:
    # resolve before launch: the first backward of a key runs the general path and records it as a plan, the later
    # ones replay the plan: the launches of the general path (BWD_PLAN off; stage, grid, stream, every value's shape,
    # strides and dtype, every scalar) and its bits on every output
    run, names = _plan_case(case, device)
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
    log = _logged_launches(monkeypatch)
    replays = _counted_replays(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "BWD_PLAN", False)
        general = run()
    assert not cake_chain._BWD_PLANS
    general_launches = list(log)
    log.clear()
    recorded = run()
    assert log == general_launches and not replays
    (plan,) = cake_chain._BWD_PLANS.values()
    assert plan.ops is not None, "this call's backward has a plan"
    log.clear()
    runs = [run() for _ in range(2)]
    assert log == general_launches * 2
    assert len(replays) == 2 and all(p is plan for p in replays)
    for got in (recorded, *runs):
        _assert_same_bits(got, general, names)


def test_a_backward_plan_replays_only_for_its_heads(device, monkeypatch) -> None:
    # H = 2 and H = 1 at HV = 4 differ in the q / k heads only (the gate shapes, v's strides and the pins agree), and
    # on sm_100a this layout's chunk tables are one ChunkMeta for both groups: each call replays only the plan of its
    # own heads, in either order, with the general path's bits (a plan of the other H returns dq / dk of its shape)
    seq_lens, pins = TAIL_LAYOUT, {"tail": "tail"}
    inputs = {
        heads: _inputs(seq_lens, heads, torch.float32, device, seed=7109)
        for heads in ((2, 4), (1, 4))
    }
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "BWD_PLAN", False)
        general = {heads: _run(x, seq_lens, **pins) for heads, x in inputs.items()}
    for order in (((2, 4), (1, 4)), ((1, 4), (2, 4))):
        monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
        replays = _counted_replays(monkeypatch)
        for heads in order:
            # the general path records the plan of the key, the next call replays it
            for _ in range(2):
                _assert_same_bits(
                    _run(inputs[heads], seq_lens, **pins), general[heads], OUTPUT_NAMES
                )
        plans = list(cake_chain._BWD_PLANS.values())
        assert len(plans) == 2 and all(p.ops is not None for p in plans), plans
        assert len(replays) == 2 and all(
            a is b for a, b in zip(replays, plans, strict=True)
        )


def _saved_layouts(monkeypatch) -> list:
    """The layout (:func:`cake_chain._slot_layout`) of the saved tensors and ``do`` of every backward, in order."""
    layouts = []
    backward = cake_chain._function_backward

    def recording(ctx, do, *args):
        layouts.append(cake_chain._slot_layout((*ctx.saved_tensors, do)))
        return backward(ctx, do, *args)

    monkeypatch.setattr(cake_chain, "_function_backward", recording)
    return layouts


def test_a_backward_plan_replays_only_for_its_saved_layout(device, monkeypatch) -> None:
    # under torch.autograd.graph.save_on_cpu the backward reads the saved tensors as the hook unpacks them: the
    # packed row's strided v comes back contiguous while the forward's key names the strided view.  Calls with and
    # without the hook alternate, in either order: each replays only a plan of its own saved layout and gives the
    # general path's bits
    seq_lens, heads = BWD_PLAN_CASES["packed-g3"][:2]
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7111)

    def run(offload):
        hook = (
            torch.autograd.graph.save_on_cpu() if offload else contextlib.nullcontext()
        )
        with hook:
            return _run_packed(inputs, seq_lens, heads)

    layouts = _saved_layouts(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "BWD_PLAN", False)
        general = {offload: run(offload) for offload in (False, True)}
    assert layouts[0] != layouts[1], (
        "the hook gives the backward saved tensors of another layout"
    )
    for order in ((False, True), (True, False)):
        monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
        replays = _counted_replays(monkeypatch)
        for offload in (*order, order[0]):
            for _ in range(2):
                _assert_same_bits(run(offload), general[offload], OUTPUT_NAMES)
        assert replays, "calls of one layout still replay their plan"


def test_a_backward_plan_takes_a_misaligned_do(device, monkeypatch) -> None:
    # the loss reads o through a concatenation behind one element, so o's gradient is a contiguous slice 2 bytes past
    # a 16-byte boundary: after two calls with an aligned do (the plan and its replay), the call with that do replays
    # the plan on the aligned copy the general path reads, with the general path's bits
    seq_lens, heads = TAIL_LAYOUT, (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7113)
    cu = _cu(seq_lens)

    def run(misaligned):
        leaves = {
            n: inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        }
        o, _ = cake_gdn_nondet(*(leaves[n] for n in ("q", "k", "v", "g", "beta")), scale=1.0 / math.sqrt(HEAD_DIM),
                               cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                               cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64),
                               use_qk_l2norm_in_kernel=True)  # fmt: skip
        do = inputs["do"].to(o.dtype)
        if misaligned:
            pad = torch.zeros(1, dtype=o.dtype, device=device)
            weight = torch.cat([pad, do.flatten()])
            (torch.cat([pad, o.flatten()]) * weight).float().sum().backward()
        else:
            o.backward(do)
        torch.cuda.synchronize()
        return {
            "o": o.detach(),
            **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
        }

    layouts = _saved_layouts(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "BWD_PLAN", False)
        general = {misaligned: run(misaligned) for misaligned in (False, True)}
    do_stride, do_offset = layouts[1][-1][1], layouts[1][-1][3]
    assert do_offset != 0 and do_stride == layouts[0][-1][1], layouts[1][-1]
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
    replays = _counted_replays(monkeypatch)
    for misaligned in (False, False, True, False):
        _assert_same_bits(run(misaligned), general[misaligned], OUTPUT_NAMES)
    assert len(replays) == 3 and len(set(map(id, replays))) == 1


def test_the_backward_plan_holds_no_tensor_of_a_call(device, monkeypatch) -> None:
    # a plan's tensors are views of the stream's pooled workspaces and of the layout's chunk tables: a replayed call
    # leaves the device memory where it was once its outputs and gradients are dropped
    seq_lens, heads = (300, 77, 640, 17), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7103)
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
    replays = _counted_replays(monkeypatch)
    for _ in range(2):  # the workspaces, the tables and the plan
        _run_packed(inputs, seq_lens, heads)
    (plan,) = cake_chain._BWD_PLANS.values()
    constants: set = set()
    for buffers in cake_chain._POOLED.tables[plan.stream].values():
        cake_chain._storages(list(buffers.values()), constants)
    cake_chain._storages(
        list(cake_chain._POOLED.zeroed.get(plan.stream, {}).values()), constants
    )
    cake_chain._storages(plan.meta, constants)
    held = [
        v
        for op in plan.ops
        if op[0] == cake_chain._OP_LAUNCH
        for v in op[4]
        if isinstance(v, torch.Tensor)
    ]
    assert held and all(v.untyped_storage().data_ptr() in constants for v in held)
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(device)
    got = _run_packed(inputs, seq_lens, heads)
    assert len(replays) == 2
    del got
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(device) == before


def test_the_backward_plan_follows_the_module_globals_and_the_pool(
    device, monkeypatch
) -> None:
    # a plan replays only with the module globals, cake_backend.device_arch and the stream's pool tables it was
    # recorded with: another object in their place, or a cleared pool, sends the call to the general path, which
    # records a new plan; the bits never change.  The package tail ("tail") takes zeroed buffers of the pool too
    seq_lens, heads = (300, 77, 640, 17), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7104)
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
    replays = _counted_replays(monkeypatch)

    def run():
        return _run(inputs, seq_lens, tail="tail")

    want = run()
    arch_of, segments = cake_backend.device_arch, cake_chain.tail_segments
    changes = (
        lambda: monkeypatch.setattr(
            cake_backend, "device_arch", lambda index: arch_of(index)
        ),
        lambda: monkeypatch.setattr(
            cake_chain, "tail_segments", lambda *a, **k: segments(*a, **k)
        ),
        lambda: cake_chain._POOLED.tables.clear(),
        lambda: cake_chain._POOLED.zeroed.clear(),
    )
    _assert_same_bits(run(), want, OUTPUT_NAMES)
    assert len(replays) == 1
    for i, change in enumerate(changes):
        change()
        _assert_same_bits(run(), want, OUTPUT_NAMES)  # the general path records
        assert len(replays) == 1 + i, i
        _assert_same_bits(run(), want, OUTPUT_NAMES)
        assert len(replays) == 2 + i, i


@pytest.mark.parametrize(
    "why", ["device-route", "no-host-cache", "no-save-state", "dht", "off"]
)
def test_calls_without_a_backward_plan_keep_the_general_path(
    why, device, monkeypatch
) -> None:
    # no plan for the device route (a new table per call), without the host cache or the kept tape, with a
    # final-state gradient, or with BWD_PLAN off: every backward runs the general path, run to run the same bits
    seq_lens, heads = (300, 77, 640, 17), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7105)
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
    replays = _counted_replays(monkeypatch)
    kwargs: dict = {
        "device-route": {"route": "device"},
        "no-host-cache": {"host_cache": False},
        "no-save-state": {"save_state": False},
        "off": {},
    }.get(why, {})
    names = OUTPUT_NAMES
    if why == "dht":
        gen = torch.Generator().manual_seed(7106)
        dht = torch.randn(
            len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen
        ).to(device)
        kwargs = {"output_final_state": True, "dht": dht}
        names = OUTPUT_NAMES + ("final_state",)
    elif why == "off":
        monkeypatch.setattr(cake_chain, "BWD_PLAN", False)
    runs = [_run(inputs, seq_lens, **kwargs) for _ in range(3)]
    assert not replays and not cake_chain._BWD_PLANS
    for got in runs[1:]:
        _assert_same_bits(got, runs[0], names)


def test_the_backward_plan_under_activation_checkpointing(device, monkeypatch) -> None:
    # torch.utils.checkpoint (no reentrant): the backward recomputes the forward for its saved list and replays the
    # plan of the first forward's key; the bits of the general path
    seq_lens, heads = (300, 77, 640, 17), (2, 6)
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=7107)
    cu = _cu(seq_lens)
    cu_dev, cu_cpu = (
        torch.tensor(cu, dtype=torch.int64, device=device),
        torch.tensor(cu, dtype=torch.int64),
    )
    monkeypatch.setattr(cake_chain, "_BWD_PLANS", {})
    replays = _counted_replays(monkeypatch)

    def f(q, k, v, g, beta):
        o, _ = cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_dev,
                               cu_seqlens_cpu=cu_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip
        return o

    def step():
        leaves = [
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        o = checkpoint(f, *leaves, use_reentrant=False)
        (o.float() * inputs["do"].float()).sum().backward()
        torch.cuda.synchronize()
        return {
            "o": o.detach(),
            **{name: t.grad for name, t in zip(GRAD_NAMES, leaves, strict=True)},
        }

    with monkeypatch.context() as m:
        m.setattr(cake_chain, "BWD_PLAN", False)
        want = step()
    runs = [step() for _ in range(3)]
    assert len(replays) == 2
    for got in runs:
        _assert_same_bits(got, want, OUTPUT_NAMES)


# --------------------------------------------------------------------------- the forward on the WY stage's A
# Layouts whose forward segment table is serial on every device of the package (more sequences x value heads than SMs,
# 192 CTAs): partial chunks, a one-token sequence, value-head groups of three, two and one.
FWD_A_CASES = {
    "g3": ((16, 48), (300, 77, 1, 130)),
    "g2": ((16, 32), (300, 77, 1, 130, 64, 65)),
    "g1": ((16, 16), (300, 77, 1, 130, 64, 65, 200, 17, 129, 5, 63, 250)),
}


def _fwd_a_or_skip(device) -> str:
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if arch not in cake_chain.FWD_A_ARCHS:
        pytest.skip(f"the forward on the WY stage's A is not routed on {arch}")
    return arch


def _record_launch_values(monkeypatch) -> list[tuple[str, tuple]]:
    """The (stage, values) of every launch from here on, in order."""
    launches: list[tuple[str, tuple]] = []
    launch = cake_backend.launch_stage

    def recording(stage, grid, values, **kwargs):
        launches.append((stage, tuple(values)))
        return launch(stage, grid, values, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_stage", recording)
    return launches


def _tape_words(launches) -> list[int]:
    """The tape word of every fwd_ho_a launch (its last argument at the call site, wherever the record puts it)."""
    names = cake_backend.parameters(
        cake_backend.stage_record(
            "fwd_ho_a", cake_backend.device_arch(torch.cuda.current_device())
        )
    )
    return [
        int(values[names.index("tape")])
        for stage, values in launches
        if stage == "fwd_ho_a"
    ]


def test_fwd_a_route_rule(monkeypatch) -> None:
    # the forward on the WY stage's A takes a host-built table whose forward segment table is serial, FP32 beta,
    # normalized q / k, a value-head group of at most three, no prep pin and 16-byte aligned q / k
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)

    def meta(seq_lens, segment_len=0, num_segments=None):
        segs = types.SimpleNamespace(
            segment_len=segment_len,
            num_segments=len(seq_lens) if num_segments is None else num_segments,
        )
        return types.SimpleNamespace(
            seq_lens=seq_lens, num_seqs=len(seq_lens), fwd_segments=segs
        )

    def tensors(h, hv, beta_dtype=torch.float32):
        q = torch.empty(1, 64, h, HEAD_DIM, dtype=torch.bfloat16, device="meta")
        v = torch.empty(1, 64, hv, HEAD_DIM, dtype=torch.bfloat16, device="meta")
        return q, q, v, torch.empty(1, 64, hv, dtype=beta_dtype, device="meta")

    whole = meta((300, 77, 1, 130))
    route = cake_chain.fwd_a_route
    opts = cake_chain.make_options()
    assert route(whole, "sm_100a", *tensors(16, 48), True, opts)
    assert route(whole, "sm_103a", *tensors(16, 48), True, opts)
    assert route(whole, "sm_107a", *tensors(16, 48), True, opts)
    assert not route(whole, "sm_120a", *tensors(16, 48), True, opts)
    assert not route(whole, "sm_100a", *tensors(16, 48, torch.bfloat16), True, opts)
    assert not route(whole, "sm_100a", *tensors(16, 48), False, opts)
    assert not route(whole, "sm_100a", *tensors(12, 48), True, opts)  # a group of four
    assert not route(
        whole, "sm_100a", *tensors(16, 48), True, cake_chain.make_options(prep="tma")
    )
    # long sequences on a serial forward segment table route; the same lengths on a split table do not
    assert route(meta((11000, 12000, 9768)), "sm_100a", *tensors(16, 48), True, opts)
    assert not route(
        meta((11000, 12000, 9768), 64, 9), "sm_100a", *tensors(16, 48), True, opts
    )
    assert not route(
        meta((131071, 87531, 43385), 128, 33), "sm_100a", *tensors(2, 6), True, opts
    )
    assert route(meta((4096,) * 64), "sm_100a", *tensors(2, 6), True, opts)
    assert route(meta((64, 17, 1)), "sm_100a", *tensors(1, 3), True, opts)
    # a split table of short sequences and the device-built table (no lengths) keep the forward of the prep
    assert not route(
        meta((4096,) * 64, 16, 1024), "sm_100a", *tensors(2, 6), True, opts
    )
    device_built = types.SimpleNamespace(
        seq_lens=None,
        num_seqs=2,
        fwd_segments=types.SimpleNamespace(segment_len=0, num_segments=2),
    )
    assert not route(device_built, "sm_100a", *tensors(2, 6), True, opts)


@pytest.mark.parametrize("case", sorted(FWD_A_CASES))
def test_fwd_a_matches_the_fp64_recurrence(case, device, monkeypatch) -> None:
    # wy_a and fwd_ho_a replace the prep, the WY stage and fwd_ho; o, the final state and every gradient (the initial
    # state's too) within the accuracy gate of the FP64 recurrence
    _fwd_a_or_skip(device)
    heads, seq_lens = FWD_A_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=401)
    gen = torch.Generator().manual_seed(402)
    h0 = (
        torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen) * 0.1
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    fwd = stages[: stages.index("fwd_ho_a") + 1]
    assert fwd[-2:] == ["wy_a", "fwd_ho_a"], fwd
    assert not {"prep", "prep_tma", "wy_tc2", "fwd_ho"} & set(stages), stages
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


@pytest.mark.parametrize("case", sorted(FWD_A_CASES))
def test_fwd_a_within_the_budgets_of_the_prep_forward(
    case, device, monkeypatch
) -> None:
    # the forward on the WY stage's A against the forward of the prep, the WY stage and fwd_ho on the same inputs
    _fwd_a_or_skip(device)
    heads, seq_lens = FWD_A_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=403)
    got = _run(inputs, seq_lens, output_final_state=True)
    monkeypatch.setattr(cake_chain, "FWD_A_ARCHS", ())
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens, output_final_state=True)
    assert "fwd_ho" in stages and "fwd_ho_a" not in stages, stages
    _assert_within_budget(got, want, OUTPUT_NAMES + ("final_state",))


def test_fwd_a_tape_off_keeps_the_output_bits(device, monkeypatch) -> None:
    # a call that records no backward (no_grad, or no input that requires grad) writes no tape (word 0); o and the final
    # state are the bits of the taped call (word 11: h, v_new and w)
    _fwd_a_or_skip(device)
    heads, seq_lens = FWD_A_CASES["g3"]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=405)
    cu = _cu(seq_lens)
    kwargs = {
        "scale": 1.0 / math.sqrt(HEAD_DIM),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": torch.tensor(cu, dtype=torch.int64, device=device),
        "cu_seqlens_cpu": torch.tensor(cu, dtype=torch.int64),
    }
    launches = _record_launch_values(monkeypatch)
    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    taped = cake_gdn_nondet(*leaves, **kwargs)
    with torch.no_grad():
        no_grad = cake_gdn_nondet(*leaves, **kwargs)
    plain = cake_gdn_nondet(
        *(inputs[n] for n in ("q", "k", "v", "g", "beta")), **kwargs
    )
    torch.cuda.synchronize()
    assert _tape_words(launches) == [11, 0, 0], _tape_words(launches)
    for got in (no_grad, plain):
        _assert_same_bits(
            {"o": got[0], "final_state": got[1]},
            {"o": taped[0].detach(), "final_state": taped[1].detach()},
            ("o", "final_state"),
        )


def test_fwd_a_unnormalized_q_k_keep_the_prep_forward(device, monkeypatch) -> None:
    # wy_a always normalizes q / k: a call without the in-kernel normalization runs the prep, wy_tc2 and fwd_ho on a
    # layout fwd_a_route takes otherwise, with the bits of the package whose forward is the prep's everywhere
    _fwd_a_or_skip(device)
    heads, seq_lens = FWD_A_CASES["g3"]
    # unit-norm q / k rows, as a caller that normalizes them itself passes them
    inputs = _inputs(
        seq_lens, heads, torch.float32, device, seed=409, normalized_qk=True
    )
    stages = _record_launches(monkeypatch)
    _run(inputs, seq_lens)
    assert "fwd_ho_a" in stages, stages
    stages.clear()
    got = _run(inputs, seq_lens, use_qk_l2norm_in_kernel=False)
    assert "fwd_ho_a" not in stages and "wy_a" not in stages, stages
    assert "fwd_ho" in stages or "fwd_ho_vh2" in stages, stages
    monkeypatch.setattr(cake_chain, "FWD_A_ARCHS", ())
    want = _run(inputs, seq_lens, use_qk_l2norm_in_kernel=False)
    assert all(bool(torch.isfinite(got[name].float()).all()) for name in OUTPUT_NAMES)
    _assert_same_bits(got, want, OUTPUT_NAMES)


@pytest.mark.parametrize("reentrant", [False, True], ids=["non-reentrant", "reentrant"])
def test_fwd_a_checkpoint_writes_the_tape_once(reentrant, device, monkeypatch) -> None:
    # under torch.utils.checkpoint the first pass keeps no tape (its saved tensors are dropped either way) and the
    # recompute writes it: the gradients are the bits of the plain call
    _fwd_a_or_skip(device)
    heads, seq_lens = FWD_A_CASES["g3"]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=407)
    want = _run(inputs, seq_lens)
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def layer(q, k, v, g, beta):
        o, _ = cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                               cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip
        return o

    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    launches = _record_launch_values(monkeypatch)
    o = checkpoint(layer, *leaves.values(), use_reentrant=reentrant)
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    assert _tape_words(launches) == [0, 11], _tape_words(launches)
    got = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_same_bits(got, want, OUTPUT_NAMES)


@pytest.mark.parametrize("case", sorted(FWD_A_CASES))
def test_fwd_a_packed_views_give_the_contiguous_bits(case, device, monkeypatch) -> None:
    # q / k / v as head views of one packed row: wy_a reads q / k and fwd_ho_a reads v in place (no copy launch), the
    # bits of contiguous inputs on every output and gradient
    _fwd_a_or_skip(device)
    heads, seq_lens = FWD_A_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=409)
    want = _run(inputs, seq_lens)
    launches = _record_launch_values(monkeypatch)
    got = _run_packed(inputs, seq_lens, heads)
    _assert_same_bits(got, want, OUTPUT_NAMES)
    wy_a = next(values for stage, values in launches if stage == "wy_a")
    assert any(not t.is_contiguous() for t in wy_a if isinstance(t, torch.Tensor)), (
        "q / k were copied"
    )


def test_fwd_a_saved_list_keeps_its_arity(device) -> None:
    # the saved list of the forward on the WY stage's A has the entries, shapes, dtypes and devices of the prep
    # forward's: the T_full state (w, h, v_new) and a placeholder for u
    arch = _fwd_a_or_skip(device)
    heads, seq_lens = FWD_A_CASES["g3"]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=411)
    cu = _cu(seq_lens)
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                           scale=1.0 / math.sqrt(HEAD_DIM), use_qk_l2norm_in_kernel=True,
                           cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                           cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64))  # fmt: skip
    saved = _saved_list(o)
    assert len(saved) == 10 + len(cake_plan.STATE_NAMES)
    kept = cake_plan.tape_state(arch, True)
    total, hv = sum(seq_lens), heads[1]
    chunks = sum(-(-n // 64) for n in seq_lens)
    shapes = {"w": (total, hv, HEAD_DIM), "u": (1, total, hv, HEAD_DIM), "h": (chunks, hv, HEAD_DIM, HEAD_DIM),
              "v_new": (1, total, hv, HEAD_DIM)}  # fmt: skip
    for name, t in zip(cake_plan.STATE_NAMES, saved[9:-1], strict=True):
        placeholder = (
            all(s == 0 for s in t.stride())
            and t.untyped_storage().nbytes() == t.element_size()
        )
        assert placeholder is (name not in kept), (name, t.stride())
        assert tuple(t.shape) == shapes[name] and t.dtype == torch.bfloat16, (
            name,
            t.shape,
            t.dtype,
        )
    assert saved[7].shape == (total, hv, 64) and saved[7].dtype == torch.bfloat16  # A


# --------------------------------------------------------------------------- the split forward on the piece table
# Layouts whose forward segment table splits a sequence on every device of the package: split sequences beside
# sequences of one piece (one of them a single token), value-head groups of three and two.  Their pieces (30..42
# chunks) are shorter than the split's shape class (cake_plan.FORWARD_SPLIT_MIN_PIECE): the tests of the split
# forward's tables, bits and accuracy run them with the class lowered to FWD_SPLIT_TEST_MIN_PIECE (the same kernels and
# tables on fewer tokens than a row of the class); test_fwd_split_shape_class pins the class itself.
FWD_SPLIT_CASES = {
    "g3": ((2, 6), (40000, 20000, 1000, 64, 1)),
    "g2": ((4, 8), (30000, 9000, 100)),
}
FWD_SPLIT_TEST_MIN_PIECE = 16
# the pair rows: three sequences of about 11k tokens
PAIR_SEQ_LENS = (11000, 12000, 9768)
# the pair row of one key head: pieces of about ten chunks, so the prep forward's split scan keeps it
FWD_SPLIT_SHORT = ((1, 3), PAIR_SEQ_LENS)
# (H, HV), sequence lengths and whether the forward runs the split, for the rows that set the split's shape class:
# long sequences at HV 6 (pieces of 132..158 chunks) and the pair rows (HV 24: 63; HV 12: 38; HV 6: 20; HV 3: 11)
FWD_SPLIT_CLASS_ROWS = {
    "long3_hv6": ((2, 6), (131071, 87531, 43385), True),
    "long2_hv6": ((2, 6), (128147, 131027), True),
    "long6_hv6": ((2, 6), (126565, 32714, 33696, 15081, 22979, 10782), True),
    "pair_hv24": ((8, 24), PAIR_SEQ_LENS, True),
    "pair_hv12": ((4, 12), PAIR_SEQ_LENS, False),
    "pair_hv6": ((2, 6), PAIR_SEQ_LENS, False),
    "pair_hv3": ((1, 3), PAIR_SEQ_LENS, False),
}
# the host launch order: the middle pieces' summary (side stream) is issued before wave 1; on wy_a's A
# (FWD_SPLIT_A_ARCHS) both waves are fwd_ho_a and the copy of wave 1's end states into the segment state is a torch op
FWD_SPLIT_STAGES = ["wy_a", "wy_tc2", "fwd_summary", "fwd_fused", "stitch", "fwd_fused"]
FWD_SPLIT_A_STAGES = ["wy_a", "fwd_summary_a", "fwd_ho_a", "stitch", "fwd_ho_a"]


def _split_a() -> bool:
    """Whether the split forward of the current device runs on wy_a's A on the test rows (FWD_SPLIT_A_ARCHS; their
    pieces are shorter than every FWD_SPLIT_A_MIN_PIECE entry, test_fwd_split_tables_follow_the_plan)."""
    return cake_chain._fwd_split_a(
        cake_backend.device_arch(torch.cuda.current_device()), FWD_SPLIT_TEST_MIN_PIECE
    )


def _split_stages() -> list[str]:
    return FWD_SPLIT_A_STAGES if _split_a() else FWD_SPLIT_STAGES


def _split_wave() -> str:
    """The stage of the split forward's two waves on the current device."""
    return "fwd_ho_a" if _split_a() else "fwd_fused"


def _split_taped_word() -> int:
    """The tape word of a taped wave of the split forward on the current device (its waves' tape table)."""
    arch = cake_backend.device_arch(torch.cuda.current_device())
    words = cake_chain.FWD_A_TAPE if _split_a() else cake_chain.FWD_FUSED_TAPE
    return sum(
        words[name] for name in cake_plan.tape_state(arch, True) if name in words
    )


def _fwd_split_or_skip(device) -> str:
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    if arch not in cake_chain.FWD_SPLIT_ARCHS:
        pytest.skip(f"the split forward is not routed on {arch}")
    return arch


@pytest.fixture
def fwd_split_test_class(monkeypatch) -> None:
    """The split forward's shape class lowered to FWD_SPLIT_TEST_MIN_PIECE (the layouts of FWD_SPLIT_CASES)."""
    monkeypatch.setattr(
        cake_plan,
        "FORWARD_SPLIT_MIN_PIECE",
        types.MappingProxyType(
            {arch: FWD_SPLIT_TEST_MIN_PIECE for arch in cake_chain.FWD_SPLIT_ARCHS}
        ),
    )


def _forward_stages(stages) -> list[str]:
    """The forward's launches of one call (up to its last forward stage)."""
    last = max(
        i
        for i, s in enumerate(stages)
        if s in ("fwd_fused", "fwd_ho", "fwd_ho_vh2", "fwd_ho_a")
    )
    return list(stages[: last + 1])


def _fused_tape_words(launches) -> list[int]:
    """The tape word of every launch of the split forward's waves (:func:`_split_wave`)."""
    wave = _split_wave()
    names = cake_backend.parameters(
        cake_backend.stage_record(
            wave, cake_backend.device_arch(torch.cuda.current_device())
        )
    )
    return [
        int(values[names.index("tape")]) for stage, values in launches if stage == wave
    ]


def test_fwd_split_lists() -> None:
    # pieces by segment_bounds; wave 1 walks each split sequence's first piece (its end chunk in first_end), the
    # summary the middle pieces, wave 2 every other piece and each sequence of one piece (an empty one too)
    scs, seq, sss, first_end, first, middle, rest = cake_chain.forward_split_lists(
        (10, 3, 0, 7), 4
    )
    assert scs == [0, 3, 6, 10, 13, 13, 16, 20]
    assert seq == [0, 0, 0, 1, 2, 3, 3]
    assert sss == [0, 3, 4, 5, 7]
    assert first_end == [0, 3, 13, 13, 16]
    assert first == [0, 5] and middle == [1] and rest == [1, 2, 3, 4, 6]
    scs, seq, sss, first_end, first, middle, rest = cake_chain.forward_split_lists(
        (5, 2), 8
    )
    assert (scs, seq, sss, first_end, first, middle, rest) == (
        [0, 5, 7],
        [0, 1],
        [0, 1, 2],
        [0, 5, 7],
        [],
        [],
        [0, 1],
    )


def test_fwd_split_route_rule(monkeypatch) -> None:
    # the split forward takes a table that carries its split tables and the inputs of the forward on wy_a's A
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)
    split = types.SimpleNamespace(length=63)

    def meta(fwd_split):
        return types.SimpleNamespace(
            seq_lens=(40000, 20000), num_seqs=2, fwd_split=fwd_split
        )

    def tensors(h, hv, beta_dtype=torch.float32):
        q = torch.empty(1, 64, h, HEAD_DIM, dtype=torch.bfloat16, device="meta")
        v = torch.empty(1, 64, hv, HEAD_DIM, dtype=torch.bfloat16, device="meta")
        return q, q, v, torch.empty(1, 64, hv, dtype=beta_dtype, device="meta")

    route = cake_chain.fwd_split_route
    opts = cake_chain.make_options()
    assert route(meta(split), "sm_100a", *tensors(2, 6), True, opts)
    assert route(meta(split), "sm_103a", *tensors(2, 6), True, opts)
    assert not route(meta(split), "sm_107a", *tensors(2, 6), True, opts)
    assert not route(meta(None), "sm_100a", *tensors(2, 6), True, opts)
    assert not route(meta(split), "sm_100a", *tensors(2, 6, torch.bfloat16), True, opts)
    assert not route(meta(split), "sm_100a", *tensors(2, 6), False, opts)
    assert not route(
        meta(split), "sm_100a", *tensors(2, 8), True, opts
    )  # a group of four
    assert not route(
        meta(split),
        "sm_100a",
        *tensors(2, 6),
        True,
        cake_chain.make_options(prep="tma"),
    )


# the split rows whose split forward keeps the waves on the WY stage's w / u, per architecture (FWD_SPLIT_A_MIN_PIECE):
# on sm_103a the pair row of HV 24 (63-chunk pieces); every other split row runs on wy_a's A
FWD_SPLIT_W_FORM_ROWS = {"sm_103a": frozenset({"pair_hv24"})}


@pytest.mark.parametrize("sm_count", [148, 152])
@pytest.mark.parametrize("row", sorted(FWD_SPLIT_CLASS_ROWS))
def test_fwd_split_shape_class(row, sm_count, monkeypatch) -> None:
    # the forward runs the split on the long rows and on the pair row of HV 24 at the SM counts of B200 (148) and
    # GB200 / GB300 (152); the pair rows of HV 12, 6 and 3 keep the prep forward's split scan.  The class and the
    # split's form (on wy_a's A, or the waves on the WY stage's w / u) are read from the host piece table (lengths,
    # HV, SM count) alone.
    monkeypatch.setattr(cake_chain, "_available", lambda _stage, _arch: True)
    heads, seq_lens, split = FWD_SPLIT_CLASS_ROWS[row]
    table = cake_plan.pieces(seq_lens, heads[1], sm_count)
    assert table.split, table
    for arch in cake_chain.FWD_SPLIT_ARCHS:
        got = cake_plan.forward_split_length(arch, table)
        assert (got is not None) is split, (arch, table.length)
        assert got in (None, table.length)
        if got is not None:
            a_form = row not in FWD_SPLIT_W_FORM_ROWS.get(arch, frozenset())
            assert cake_chain._fwd_split_a(arch, got) is a_form, (arch, got)


@pytest.mark.usefixtures("fwd_split_test_class")
def test_fwd_split_tables_follow_the_plan(device) -> None:
    # the split tables ride on a host-built table whose forward scan splits, at the host plan's piece length; a pinned
    # scan table, a short piece and the device-built table carry none
    arch = _fwd_split_or_skip(device)
    index = cake_chain._device_index(device)

    def meta(heads, seq_lens, route="host", **pins):
        cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
        return cake_chain._chunk_meta(cu.to(device), cu if route == "host" else None, 1, sum(seq_lens), index,
                                      heads[1], cake_chain.make_options(**pins), heads[1] // heads[0])  # fmt: skip

    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    plan = cake_plan.host_plan(tuple(seq_lens), heads[1], index)
    got = meta(heads, seq_lens).fwd_split
    assert got is not None and got.length == plan.pieces.length
    assert got.length >= cake_plan.FORWARD_SPLIT_MIN_PIECE[arch]
    # the test rows cut pieces shorter than every FWD_SPLIT_A_MIN_PIECE entry (_split_a)
    for case_heads, case_lens in FWD_SPLIT_CASES.values():
        length = meta(case_heads, case_lens).fwd_split.length
        assert length < min(
            cake_chain.FWD_SPLIT_A_MIN_PIECE.values(), default=length + 1
        )
        assert cake_chain._fwd_split_a(arch, length) is _split_a(), (arch, length)
    assert got.num_segments == sum(plan.pieces.counts)
    assert got.num_first + got.num_middle + len(seq_lens) == got.num_segments
    assert meta(heads, seq_lens, segment_chunks=0).fwd_split is None
    assert meta(heads, seq_lens, route="device").fwd_split is None
    assert meta(*FWD_SPLIT_SHORT).fwd_split is None


@pytest.mark.usefixtures("fwd_split_test_class")
@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
def test_fwd_split_matches_the_fp64_recurrence(case, device, monkeypatch) -> None:
    # the split forward replaces the prep, the WY stage, the pre-scan and fwd_ho; o, the final state and every gradient
    # (the initial state's too) within the accuracy gate of the FP64 recurrence
    _fwd_split_or_skip(device)
    heads, seq_lens = FWD_SPLIT_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=421)
    gen = torch.Generator().manual_seed(422)
    h0 = (
        torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen) * 0.1
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    assert _forward_stages(stages) == _split_stages(), stages
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


@pytest.mark.usefixtures("fwd_split_test_class")
@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
def test_fwd_split_within_the_budgets_of_the_prep_forward(
    case, device, monkeypatch
) -> None:
    # the split forward against the prep forward's split scan (pre-scan, stitch, fwd_ho) on the same inputs
    _fwd_split_or_skip(device)
    heads, seq_lens = FWD_SPLIT_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=423)
    got = _run(inputs, seq_lens, output_final_state=True)
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_ARCHS", ())
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens, output_final_state=True)
    assert "fwd_ho" in stages and "fwd_fused" not in stages, stages
    _assert_within_budget(got, want, OUTPUT_NAMES + ("final_state",))


# the forward's routes through the stitch: the split forward (FWD_SPLIT_ARCHS) and the prep forward's pre-scan split
# on a pinned segment table (the route of the other architectures' split rows); a pinned table's pieces in chunks
FWD_STITCH_ROUTES = ("split", "prescan")
FWD_PRESCAN_CHUNKS = 32


@pytest.mark.usefixtures("fwd_split_test_class")
@pytest.mark.parametrize("route", FWD_STITCH_ROUTES)
def test_fwd_split_final_state_sizing(route, device, monkeypatch) -> None:
    # a forward through the stitch composes each piece's start state as x' = M x + B with x and M as hi / lo BF16
    # operand pairs (about 16 bits per piece boundary, from the third piece on): its final state against the serial
    # forward's (segment_chunks 0) and both against the FP64 recurrence, per sequence (pieces of each) and over all;
    # the SIZING lines (pytest -s) report the errors and the accuracy gate holds them
    index = cake_chain._device_index(device)
    arch = cake_backend.device_arch(index)
    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    pins = {}
    if route == "split":
        _fwd_split_or_skip(device)
        cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
        split = cake_chain._chunk_meta(cu.to(device), cu, 1, sum(seq_lens), index, heads[1],
                                       cake_chain.make_options(), heads[1] // heads[0]).fwd_split  # fmt: skip
        # a middle piece: a sequence of three pieces or more
        assert split is not None and split.num_middle >= 1
        length = split.length
    else:
        pins["segment_chunks"] = length = FWD_PRESCAN_CHUNKS
    pieces = [((n + 63) // 64 + length - 1) // length for n in seq_lens]
    assert max(pieces) >= 3, pieces
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=431)
    gen = torch.Generator().manual_seed(432)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, **pins)  # fmt: skip
    if route == "split":
        assert _forward_stages(stages) == _split_stages(), stages
    else:
        assert "stitch" in _forward_stages(stages), stages
    stages.clear()
    serial = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht, segment_chunks=0)  # fmt: skip
    assert "stitch" not in _forward_stages(stages), stages
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    tag = f"fwd_final_state {arch} route={route} piece_len={length}"
    for i, n in enumerate(seq_lens):
        for label, run in ((route, got), ("serial", serial)):
            _sizing(f"{tag} seq={i} tokens={n} pieces={pieces[i]} {label}_vs_fp64", run["final_state"][i],
                    ref["final_state"][i])  # fmt: skip
        _sizing(f"{tag} seq={i} tokens={n} pieces={pieces[i]} {route}_vs_serial", got["final_state"][i],
                serial["final_state"][i])  # fmt: skip
    _sizing(f"{tag} all {route}_vs_fp64", got["final_state"], ref["final_state"])
    _sizing(f"{tag} all serial_vs_fp64", serial["final_state"], ref["final_state"])
    _sizing(f"{tag} all {route}_vs_serial", got["final_state"], serial["final_state"])
    _assert_close_to_reference(got, ref, ("final_state",))
    _assert_close_to_reference(serial, ref, ("final_state",))
    _assert_within_budget(got, serial, ("final_state",))


@pytest.mark.usefixtures("fwd_split_test_class")
@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
def test_fwd_split_runs_give_the_same_bits(case, device) -> None:
    # every launch of the split forward has one writer per output element and a fixed order: run to run, and on one
    # stream (the summary then wave 1) as on two, the same bits
    _fwd_split_or_skip(device)
    heads, seq_lens = FWD_SPLIT_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=425)
    gen = torch.Generator().manual_seed(426)
    h0 = (
        torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen) * 0.1
    ).to(device)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    want = _run(inputs, seq_lens, initial_state=h0, output_final_state=True)
    for streams in (2, 1):
        got = _run(
            inputs, seq_lens, initial_state=h0, output_final_state=True, streams=streams
        )
        _assert_same_bits(got, want, names)


@pytest.mark.parametrize(
    "row",
    [row for row, (*_, split) in sorted(FWD_SPLIT_CLASS_ROWS.items()) if not split],
)
def test_fwd_split_short_pieces_keep_the_prep_forward(row, device, monkeypatch) -> None:
    # the pair rows outside the split's shape class (pieces shorter than FORWARD_SPLIT_MIN_PIECE): no launch of the
    # split forward
    _fwd_split_or_skip(device)
    heads, seq_lens, _split = FWD_SPLIT_CLASS_ROWS[row]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=427)
    stages = _record_launches(monkeypatch)
    _run(inputs, seq_lens)
    assert not {"fwd_fused", "fwd_summary", "fwd_summary_a"} & set(stages), stages


@pytest.mark.usefixtures("fwd_split_test_class")
def test_fwd_split_tape_off_keeps_the_output_bits(device, monkeypatch) -> None:
    # a call that records no backward writes no tape (word 0 on both waves); o and the final state are the bits of the
    # taped call (fwd_fused word 3: h and v_new, w is the WY stage's; fwd_ho_a word 11: h, v_new and w)
    _fwd_split_or_skip(device)
    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=429)
    cu = _cu(seq_lens)
    kwargs = {
        "scale": 1.0 / math.sqrt(HEAD_DIM),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": torch.tensor(cu, dtype=torch.int64, device=device),
        "cu_seqlens_cpu": torch.tensor(cu, dtype=torch.int64),
    }
    launches = _record_launch_values(monkeypatch)
    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    taped = cake_gdn_nondet(*leaves, **kwargs)
    with torch.no_grad():
        no_grad = cake_gdn_nondet(*leaves, **kwargs)
    plain = cake_gdn_nondet(
        *(inputs[n] for n in ("q", "k", "v", "g", "beta")), **kwargs
    )
    torch.cuda.synchronize()
    word = _split_taped_word()
    assert _fused_tape_words(launches) == [word, word, 0, 0, 0, 0], _fused_tape_words(
        launches
    )
    for got in (no_grad, plain):
        _assert_same_bits(
            {"o": got[0], "final_state": got[1]},
            {"o": taped[0].detach(), "final_state": taped[1].detach()},
            ("o", "final_state"),
        )


@pytest.mark.usefixtures("fwd_split_test_class")
def test_fwd_split_unnormalized_q_k_keep_the_prep_forward(device, monkeypatch) -> None:
    # wy_a always normalizes q / k: a call without the in-kernel normalization runs the prep forward's split scan on a
    # layout the split forward takes otherwise, with the bits of the package whose forward is the prep's everywhere
    _fwd_split_or_skip(device)
    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    inputs = _inputs(
        seq_lens, heads, torch.float32, device, seed=431, normalized_qk=True
    )
    stages = _record_launches(monkeypatch)
    _run(inputs, seq_lens)
    assert _forward_stages(stages) == _split_stages(), stages
    stages.clear()
    got = _run(inputs, seq_lens, use_qk_l2norm_in_kernel=False)
    split_stages = {"fwd_fused", "fwd_summary", "fwd_summary_a", "fwd_ho_a", "wy_a"}
    assert not split_stages & set(stages), stages
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_ARCHS", ())
    monkeypatch.setattr(cake_chain, "FWD_A_ARCHS", ())
    want = _run(inputs, seq_lens, use_qk_l2norm_in_kernel=False)
    assert all(bool(torch.isfinite(got[name].float()).all()) for name in OUTPUT_NAMES)
    _assert_same_bits(got, want, OUTPUT_NAMES)


@pytest.mark.usefixtures("fwd_split_test_class")
@pytest.mark.parametrize("reentrant", [False, True], ids=["non-reentrant", "reentrant"])
def test_fwd_split_checkpoint_writes_the_tape_once(
    reentrant, device, monkeypatch
) -> None:
    # under torch.utils.checkpoint the first pass keeps no tape and the recompute writes it: the gradients are the bits
    # of the plain call
    _fwd_split_or_skip(device)
    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=433)
    want = _run(inputs, seq_lens)
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def layer(q, k, v, g, beta):
        o, _ = cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                               cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip
        return o

    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    launches = _record_launch_values(monkeypatch)
    o = checkpoint(layer, *leaves.values(), use_reentrant=reentrant)
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    word = _split_taped_word()
    assert _fused_tape_words(launches) == [0, 0, word, word], _fused_tape_words(
        launches
    )
    got = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_same_bits(got, want, OUTPUT_NAMES)


@pytest.mark.usefixtures("fwd_split_test_class")
@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
def test_fwd_split_packed_views_give_the_contiguous_bits(
    case, device, monkeypatch
) -> None:
    # q / k / v as head views of one packed row: wy_a reads q / k and wy_tc2 (on wy_a's A: the waves) reads v in place
    # (no copy launch), the bits of contiguous inputs on every output and gradient
    _fwd_split_or_skip(device)
    heads, seq_lens = FWD_SPLIT_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=435)
    want = _run(inputs, seq_lens)
    launches = _record_launch_values(monkeypatch)
    got = _run_packed(inputs, seq_lens, heads)
    _assert_same_bits(got, want, OUTPUT_NAMES)
    assert _split_wave() in [stage for stage, _ in launches]
    wy_a = next(values for stage, values in launches if stage == "wy_a")
    assert any(not t.is_contiguous() for t in wy_a if isinstance(t, torch.Tensor)), (
        "q / k were copied"
    )
    reader = _split_wave() if _split_a() else "wy_tc2"
    wy = next(values for stage, values in launches if stage == reader)
    assert any(not t.is_contiguous() for t in wy if isinstance(t, torch.Tensor)), (
        "v was copied"
    )


@pytest.mark.usefixtures("fwd_split_test_class")
def test_fwd_split_a_tape_matches_the_w_form(device, monkeypatch) -> None:
    # on wy_a's A the waves write the T_full tape themselves (fwd_ho_a word 11): w the bits of the WY stage's w on the
    # same A, h and v_new within a relative L2 of 1e-2 of the waves on the WY stage's w / u, u a placeholder
    arch = _fwd_split_or_skip(device)
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_A_MIN_PIECE", types.MappingProxyType({}))
    if not cake_chain._fwd_split_a(arch, 0):
        pytest.skip(f"the split forward on wy_a's A is not routed on {arch}")
    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=437)
    cu = _cu(seq_lens)

    def saved_state() -> dict:
        leaves = [
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        o, _ = cake_gdn_nondet(*leaves, scale=1.0 / math.sqrt(HEAD_DIM), use_qk_l2norm_in_kernel=True,
                               cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
                               cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64))  # fmt: skip
        saved = _saved_list(o)
        return {
            "o": o.detach(),
            **dict(zip(cake_plan.STATE_NAMES, saved[9:-1], strict=True)),
        }

    stages = _record_launches(monkeypatch)
    got = saved_state()
    assert _forward_stages(stages) == FWD_SPLIT_A_STAGES, stages
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_A_ARCHS", ())
    stages.clear()
    want = saved_state()
    assert _forward_stages(stages) == FWD_SPLIT_STAGES, stages
    assert all(s == 0 for s in got["u"].stride()), got["u"].stride()
    assert torch.equal(got["w"], want["w"])
    for name in ("o", "h", "v_new"):
        a, b = got[name].float(), want[name].float()
        assert bool(torch.isfinite(a).all()), name
        rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
        assert rel <= 1e-2, (name, rel)


# --------------------------------------------------------------------------- M-share: the backward on the forward's maps
# The pricing rule's rows on B200 (sm_100a, 148 SMs): the priced saving (us) of the backward's first wave, M-share where
# it reaches M_SHARE_MIN_SAVING; a row without a middle piece prices none.
# --------------------------------------------------------------------------- unaligned rows, deterministic algorithms
# the forward routes: the prep forward, the forward on wy_a's A (serial table), and the split forward on wy_a's A and
# on the WY stage's w / u
ROW_ROUTES = ("prep", "a", "split_a", "split_w")


def _route_or_skip(route, device, monkeypatch):
    """Force ``route`` on the device (skip where it is not routed) and return its layout ``(heads, seq_lens)`` and the
    stage that marks it among the launches (``prep``: a prep stage)."""
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    if route == "prep":
        monkeypatch.setattr(cake_chain, "FWD_A_ARCHS", ())
        monkeypatch.setattr(cake_chain, "FWD_SPLIT_ARCHS", ())
        return FWD_A_CASES["g3"], "prep"
    if route == "a":
        _fwd_a_or_skip(device)
        return FWD_A_CASES["g3"], "fwd_ho_a"
    _fwd_split_or_skip(device)
    monkeypatch.setattr(
        cake_plan,
        "FORWARD_SPLIT_MIN_PIECE",
        types.MappingProxyType(
            {a: FWD_SPLIT_TEST_MIN_PIECE for a in cake_chain.FWD_SPLIT_ARCHS}
        ),
    )
    if route == "split_a":
        stages = cake_chain.FWD_SPLIT_A_STAGES
        if not all(cake_chain._available(stage, arch) for stage in stages):
            pytest.skip(f"the split forward on wy_a's A is not delivered for {arch}")
        monkeypatch.setattr(cake_chain, "FWD_SPLIT_A_ARCHS", (arch,))
        monkeypatch.setattr(
            cake_chain, "FWD_SPLIT_A_MIN_PIECE", types.MappingProxyType({})
        )
        return FWD_SPLIT_CASES["g3"], "fwd_summary_a"
    monkeypatch.setattr(cake_chain, "FWD_SPLIT_A_ARCHS", ())
    return FWD_SPLIT_CASES["g3"], "fwd_fused"


def _unaligned(t: torch.Tensor, offset: int) -> torch.Tensor:
    """A contiguous copy of ``t`` ``offset`` elements past the base of a fresh allocation (a leaf of its own)."""
    buf = torch.empty(t.numel() + 16, dtype=t.dtype, device=t.device)
    out = buf[offset : offset + t.numel()].view(t.shape).detach()
    out.copy_(t)
    return out


# the byte offsets past a 16-byte boundary the unaligned operands take: each operand at the nearest whole element
# (BF16 1, 2, 4, 7 elements; FP32 1, 1, 2, 3)
ROW_BYTE_OFFSETS = (2, 4, 8, 14)
# the operands placed past the boundary: the row operands (v, g, beta, the initial state, do, the final state's
# gradient), and q / k (the forwards on wy_a's A take 16-byte aligned q / k only, the plain prep 8-byte aligned rows)
ROW_OPERANDS = {
    "rows": ("v", "g", "beta", "h0", "do", "dht"),
    "qk": ("q", "k"),
}


def _run_rows(inputs, seq_lens, h0, dht, *, offset=0, names=(), ckpt=False, **pins) -> dict:  # fmt: skip
    """Forward and backward with the operands of ``names`` (of q, k, v, g, beta, h0, do, dht) as contiguous tensors
    ``offset`` bytes past a 16-byte boundary (the nearest whole element past it), the others as fresh copies;
    ``ckpt``: the forward under non-reentrant ``torch.utils.checkpoint``."""

    def place(name, t):
        if name not in names:
            return t.detach().clone()
        size = t.element_size()
        got = _unaligned(t, max(1, offset // size))
        assert (
            got.is_contiguous() and got.data_ptr() % 16 == max(1, offset // size) * size
        ), (name, offset)
        return got

    leaves = {
        n: place(n, inputs[n]).requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    h = place("h0", h0).requires_grad_(True)
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=h0.device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def layer(q, k, v, g, beta, h):
        return cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), initial_state=h,
                               output_final_state=True, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                               use_qk_l2norm_in_kernel=True, **pins)  # fmt: skip

    args = (*leaves.values(), h)
    o, final = checkpoint(layer, *args, use_reentrant=False) if ckpt else layer(*args)
    torch.autograd.backward((o, final), (place("do", inputs["do"]), place("dht", dht)))
    torch.cuda.synchronize()
    return {
        "o": o.detach(),
        "final_state": final.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
        "dh0": h.grad,
    }


@pytest.mark.parametrize("operands", sorted(ROW_OPERANDS))
@pytest.mark.parametrize("route", ROW_ROUTES)
def test_unaligned_rows_give_the_aligned_bits(
    route, operands, device, monkeypatch
) -> None:
    # contiguous operands whose base is 2, 4, 8 or 14 bytes past a 16-byte boundary are copied before a stage reads
    # them in words: v, g, beta, h0, do and dht keep the route and give the bits of aligned copies on every forward
    # route, the backward and a checkpointed step; q / k take the prep forward (the plain prep reads 8-byte aligned
    # rows in place, others are copied) with the bits of aligned q / k on the plain prep (the two preps give the
    # same bits)
    (heads, seq_lens), marker = _route_or_skip(route, device, monkeypatch)
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=451)
    gen = torch.Generator().manual_seed(452)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    names = OUTPUT_NAMES + ("final_state", "dh0")
    placed = ROW_OPERANDS[operands]
    pins = {"prep": "plain"} if operands == "qk" else {}
    stages = _record_launches(monkeypatch)
    for ckpt in (False, True):
        stages.clear()
        want = _run_rows(inputs, seq_lens, h0, dht, ckpt=ckpt, **pins)
        if marker == "prep" or operands == "qk":
            assert _preps(stages) and "fwd_ho_a" not in stages, (route, stages)
        else:
            assert marker in stages, (route, stages)
        for offset in ROW_BYTE_OFFSETS:
            stages.clear()
            got = _run_rows(
                inputs, seq_lens, h0, dht, offset=offset, names=placed, ckpt=ckpt
            )
            if operands == "qk":
                assert _preps(stages) and "fwd_ho_a" not in stages, (offset, stages)
                assert "fwd_fused" not in stages, (offset, stages)
            elif marker != "prep":
                assert marker in stages, (route, offset, stages)
            _assert_same_bits(got, want, names)


@pytest.mark.parametrize("route", ("a", "split"))
def test_deterministic_algorithms_run_the_a_and_split_routes(
    route, device, monkeypatch
) -> None:
    # every launch is fixed-order and the host ops of the forwards on wy_a's A (index_fill_, index_select,
    # index_copy_) have deterministic implementations: under torch.use_deterministic_algorithms(True) (uninitialized
    # memory filled, warnings as errors) the forward, the backward and a checkpointed step run without an alert, with
    # the bits of the default mode
    if route == "a":
        _fwd_a_or_skip(device)
        heads, seq_lens = FWD_A_CASES["g3"]
        marker = "fwd_ho_a"
    else:
        _fwd_split_or_skip(device)
        monkeypatch.setattr(
            cake_plan,
            "FORWARD_SPLIT_MIN_PIECE",
            types.MappingProxyType(
                {a: FWD_SPLIT_TEST_MIN_PIECE for a in cake_chain.FWD_SPLIT_ARCHS}
            ),
        )
        monkeypatch.setattr(cake_chain, "_META_FRONT", {})
        heads, seq_lens = FWD_SPLIT_CASES["g3"]
        marker = _split_wave()
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=455)
    gen = torch.Generator().manual_seed(456)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    names = OUTPUT_NAMES + ("final_state", "dh0")
    want = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    plain = _run(inputs, seq_lens)
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def layer(q, k, v, g, beta):
        o, _ = cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                               cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip
        return o

    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    with _deterministic_algorithms(True), warnings.catch_warnings():
        warnings.simplefilter("error")
        stages = _record_launches(monkeypatch)
        got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
        o = checkpoint(layer, *leaves.values(), use_reentrant=False)
        (o.float() * inputs["do"].float()).sum().backward()
        torch.cuda.synchronize()
    assert marker in stages, (route, stages)
    _assert_same_bits(got, want, names)
    ckpt = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_same_bits(ckpt, plain, OUTPUT_NAMES)


M_SHARE_ROWS = {
    "issue": ((2, 6), (131071, 87531, 43385), 55.194),
    "P3": ((2, 6), (128147, 131027), 55.194),
    "P1": ((2, 6), (126565, 32714, 33696, 15081, 22979, 10782), 43.676),
    "h8": ((8, 24), (11000, 12000, 9768), 13.109),
    "h4": ((4, 12), (11000, 12000, 9768), 2.034),
    "h2": ((2, 6), (11000, 12000, 9768), -8.619),
    "h16": ((16, 48), (11000, 12000, 9768), None),
    "A16": ((16, 48), (32768, 32768, 32768), None),
}
# the stages of the split adjoint walk and its composition
_ADJOINT_WALK_STAGES = (
    "dhu_tc2_dvl32_modes",
    "dhu_tc2_dvl32_gm",
    "stitch",
    "stitch_mt",
    "piece_chain",
)


def _force_m_share(monkeypatch, device) -> str:
    """Let every layout of FWD_SPLIT_CASES take M-share on the device: the split forward's shape class lowered to
    FWD_SPLIT_TEST_MIN_PIECE (as :func:`fwd_split_test_class`), the adjoint's one-wave split priced on (its pieces are
    the forward's: the device's own SM count), the B200 M-share costs under the device's (architecture, SM count) and
    no least saving; fresh layout and plan caches."""
    arch = _fwd_split_or_skip(device)
    if arch not in cake_chain.M_SHARE_ARCHS:
        pytest.skip(f"M-share is not planned on {arch}")
    stages = (*_SPLIT_STAGES, *cake_chain.M_SHARE_STAGES)
    if not all(cake_chain._available(stage, arch) for stage in stages):
        pytest.skip(f"the M-share stages are not delivered for {arch}")
    if cake_chain._adjoint_chain_stage(arch) != "stitch":
        pytest.skip(
            f"the split's end adjoints are not composed by the stitch on {arch}"
        )
    index = cake_chain._device_index(device)
    sms = int(torch.cuda.get_device_properties(index).multi_processor_count)
    monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_ARCHS", frozenset({arch}))
    costs = {"walk": 1.0, "summary": 1.0, "serial_low": 1.0, "serial_full": 1.0, "stitch": 0.0, "stitch_piece": 0.0,
             "fixed": 0.0}  # fmt: skip
    monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_COSTS", {arch: costs})
    monkeypatch.setattr(cake_chain, "ADJOINT_SPLIT_DEVICE_COSTS", {})
    monkeypatch.setattr(cake_plan, "ADJOINT_FORMS", types.MappingProxyType({}))
    m_costs = dict(cake_plan.M_SHARE_COSTS)
    m_costs.setdefault((arch, sms), cake_plan.M_SHARE_COSTS[("sm_100a", 148)])
    monkeypatch.setattr(cake_plan, "M_SHARE_COSTS", types.MappingProxyType(m_costs))
    monkeypatch.setattr(cake_plan, "M_SHARE_MIN_SAVING", float("-inf"))
    monkeypatch.setattr(
        cake_plan,
        "FORWARD_SPLIT_MIN_PIECE",
        types.MappingProxyType(
            {a: FWD_SPLIT_TEST_MIN_PIECE for a in cake_chain.FWD_SPLIT_ARCHS}
        ),
    )
    for cache in ("_META_FRONT", "_DHU_AUTO", "_SDT_AUTO", "_BWD_PLANS"):
        monkeypatch.setattr(cake_chain, cache, {})
    return arch


def _m_share_meta(device, heads, seq_lens, **pins):
    index = cake_chain._device_index(device)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
    return cake_chain._chunk_meta(cu.to(device), cu, 1, sum(seq_lens), index, heads[1],
                                  cake_chain.make_options(**pins), heads[1] // heads[0])  # fmt: skip


def _adjoint_walk(stages) -> list[str]:
    """The backward's adjoint walk launches of one call (the stages of the split walk after the forward's)."""
    return [
        s for s in stages[len(_forward_stages(stages)) :] if s in _ADJOINT_WALK_STAGES
    ]


def test_m_share_rule_on_the_bench_rows() -> None:
    # the first wave's priced saving on B200: M-share on issue, P3 and P1 (h8 saves less than M_SHARE_MIN_SAVING);
    # rows without a middle piece and devices without an entry keep the summary
    assert cake_plan.M_SHARE_MIN_SAVING == 15.0
    on = set()
    for row, (heads, seq_lens, saving) in M_SHARE_ROWS.items():
        chunks = cake_plan.chunk_counts(seq_lens)
        got = cake_plan.m_share_saving(chunks, heads[1], "sm_100a", 148)
        if saving is None:
            assert got is None, row
        else:
            assert got == pytest.approx(saving, abs=1e-3), (row, got)
        if cake_plan.m_share(chunks, heads[1], "sm_100a", 148):
            on.add(row)
        assert cake_plan.m_share_saving(chunks, heads[1], "sm_100a", 152) is None, row
        assert not cake_plan.m_share(chunks, heads[1], "sm_107a", 148), row
    assert on == {"issue", "P3", "P1"}
    assert frozenset({"sm_100a", "sm_103a"}) == cake_chain.M_SHARE_ARCHS
    assert cake_chain.M_SHARE_STAGES == ("dhu_tc2_dvl32_modes", "stitch_mt")
    assert cake_chain.M_SHARE_ONE_LAUNCH is False


def test_m_share_entry_tables() -> None:
    # beside the warm split's entries (summaries of every piece but each sequence's first, under WARM): gonly, each split
    # sequence's middle pieces walked from zero for their start adjoint alone (OUT); ms_walk, the last pieces' walks, a
    # split sequence's storing its start adjoint (OUT); pass1, ms_walk's rows, then gonly's; the stitch's host seg_flag,
    # 0 at each sequence's last piece
    split = cake_chain.adjoint_split_tables([0, 3, 4, 8], 2, "cpu")
    out, warm = cake_chain.SPLIT_OUT, cake_chain.SPLIT_WARM
    walk = cake_chain.SPLIT_SEED | cake_chain.SPLIT_SEED_SEQ | cake_chain.SPLIT_TAPE
    assert split.summary.tolist() == [[1, 2, 5, 6, 7], [warm] * 5]
    assert split.num_gonly == 3
    assert split.gonly.tolist() == [[1, 5, 6], [out] * 3]
    assert split.ms_walk.tolist() == [[2, 3, 7], [walk | out, walk, walk | out]]
    assert split.walk.tolist() == [[2, 3, 7], [walk] * 3]
    assert split.pass1.tolist() == [
        split.ms_walk[0].tolist() + [1, 5, 6],
        split.ms_walk[1].tolist() + [out] * 3,
    ]
    assert split.seg_flag.tolist() == [
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
    ]


def test_m_share_plan_follows_the_rule(device, monkeypatch) -> None:
    # a host layout whose forward and backward split on one piece table takes M-share where the rule prices it; the
    # default rule keeps FWD_SPLIT_CASES on the summary (they save less than M_SHARE_MIN_SAVING on B200), and so do a
    # device-built table, an architecture outside M_SHARE_ARCHS and a missing stitch_mt
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    for heads, seq_lens in FWD_SPLIT_CASES.values():
        assert not _m_share_meta(device, heads, seq_lens).m_share
    _force_m_share(monkeypatch, device)
    for case, (heads, seq_lens) in FWD_SPLIT_CASES.items():
        meta = _m_share_meta(device, heads, seq_lens)
        split, split1, segs = meta.fwd_split, meta.bwd_split, meta.bwd_segments
        assert split is not None and split1 is not None, case
        assert meta.m_share, case
        assert (
            split.length == segs.segment_len and split.num_segments == segs.num_segments
        ), case
        assert split.num_middle == split1.num_gonly > 0, case
        assert split.middle.tolist() == split1.gonly[0].tolist(), case
    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    index = cake_chain._device_index(device)
    cu = torch.tensor(_cu(seq_lens), dtype=torch.int64)
    device_built = cake_chain._chunk_meta(cu.to(device), None, 1, sum(seq_lens), index, heads[1],
                                          cake_chain.make_options(), heads[1] // heads[0])  # fmt: skip
    assert not device_built.m_share
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "M_SHARE_ARCHS", frozenset())
        assert not _m_share_meta(device, heads, seq_lens).m_share
    with monkeypatch.context() as m:
        available = cake_chain._available
        m.setattr(
            cake_chain,
            "_available",
            lambda stage, a: stage != "stitch_mt" and available(stage, a),
        )
        m.setattr(cake_chain, "_META_FRONT", {})
        assert not _m_share_meta(device, heads, seq_lens).m_share
    assert arch in cake_chain.M_SHARE_ARCHS


@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
def test_m_share_matches_the_fp64_recurrence(case, device, monkeypatch) -> None:
    # the backward on the forward's maps (wave 1 without the summary, the stitch's transpose_m build) keeps o, the final
    # state and every gradient (the initial state's too) within the accuracy gate of the FP64 recurrence
    _force_m_share(monkeypatch, device)
    heads, seq_lens = FWD_SPLIT_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=441)
    gen = torch.Generator().manual_seed(442)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    dht = torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen).to(
        device
    )
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, dht=dht)
    assert _forward_stages(stages) == _split_stages(), stages
    walk = _adjoint_walk(stages)
    assert walk == [
        "dhu_tc2_dvl32_modes",
        "dhu_tc2_dvl32_modes",
        "stitch_mt",
        "dhu_tc2_dvl32_modes",
    ], walk
    ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
    _assert_close_to_reference(got, ref, OUTPUT_NAMES + ("final_state", "dh0"))


@pytest.mark.parametrize("states", [False, True], ids=["no-states", "h0-dht"])
@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
def test_m_share_within_the_budgets_of_the_summary(
    case, states, device, monkeypatch
) -> None:
    # M-share against the same split backward on the adjoint's own maps (dhu_tc2_dvl32_gm and the plain stitch): o and
    # the final state the same bits (the forward's segment state is only kept), every gradient within the budgets
    _force_m_share(monkeypatch, device)
    heads, seq_lens = FWD_SPLIT_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=443)
    h0 = dht = None
    if states:
        gen = torch.Generator().manual_seed(444)
        h0 = (
            0.1
            * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
        ).to(device)
        dht = torch.randn(
            len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen
        ).to(device)
    names = OUTPUT_NAMES + (("final_state", "dh0") if states else ())
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=states, dht=dht)
    monkeypatch.setattr(cake_chain, "M_SHARE_ARCHS", frozenset())
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens, initial_state=h0, output_final_state=states, dht=dht)
    walk = _adjoint_walk(stages)
    assert walk == [
        "dhu_tc2_dvl32_gm",
        "dhu_tc2_dvl32_modes",
        "stitch",
        "dhu_tc2_dvl32_modes",
    ], walk
    _assert_same_bits(got, want, ("o",) + (("final_state",) if states else ()))
    _assert_within_budget(got, want, names)


@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
def test_m_share_runs_give_the_same_bits(case, device, monkeypatch) -> None:
    # one writer per output element and a fixed order: run to run (the second run on, the recorded backward plan), on
    # one stream as on two, and with wave 1 as one launch (M_SHARE_ONE_LAUNCH), the same bits
    _force_m_share(monkeypatch, device)
    heads, seq_lens = FWD_SPLIT_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=445)
    gen = torch.Generator().manual_seed(446)
    h0 = (
        0.1 * torch.randn(len(seq_lens), heads[1], HEAD_DIM, HEAD_DIM, generator=gen)
    ).to(device)
    names = OUTPUT_NAMES + ("final_state", "dh0")
    want = _run(inputs, seq_lens, initial_state=h0, output_final_state=True)
    for _ in range(2):
        _assert_same_bits(
            _run(inputs, seq_lens, initial_state=h0, output_final_state=True),
            want,
            names,
        )
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True, streams=1)
    _assert_same_bits(got, want, names)
    monkeypatch.setattr(cake_chain, "M_SHARE_ONE_LAUNCH", True)
    stages = _record_launches(monkeypatch)
    got = _run(inputs, seq_lens, initial_state=h0, output_final_state=True)
    walk = _adjoint_walk(stages)
    assert walk == ["dhu_tc2_dvl32_modes", "stitch_mt", "dhu_tc2_dvl32_modes"], walk
    _assert_same_bits(got, want, names)


def test_m_share_keeps_the_segment_state_on_the_tape(device, monkeypatch) -> None:
    # the taped forward's segment state is the saved list's last entry (FP32 [segments, HV, 128, 256]); a call that
    # keeps no M-share saves a placeholder there; retain_graph: a second backward on the same tape gives the bits of
    # the first (wave 1 rewrites the B halves, the M halves stay the forward's)
    _force_m_share(monkeypatch, device)
    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=447)
    meta = _m_share_meta(device, heads, seq_lens)
    cu = _cu(seq_lens)
    kwargs = {"scale": 1.0 / math.sqrt(HEAD_DIM), "use_qk_l2norm_in_kernel": True,
              "cu_seqlens": torch.tensor(cu, dtype=torch.int64, device=device),
              "cu_seqlens_cpu": torch.tensor(cu, dtype=torch.int64)}  # fmt: skip
    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    o, _ = cake_gdn_nondet(*leaves.values(), **kwargs)
    saved = _saved_list(o)
    assert len(saved) == 10 + len(cake_plan.STATE_NAMES)
    seg_state = saved[-1]
    assert tuple(seg_state.shape) == (
        meta.fwd_split.num_segments,
        heads[1],
        HEAD_DIM,
        2 * HEAD_DIM,
    )
    assert seg_state.dtype == torch.float32 and seg_state.is_contiguous()
    loss = (o.float() * inputs["do"].float()).sum()
    loss.backward(retain_graph=True)
    first = {"d" + n: leaves[n].grad.clone() for n in leaves}
    for t in leaves.values():
        t.grad = None
    loss.backward()
    second = {"d" + n: leaves[n].grad for n in leaves}
    _assert_same_bits(second, first, GRAD_NAMES)
    with monkeypatch.context() as m:
        m.setattr(cake_chain, "M_SHARE_ARCHS", frozenset())
        o, _ = cake_gdn_nondet(*leaves.values(), **kwargs)
        assert all(s == 0 for s in _saved_list(o)[-1].stride())


@pytest.mark.parametrize("reentrant", [False, True], ids=["non-reentrant", "reentrant"])
def test_m_share_checkpoint_keeps_the_maps_of_the_recompute(
    reentrant, device, monkeypatch
) -> None:
    # under torch.utils.checkpoint the first pass keeps no tape (and no segment state); the recompute keeps them and
    # the backward reads its maps: the gradients are the bits of the plain call
    _force_m_share(monkeypatch, device)
    heads, seq_lens = FWD_SPLIT_CASES["g3"]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=449)
    want = _run(inputs, seq_lens)
    cu = _cu(seq_lens)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64, device=device)
    cu_seqlens_cpu = torch.tensor(cu, dtype=torch.int64)

    def layer(q, k, v, g, beta):
        o, _ = cake_gdn_nondet(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                               cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True)  # fmt: skip
        return o

    leaves = {
        n: inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    }
    stages = _record_launches(monkeypatch)
    o = checkpoint(layer, *leaves.values(), use_reentrant=reentrant)
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    assert stages.count("stitch_mt") == 1 and "dhu_tc2_dvl32_gm" not in stages, stages
    got = {
        "o": o.detach(),
        **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }
    _assert_same_bits(got, want, OUTPUT_NAMES)


@pytest.mark.parametrize("case", sorted(FWD_SPLIT_CASES))
def test_m_share_packed_views_give_the_contiguous_bits(
    case, device, monkeypatch
) -> None:
    # q / k / v as head views of one packed row: the bits of contiguous inputs on every output and gradient
    _force_m_share(monkeypatch, device)
    heads, seq_lens = FWD_SPLIT_CASES[case]
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=451)
    want = _run(inputs, seq_lens)
    stages = _record_launches(monkeypatch)
    got = _run_packed(inputs, seq_lens, heads)
    assert "stitch_mt" in stages, stages
    _assert_same_bits(got, want, OUTPUT_NAMES)


# --------------------------------------------------------------------------- #
# cached device tables and workspaces across streams and CUDA graph captures
# --------------------------------------------------------------------------- #
# the caches of host-built device tables: the chunk tables, the split tables of the adjoint walk and of the forward,
# the identity walks of the forward scan
TABLE_CACHES = ("_HOST_META", "_ADJOINT_SPLITS", "_FORWARD_SPLITS", "_WALKS")
# row -> (the cache of a table that a call reads, sequence lengths, (H, HV), pins, setup): "chunk" rows reach the chunk
# tables through _META_FRONT or, under host_cache=False, through _HOST_META alone; setup "streams" lets the planner see
# the SMs of the two-stream groups, "adjoint-split" / "forward-split" route the layout through the one-wave split of
# the adjoint walk / of the forward scan (the split forward, the only reader of the forward split's tables, takes FP32
# beta: _table_inputs)
TABLE_ROWS = {
    "chunk": ("_HOST_META", (300, 77, 640), (2, 4), {}, None),
    "chunk-uncached-front": ("_HOST_META", (300, 77, 640), (2, 4), {"host_cache": False}, None),
    "chunk-two-streams": ("_HOST_META", STREAM_LAYOUT, (2, 4), STREAM_PINS, "streams"),
    "walk": ("_WALKS", (300, 77, 640), (2, 4), {"lpt": False}, None),
    "adjoint-split": ("_ADJOINT_SPLITS", SPLIT_LAYOUT, SPLIT_HEADS, {}, "adjoint-split"),
    "adjoint-split-uncached-front": ("_ADJOINT_SPLITS", SPLIT_LAYOUT, SPLIT_HEADS, {"host_cache": False},
                                     "adjoint-split"),
    "forward-split": ("_FORWARD_SPLITS", FWD_SPLIT_CASES["g2"][1], FWD_SPLIT_CASES["g2"][0], {}, "forward-split"),
    "forward-split-uncached-front": ("_FORWARD_SPLITS", FWD_SPLIT_CASES["g2"][1], FWD_SPLIT_CASES["g2"][0],
                                     {"host_cache": False}, "forward-split"),
}  # fmt: skip
# layout -> (sequence lengths, (H, HV), setup, the cache of its own table) of the eviction and capture tests; their
# calls take lpt=False, so a forward scan that walks its segments in order reads the cached identity walk
TABLE_LAYOUTS = {
    "chunk-walk": ((300, 77, 640), (2, 4), None, "_WALKS"),
    "adjoint-split": (SPLIT_LAYOUT, SPLIT_HEADS, "adjoint-split", "_ADJOINT_SPLITS"),
    "forward-split": (FWD_SPLIT_CASES["g2"][1], FWD_SPLIT_CASES["g2"][0], "forward-split", "_FORWARD_SPLITS"),
}  # fmt: skip


def _table_setup(monkeypatch, device, setup) -> None:
    """The device view of a :data:`TABLE_ROWS` setup."""
    if setup == "streams":
        _small_device(monkeypatch, device, STREAM_CASES[(2, 4)])
    elif setup == "adjoint-split":
        _split_device(monkeypatch, device, split=True)
    elif setup == "forward-split":
        _fwd_split_or_skip(device)
        monkeypatch.setattr(
            cake_plan,
            "FORWARD_SPLIT_MIN_PIECE",
            types.MappingProxyType(
                {arch: FWD_SPLIT_TEST_MIN_PIECE for arch in cake_chain.FWD_SPLIT_ARCHS}
            ),
        )


def _table_inputs(seq_lens, heads, setup, device, seed) -> dict:
    """The operands of a :data:`TABLE_ROWS` / :data:`TABLE_LAYOUTS` call: FP32 beta under setup "forward-split" (a
    condition of the split forward), else BF16."""
    beta_dtype = torch.float32 if setup == "forward-split" else torch.bfloat16
    return _inputs(seq_lens, heads, beta_dtype, device, seed=seed)


def _check_forward_route(stages, setup) -> None:
    """Under setup "forward-split", the call whose launches are ``stages`` ran the split forward."""
    if setup == "forward-split":
        assert _forward_stages(stages) == _split_stages(), stages


# cache -> the builder of its tables.  A package version whose builders cache their own tables
# (``functools.lru_cache``, without these dicts) is emptied and counted through the builders, so the tests below
# also run on it
TABLE_BUILDERS = {
    "_HOST_META": "_host_chunk_meta",
    "_ADJOINT_SPLITS": "_host_adjoint_split",
    "_FORWARD_SPLITS": "forward_split_tables",
}


def _builder_cache(cache: str):
    """The builder of ``cache``'s tables when it caches them itself, else None."""
    builder = getattr(cake_chain, TABLE_BUILDERS.get(cache, ""), None)
    return builder if hasattr(builder, "cache_info") else None


def _clear_builder_caches(caches=TABLE_CACHES) -> None:
    for cache in caches:
        builder = _builder_cache(cache)
        if builder is not None:
            builder.cache_clear()


def _fresh_table_caches(monkeypatch) -> None:
    """Empty device-table caches, layout front, backward plans and tables kept by captures."""
    for name in ("_META_FRONT", *TABLE_CACHES, "_GRAPH_TABLES", "_BWD_PLANS"):
        monkeypatch.setattr(cake_chain, name, {}, raising=False)
    _clear_builder_caches()


def _table_cache_size(cache: str) -> int:
    builder = _builder_cache(cache)
    if builder is not None:
        return builder.cache_info().currsize
    return len(getattr(cake_chain, cache))


def _cached_entries() -> int:
    """The entries of the layout front and of every device-table cache."""
    return sum(_table_cache_size(name) for name in ("_META_FRONT", *TABLE_CACHES))


def _drop_table_cache(monkeypatch, cache: str) -> None:
    """Empty ``cache`` (and the layout front, which holds the tables of the other caches), so the next call builds
    that table again."""
    if cache != "_WALKS":
        monkeypatch.setattr(cake_chain, "_META_FRONT", {})
    monkeypatch.setattr(cake_chain, cache, {}, raising=False)
    _clear_builder_caches((cache,))


def _cached_tables(caches=TABLE_CACHES) -> list[torch.Tensor]:
    """One view of every storage of the device tables in ``caches`` (none of zero bytes), from the caches and from the
    layout front's ChunkMetas (which hold the tables of every cache, also of builders that cache them themselves)."""
    metas = list(cake_chain._META_FRONT.values())
    tables: list[torch.Tensor] = []
    if "_HOST_META" in caches:
        metas += list(getattr(cake_chain, "_HOST_META", {}).values())
        tables += [m.chunk_start for m in metas]
    if "_ADJOINT_SPLITS" in caches:
        splits = list(getattr(cake_chain, "_ADJOINT_SPLITS", {}).values())
        splits += [m.bwd_split for m in metas if m.bwd_split is not None]
        tables += [s.walk for s in splits]
    if "_FORWARD_SPLITS" in caches:
        splits = list(getattr(cake_chain, "_FORWARD_SPLITS", {}).values())
        splits += [m.fwd_split for m in metas if m.fwd_split is not None]
        for s in splits:
            tables += [s.seg_chunk_start, s.first_index]
    if "_WALKS" in caches:  # an identity walk, or (walk, ready)
        tables += [
            w if isinstance(w, torch.Tensor) else w[0]
            for w in cake_chain._WALKS.values()
        ]
    unique = {t.untyped_storage().data_ptr(): t for t in tables}
    return [t for t in unique.values() if t.untyped_storage().nbytes()]


def _drop_cached_tables() -> None:
    """Drop every reference the package keeps to a cached device table: the caches, the layout front and the backward
    plans (a plan holds its layout's ChunkMeta)."""
    for name in ("_META_FRONT", *TABLE_CACHES, "_BWD_PLANS"):
        getattr(cake_chain, name, {}).clear()
    _clear_builder_caches()


def _storage_addresses(tables) -> list[int]:
    return sorted({t.untyped_storage().data_ptr() for t in tables})


def _issue(inputs, cu_seqlens, cu_seqlens_cpu, *, backward=True, **pins) -> dict:
    """One call on the current stream without a synchronization (``cu_seqlens`` already on the device): the output
    and, with ``backward``, the gradients."""
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(backward)
        for name in ("q", "k", "v", "g", "beta")
    }
    o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                           scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                           use_qk_l2norm_in_kernel=True, **pins)  # fmt: skip
    if not backward:
        return {"o": o}
    (o.float() * inputs["do"].float()).sum().backward()
    return {
        "o": o.detach(),
        **{"d" + name: leaves[name].grad for name in ("q", "k", "v", "g", "beta")},
    }


def _cu_tensors(seq_lens, device):
    cu = _cu(seq_lens)
    return torch.tensor(cu, dtype=torch.int64, device=device), torch.tensor(
        cu, dtype=torch.int64
    )


def _overlapping_streams(device):
    """Two streams whose work overlaps: a launch on the second completes while the first sleeps.  Two streams can share
    one hardware queue, which runs their work in issue order and hides the ordering under test; such a pair is replaced
    by the next streams of the pool."""
    current = torch.cuda.current_stream(device)
    for _ in range(16):
        s1, s2 = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
        s1.wait_stream(current)
        s2.wait_stream(current)
        woke, probe = torch.cuda.Event(), torch.cuda.Event()
        with torch.cuda.stream(s1):
            torch.cuda._sleep(100_000_000)
            woke.record(s1)
        with torch.cuda.stream(s2):
            torch.ones(1, device=device).add_(1)
            probe.record(s2)
        probe.synchronize()
        overlap = not woke.query()
        torch.cuda.synchronize()
        if overlap:
            return s1, s2
    pytest.skip("no two streams of the pool run concurrently on this device")


def _block_states(addresses) -> dict:
    """The caching allocator's state of the block holding each address (``active_allocated``, ``active_pending_free``
    while a stream recorded on a freed tensor may still use it, ``inactive`` once free)."""
    states = {}
    for segment in torch.cuda.memory_snapshot():
        for block in segment["blocks"]:
            for address in addresses:
                if block["address"] <= address < block["address"] + block["size"]:
                    states[address] = block["state"]
    return states


def _native_allocator_or_skip() -> None:
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("reads the state of the native caching allocator's blocks")


@pytest.mark.parametrize("row", sorted(TABLE_ROWS))
def test_a_cached_table_is_read_after_its_write_on_another_stream(
    row, device, monkeypatch
) -> None:
    # Stream s1 sleeps, then its call rebuilds one cached table (its write queued behind the sleep); a call on stream
    # s2 then hits that table.  s2 can run while s1 sleeps (a probe launch on s2 completes), but the s2 call must wait
    # for the write: its work is still queued, and both calls give the bits of a call on the written table.  The
    # adjoint split's tables are read by the backward alone
    cache, seq_lens, heads, pins, setup = TABLE_ROWS[row]
    _table_setup(monkeypatch, device, setup)
    _fresh_table_caches(monkeypatch)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=113)
    stages = _record_launches(monkeypatch)
    want = _run(inputs, seq_lens, **pins)
    _check_forward_route(stages, setup)
    assert _table_cache_size(cache) == 1
    if setup == "streams":
        assert any(
            m.fwd_groups is not None or m.bwd_groups is not None
            for m in cake_chain._META_FRONT.values()
        )
    _drop_table_cache(monkeypatch, cache)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    s1, s2 = _overlapping_streams(device)
    woke, probe, done = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    with torch.cuda.stream(s1):
        torch.cuda._sleep(2_000_000_000)
        woke.record(s1)
        first = _issue(inputs, cu_seqlens, cu_seqlens_cpu, **pins)
    assert _table_cache_size(cache) == 1  # s1 rebuilt the table
    with torch.cuda.stream(s2):
        torch.ones(1, device=device).add_(1)
        probe.record(s2)
        second = _issue(inputs, cu_seqlens, cu_seqlens_cpu, **pins)
        done.record(s2)
    probe.synchronize()
    time.sleep(0.1)  # time for the s2 call to finish if it does not wait for the write
    asleep, queued = not woke.query(), not done.query()
    try:
        assert asleep, "s1 woke before the check (s2 did not run while s1 slept)"
        assert queued, "the call on s2 ran before the table write on s1"
    finally:
        torch.cuda.synchronize()
    _assert_same_bits(first, want, OUTPUT_NAMES)
    _assert_same_bits(second, want, OUTPUT_NAMES)


@pytest.mark.parametrize("layout", sorted(TABLE_LAYOUTS))
def test_an_evicted_table_waits_for_reads_on_another_stream(
    layout, device, monkeypatch
) -> None:
    # A call on stream s2, queued behind a sleep, hits tables built on stream s1; the package then drops every table.
    # The caching allocator must keep each table's block until s2's launches have read it (so no later allocation can
    # overwrite it), and free it after them
    _native_allocator_or_skip()
    seq_lens, heads, setup, cache = TABLE_LAYOUTS[layout]
    _table_setup(monkeypatch, device, setup)
    _fresh_table_caches(monkeypatch)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=137)
    s1, s2 = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    s1.wait_stream(torch.cuda.current_stream(device))
    stages = _record_launches(monkeypatch)
    with torch.cuda.stream(s1):
        want = _run(inputs, seq_lens, lpt=False)
    _check_forward_route(stages, setup)
    assert _table_cache_size(cache) == 1
    addresses = _storage_addresses(_cached_tables())
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    s2.wait_stream(s1)
    with torch.cuda.stream(s2):
        torch.cuda._sleep(2_000_000_000)
        got = _issue(inputs, cu_seqlens, cu_seqlens_cpu, lpt=False)
    _drop_cached_tables()
    gc.collect()
    dropped = _block_states(addresses)
    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        torch.empty(
            1, device=device
        )  # an allocation frees the blocks whose recorded uses are done
    read = _block_states(addresses)
    assert dropped == dict.fromkeys(addresses, "active_pending_free"), dropped
    assert read == dict.fromkeys(addresses, "inactive"), read
    _assert_same_bits(got, want, OUTPUT_NAMES)


@pytest.mark.parametrize("layout", sorted(TABLE_LAYOUTS))
def test_tables_built_inside_a_capture_are_not_cached(
    layout, device, monkeypatch
) -> None:
    # A layout's first call inside a CUDA graph capture writes its tables (the chunk tables, the split tables, the
    # identity walk) only when the graph replays: the capture caches none of them, so an eager call of that layout
    # before any replay builds its own (its backward reads the adjoint split's tables); the replays match too
    seq_lens, heads, setup, _cache = TABLE_LAYOUTS[layout]
    _table_setup(monkeypatch, device, setup)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=127)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    _fresh_table_caches(monkeypatch)
    stages = _record_launches(monkeypatch)
    want = _issue(inputs, cu_seqlens, cu_seqlens_cpu, lpt=False)
    torch.cuda.synchronize()
    _check_forward_route(stages, setup)
    # the stages stay loaded; every table of the layout is now built inside the capture
    _fresh_table_caches(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = _issue(inputs, cu_seqlens, cu_seqlens_cpu, backward=False, lpt=False)["o"]
    assert _cached_entries() == 0
    eager = _issue(inputs, cu_seqlens, cu_seqlens_cpu, lpt=False)
    torch.cuda.synchronize()
    _assert_same_bits(eager, want, OUTPUT_NAMES)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(o, want["o"])


def _graph_step(inputs, leaves, cu_seqlens, cu_seqlens_cpu, **pins):
    """Forward and backward of one call on the current stream into ``leaves``' fresh gradients: the output."""
    for leaf in leaves.values():
        leaf.grad = None
    o, _ = cake_gdn_nondet(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                           scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                           use_qk_l2norm_in_kernel=True, **pins)  # fmt: skip
    (o.float() * inputs["do"].float()).sum().backward()
    return o


def _step_results(o, leaves, clone=False) -> dict:
    got = {"o": o, **{"d" + n: leaves[n].grad for n in ("q", "k", "v", "g", "beta")}}
    return {name: t.detach().clone() for name, t in got.items()} if clone else got


def _leaves(inputs) -> dict:
    return {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }


@pytest.mark.parametrize("layout", sorted(TABLE_LAYOUTS))
def test_a_capture_keeps_the_cached_tables_it_reads(
    layout, device, monkeypatch
) -> None:
    # A capture of a warmed-up layout's forward and backward reads its cached tables (the backward the adjoint split's
    # too).  Once the captured call's autograd graph and every cached table are dropped, the capture alone must keep
    # the tables it reads: their blocks stay allocated, and reusing memory of their sizes (zero-filled) leaves the
    # replays' bits unchanged
    _native_allocator_or_skip()
    seq_lens, heads, setup, cache = TABLE_LAYOUTS[layout]
    _table_setup(monkeypatch, device, setup)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=131)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    _fresh_table_caches(monkeypatch)
    leaves = _leaves(inputs)
    stages = _record_launches(monkeypatch)
    want = _step_results(
        _graph_step(inputs, leaves, cu_seqlens, cu_seqlens_cpu, lpt=False), leaves, True
    )
    torch.cuda.synchronize()
    _check_forward_route(stages, setup)
    assert _table_cache_size(cache) == 1
    tables = _cached_tables()
    addresses = _storage_addresses(tables)
    own = _storage_addresses(_cached_tables(("_HOST_META", cache)))
    sizes = [t.untyped_storage().nbytes() for t in tables]
    del tables
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = _graph_step(inputs, leaves, cu_seqlens, cu_seqlens_cpu, lpt=False)
    got = _step_results(o.detach(), leaves)
    # o's autograd node holds the call's context, with the layout's ChunkMeta and plan
    del o
    _drop_cached_tables()
    gc.collect()
    kept = _block_states(own)
    assert kept == dict.fromkeys(own, "active_allocated"), kept
    junk = [
        torch.zeros(n // 4, dtype=torch.int32, device=device)
        for n in sizes
        for _ in range(4)
    ]
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        _assert_same_bits(got, want, OUTPUT_NAMES)
    # the layout's chunk tables and its own table are kept, nothing but cached tables is
    assert set(own) <= set(cake_chain._GRAPH_TABLES) <= set(addresses)
    del junk


def test_a_capture_takes_its_own_workspaces(device, monkeypatch) -> None:
    # Eager calls on stream s fill s's pooled workspaces (and record and replay the backward's plan over them); a
    # capture on s must take workspaces of its own: once s's pooled buffers are filled with a byte pattern, the
    # replays leave the pattern in place and give the eager bits.  (The self-cleaning zeroed buffers are left alone:
    # every launch leaves them zero.)
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 4), torch.bfloat16, device, seed=139)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    _fresh_table_caches(monkeypatch)
    monkeypatch.setattr(cake_chain, "_POOLED", cake_chain.Workspaces())
    replays = _counted_replays(monkeypatch)
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    leaves = _leaves(inputs)
    with torch.cuda.stream(s):
        for _ in range(2):
            o = _graph_step(inputs, leaves, cu_seqlens, cu_seqlens_cpu)
        want = _step_results(o, leaves, True)
    torch.cuda.synchronize()
    assert len(replays) == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        o = _graph_step(inputs, leaves, cu_seqlens, cu_seqlens_cpu)
    got = _step_results(o, leaves)
    pooled = [
        b
        for table in cake_chain._POOLED.tables.values()
        for role in table.values()
        for b in role.values()
    ]
    assert pooled
    with torch.cuda.stream(s):
        for b in pooled:
            b.view(torch.uint8).fill_(0x5A)
    torch.cuda.synchronize()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        written = [b for b in pooled if not bool((b.view(torch.uint8) == 0x5A).all())]
        assert not written, f"a replay wrote {len(written)} of s's pooled buffers"
        _assert_same_bits(got, want, OUTPUT_NAMES)


def test_a_replayed_backward_plan_reads_tables_kept_for_its_stream(
    device, monkeypatch
) -> None:
    # The tables of a split-adjoint layout are built on stream s1; on stream s2 a call records the backward's plan and
    # a second call, queued behind a sleep, replays it (the plan reads the adjoint split's tables without a lookup of
    # its own).  The package (and the test) then drop every reference to the tables: the caching allocator must keep
    # their blocks until s2's launches have read them, since s2's use was recorded when its forward found them; s1
    # zero-fills memory of their sizes while s2 sleeps, and both calls on s2 give the bits of s1's call
    _native_allocator_or_skip()
    seq_lens, heads, setup, cache = TABLE_LAYOUTS["adjoint-split"]
    _table_setup(monkeypatch, device, setup)
    _fresh_table_caches(monkeypatch)
    replays = _counted_replays(monkeypatch)
    inputs = _table_inputs(seq_lens, heads, setup, device, seed=149)
    s2, s1 = _overlapping_streams(device)  # s1's launches complete while s2 sleeps
    with torch.cuda.stream(s1):
        want = _run(inputs, seq_lens)
    assert _table_cache_size(cache) == 1
    # the tables the backward's plan reads
    tables = _cached_tables(("_HOST_META", cache))
    addresses = _storage_addresses(tables)
    sizes = [t.untyped_storage().nbytes() for t in tables]
    del tables
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    s2.wait_stream(s1)
    woke, filled = torch.cuda.Event(), torch.cuda.Event()
    with torch.cuda.stream(s2):
        first = _issue(inputs, cu_seqlens, cu_seqlens_cpu)
        torch.cuda._sleep(2_000_000_000)
        woke.record(s2)
        second = _issue(inputs, cu_seqlens, cu_seqlens_cpu)
    assert len(replays) == 1, "the second call on s2 replays the plan of the first"
    replays.clear()  # a counted plan holds its layout's ChunkMeta
    _drop_cached_tables()
    gc.collect()
    dropped = _block_states(addresses)
    with torch.cuda.stream(s1):
        junk = [
            torch.zeros(n // 4, dtype=torch.int32, device=device)
            for n in sizes
            for _ in range(4)
        ]
        filled.record(s1)
    filled.synchronize()
    try:
        assert dropped == dict.fromkeys(addresses, "active_pending_free"), dropped
        assert not woke.query(), "s1's zero fill did not run while s2 slept"
    finally:
        torch.cuda.synchronize()
    del junk
    _assert_same_bits(first, want, OUTPUT_NAMES)
    _assert_same_bits(second, want, OUTPUT_NAMES)


def test_a_capture_neither_replays_nor_records_a_backward_plan(
    device, monkeypatch
) -> None:
    # A plan's launches read the pooled workspaces of its stream.  The first call of a key inside a capture on stream s
    # records no plan, eager calls on s then record and replay their own, and a later capture on s does not replay it;
    # both graphs replay the eager bits
    seq_lens = (300, 77, 640)
    inputs = _inputs(seq_lens, (2, 4), torch.bfloat16, device, seed=151)
    cu_seqlens, cu_seqlens_cpu = _cu_tensors(seq_lens, device)
    _fresh_table_caches(monkeypatch)
    monkeypatch.setattr(cake_chain, "_POOLED", cake_chain.Workspaces())
    replays = _counted_replays(monkeypatch)
    leaves = _leaves(inputs)
    warm, s = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    warm.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warm):
        for _ in range(2):  # the tables, the libraries and the warm-up stream's plan
            o = _graph_step(inputs, leaves, cu_seqlens, cu_seqlens_cpu)
        want = _step_results(o, leaves, True)
    torch.cuda.synchronize()
    assert len(replays) == 1 and len(cake_chain._BWD_PLANS) == 1
    graphs = []
    for eager_first in (False, True):
        if eager_first:
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s):
                for _ in range(2):  # the plan of s's key, recorded and replayed
                    o = _graph_step(inputs, leaves, cu_seqlens, cu_seqlens_cpu)
                got = _step_results(o, leaves, True)
            torch.cuda.synchronize()
            assert len(replays) == 2 and len(cake_chain._BWD_PLANS) == 2
            _assert_same_bits(got, want, OUTPUT_NAMES)
        plans, count = dict(cake_chain._BWD_PLANS), len(replays)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            o = _graph_step(inputs, leaves, cu_seqlens, cu_seqlens_cpu)
        assert len(replays) == count, "a capture replayed a backward plan"
        assert plans == cake_chain._BWD_PLANS, "a capture recorded a backward plan"
        graphs.append((graph, _step_results(o, leaves)))
    for graph, got in graphs:
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            _assert_same_bits(got, want, OUTPUT_NAMES)
