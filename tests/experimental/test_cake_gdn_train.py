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
(``flashinfer.experimental.cake.cake_gdn_train``).

The suite mirrors the deterministic package's suite (``test_cake_gdn_det_train.py``) with every bitwise contract
replaced by the accuracy budgets of that suite (the same constants, declared before the first run, never derived from a
non-deterministic output): two runs, the two table routes, the longest-first order, two streams, the two preps, the
CUDA graph replay and the equal-length batch agree within the budgets.  It adds the non-deterministic contract: the
explicit ``deterministic=False`` on every entry and the backward's refusal of a tape of the other mode, the interaction
with PyTorch's deterministic algorithms, the det package's names left out, the launches served by this package's
records, the five source head pairs, strided packed QKV views, non-reentrant checkpointing and its zero retained bytes,
gradient accumulation into a nonzero ``.grad``, changed shapes over poisoned reused workspaces, the worst case over
repeated runs, two concurrent caller streams, a resident side-stream kernel and the same-input comparison with the
deterministic package.  The tail's own contract: it serves every value-head group, leaves its counters zero after
every launch, launches in segments that give the bits of one launch, keeps the deterministic bits of ``o``, ``dv``,
``dg`` and ``dbeta`` and its own bits run to run, and protocol controls (stale counters must break ``dq`` / ``dk``,
stale workspace slots must not).

The suite names its packages through the shared alias helper and imports nothing else from flashinfer.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import re
import types
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_gdn_train", "cake_backend")
cake_chain = cake_package("cake.cake_gdn_train", "cake_chain")
cake_jit = cake_package("cake.cake_gdn_train", "cake_jit")
cake_gdn_train = cake_package("cake.cake_gdn_train")
# The deterministic package: the same-input comparison point (never a source of budgets).
cake_gdn_det_train = cake_package("cake_gdn_det_train")
det_backend = cake_package("cake_gdn_det_train", "cake_backend")


def chunk_gated_delta_rule_nd(*args, **kwargs):
    """The package's entry point with the mode it requires."""
    return cake_gdn_train.chunk_gated_delta_rule(*args, deterministic=False, **kwargs)


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
    entry = (
        cake_gdn_det_train.chunk_gated_delta_rule_det
        if det
        else chunk_gated_delta_rule_nd
    )
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
FUSED_TAIL_STAGES = {"tail_g2": 2, "tail_g2_bf16": 2, "tail_g3": 3, "tail_g3_bf16": 3}


def test_registry_records_are_well_formed() -> None:
    assert len(cake_jit.STAGES) == 34 and len(set(cake_jit.STAGES)) == 34
    assert set(cake_chain.LAUNCH_ARGS) == set(cake_jit.STAGES)
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
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        assert launch["cluster"] == (FUSED_TAIL_STAGES.get(rec["stage"], 1), 1, 1), name
        assert len(launch["block"]) == 3 and launch["block"][0] % 32 == 0, name
        assert launch["dynamic_smem_bytes"] >= 0, name
        # the record's kernel parameters are the call site's, possibly reordered (the chain reorders them)
        params = cake_backend.parameters(name)
        assert sorted(params) == sorted(cake_chain.LAUNCH_ARGS[rec["stage"]]), name
        libraries.add(rec["library"])
    # one program per record: a kernel swap renames exactly the records it touches
    assert len(libraries) == len(cake_jit.MODULES)


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


@pytest.mark.parametrize("save_state", [True, False], ids=["saved", "recompute"])
@pytest.mark.parametrize("lpt", [True, False], ids=["lpt", "caller-order"])
@pytest.mark.parametrize("heads", sorted(STREAM_CASES), ids=["g2", "g3"])
def test_two_streams_match_one_stream(
    heads, lpt, save_state, device, monkeypatch
) -> None:
    seq_lens = STREAM_LAYOUT
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
        # the side stream: wy_tc and fwd_ho of the forward's group A, dhu_tc2 of the adjoint's group B (the fused
        # tail of a value-head group of two only)
        expected = ["wy_tc", "fwd_ho"] + (
            ["dhu_tc2"] if heads[1] == 2 * heads[0] else []
        )
        assert side == (expected if streams == 2 else []), (streams, launches[start:])
    _assert_within_budget(runs[0], runs[1], names)
    _assert_within_budget(runs[0], runs[2], names)
    if lpt and save_state:
        ref = _reference(inputs, seq_lens, initial_state=h0, dht=dht)
        _assert_close_to_reference(runs[0], ref, names)


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
    assert side_launches == ["wy_tc", "fwd_ho", "dhu_tc2"] * 16, side_launches


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
        o, _ = chunk_gated_delta_rule_nd(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
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
        o, _ = chunk_gated_delta_rule_nd(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
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
        chunk_gated_delta_rule_nd(
            inputs["q"].half(), inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        )
    cu = torch.tensor([0, 64], dtype=torch.int64, device=device)
    two = {name: torch.cat([t, t], 0) for name, t in inputs.items()}
    with pytest.raises(ValueError):
        chunk_gated_delta_rule_nd(
            two["q"], two["k"], two["v"], two["g"], two["beta"], cu_seqlens=cu
        )
    with pytest.raises(ValueError):
        chunk_gated_delta_rule_nd(
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
        chunk_gated_delta_rule_nd(
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
    o, final = chunk_gated_delta_rule_nd(
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
        o, final = chunk_gated_delta_rule_nd(
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


# --------------------------------------------------------------------------- the mode contract (no device)

ENTRY_NAMES = (
    "chunk_gated_delta_rule",
    "ChunkGatedDeltaRuleFunction",
    "gdn_chunk_forward",
    "gdn_chunk_backward",
)
DET_NAMES = ("chunk_gated_delta_rule_det", "ChunkGatedDeltaRuleDetFunction", "gdn_chunk_det_forward",
             "gdn_chunk_det_backward")  # fmt: skip
_ALERT = "does not have a deterministic implementation"


def test_the_package_exports_the_det_free_names_only() -> None:
    for name in ENTRY_NAMES:
        for module in (cake_gdn_train, cake_chain):
            assert name in module.__all__ and getattr(module, name) is getattr(
                cake_chain, name
            ), name
    for name in DET_NAMES:
        for module in (cake_gdn_train, cake_chain):
            assert not hasattr(module, name) and name not in module.__all__, name
    # the deterministic package keeps its names and its aliases, distinct objects from this package's entries
    assert (
        cake_gdn_det_train.chunk_gated_delta_rule
        is cake_gdn_det_train.chunk_gated_delta_rule_det
    )
    assert (
        cake_gdn_train.chunk_gated_delta_rule
        is not cake_gdn_det_train.chunk_gated_delta_rule
    )
    assert cake_jit.PACKAGE == "flashinfer.experimental.cake.cake_gdn_train"
    assert cake_jit.LIBRARY_PREFIX == "cake_gdn_train"


def _entry_calls(**mode) -> dict:
    """One call per entry with ``mode`` (``{}``: the default); with an accepted mode each fails after its mode check."""
    q = torch.zeros(1, 64, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros(1, 64, 2, HEAD_DIM, dtype=torch.bfloat16)
    g, beta = torch.zeros(1, 64, 2), torch.zeros(1, 64, 2)
    options = cake_chain.make_options()
    apply_args = (
        q,
        q,
        v,
        g,
        beta,
        1.0,
        None,
        False,
        None,
        None,
        True,
        options,
        *mode.values(),
    )
    return {
        "chunk_gated_delta_rule": lambda: cake_gdn_train.chunk_gated_delta_rule(
            q, q, v, g, beta, **mode
        ),
        "ChunkGatedDeltaRuleFunction": lambda: cake_gdn_train.ChunkGatedDeltaRuleFunction.apply(
            *apply_args
        ),
        "gdn_chunk_forward": lambda: cake_chain.gdn_chunk_forward(
            q, q, v, g, beta, meta=None, arch=None, **mode
        ),
        "gdn_chunk_backward": lambda: cake_chain.gdn_chunk_backward(
            {"deterministic": False}, None, **mode
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


@pytest.mark.parametrize("mode", [{}, {"deterministic": True}, {"deterministic": None}, {"deterministic": 0}],
                         ids=["default", "true", "none", "zero"])  # fmt: skip
def test_every_entry_requires_deterministic_false(mode) -> None:
    for name, call in _entry_calls(**mode).items():
        with pytest.raises(
            ValueError,
            match=f"^{name} is the non-deterministic route and takes deterministic=False",
        ):
            call()
    # the refusal comes first, also with PyTorch's deterministic algorithms on
    with _deterministic_algorithms(True):
        for name, call in _entry_calls(**mode).items():
            with pytest.raises(
                ValueError, match=f"^{name} is the non-deterministic route"
            ):
                call()


def test_torch_deterministic_algorithms_do_not_alert_the_forward_entries() -> None:
    # the forward entries run fixed-order programs: under PyTorch's deterministic algorithms (an alert as an error,
    # a warning as an error) each proceeds to its operand checks, which refuse the host operands here
    for warn_only in (False, True):
        with (
            _deterministic_algorithms(True, warn_only=warn_only),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            for name, call in _entry_calls(deterministic=False).items():
                if name == "gdn_chunk_backward":
                    continue
                with pytest.raises(Exception) as info:  # noqa: PT011 - any operand refusal, never the alert
                    call()
                assert _ALERT not in str(info.value), (name, warn_only, info.value)


@pytest.mark.parametrize(
    "tape",
    [{}, {"deterministic": True}, {"deterministic": None}],
    ids=["det-tape", "true", "none"],
)
def test_backward_refuses_a_tape_of_the_other_mode(tape) -> None:
    with pytest.raises(ValueError, match="this tape records the deterministic mode"):
        cake_chain.gdn_chunk_backward(
            {"arch": ARCHES[0], **tape}, None, deterministic=False
        )


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

    tape = cake_chain.gdn_chunk_forward(
        q, k, v, g, beta, meta=meta_of(cake_chain), arch=arch, deterministic=False
    )
    assert tape["deterministic"] is False
    grads = cake_chain.gdn_chunk_backward(tape, do, deterministic=False)
    assert all(bool(torch.isfinite(grads[n]).all()) for n in GRAD_NAMES)
    det_tape = det_chain.gdn_chunk_det_forward(
        q, k, v, g, beta, meta=meta_of(det_chain), arch=arch
    )
    with pytest.raises(ValueError, match="this tape records the deterministic mode"):
        cake_chain.gdn_chunk_backward(det_tape, do, deterministic=False)


# (H, HV) and tail pins: every tail the rule dispatches or a pin forces
FIXED_ORDER_CASES = (
    ((2, 4), {}),
    ((2, 6), {}),
    ((1, 3), {}),
    ((1, 4), {"crossover0": True}),  # the rule with TAIL_MIN_CTAS = 0: the package tail
    ((2, 2), {"tail": "tail"}),
    ((2, 6), {"tail": "tail"}),
    ((2, 4), {"tail": "tail_g2"}),
    ((2, 6), {"tail": "tail_g3"}),
    ((2, 6), {"tail": "unfused"}),
)


@pytest.mark.parametrize(
    "heads, pins",
    FIXED_ORDER_CASES,
    ids=["rule-g2", "rule-g3", "rule-g3-h1", "rule-g4-large", "g1-tail", "g3-tail", "g2-tail-g2", "g3-tail-g3",
         "g3-unfused"],
)  # fmt: skip
def test_torch_deterministic_algorithms_change_nothing(
    heads, pins, device, monkeypatch
) -> None:
    # every launch of the package is fixed-order: under PyTorch's deterministic algorithms (an alert as an error,
    # or as a warning) the autograd route and the low-level route run without an alert, to the same bits
    seq_lens = (300, 77)
    pins = dict(pins)
    if pins.pop("crossover0", False):
        monkeypatch.setattr(cake_chain, "TAIL_MIN_CTAS", _min_ctas_table(0))
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
        tape = cake_chain.gdn_chunk_forward(q, k, v, g, beta, meta=meta, arch=arch, options=options,
                                            deterministic=False)  # fmt: skip
        cake_chain.gdn_chunk_backward(tape, do, deterministic=False)


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
            set(DET_TAIL_STAGES) | set(TAIL_STAGES) - {tail}
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


# (H, HV) -> the tail the rule dispatches on a large call, per architecture: tail_g2 at G = 2 and tail_g3 at G = 3
# (both measured faster than the package tail on B200 and GB300; sm_107a as sm_103a until measured), else the
# package tail
TAIL_RULE = {
    arch: {
        (2, 4): "tail_g2",
        (8, 16): "tail_g2",
        (16, 32): "tail_g2",
        (1, 3): "tail_g3",
        (2, 6): "tail_g3",
        (16, 48): "tail_g3",
        (2, 2): "tail",
        (16, 16): "tail",
        (1, 4): "tail",
        (4, 16): "tail",
    }
    for arch in ("sm_100a", "sm_103a", "sm_107a")
}
# the package tail's (chunk, value head) CTA floor per architecture and group class (1: G = 1; 4: the larger groups
# without a cluster tail), from the backward crossovers measured on B200, GB300 and R200
TAIL_MIN_CTAS_TABLE = {
    "sm_100a": {1: 2048, 4: 2048},
    "sm_103a": {1: 2048, 4: 1024},
    "sm_107a": {1: 1024, 4: 1024},
}


def _min_ctas_table(value: int) -> dict[str, dict[int, int]]:
    return {arch: {1: value, 4: value} for arch in TAIL_MIN_CTAS_TABLE}


def test_tail_min_ctas_table() -> None:
    assert cake_chain.TAIL_MIN_CTAS == TAIL_MIN_CTAS_TABLE
    for arch, floors in TAIL_MIN_CTAS_TABLE.items():
        assert cake_chain.tail_min_ctas(arch, 1) == floors[1], arch
        for group in (3, 4, 5, 8, 16):
            assert cake_chain.tail_min_ctas(arch, group) == floors[4], (arch, group)
        # the rule at the floor: the package tail; one chunk below: unfused
        for h, hv in ((4, 4), (1, 4), (2, 16)):
            edge = floors[1 if hv == h else 4]
            assert (
                cake_chain.tail_form(h, hv, arch=arch, num_chunks=edge // hv)
                == cake_chain.TAIL_DEFAULT
            ), (arch, h, hv)
            assert (
                cake_chain.tail_form(h, hv, arch=arch, num_chunks=edge // hv - 1)
                == "unfused"
            ), (arch, h, hv)


# the stages a backward launches per tail (beyond the shared scans), and the ones it never launches
_TAIL_SETS = {f: {f, f + "_bf16"} for f in ("tail", "tail_g2", "tail_g3")}
_TAIL_SETS["unfused"] = {"dqkwg", "finalize", "finalize_mlp"}
TAIL_LAUNCHES = {
    form: (want, set().union(*(other for f, other in _TAIL_SETS.items() if f != form)))
    for form, want in _TAIL_SETS.items()
}


@pytest.mark.parametrize("min_tiles", [None, 0], ids=["rule", "crossover0"])
@pytest.mark.parametrize(
    "heads",
    [(2, 4), (1, 3), (2, 6), (2, 2), (1, 4)],
    ids=["g2", "g3-h1", "g3", "g1", "g4"],
)
def test_the_dispatched_tail_runs(heads, min_tiles, device, monkeypatch) -> None:
    # without a pin the backward launches the tail the rule names, and only that one; the pins launch theirs
    seq_lens = TAIL_LAYOUT
    if min_tiles is not None:
        monkeypatch.setattr(cake_chain, "TAIL_MIN_CTAS", _min_ctas_table(min_tiles))
    inputs = _inputs(seq_lens, heads, torch.float32, device, seed=289)
    launches = _record_tail_launches(monkeypatch, device)
    arch = cake_backend.device_arch(cake_chain._device_index(device))
    group = heads[1] // heads[0]
    forms = [None, "tail", "unfused"] + ([f"tail_g{group}"] if group in (2, 3) else [])
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
            edge = cake_chain.tail_min_ctas(arch, hv // h)
            assert (
                cake_chain.tail_form(h, hv, arch=arch, num_chunks=-(-edge // hv))
                == cake_chain.TAIL_DEFAULT
            ), (arch, h, hv)
            assert (
                cake_chain.tail_form(h, hv, arch=arch, num_chunks=-(-edge // hv) - 1)
                == "unfused"
            ), (arch, h, hv)
        # a cluster tail serves its group at any size
        assert (
            cake_chain.tail_form(2, 4, arch=arch, num_chunks=1)
            == cake_chain.tail_form(2, 4, arch=arch)
            == "tail_g2"
        )
        assert (
            cake_chain.tail_form(2, 6, arch=arch, num_chunks=1)
            == cake_chain.tail_form(2, 6, arch=arch)
            == "tail_g3"
        )
    # an architecture without a cluster tail for a group of three: tail_g3 below the package tail's crossover
    monkeypatch.setitem(cake_chain.CLUSTER_TAILS, "sm_107a", {2: "tail_g2"})
    edge = cake_chain.tail_min_ctas("sm_107a", 3)
    assert (
        cake_chain.tail_form(2, 6, arch="sm_107a", num_chunks=-(-edge // 6) - 1)
        == "tail_g3"
    )
    assert (
        cake_chain.tail_form(2, 6, arch="sm_107a", num_chunks=-(-edge // 6))
        == cake_chain.TAIL_DEFAULT
    )
    assert (
        cake_chain.tail_form(2, 4, "tail_g2", arch="sm_100a") == "tail_g2"
        and cake_chain.tail_form(2, 6, "unfused", arch="sm_103a") == "unfused"
        and cake_chain.tail_form(2, 6, "tail_g3", arch="sm_100a") == "tail_g3"
    )
    with pytest.raises(
        ValueError, match="tail_g2 form serves a value-head group of two"
    ):
        cake_chain.tail_form(2, 6, "tail_g2", arch="sm_103a")
    with pytest.raises(
        ValueError, match="tail_g3 form serves a value-head group of three"
    ):
        cake_chain.tail_form(2, 4, "tail_g3", arch="sm_103a")
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
    # and the same bits run to run (the planner's tables, as the deterministic package plans them)
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
    for name in ("o", "dv", "dg", "dbeta"):
        assert torch.equal(got[name], det[name]), name
        assert all(torch.equal(got[name], run[name]) for run in pinned), name
    for run in pinned:  # the fixed-order tails keep the deterministic package's bits
        assert all(torch.equal(run[name], det[name]) for name in OUTPUT_NAMES)
    _assert_within_budget(got, det, ("dq", "dk"))


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
    # same segment table: the same bits on every output (the rule's fixed-order cluster tails at G = 2 / 3), with an
    # initial state and a final-state gradient
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
        if rec["select"]:
            assert rec["saving"] >= cake_chain.TC_PLAN_MIN_SAVING, (row, rec)
            assert rec["best"]["T"] <= cake_chain.TC_PLAN_GAIN * rec["plan_T"], (
                row,
                rec,
            )
            assert (
                0 <= rec["segment_len"] <= cap
                and tuple(bwd.config) == cake_chain.SDT_CONFIG
            ), (row, rec)
        if arch in _TC_BOTH:
            assert rec["select"] is (arch in replaced_on), (row, rec)


@pytest.mark.parametrize("heads", [(8, 16), (8, 24)], ids=["g2", "g3"])
def test_tc_plan_changes_only_the_adjoint_table(heads, device, monkeypatch) -> None:
    # one 16384-token sequence (the r07 form; gates kept above the floors with TC_PLAN_GAIN = 1, no saving floor):
    # the tc-priced adjoint table leaves the forward's bits, runs on dhu_tc2 (no dv_local stage), repeats bitwise
    # (fixed-order cluster tails) and stays within the budgets of the planner's table; a plan pin keeps the planner
    seq_lens = (16384,)
    index = cake_chain._device_index(device)
    monkeypatch.setattr(cake_chain, "TC_PLAN_MIN_SAVING", 0.0)
    monkeypatch.setattr(cake_chain, "TC_PLAN_GAIN", 1.0)
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
        want = ("dhu_pre_dvl_r64k1", "dhu_tc2") if rec["segment_len"] else ("dhu_tc2",)
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
        o, _ = chunk_gated_delta_rule_nd(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
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
    o, _ = chunk_gated_delta_rule_nd(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM),
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
        o, _ = chunk_gated_delta_rule_nd(q, k, v, g, beta, scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
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
        o, _ = chunk_gated_delta_rule_nd(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
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


@pytest.mark.parametrize("heads", [(2, 4), (2, 6)], ids=["g2", "g3"])
def test_cuda_graph_replay_matches_eager(heads, device) -> None:
    seq_lens = STREAM_LAYOUT
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=269)
    eager = _run(inputs, seq_lens, streams=1)
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
        o, _ = chunk_gated_delta_rule_nd(leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                                         scale=1.0 / math.sqrt(HEAD_DIM), cu_seqlens=cu_seqlens,
                                         cu_seqlens_cpu=cu_seqlens_cpu, use_qk_l2norm_in_kernel=True,
                                         streams=1)  # fmt: skip
        (o.float() * inputs["do"].float()).sum().backward()
        return o

    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup):
        for _ in range(2):
            step()
    torch.cuda.current_stream(device).wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = step()
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
