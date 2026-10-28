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

Backend suite of the Cake deterministic GDN chunked-training package
(``flashinfer.experimental.cake_gdn_det_train``).

Without a device: the registry (one record per stage and architecture, the argument plans of the call sites, the
scan constants, the ``dhu_tc2`` selection constants and the ``prep_tma`` window), the build manifest (every record's
library, the recorded source digests), the plan pins, the finalize rule, the ``dhu_tc2`` selection rule on the
campaign rows, the ``prep_tma`` dispatch window and the host chunk / segment tables.  On SM100 / SM103: the six outputs ``o, dq, dk, dv, dg, dbeta`` against an FP64 token recurrence on
packed layouts (host and device table routes, FP32 and BF16 ``beta``, every register-scan configuration, the serial
and segment-parallel scans, both adjoint scans, both preps, the fused tail of a value-head group of two and the
unfused tail with both finalize stages, the recompute and uncached forms, the un-normalized ``q`` / ``k`` form), the
initial / final state with its gradients, the pre-scan capacity, and the bitwise contracts: two runs, the two table
routes on one plan, the two preps, ``q`` / ``k`` off the tensor maps' 16-byte alignment (the plain prep), and an
equal-length batch against the same sequences packed.

The suite names its package once, through the shared alias helper, and imports nothing else from flashinfer: a build
that vendors the package under another import path runs it unchanged after pointing the ``PACKAGES`` entry at its
own package.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake_gdn_det_train", "cake_backend")
cake_chain = cake_package("cake_gdn_det_train", "cake_chain")
cake_jit = cake_package("cake_gdn_det_train", "cake_jit")
chunk_gated_delta_rule = cake_package("cake_gdn_det_train").chunk_gated_delta_rule

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
ARCHES = tuple(cake_jit.ARCHES)
HEAD_DIM = 128
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
OUTPUT_NAMES = ("o",) + GRAD_NAMES

# Accuracy gate against the FP64 recurrence of the same BF16 / FP32 inputs, declared before the first run: per output
# tensor, the relative RMS error at most REL_RMS_MAX and the largest absolute error at most MAX_ABS_REL times the
# largest reference magnitude.  The chain stores o / dq / dk / dv in BF16 (a 2^-8 relative rounding step); its
# campaign rows measured at most 8.3e-3 of the largest reference magnitude on every output.
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
         use_qk_l2norm_in_kernel=True, leaves=None, **pins):  # fmt: skip
    """Forward and backward through the package; returns the outputs and gradients (and the final state / its
    initial-state gradient when asked).  ``leaves`` replaces the fresh leaf copies of the inputs it names."""
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
    o, final = chunk_gated_delta_rule(
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


def _assert_bitwise(a: dict, b: dict, names) -> None:
    differing = [n for n in names if not torch.equal(a[n], b[n])]
    assert not differing, f"not bitwise equal: {differing}"


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


FUSED_TAIL_STAGES = (
    "tail_g2",
    "tail_g2_bf16",
)  # two-CTA clusters: the two value heads of a key head


def test_registry_records_are_well_formed() -> None:
    assert len(cake_jit.STAGES) == 29 and len(set(cake_jit.STAGES)) == 29
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
        assert re.fullmatch(r"cake_gdn_det_train_[0-9a-f]{20}", rec["library"]), name
        assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        launch = rec["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        assert launch["cluster"] == (
            (2, 1, 1) if rec["stage"] in FUSED_TAIL_STAGES else (1, 1, 1)
        ), name
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
        ctas_per_sm, window = cake_chain._prep_tma_constants(arch)
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


# The campaign rows the selection rule decides on (serial adjoint plans on both architectures), and the decision
# recorded with the rule's constants: dhu_tc2 on r01, r03, r05, r06, r08, r09 (r01: T_tc = 64 steps x t(64 CTAs),
# B200 154.9 us, GB300 141.9 us).
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
}


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
        assert bwd.segment_len == 0 or max(chunks) <= bwd.segment_len, (row, bwd)
        rec = cake_chain.dhu_tc_pricing(chunks, num_v_heads, 0)
        assert rec["select"] is selected, (row, rec)
        if row == "r01":
            assert abs(rec["T_tc"] - r01_tc) < 0.1, rec


# The prep dispatch window on the two architectures' SM counts (slots = 6 x SMs): the plain prep inside
# (slots, 2.5 x slots], prep_tma outside; 4 x 4096 tokens at H = 8 (2048 CTAs) lie inside on both.
@pytest.mark.parametrize("arch", sorted(RULE_DEVICES))
def test_prep_rule(arch, monkeypatch) -> None:
    props, _r01 = RULE_DEVICES[arch]
    monkeypatch.setattr(cake_chain, "_device_props", lambda _index: props)
    monkeypatch.setattr(cake_backend, "device_arch", lambda _index: arch)
    rule = cake_chain.use_prep_tma
    slots = 6 * props[0]
    assert rule(slots, 1, 0) and not rule(slots + 1, 1, 0)
    assert not rule(int(2.5 * slots), 1, 0) and rule(int(2.5 * slots) + 1, 1, 0)
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
    ("finalize-forced", (100, 640, 77, 300), (2, 6), torch.bfloat16, "host", {"finalize": "finalize"}, True),
    ("finalize-mlp-forced", (100, 640, 77, 300), (2, 6), torch.bfloat16, "host", {"finalize": "finalize_mlp"}, True),
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
def test_long_device_route_runs_and_is_deterministic(layout, device) -> None:
    _id, seq_lens, heads = layout
    inputs = _inputs(seq_lens, heads, torch.bfloat16, device, seed=13)
    first = _run(inputs, seq_lens, route="device")
    _assert_bitwise(first, _run(inputs, seq_lens, route="device"), OUTPUT_NAMES)
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


# --------------------------------------------------------------------------- bitwise contracts


@pytest.mark.parametrize(
    "pins",
    [
        {"scan_config": (32, 2), "segment_chunks": 0, "dhu_scan": "reg"},
        {"scan_config": (64, 1), "segment_chunks": 0, "dhu_scan": "tc"},
        {"scan_config": (32, 2), "segment_chunks": 4, "dhu_scan": "reg"},
        {"scan_config": (64, 1), "segment_chunks": 6, "dhu_scan": "reg"},
    ],
    ids=["serial-32x2", "serial-dhu-tc", "split-32x2", "split-64x1"],
)
def test_table_routes_agree_bitwise(pins, device) -> None:
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=11)
    host = _run(inputs, seq_lens, route="host", **pins)
    dev = _run(inputs, seq_lens, route="device", **pins)
    _assert_bitwise(host, dev, OUTPUT_NAMES)


@pytest.mark.parametrize("route", ["host", "device"])
def test_run_to_run_bitwise(route, device) -> None:
    seq_lens = (2048, 100, 513)
    inputs = _inputs(seq_lens, (2, 6), torch.bfloat16, device, seed=5)
    first = _run(inputs, seq_lens, route=route)
    for _ in range(2):
        _assert_bitwise(first, _run(inputs, seq_lens, route=route), OUTPUT_NAMES)
    uncached = _run(inputs, seq_lens, route=route, host_cache=False)
    _assert_bitwise(first, uncached, OUTPUT_NAMES)


@pytest.mark.parametrize("route", ["host", "device"])
@pytest.mark.parametrize(
    "beta_dtype", [torch.bfloat16, torch.float32], ids=["bf16beta", "fp32beta"]
)
def test_prep_forms_agree_bitwise(route, beta_dtype, device, monkeypatch) -> None:
    seq_lens = (1500, 77, 640)
    inputs = _inputs(seq_lens, (2, 6), beta_dtype, device, seed=19)
    stages = _record_launches(monkeypatch)
    tma = _run(inputs, seq_lens, route=route, prep="tma")
    tma_preps = _preps(stages)
    stages.clear()
    plain = _run(inputs, seq_lens, route=route, prep="plain")
    suffix = "_bf16beta" if beta_dtype == torch.bfloat16 else ""
    assert tma_preps == {"prep_tma" + suffix} and _preps(stages) == {"prep" + suffix}
    _assert_bitwise(tma, plain, OUTPUT_NAMES)


def test_unaligned_qk_take_the_plain_prep(device, monkeypatch) -> None:
    # q / k as contiguous views 8 bytes past a 16-byte boundary: the tensor maps of prep_tma need 16-byte aligned
    # addresses, the plain prep's 4-element loads 8; a prep_tma pin still runs the plain prep, with the same bits.
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
    _assert_bitwise(shifted, aligned, OUTPUT_NAMES)


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
    _assert_bitwise(batched, packed, OUTPUT_NAMES)


# --------------------------------------------------------------------------- input contract


def test_rejects_unsupported_inputs(device) -> None:
    inputs = _inputs((64,), (1, 2), torch.float32, device)
    with pytest.raises(NotImplementedError):
        chunk_gated_delta_rule(
            inputs["q"].half(), inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        )
    cu = torch.tensor([0, 64], dtype=torch.int64, device=device)
    two = {name: torch.cat([t, t], 0) for name, t in inputs.items()}
    with pytest.raises(ValueError):
        chunk_gated_delta_rule(
            two["q"], two["k"], two["v"], two["g"], two["beta"], cu_seqlens=cu
        )
    with pytest.raises(ValueError):
        chunk_gated_delta_rule(
            inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
            segment_chunks=-1,
        )  # fmt: skip
