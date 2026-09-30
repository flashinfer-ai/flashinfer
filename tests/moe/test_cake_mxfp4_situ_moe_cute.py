#
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tests of the ``backend="cake_cute"`` package of the Cake MXFP4 SiTU routed MoE.

Host-only tests pin the manifest (schema, backend, per-file SHA-256), the plan's
decision table against the hand-written plan's observed decisions, and that every
form the plan can select over the served geometry is carried by the package.  GPU
tests need ``sm_103a`` (B300 / GB300) and run the exported chain end to end:
the contract's gate-oracle criterion on the contract's input domain, run-to-run and
CUDA-graph-replay identity on the deterministic path (tokens with at most one local
contribution) with the BF16 accumulation budget on the reduced tokens, the same
class between the two backends (the same Cake programs lowered twice) and, when the
installed FlashInfer exposes the hand-written wrapper, against it.
"""

import hashlib
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

import flashinfer.fused_moe.cake_cute as pkg
from flashinfer.fused_moe.cake_cute import cake_mxfp4_situ_moe_kernels as kernels
from flashinfer.fused_moe.cake_cute import cake_mxfp4_situ_moe_plan as plan

BACKEND = "cake_cute"
SIBLING_MODULE = "flashinfer.fused_moe.cake"

# Kimi K3 geometry of the served rows (896 experts, top-16, H 7168, I 3072).
EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 896, 16, 7168, 3072
EP8 = dict(num_local_experts=112, local_expert_offset=336)
TP8 = dict(
    num_local_experts=EXPERTS,
    local_expert_offset=0,
    intermediate_shard=INTERMEDIATE // 8,
)
MODES = tuple(
    kernels.manifest()["contract"]["routing_modes"]
)  # routing modes carried by this package


def _runner(layout, **overrides):
    return plan.CakeMxfp4MoEWrapper(
        EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, backend=BACKEND, **layout, **overrides
    )


def _checkout_root():
    return Path(kernels.__file__).resolve().parents[3]


# --- manifest ----------------------------------------------------------------------


def test_manifest_is_authentic():
    manifest = kernels.manifest()
    assert manifest["schema"] == "cake.library_export.v5"
    assert manifest["contract"]["backend"] == BACKEND
    assert manifest["contract"]["arch"] == kernels.ARCH == "sm_103a"
    assert len(manifest["producer_revision"]) == 40
    assert manifest["contract"]["cake_revision"] == manifest["producer_revision"]
    for entry in manifest["files"]:
        path = kernels.source_path(entry["path"])
        assert path.stat().st_size == entry["bytes"]
    stages = [item["route"]["stage"] for item in kernels.modules()]
    assert len(stages) == len(set(stages)) == len(manifest["contract"]["forms"])
    compiler_backend = {"cake": "cuda_cpp", "cake_cute": "cutedsl"}[BACKEND]
    for item in kernels.modules():
        form = item["route"]["form"]
        assert form["gate"] in {"e2e_registry", "plan_gate", "plan_variant"}
        assert compiler_backend in form["backends"]
    with pytest.raises(KeyError):
        kernels.record("not_a_stage")


def test_integration_files_match_their_manifest_hashes():
    root = _checkout_root()
    manifest = kernels.manifest()
    listed = {entry["path"] for entry in manifest["contract"]["integration_files"]}
    assert Path(plan.__file__).resolve().relative_to(root).as_posix() in listed
    assert Path(kernels.__file__).resolve().relative_to(root).as_posix() in listed
    for entry in manifest["contract"]["integration_files"]:
        path = root / entry["path"]
        if not path.is_file():  # installed wheels do not carry tests/ and docs/
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry[
            "path"
        ]


def test_unsupported_forms_are_declared_not_faked():
    unsupported = kernels.unsupported_forms()
    stages = {item["route"]["stage"] for item in kernels.modules()}
    assert not (set(unsupported) & stages)
    if BACKEND == "cake_cute":
        assert {"moe_sort_init", "moe_sort_coop_dual_mixed"} <= set(unsupported)
        assert all("grid.sync" in reason for reason in unsupported.values())
    else:
        assert unsupported == {}
        assert {"moe_sort_init", "moe_sort_coop_dual_mixed"} <= stages
        assert kernels.load("moe_sort_coop_dual_mixed").cooperative


# --- decision table (hand-written plan facts) ----------------------------------------


@pytest.mark.parametrize(
    "layout, T, gemm1, gemm2, tiles, workspace",
    [
        (EP8, 1, (8, 8), (8, 8, 1), 16, 441600),
        (EP8, 8, (8, 8), (8, 8, 1), 114, 2929920),
        (EP8, 16, (8, 8), (8, 8, 1), 130, 3337984),
        (EP8, 17, (16, 4), (16, 8, 1), 122, 6231040),
        (EP8, 128, (16, 4), (16, 8, 1), 233, 11886336),
        (EP8, 129, (32, 4), (32, 8, 1), 173, 17621504),
        (EP8, 256, (32, 4), (32, 8, 1), 236, 24040448),
        (EP8, 257, (32, 4), (32, 12, 1), 237, 24142848),
        (EP8, 512, (32, 4), (32, 12, 1), 364, 37083136),
        (EP8, 1024, (32, 4), (32, 12, 1), 620, 63168512),
        (TP8, 1, (8, 8), (8, 12, 1), 16, 92928),
        (TP8, 8, (8, 8), (8, 12, 1), 128, 452608),
        (TP8, 16, (8, 8), (8, 4, 2), 256, 864768),
    ],
)
def test_decision_table_pins_the_hand_written_plan(
    layout, T, gemm1, gemm2, tiles, workspace
):
    runner = _runner(layout)
    decision = runner.decide(T)
    assert decision.supported, decision.reason
    assert (decision.gemm1.n_tile, decision.gemm1.kbps) == gemm1
    assert (decision.gemm2.n_tile, decision.gemm2.kbps, decision.gemm2.m_group) == gemm2
    assert decision.tiles == tiles
    assert runner.get_workspace_size(T) == workspace
    assert decision.fused_routing and decision.clear_output and decision.pdl
    assert decision.path == "plain" and not decision.missing_forms
    assert decision.launches == (
        "routing_fused",
        "gemm1_swapab_situ",
        "gemm2_swapab_finalize",
    )
    assert all(launch.traced for launch in decision.launch_plan)


@pytest.mark.parametrize(
    "layout, T, needle",
    [
        (EP8, 1025, "dense grouped-GEMM"),
        (EP8, 4096, "dense grouped-GEMM"),
        (TP8, 17, "two-stage finalize"),
        (TP8, 128, "split form"),
        (TP8, 1024, "split form"),
        (TP8, 1025, "hybrid form"),
    ],
)
def test_refused_rows_carry_the_hand_written_reason(layout, T, needle):
    runner = _runner(layout)
    decision = runner.decide(T)
    assert not decision.supported
    assert needle in decision.reason
    # The refused row still carries the hand-written launch plan; the launches without a traced form name
    # the missing IR form (never substituted), and ``plan`` refuses with the same reason.
    assert decision.path != "plain" and decision.launches and decision.missing_forms
    untraced = [launch for launch in decision.launch_plan if not launch.traced]
    assert untraced and all(launch.missing for launch in untraced)
    assert "missing IR forms" in decision.reason


def test_layouts_resolve():
    assert plan.resolve_layout(EXPERTS, INTERMEDIATE, **EP8).mode == "expert_parallel"
    assert (
        plan.resolve_layout(EXPERTS, INTERMEDIATE, **TP8).mode == "moe_tensor_parallel"
    )
    assert plan.resolve_layout(EXPERTS, INTERMEDIATE).mode == "single"
    with pytest.raises(ValueError):
        plan.resolve_layout(
            EXPERTS,
            INTERMEDIATE,
            num_local_experts=112,
            local_expert_offset=0,
            intermediate_shard=INTERMEDIATE // 8,
        )


def test_wrapper_refuses_the_other_backend():
    with pytest.raises(ValueError):
        plan.CakeMxfp4MoEWrapper(
            EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, backend="not-a-backend", **EP8
        )


# --- coverage: every plan selection over the served geometry is in the package -------------


def _served_decisions():
    for layout in (EP8, TP8):
        runner = _runner(layout)
        for T in range(1, runner.swapab_max_tokens + 1):
            decision = runner.decide(T)
            if decision.supported:
                yield runner, decision


def test_every_plan_selection_resolves_to_a_module():
    seen = set()
    for runner, decision in _served_decisions():
        g1, g2 = decision.gemm1, decision.gemm2
        seen.add(kernels.select_gemm1(g1.n_tile, g1.kbps).stage)
        seen.add(kernels.select_gemm2(g2.n_tile, g2.kbps, g2.m_group).stage)
        for mode in MODES:
            cfg = plan.routing_config(runner.policy, decision, mode=mode)
            kernel = kernels.select_routing(cfg)
            assert kernel.form["cluster"] == cfg.cluster and kernel.form["mode"] == mode
            seen.add(kernel.stage)
    assert "separate_bf16" in MODES and len(seen) >= 9 + 10 * len(MODES)
    with pytest.raises(KeyError, match="not in this package"):
        kernels.select_gemm2(192, 4, 7)


def test_gemm_forms_carry_the_trace_time_constants():
    kernel = kernels.select_gemm2(32, 12, 1)
    form = kernel.form
    assert form["kind"] == "gemm2_swapab" and form["is_situ"] is False
    assert form["n_tile"] == 32 and form["kbps"] == 12 and form["m_group"] == 1
    # The launch's dynamic shared memory is the form's smem plus the lowering's alignment slack.
    assert kernel.block[0] == form["threads"]
    assert form["smem_bytes"] <= kernel.dynamic_smem_bytes <= form["smem_bytes"] + 4096
    assert kernel.use_pdl is True
    situ = kernels.select_gemm1(8, 8).form
    assert (
        situ["kind"] == "gemm1_swapab" and situ["is_situ"] is True and situ["kbps"] == 8
    )


# --- GPU -----------------------------------------------------------------------------


def _sm_103a():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (10, 3)


gpu = pytest.mark.skipif(
    not _sm_103a() or not kernels.is_available(),
    reason="needs sm_103a and the package kernels",
)

# --- contract-domain fixture (ported from the producer's contract evaluator) -------------------
#
# The GPU tests exercise the contract's input domain: MXFP8 activations quantized from unit-scale
# Gaussians, MXFP4 expert weights quantized from Gaussians scaled by 1/sqrt(fan-in), positive BF16
# route weights normalized per token.  Uniform random weight bytes (``_adversarial_inputs``) drive
# the chain to O(1e3) outputs with heavy cancellation, outside the domain the gate oracle models;
# they are kept for finite / non-zero / form-selection asserts only, never for a reference-class
# assert.

GROUP = 32
E4M3_MAX = 448.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0
# The contract's correctness criterion against its gate oracle (full-element atol / rtol).
GATE_ATOL, GATE_RTOL = 1.0, 0.1


def _seed(base, role, index=0):
    role_ids = {"routing_weights": 1, "activations": 2, "w1": 3, "w2": 4}
    return (int(base) * 1_000_003 + role_ids[role] * 10_007 + int(index) * 97) & (
        (1 << 63) - 1
    )


def decode_ue8m0(scales, dtype=torch.float64):
    codes = scales.view(torch.uint8).to(torch.int32)
    values = torch.ldexp(torch.ones_like(codes, dtype=dtype), codes - 127)
    return torch.where(codes == 255, torch.full_like(values, float("nan")), values)


def decode_mxfp4(packed, scales, dtype=torch.float64):
    packed = packed.to(torch.uint8)
    table = torch.tensor(
        list(E2M1_VALUES) + [-v for v in E2M1_VALUES], device=packed.device, dtype=dtype
    )
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    blocks = table[nibbles.long()].reshape(*nibbles.shape[:-1], -1, GROUP)
    return (
        blocks * decode_ue8m0(scales.to(packed.device), dtype).unsqueeze(-1)
    ).flatten(-2)


def decode_mxfp8(values, scales, dtype=torch.float64):
    blocks = values.to(dtype).reshape(*values.shape[:-1], -1, GROUP)
    return (blocks * decode_ue8m0(scales, dtype).unsqueeze(-1)).flatten(-2)


def quantize_mxfp8(values):
    """FP32 -> MXFP8 (group 32, upward UE8M0 scale, nearest E4M3); the gate oracle's requantizer."""
    shape = values.shape
    blocks = values.float().reshape(*shape[:-1], -1, GROUP)
    scale = blocks.abs().amax(dim=-1) * (1.0 / E4M3_MAX)
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    code = (exponent + bump.to(torch.int32)).clamp(0, 254)
    code = torch.where(scale <= 0, torch.zeros_like(code), code).to(torch.uint8)
    inverse = torch.ldexp(torch.ones_like(scale, dtype=torch.float64), 127 - code.int())
    scaled = blocks.double() * inverse.unsqueeze(-1)
    scaled = torch.where((scale == 0).unsqueeze(-1), torch.zeros_like(scaled), scaled)
    quantized = scaled.clamp(-E4M3_MAX, E4M3_MAX).float().to(torch.float8_e4m3fn)
    return quantized.reshape(shape).contiguous(), code.contiguous()


def quantize_mxfp4(values):
    """FP32 -> packed E2M1 with the smallest power-of-two UE8M0 scale (even K index in the low nibble)."""
    shape = values.shape
    blocks = values.float().reshape(*shape[:-1], -1, GROUP)
    amax = blocks.abs().amax(dim=-1)
    ratio = amax / E2M1_VALUES[-1]
    mantissa, exponent = torch.frexp(ratio)
    ceil_log2 = torch.where(mantissa == 0.5, exponent - 1, exponent)
    code = torch.where(
        amax > 0, (127 + ceil_log2).clamp(0, 254), torch.full_like(ceil_log2, 127)
    ).to(torch.uint8)
    scale = torch.ldexp(torch.ones_like(amax), code.int() - 127)
    scaled = blocks / scale.unsqueeze(-1)
    midpoints = torch.tensor(E2M1_MIDPOINTS, device=values.device, dtype=torch.float32)
    magnitude = torch.bucketize(scaled.abs().contiguous(), midpoints).to(torch.uint8)
    nibbles = magnitude | ((scaled < 0).to(torch.uint8) << 3)
    nibbles = nibbles.reshape(*shape[:-1], -1, 2)
    packed = nibbles[..., 0] | (nibbles[..., 1] << 4)
    return packed.contiguous(), code.contiguous()


def situ(up, gate, beta, linear_beta):
    gated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    return gated * (linear_beta * torch.tanh(up / linear_beta))


def _layout_geometry(layout):
    local = layout["num_local_experts"]
    shard = layout.get("intermediate_shard", INTERMEDIATE)
    return dict(
        local_experts=local,
        local_expert_offset=layout["local_expert_offset"],
        intermediate_shard=shard,
        shard_offset=0,
    )


_BANK_CACHE = {}
_BANK_CACHE_LIMIT = 2
BANK_SEED = 20260921  # the contract's canonical weight bank
EP8_RANK3_INTERVAL = (336, 448)
MANY_EMPTY_POOL = tuple(range(336, 344)) + tuple(range(0, 8))


def _weight_bank(layout, bank_seed=BANK_SEED, device="cuda"):
    """Rank-local MXFP4 bank of the contract's canonical experts: dense ``W1 [2I, H]`` (``[up, gate]``,
    Gaussian * H**-0.5) and ``W2 [H, I]`` (Gaussian * I**-0.5) from per-expert generators, quantized
    group-wise; the intermediate shard is group aligned, so slicing commutes with quantization."""
    geo = _layout_geometry(layout)
    key = (
        bank_seed,
        geo["local_experts"],
        geo["local_expert_offset"],
        geo["intermediate_shard"],
    )
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    local, shard, off = (
        geo["local_experts"],
        geo["intermediate_shard"],
        geo["shard_offset"],
    )
    dev = torch.device(device)
    bank = dict(
        w1=torch.empty((local, 2 * shard, HIDDEN // 2), dtype=torch.uint8, device=dev),
        w1_scale=torch.empty(
            (local, 2 * shard, HIDDEN // GROUP), dtype=torch.uint8, device=dev
        ),
        w2=torch.empty((local, HIDDEN, shard // 2), dtype=torch.uint8, device=dev),
        w2_scale=torch.empty(
            (local, HIDDEN, shard // GROUP), dtype=torch.uint8, device=dev
        ),
    )
    g = torch.Generator(device=dev)
    for i in range(local):
        expert = geo["local_expert_offset"] + i
        g.manual_seed(_seed(bank_seed, "w1", expert))
        w1 = torch.randn((2 * INTERMEDIATE, HIDDEN), generator=g, device=dev) * (
            HIDDEN**-0.5
        )
        rows = torch.cat(
            (
                torch.arange(off, off + shard, device=dev),
                torch.arange(
                    INTERMEDIATE + off, INTERMEDIATE + off + shard, device=dev
                ),
            )
        )
        q, sc = quantize_mxfp4(w1.index_select(0, rows))
        bank["w1"][i].copy_(q)
        bank["w1_scale"][i].copy_(sc)
        g.manual_seed(_seed(bank_seed, "w2", expert))
        w2 = torch.randn((HIDDEN, INTERMEDIATE), generator=g, device=dev) * (
            INTERMEDIATE**-0.5
        )
        q, sc = quantize_mxfp4(w2[:, off : off + shard])
        bank["w2"][i].copy_(q)
        bank["w2_scale"][i].copy_(sc)
        del w1, w2
    while len(_BANK_CACHE) >= _BANK_CACHE_LIMIT:
        _BANK_CACHE.pop(next(iter(_BANK_CACHE)))
    _BANK_CACHE[key] = bank
    return bank


def route_ids(profile, T):
    """The contract's deterministic route tables (global expert IDs, ``[T, top_k]``, int64 on the host)."""
    rows = torch.arange(T, dtype=torch.int64).reshape(-1, 1)
    slots = torch.arange(TOP_K, dtype=torch.int64).reshape(1, -1)
    if profile == "uniform_cycle":
        return (rows * TOP_K + slots) % EXPERTS
    if profile == "hotset_skew":
        hot_size, hot_count = 32, 3 * TOP_K // 4
        hot_ids = (rows * 5 + slots) % hot_size
        cold_ids = hot_size + (rows * (TOP_K - hot_count) + (slots - hot_count)) % (
            EXPERTS - hot_size
        )
        return torch.where(slots < hot_count, hot_ids, cold_ids)
    if profile == "uniform_spread":
        return (rows + slots * (EXPERTS // TOP_K)) % EXPERTS
    if profile == "many_empty":
        pool = torch.tensor(MANY_EMPTY_POOL, dtype=torch.int64)
        return pool[(rows * TOP_K + slots) % len(pool)]
    assert profile == "remote_dominated", profile
    lo, hi = EP8_RANK3_INTERVAL
    remote = torch.tensor(
        [e for e in range(EXPERTS) if not lo <= e < hi], dtype=torch.int64
    )
    local = lo + (rows % (hi - lo))
    remote_ids = remote[(rows * (TOP_K - 1) + (slots - 1)) % len(remote)]
    return torch.where(slots == 0, local.expand(T, TOP_K), remote_ids)


def _contract_inputs(row, *, device="cuda"):
    """One canonical row of the contract: ``row = (layout, T, route_profile, seed)``."""
    layout, T, profile, seed = row
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(_seed(seed, "activations"))
    x, x_sf = quantize_mxfp8(torch.randn((T, HIDDEN), generator=g, device=dev))
    ids = route_ids(profile, T)
    assert bool((ids.sort(dim=-1).values.diff(dim=-1) != 0).all()), profile
    gw = torch.Generator().manual_seed(_seed(seed, "routing_weights"))
    raw = torch.rand((T, TOP_K), generator=gw, dtype=torch.float32) + 0.125
    weights = (raw / raw.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    bank = _weight_bank(layout)
    L = layout["num_local_experts"]
    return dict(
        x=x,
        x_sf=x_sf,
        topk_ids=ids.to(torch.int32).to(dev).contiguous(),
        topk_weights=weights.to(dev).contiguous(),
        w1=bank["w1"],
        w1_scale=bank["w1_scale"],
        w2=bank["w2"],
        w2_scale=bank["w2_scale"],
        beta=torch.full((L,), SITU_BETA, dtype=torch.float32, device=dev),
        linear_beta=torch.full((L,), SITU_LINEAR_BETA, dtype=torch.float32, device=dev),
    )


def _adversarial_inputs(T, layout, seed, *, device="cuda"):
    """Uniform random operand bytes: every E2M1 / E4M3 code, scales up to 2**1; O(1e3) outputs."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    L = layout["num_local_experts"]
    I = layout.get("intermediate_shard", INTERMEDIATE)
    x = (torch.randn(T, HIDDEN, generator=g) * 0.5).to(torch.float8_e4m3fn).to(device)
    x_sf = torch.randint(
        124, 129, (T, HIDDEN // 32), generator=g, dtype=torch.uint8
    ).to(device)
    ids = (
        torch.stack([torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(T)])
        .to(torch.int32)
        .to(device)
    )
    raw = torch.rand((T, TOP_K), generator=g) + 0.125
    weights = (raw / raw.sum(dim=-1, keepdim=True)).to(torch.bfloat16).to(device)
    w1 = torch.randint(
        0, 256, (L, 2 * I, HIDDEN // 2), generator=g, dtype=torch.uint8
    ).to(device)
    w1_scale = torch.randint(
        120, 128, (L, 2 * I, HIDDEN // 32), generator=g, dtype=torch.uint8
    ).to(device)
    w2 = torch.randint(0, 256, (L, HIDDEN, I // 2), generator=g, dtype=torch.uint8).to(
        device
    )
    w2_scale = torch.randint(
        120, 128, (L, HIDDEN, I // 32), generator=g, dtype=torch.uint8
    ).to(device)
    return dict(
        x=x,
        x_sf=x_sf,
        topk_ids=ids,
        topk_weights=weights,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        beta=torch.full((L,), SITU_BETA, dtype=torch.float32, device=device),
        linear_beta=torch.full(
            (L,), SITU_LINEAR_BETA, dtype=torch.float32, device=device
        ),
    )


def _local_counts(inputs, layout):
    ids = inputs["topk_ids"]
    lo = layout["local_expert_offset"]
    return ((ids >= lo) & (ids < lo + layout["num_local_experts"])).sum(dim=1)


def _gate_oracle(inputs, layout):
    """The contract's gate oracle: FP64 GEMM1 + SiTU, MXFP8 requantization of the activation, FP64 GEMM2,
    route-weighted sum.  Returns ``(out_fp64, sum_k |contribution_k|)``."""
    dev = inputs["x"].device
    x = decode_mxfp8(inputs["x"], inputs["x_sf"])
    ids = inputs["topk_ids"].cpu().long()
    weights = inputs["topk_weights"].to(device=dev, dtype=torch.float64)
    out = torch.zeros((ids.shape[0], HIDDEN), dtype=torch.float64, device=dev)
    out_abs = torch.zeros_like(out)
    beta = torch.tensor(SITU_BETA, dtype=torch.float64, device=dev)
    linear_beta = torch.tensor(SITU_LINEAR_BETA, dtype=torch.float64, device=dev)
    lo = layout["local_expert_offset"]
    for local in range(layout["num_local_experts"]):
        pairs = (ids == lo + local).nonzero()
        if pairs.shape[0] == 0:
            continue
        token, slot = pairs[:, 0].to(dev), pairs[:, 1].to(dev)
        w1 = decode_mxfp4(inputs["w1"][local], inputs["w1_scale"][local])
        fc1 = x[token] @ w1.t()
        del w1
        up, gate = fc1.chunk(2, dim=-1)
        act = situ(up, gate, beta, linear_beta)
        q, sf = quantize_mxfp8(act.float())
        middle = decode_mxfp8(q, sf)
        w2 = decode_mxfp4(inputs["w2"][local], inputs["w2_scale"][local])
        contrib = (middle @ w2.t()) * weights[token, slot, None]
        out.index_add_(0, token, contrib)
        out_abs.index_add_(0, token, contrib.abs())
        del w2, fc1, act, middle, contrib
    return out, out_abs


def bf16_accumulation_tolerance(ref, ref_abs, local_count):
    """The producer gate's per-element budget of a BF16 ``red.add`` reduction: the BF16 class
    ``1e-2 + 1e-2 |ref|`` widened by the rounding of ``local_count`` BF16 partial sums, each bounded by
    the sum of the contribution magnitudes (``local_count * 2**-9 * ref_abs``)."""
    k = local_count.to(torch.float64).view(-1, 1)
    return 1e-2 + 1e-2 * ref.abs() + k * (2.0**-9) * ref_abs


def _assert_gate_class(out, ref, label, record_property=None):
    """The contract's criterion: ``|out - bf16(ref)| <= atol + rtol * |bf16(ref)|`` on every element;
    the strict BF16 verdict (``1e-2 + 1e-2 |ref|``) is reported, not asserted."""
    expected = ref.to(torch.bfloat16).double()
    err = (out.double() - expected).abs()
    assert torch.isfinite(out.float()).all(), label
    over = err > GATE_ATOL + GATE_RTOL * expected.abs()
    assert not bool(over.any()), (
        f"{label}: {int(over.sum())} elements beyond the gate criterion "
        f"(max err {float(err.max()):.4g}, max |ref| {float(expected.abs().max()):.4g})"
    )
    strict = bool((err <= 1e-2 + 1e-2 * expected.abs()).all())
    if record_property is not None:
        record_property(f"{label}_strict_bf16_passed", strict)
        record_property(f"{label}_max_err", float(err.max()))
    return strict


def _assert_same_accumulation_class(
    a, b, ref, ref_abs, local_count, label, record_property=None
):
    """Two outputs of the same chain (or the two backends): bit-identical on tokens with at most one
    local contribution (a deterministic path) and within the BF16 accumulation budget on tokens with
    two or more (the finalize epilogue reduces the contributions with ``red.global.add.noftz.bf16x2``,
    whose arrival order is not fixed); the differing set must be a subset of the multi-local tokens."""
    single = (local_count <= 1).view(-1, 1).expand_as(a)
    assert torch.equal(a[single], b[single]), (
        f"{label}: differs on a token with <= 1 local contribution"
    )
    diff = (a.float() - b.float()).abs().double()
    tol = bf16_accumulation_tolerance(ref, ref_abs, local_count)
    assert bool((diff <= tol).all()), (
        f"{label}: multi-local difference beyond the BF16 accumulation budget "
        f"(max {float(diff.max()):.4g}, max over budget {float((diff / tol).max()):.3g})"
    )
    differing_tokens = (diff > 0).any(dim=1)
    assert bool((local_count[differing_tokens] >= 2).all()), label
    if record_property is not None:
        record_property(f"{label}_bit_exact", bool(not differing_tokens.any()))
        record_property(f"{label}_differing_tokens", int(differing_tokens.sum()))
        record_property(f"{label}_max_diff", float(diff.max()))


def _run_chain(module, inputs, T, layout):
    runner = module.CakeMxfp4MoEWrapper(
        EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, backend=module.BACKEND, **layout
    )
    weights = module.prepare_cake_mxfp4_weights(
        inputs["w1"], inputs["w1_scale"], inputs["w2"], inputs["w2_scale"]
    )
    workspace = torch.empty(
        runner.get_workspace_size(T), dtype=torch.uint8, device="cuda"
    )
    out = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
    p = runner.plan(
        inputs["x"],
        inputs["x_sf"],
        inputs["topk_ids"],
        inputs["topk_weights"],
        weights["w1"],
        weights["w1_sf"],
        weights["w2"],
        weights["w2_sf"],
        beta=inputs["beta"],
        linear_beta=inputs["linear_beta"],
        workspace=workspace,
        output=out,
    )
    return p, out


def _hand_written():
    """The hand-written CuTe DSL MXFP4 wrapper this family reproduces, when the installed FlashInfer
    exposes it (``CuteDslMxfp4MoEWrapper`` + ``prepare_cute_dsl_mxfp4_weights``); ``None`` otherwise."""
    try:
        cute_dsl = importlib.import_module("flashinfer.fused_moe.cute_dsl")
        prepare = importlib.import_module("flashinfer.fused_moe.prepare")
        enums = importlib.import_module("flashinfer.tllm_enums")
        wrapper = cute_dsl.CuteDslMxfp4MoEWrapper
        prepare_fn = prepare.prepare_cute_dsl_mxfp4_weights
        activation = enums.ActivationType.Situ
    except (ImportError, AttributeError):
        return None
    return wrapper, prepare_fn, activation


def _run_hand_written(hw, inputs, T, layout):
    wrapper, prepare_fn, activation = hw
    weights = prepare_fn(
        inputs["w1"], inputs["w1_scale"], inputs["w2"], inputs["w2_scale"]
    )
    runner = wrapper(
        EXPERTS,
        TOP_K,
        HIDDEN,
        layout.get("intermediate_shard", INTERMEDIATE),
        num_local_experts=layout["num_local_experts"],
        local_expert_offset=layout["local_expert_offset"],
        quantization="mxfp4_w4a8",
        activation_type=activation,
    )
    workspace = torch.empty(
        runner.get_workspace_size(T), dtype=torch.uint8, device="cuda"
    )
    out = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
    p = runner.plan(
        inputs["x"],
        inputs["x_sf"],
        inputs["topk_ids"],
        inputs["topk_weights"],
        *weights,
        beta=inputs["beta"],
        linear_beta=inputs["linear_beta"],
        workspace=workspace,
        output=out,
    )
    return p, out


# Canonical contract rows (layout, T, route profile, seed); the EP8 rows are rank 3, the TP8 row rank 0.
# ``remote_dominated`` routes exactly one local expert per token (the deterministic path);
# ``uniform_cycle`` at T >= 128 routes whole tokens to the local rank (16 local contributions each);
# ``uniform_cycle`` at T = 16 routes no work to rank 3 (the output must be zero-filled).
ROWS = {
    "ep8_1": (EP8, 1, "remote_dominated", 100015),
    "ep8_16": (EP8, 16, "remote_dominated", 100165),
    "ep8_16_empty": (EP8, 16, "uniform_cycle", 100161),
    "ep8_128": (EP8, 128, "uniform_cycle", 101281),
    "ep8_512": (EP8, 512, "uniform_cycle", 105121),
    "ep8_512_skew": (EP8, 512, "hotset_skew", 105122),
    "tp8_16": (TP8, 16, "uniform_cycle", 200161),
}
GATE_ROWS = ["ep8_1", "ep8_16", "ep8_16_empty", "ep8_128", "ep8_512", "tp8_16"]
CLASS_ROWS = ["ep8_1", "ep8_16", "ep8_128", "ep8_512", "ep8_512_skew"]


def _run_row(module, row):
    layout, T = row[0], row[1]
    inputs = _contract_inputs(row)
    p, out = _run_chain(module, inputs, T, layout)
    torch.cuda.synchronize()
    return inputs, p, out


@gpu
@pytest.mark.parametrize("row", GATE_ROWS)
def test_chain_matches_the_gate_oracle(row, record_property):
    layout = ROWS[row][0]
    inputs, _p, out = _run_row(pkg, ROWS[row])
    ref, _ref_abs = _gate_oracle(inputs, layout)
    if int(_local_counts(inputs, layout).sum()) == 0:
        assert torch.equal(out, torch.zeros_like(out)), (
            "an empty local set must zero-fill"
        )
    else:
        assert out.abs().sum() > 0
    _assert_gate_class(out, ref, BACKEND, record_property)


@gpu
@pytest.mark.parametrize("row", GATE_ROWS)
def test_hand_written_kernel_passes_the_same_gate_and_matches_the_deterministic_path(
    row, record_property
):
    hw = _hand_written()
    if hw is None:
        pytest.skip(
            "the installed FlashInfer does not expose the hand-written MXFP4 wrapper"
        )
    layout, T = ROWS[row][0], ROWS[row][1]
    inputs, _p, out = _run_row(pkg, ROWS[row])
    _hp, hw_out = _run_hand_written(hw, inputs, T, layout)
    torch.cuda.synchronize()
    ref, ref_abs = _gate_oracle(inputs, layout)
    _assert_gate_class(hw_out, ref, "hand_written", record_property)
    _assert_same_accumulation_class(
        out,
        hw_out,
        ref,
        ref_abs,
        _local_counts(inputs, layout),
        f"{BACKEND}_vs_hand_written",
        record_property,
    )


@gpu
@pytest.mark.parametrize("row", CLASS_ROWS)
def test_chain_runs_deterministically_and_replays_under_graph_capture(
    row, record_property
):
    layout = ROWS[row][0]
    inputs, p, out = _run_row(pkg, ROWS[row])
    local_count = _local_counts(inputs, layout)
    first = out.clone()
    assert torch.isfinite(first.float()).all()
    assert first.abs().sum() > 0
    ref, ref_abs = _gate_oracle(inputs, layout)
    p.run()
    torch.cuda.synchronize()
    _assert_same_accumulation_class(
        out, first, ref, ref_abs, local_count, "run2_vs_run1", record_property
    )
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        p.run()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_same_accumulation_class(
        out, first, ref, ref_abs, local_count, "replay_vs_run1", record_property
    )
    if bool((local_count <= 1).all()):
        # Only the deterministic path was exercised: the verdict must be bit identity.
        assert torch.equal(out, first)
    assert plan.kernel_stages(p)[1].startswith(
        "gemm1_swapab_situ_n"
    ) and plan.kernel_stages(p)[2].startswith("gemm2_swapab_finalize_n")


@gpu
def test_both_backends_agree(record_property):
    if importlib.util.find_spec(SIBLING_MODULE) is None:
        pytest.skip("sibling package is not installed")
    sibling = importlib.import_module(SIBLING_MODULE)
    if not sibling.is_available():
        pytest.skip("sibling package kernels are not available on this machine")
    for row in CLASS_ROWS:
        layout = ROWS[row][0]
        inputs, _p1, out = _run_row(pkg, ROWS[row])
        _p2, other = _run_chain(sibling, inputs, ROWS[row][1], layout)
        torch.cuda.synchronize()
        ref, ref_abs = _gate_oracle(inputs, layout)
        _assert_same_accumulation_class(
            out,
            other,
            ref,
            ref_abs,
            _local_counts(inputs, layout),
            f"{row}_{BACKEND}_vs_{sibling.BACKEND}",
            record_property,
        )


@gpu
@pytest.mark.parametrize("T", [8, 128])
def test_adversarial_operand_bytes_run_finite(T):
    """Every E2M1 / E4M3 code and large scales: the chain stays finite and non-zero (no reference-class
    assert on this out-of-domain fixture)."""
    inputs = _adversarial_inputs(T, EP8, seed=T)
    p, out = _run_chain(pkg, inputs, T, EP8)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert out.abs().sum() > 0
    assert plan.kernel_stages(p)[0].startswith("routing_")


@gpu
def test_tp8_row_selects_the_grouped_gemm2_form():
    inputs = _adversarial_inputs(16, TP8, seed=1616)
    p, out = _run_chain(pkg, inputs, 16, TP8)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert p.decision.gemm2.m_group == 2 and plan.kernel_stages(p)[2].endswith("_m2")
