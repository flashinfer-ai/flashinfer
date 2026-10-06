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
"""

import pytest
import torch

from flashinfer.experimental.kimi_k3_latent_moe import cake_backend as cb
from flashinfer.experimental.kimi_k3_latent_moe import cake_jit
from flashinfer.experimental.kimi_k3_latent_moe.cake_jit import KERNELS, MODULES
from flashinfer.experimental.kimi_k3_latent_moe.cake_backend import (
    DECODE_MAX_T,
    HIDDEN,
    LATENT,
    NUM_EXPERTS,
    RMS_EPS,
    SHARED_INTERMEDIATE,
    SM_COUNT,
    SUPPORTED_COMPUTE_CAPABILITIES,
    SUPPORTED_TP,
    decode_front_plan,
    decode_kernel_key,
    decode_tail_plan,
    front_split_plan,
    i_local_for_tp,
    prefill_front_plan,
    prefill_tail_plan,
    route_kernel_keys,
    split_plan,
)
from flashinfer.kimi_k3_latent_moe import (
    kimi_k3_latent_moe_front,
    kimi_k3_latent_moe_tail,
    prepare_kimi_k3_latent_moe_front,
    prepare_kimi_k3_latent_moe_tail,
)

# Tolerances of the Cake evaluation contract (never looser): BF16 outputs 1e-2, FP32 logits 1e-3.
ATOL = RTOL = 1e-2
LOGITS_ATOL = LOGITS_RTOL = 1e-3
WEIGHT_SEED = 621
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
# Representative subset of the 60 validated rows: both stages x TP {1, 8} x these token counts.
SMOKE_TOKENS = (
    1,
    8,
    16,
    128,
    256,
    4096,
    8192,
    16384,
)  # decode + prefill routes, incl. the power-capped GEMM rows


# ---------------------------------------------------------------------------
# Host plan (CPU)
# ---------------------------------------------------------------------------


def test_decode_plan_rules():
    # Front: TP1 streams 7 + 28 + 96 = 131 tiles (one CTA per tile), TP8 7 + 28 + 12 = 47 tiles as 94 cluster-pair CTAs.
    tp1 = decode_front_plan(1, i_local_for_tp(1))
    assert (tp1["tiles"], tp1["grid"], tp1["cluster"], tp1["n_pad"]) == (131, 131, 1, 8)
    tp8 = decode_front_plan(1, i_local_for_tp(8))
    assert (tp8["tiles"], tp8["grid"], tp8["cluster"], tp8["n_pad"]) == (47, 94, 2, 8)
    assert decode_front_plan(9, i_local_for_tp(8))["n_pad"] == 16
    assert decode_front_plan(128, i_local_for_tp(1))["n_pad"] == 128
    # Front K depth 2 while its ring keeps >= 4 (one CTA per tile) / 6 (cluster pairs) stages: TP1 T <= 64, TP8 T <= 16.
    for plan in (tp1, tp8):
        assert (
            plan["kdepth"] == 2
            and 1 <= plan["stages"] <= cb.MAX_STAGES
            and not plan["fused"]
        )
    assert decode_front_plan(64, i_local_for_tp(1))["kdepth"] == 2
    assert decode_front_plan(128, i_local_for_tp(1))["kdepth"] == 1
    assert decode_front_plan(16, i_local_for_tp(8))["kdepth"] == 2
    assert decode_front_plan(32, i_local_for_tp(8))["kdepth"] == 1
    assert decode_front_plan(32, i_local_for_tp(8))["stages"] == cb.plan_stages(
        n_pad=32, kdepth=1, cluster=2
    )
    # Tail: 56 output tiles -> 112 cluster-pair CTAs; the fused norm is always on.
    for tp in SUPPORTED_TP:
        for tokens in (1, 8, 16, 128):
            plan = decode_tail_plan(tokens, i_local_for_tp(tp), tp)
            assert (plan["tiles"], plan["grid"], plan["cluster"]) == (56, 112, 2)
            assert (
                plan["fused"]
                and plan["k1"] == LATENT // tp // 64
                and plan["k2"] == SHARED_INTERMEDIATE // tp // 64
            )
            if plan["rows_smem"]:
                assert plan["smem_b1"]
    # The resident B operand with staged rows serves the TP8 tails up to T16 (5 ring stages left); T32 falls back to
    # the global-y protocol, and so does TP1 (long shared-down window).
    assert decode_tail_plan(1, i_local_for_tp(8), 8)["smem_b1"]
    assert decode_tail_plan(1, i_local_for_tp(8), 8)["rows_smem"]
    assert decode_tail_plan(8, i_local_for_tp(8), 8)["smem_b1"]
    t16 = decode_tail_plan(16, i_local_for_tp(8), 8)
    assert t16["smem_b1"] and t16["rows_smem"] and t16["stages"] == 5
    assert (
        not t16["tmap_prefetch"]
        and decode_tail_plan(8, i_local_for_tp(8), 8)["tmap_prefetch"]
    )
    # Round-7 lever 3b: the same instance defers the cluster rendezvous to its epilogue warps.
    assert (
        t16["wait_warps"]
        and not decode_tail_plan(8, i_local_for_tp(8), 8)["wait_warps"]
    )
    assert not decode_tail_plan(32, i_local_for_tp(8), 8)["wait_warps"]
    assert not decode_front_plan(16, i_local_for_tp(8))["wait_warps"]
    # one ring stage fewer than the smem budget for the small-T tail (T16 keeps its 5)
    assert decode_tail_plan(1, i_local_for_tp(8), 8)["stages"] == 8
    assert decode_tail_plan(8, i_local_for_tp(8), 8)["stages"] == 8
    assert decode_tail_plan(1, i_local_for_tp(1), 1)["stages"] == 10
    assert not decode_tail_plan(32, i_local_for_tp(8), 8)["smem_b1"]
    assert not decode_tail_plan(32, i_local_for_tp(1), 1)["smem_b1"]
    assert not decode_tail_plan(16, i_local_for_tp(1), 1)["rows_smem"]
    # Round-7 landing-zone alias (lever 6a): only the N_PAD 128 cluster instances streaming >= 32 chunk units per CTA
    # (tail TP1 T=128: 76 units, front TP8 T=128: 56) alias the DSMEM landing zone on the drained A ring and gain two
    # ring stages (5 -> 7); the short TP8 tail (9 units) and the single-CTA TP1 front keep their own zone.
    la_tail = decode_tail_plan(128, i_local_for_tp(1), 1)
    assert la_tail["land_alias"] and la_tail["cluster"] == 2 and la_tail["stages"] == 7
    la_front = decode_front_plan(128, i_local_for_tp(8))
    assert (
        la_front["land_alias"] and la_front["cluster"] == 2 and la_front["stages"] == 7
    )
    assert not decode_tail_plan(128, i_local_for_tp(8), 8)["land_alias"]
    assert not decode_tail_plan(64, i_local_for_tp(1), 1)["land_alias"]
    assert not decode_front_plan(128, i_local_for_tp(1))["land_alias"]
    assert cb.land_alias_auto(128, 152) and not cb.land_alias_auto(128, 19)
    assert not cb.land_alias_auto(64, 152)
    # A second routed partial disables the staged rows (P == 1 only).
    assert not decode_tail_plan(1, i_local_for_tp(8), 8, num_partials=2)["rows_smem"]
    with pytest.raises(ValueError):
        cb.n_pad_for(DECODE_MAX_T + 1)


def test_split_plan_rules():
    # No remainder wave, short K loops (TP8: 19 iterations) or a small-reuse region keep whole tiles.
    whole = split_plan(148, 152, SM_COUNT, 1)
    assert whole == dict(
        num_items=148, full_items=148, sk_ipc=152, sk_max_seg=1, sk_total=0, sk_tiles=0
    )
    assert split_plan(56, 19, SM_COUNT, 1)["sk_tiles"] == 0
    assert split_plan(60, 152, SM_COUNT, 4)["sk_tiles"] == 0
    # Partial reuse (2-4 pair rows per column): only a remainder wave after a full wave is split, and only
    # when it fills at most 55 % of the clusters (TP1 T=1024: 74 whole + 38 tiles as 74 x 79 iterations).
    rem = split_plan(112, 152, SM_COUNT, 4)
    assert rem["full_items"] == 74 and rem["sk_tiles"] == 38
    assert rem["sk_ipc"] == 79 and rem["num_items"] == 148
    assert split_plan(56, 152, SM_COUNT, 2)["sk_tiles"] == 0
    assert split_plan(140, 152, SM_COUNT, 5)["sk_tiles"] == 0
    # TP1 T=2048 (224 pair tiles on 74 resident clusters): 2 x 74 whole + one stream-K wave (module docstring).
    sk = split_plan(224, 152, SM_COUNT, 8)
    assert sk["full_items"] == 148 and sk["sk_tiles"] == 76 and sk["sk_ipc"] == 157
    assert (
        sk["num_items"] == 148 + -(-76 * 152 // 157)
        and 1 < sk["sk_max_seg"] <= cb.MAX_SEG
    )


def test_front_split_plan_rules():
    # Round-8 rule: the trailing wave's tiles become two aligned K halves when the halves fit one wave of the
    # 74 resident clusters and the modelled saving is >= 4 % of the whole-tile cost; every other shape runs whole.
    whole = dict(
        num_items=48, full_items=48, sk_ipc=112, sk_max_seg=1, sk_total=0, sk_tiles=0
    )
    assert (
        front_split_plan(48, SM_COUNT) == whole
    )  # TP8 T=512: 96 halves would need two waves
    assert front_split_plan(24, SM_COUNT) == dict(
        num_items=48,
        full_items=0,
        sk_ipc=56,
        sk_max_seg=2,
        sk_total=24 * 112,
        sk_tiles=24,
    )  # TP8 T=256
    assert front_split_plan(96, SM_COUNT) == dict(
        num_items=118,
        full_items=74,
        sk_ipc=56,
        sk_max_seg=2,
        sk_total=22 * 112,
        sk_tiles=22,
    )  # TP8 T=1024
    assert (
        front_split_plan(384, SM_COUNT)["sk_tiles"] == 14
    )  # TP8 T=4096: 370 whole + 14 x 2
    assert (
        front_split_plan(528, SM_COUNT)["sk_tiles"] == 10
    )  # TP1 T=2048: 518 whole + 10 x 2
    assert (
        front_split_plan(768, SM_COUNT)["sk_tiles"] == 0
    )  # TP8 T=8192: 3.1 % modelled -> whole
    assert (
        front_split_plan(1056, SM_COUNT)["sk_tiles"] == 0
    )  # TP1 T=4096: 2.3 % modelled -> whole
    for tiles in (66, 132, 264, 192, 1536, 2112, 4224):
        assert front_split_plan(tiles, SM_COUNT)["sk_max_seg"] == 1
    plan = prefill_front_plan(1024, i_local_for_tp(8))
    assert (
        plan["cluster_tiles"] == 96
        and plan["grid"] == 118 * 2
        and not plan["evict_first"]
    )
    assert prefill_front_plan(256, i_local_for_tp(1))["evict_first"]
    assert prefill_front_plan(512, i_local_for_tp(8))["evict_first"]
    assert not prefill_front_plan(1024, i_local_for_tp(1))["evict_first"]
    assert prefill_front_plan(256, i_local_for_tp(1))["grid"] == cb.front_grid(
        cb.m_tiles_for(256), i_local_for_tp(1)
    )
    assert cb.front_evict_first(4) and not cb.front_evict_first(6)


def test_prefill_tail_plan_and_trigger():
    plan = prefill_tail_plan(256, 8)
    assert (
        plan["m_tiles"] == 2 and plan["cluster_tiles"] == 28 and plan["norm_grid"] == 64
    )
    assert plan["num_k"] == (LATENT // 8 + SHARED_INTERMEDIATE // 8) // 64 == 19
    assert plan["sk_tiles"] == 0 and plan["gemm_grid"] == 56 and plan["early_trigger"]
    # TP1 T=256: 112 GEMM CTAs next to 64 norm CTAs do not fit 148 SMs -> late trigger.
    tp1 = prefill_tail_plan(256, 1)
    assert tp1["gemm_grid"] == (tp1["num_items"]) * 2 and not tp1["early_trigger"]
    assert plan["weights_evict_first"] and tp1["weights_evict_first"]
    assert not prefill_tail_plan(2048, 1)["weights_evict_first"]
    # Round-7 N128 rule (TP8 T=512): 128-wide pair tile, 56 column tiles, 224 CTAs, 9-deep ring, no stream-K region.
    n128 = prefill_tail_plan(512, 8)
    assert (n128["block_n"], n128["n_tiles"], n128["num_stages"]) == (128, 56, 9)
    assert (
        n128["cluster_tiles"] == 112
        and n128["gemm_grid"] == 224
        and n128["sk_tiles"] == 0
    )
    assert (
        not n128["weights_evict_first"]
        and n128["early_trigger"]
        and not n128["fused_norm"]
    )
    assert (
        prefill_tail_plan(256, 8)["block_n"] == 256
        and prefill_tail_plan(256, 8)["n_tiles"] == 28
    )
    # Round-10 rule: every 256-wide instance stages the CTA's final item through a TMA store (``final_ts``, a
    # separate kernel instance that binds the ``out`` tensor map beside the pointer); the 128-wide TP8 T=512
    # instance keeps the direct stores.
    assert (
        plan["final_ts"] and tp1["final_ts"] and prefill_tail_plan(2048, 1)["final_ts"]
    )
    assert not n128["final_ts"]
    assert cb.tail_gemm_final_ts(1024, 8) and not cb.tail_gemm_final_ts(512, 8)
    # Fused norm: TP1 T = 256 / 512 (single wave, K2 = 96 blocks); TP8 (K2 = 12) and multi-wave grids do not fuse.
    assert tp1["fused_norm"] and prefill_tail_plan(512, 1)["fused_norm"]
    assert not plan["fused_norm"] and not prefill_tail_plan(1024, 1)["fused_norm"]
    assert cb.use_fused_norm(256, 112, SM_COUNT, 96) and not cb.use_fused_norm(
        256, 56, SM_COUNT, 12
    )
    assert cb.weights_evict_first(112, SM_COUNT) and not cb.weights_evict_first(
        224, SM_COUNT
    )
    assert cb.norm_early_trigger(300, 64, SM_COUNT) and not cb.norm_early_trigger(
        112, 64, SM_COUNT
    )
    assert cb.front_grid(cb.m_tiles_for(256), i_local_for_tp(1)) == 1 * 66 * 2
    assert cb.front_grid(cb.m_tiles_for(300), i_local_for_tp(8)) == 2 * 24 * 2


# Every token count the public entry points accept, both tensor-parallel degrees and the partial
# counts the decode tail plans on (P >= 2 selects the un-staged instances).
COVERAGE_TOKENS = tuple(range(1, 1025)) + tuple(range(1025, 16385, 97)) + (16384,)
COVERAGE_PARTIALS = (1, 2, 4)


def test_route_keys_are_registered_for_every_token_count():
    """Every route of the public contract resolves to a registered program (no token-range holes)."""
    assert MODULES and KERNELS
    arches = cake_jit.registered_arches()
    assert set(arches) <= set(SUPPORTED_COMPUTE_CAPABILITIES.values()) and arches
    reached = set()
    for stage in ("front", "tail"):
        for tp in SUPPORTED_TP:
            for tokens in COVERAGE_TOKENS:
                keys = route_kernel_keys(stage, tp, tokens)
                assert 1 <= len(keys) <= 2
                for key in keys:
                    assert key in KERNELS, (
                        f"{stage} tp{tp} T={tokens}: {key} is not registered"
                    )
                    reached.add(KERNELS[key])
                for arch in arches:
                    assert cake_jit.route_available(arch, keys)
    for tp in SUPPORTED_TP:
        for num_partials in COVERAGE_PARTIALS:
            for tokens in range(1, DECODE_MAX_T + 1):
                key = decode_kernel_key(
                    decode_tail_plan(tokens, i_local_for_tp(tp), tp, num_partials)
                )
                assert key in KERNELS, f"tail tp{tp} T={tokens} P={num_partials}: {key}"
                reached.add(KERNELS[key])
    # Every registered program is reachable, serves every registered architecture once and carries
    # a complete launch contract.
    assert reached == set(MODULES)
    for name, record in MODULES.items():
        assert tuple(record["arches"]) == arches, name
        assert len(record["sources"]) == 2 and record["ffi_entry"] == "run"
        assert {"block", "cluster", "cooperative", "dynamic_smem_bytes"} <= set(
            record["launch"]
        )
        assert all(
            kind in {"tma_buffer", "buffer", "parameter", "grid"}
            for kind, _ in record["arg_plan"]
        )
    for key, defines in cake_jit.SPECIALIZATIONS.items():
        assert (
            key in KERNELS
            and defines
            and all(isinstance(v, int) for v in defines.values())
        )
    with pytest.raises(ValueError):
        route_kernel_keys("front", 4, 8)
    with pytest.raises(ValueError):
        route_kernel_keys("block", 1, 8)


def test_route_rules_select_the_expected_programs():
    """Planner rules (behaviour, not registry text): route lengths, fused norm, cache policy, ring depth."""
    assert route_kernel_keys("front", 1, 128)[0].startswith("decode:")
    assert len(route_kernel_keys("front", 1, 1024)) == 1
    assert cb.prefill_front_plan(256, i_local_for_tp(1))["evict_first"]
    assert not cb.prefill_front_plan(1024, i_local_for_tp(1))["evict_first"]
    tail = cb.prefill_tail_plan(256, 1)
    assert tail["fused_norm"] and len(route_kernel_keys("tail", 1, 256)) == 1
    assert cb.tail_gemm_num_stages(256, 1) == 6 and cb.tail_gemm_num_stages(512, 1) == 7
    assert cb.tail_gemm_config(512, 8) == (9, 128, False)
    assert cb.tail_gemm_config(512, 1) == (7, 256, True)
    for tokens in (300, 511):
        plan = cb.prefill_tail_plan(tokens, 8)
        assert not plan["fused_norm"] and not plan["early_trigger"]
        assert plan["weights_evict_first"]
        assert len(route_kernel_keys("tail", 8, tokens)) == 2
    assert cb.prefill_tail_plan(512, 8)["early_trigger"]
    assert len(route_kernel_keys("tail", 1, 16384)) == 2
    # P >= 2 routed partials leave the staged-rows / resident-B decode instances (T <= 16, TP8).
    one = decode_tail_plan(16, i_local_for_tp(8), 8, 1)
    two = decode_tail_plan(16, i_local_for_tp(8), 8, 2)
    assert one["rows_smem"] and one["stagger"] == cb.TAIL_ROWS_STAGGER
    assert not two["rows_smem"] and not two["smem_b1"] and two["stagger"] == 0
    assert decode_kernel_key(one) != decode_kernel_key(two)
    assert decode_kernel_key(
        decode_tail_plan(32, i_local_for_tp(8), 8, 1)
    ) == decode_kernel_key(decode_tail_plan(32, i_local_for_tp(8), 8, 2))


# ---------------------------------------------------------------------------
# Torch reference (nvidia/Kimi-K3-NVFP4 modeling_kimi_linear.py semantics)
# ---------------------------------------------------------------------------


def make_weights(device, seed=WEIGHT_SEED):
    """Synthetic checkpoint-shaped BF16 weights (variance ~ 1 / fan_in)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def lin(n, k):
        return (
            (torch.randn(n, k, generator=gen, dtype=torch.float32) * (k**-0.5))
            .to(torch.bfloat16)
            .to(device)
        )

    gate_weight = lin(NUM_EXPERTS, HIDDEN)
    torch.randn(
        NUM_EXPERTS, generator=gen, dtype=torch.float32
    )  # gate bias (router only; keeps the seed stream)
    norm_weight = (
        (1.0 + 0.1 * torch.randn(LATENT, generator=gen, dtype=torch.float32))
        .to(torch.bfloat16)
        .to(device)
    )
    return dict(
        gate_weight=gate_weight,
        norm_weight=norm_weight,
        down_weight=lin(LATENT, HIDDEN),
        up_weight=lin(HIDDEN, LATENT),
        shared_gate_weight=lin(SHARED_INTERMEDIATE, HIDDEN),
        shared_up_weight=lin(SHARED_INTERMEDIATE, HIDDEN),
        shared_down_weight=lin(HIDDEN, SHARED_INTERMEDIATE),
    )


def shard(weights, tp, rank):
    i_local = SHARED_INTERMEDIATE // tp
    rows = slice(rank * i_local, (rank + 1) * i_local)
    return dict(
        weights,
        shared_gate_weight=weights["shared_gate_weight"][rows].contiguous(),
        shared_up_weight=weights["shared_up_weight"][rows].contiguous(),
        shared_down_weight=weights["shared_down_weight"][:, rows].contiguous(),
    )


def situ_and_mul(gate_up):
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    a = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    b = SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
    return (a * b).to(gate_up.dtype)


def rmsnorm(x, weight, eps=RMS_EPS):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def front_reference(x, w):
    gate_up = torch.cat(
        [
            torch.nn.functional.linear(x, w["shared_gate_weight"]),
            torch.nn.functional.linear(x, w["shared_up_weight"]),
        ],
        dim=-1,
    )
    return dict(
        logits=torch.nn.functional.linear(x.float(), w["gate_weight"].float()),
        latent=torch.nn.functional.linear(x, w["down_weight"]),
        shared_act=situ_and_mul(gate_up),
    )


def tail_reference(routed, shared_act, w, tp, rank):
    acc = routed[0].float()
    for p in range(1, routed.shape[0]):
        acc = acc + routed[p].float()
    y = rmsnorm(acc.to(torch.bfloat16), w["norm_weight"])
    k_up = LATENT // tp
    cols = slice(rank * k_up, (rank + 1) * k_up)
    up = torch.nn.functional.linear(y[:, cols].float(), w["up_weight"][:, cols].float())
    shared = torch.nn.functional.linear(
        shared_act.float(), w["shared_down_weight"].float()
    )
    return dict(y=y, out=(up + shared).to(torch.bfloat16))


def _gpu_arch():
    if not torch.cuda.is_available():
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(0))


def _require_program(stage, tp, tokens):
    arch = _gpu_arch()
    if arch is None:
        pytest.skip("the Kimi-K3 LatentMoE programs require an SM100/SM103 GPU")
    device = torch.device("cuda", 0)
    if int(torch.cuda.get_device_properties(0).multi_processor_count) != SM_COUNT:
        pytest.skip(f"the plan rules were frozen for {SM_COUNT} SMs")
    assert cb.generated_program_available(device, stage, tp, tokens), (
        f"the generated {stage} program for {arch} (tp {tp}, T {tokens}) is not registered"
    )
    return device


_WEIGHTS = {}


def _weights(device):
    key = str(device)
    if key not in _WEIGHTS:
        _WEIGHTS[key] = make_weights(device)
    return _WEIGHTS[key]


def _graph_replay(runner):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner()
    return graph


def _assert_close(name, actual, expected, atol, rtol):
    assert torch.isfinite(actual.float()).all(), f"{name}: non-finite values"
    err = (actual.float() - expected.float()).abs()
    tol = atol + rtol * expected.float().abs()
    bad = int((err > tol).sum())
    assert bad == 0, (
        f"{name}: {bad} elements outside atol={atol} rtol={rtol} (max abs err {float(err.max()):.4g})"
    )


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", SMOKE_TOKENS)
@pytest.mark.parametrize("tp", SUPPORTED_TP)
def test_front_matches_reference(tp, tokens):
    device = _require_program("front", tp, tokens)
    w = shard(_weights(device), tp, 0)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(621 + tokens)
    x = (
        torch.randn(tokens, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )
    shared_gate_up = torch.cat(
        [w["shared_gate_weight"], w["shared_up_weight"]], dim=0
    ).contiguous()
    logits = torch.full(
        (tokens, NUM_EXPERTS), float("nan"), dtype=torch.float32, device=device
    )
    latent = torch.full(
        (tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device
    )
    shared_act = torch.full(
        (tokens, i_local), float("nan"), dtype=torch.bfloat16, device=device
    )
    runner = prepare_kimi_k3_latent_moe_front(
        x,
        w["gate_weight"],
        w["down_weight"],
        shared_gate_up,
        logits,
        latent,
        shared_act,
    )
    assert runner.route == ("decode" if tokens <= DECODE_MAX_T else "prefill")
    assert runner.kernel_keys == route_kernel_keys("front", tp, tokens)
    runner()
    torch.cuda.synchronize()
    expected = front_reference(x, w)
    _assert_close("logits", logits, expected["logits"], LOGITS_ATOL, LOGITS_RTOL)
    _assert_close("latent", latent, expected["latent"], ATOL, RTOL)
    _assert_close("shared_act", shared_act, expected["shared_act"], ATOL, RTOL)
    first = (logits.clone(), latent.clone(), shared_act.clone())
    # Idempotent re-launch and bit-identical CUDA-graph replay (the deployment form).
    runner()
    torch.cuda.synchronize()
    assert all(
        torch.equal(a, b)
        for a, b in zip(first, (logits, latent, shared_act), strict=True)
    )
    graph = _graph_replay(runner)
    for t in (logits, latent, shared_act):
        t.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert all(
        torch.equal(a, b)
        for a, b in zip(first, (logits, latent, shared_act), strict=True)
    )
    # The one-shot entry point writes the same values.
    for t in (logits, latent, shared_act):
        t.fill_(float("nan"))
    kimi_k3_latent_moe_front(
        x,
        w["gate_weight"],
        w["down_weight"],
        shared_gate_up,
        logits,
        latent,
        shared_act,
    )
    torch.cuda.synchronize()
    assert all(
        torch.equal(a, b)
        for a, b in zip(first, (logits, latent, shared_act), strict=True)
    )


@pytest.mark.parametrize("tokens", SMOKE_TOKENS)
@pytest.mark.parametrize("tp", SUPPORTED_TP)
def test_tail_matches_reference(tp, tokens):
    device = _require_program("tail", tp, tokens)
    rank = 0
    w = shard(_weights(device), tp, rank)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(628 + tokens)
    routed = (
        (torch.randn(1, tokens, LATENT, generator=gen) * 0.7)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    shared_act = (
        (torch.randn(tokens, i_local, generator=gen) * 0.6)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    out = torch.full(
        (tokens, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
    )
    y = torch.full((tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device)
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == ("decode" if tokens <= DECODE_MAX_T else "prefill")
    assert runner.kernel_keys == route_kernel_keys("tail", tp, tokens)
    assert runner.launch_count == (
        1 if tokens <= DECODE_MAX_T or runner.plan.get("fused_norm") else 2
    )
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    # Decode route: the fused norm reproduces the reference rounding exactly (FP32 statistics, BF16
    # round, BF16 weight product).  Prefill route: the one-pass RMSNorm kernel reduces the row in a
    # different FP32 order and lands within one BF16 ulp on a few elements (the Cake contract
    # receipts record the same for the production kernel), so it is held to the BF16 tolerance.
    # Both routes are bit-identical across re-launch and CUDA-graph replay below.
    _assert_close("y", y, expected["y"], ATOL, RTOL)
    if tokens <= DECODE_MAX_T:
        assert torch.equal(y, expected["y"]), (
            f"y differs from the reference in {int((y != expected['y']).sum())} elements"
        )
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    runner()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)
    graph = _graph_replay(runner)
    y.fill_(float("nan"))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)
    y.fill_(float("nan"))
    out.fill_(float("nan"))
    kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)


def test_tail_rank_slice_tp8():
    """A non-zero rank multiplies its own latent slice (k1_off) and its own shared-down shard."""
    tokens, tp, rank = 4, 8, 5
    device = _require_program("tail", tp, tokens)
    w = shard(_weights(device), tp, rank)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(777)
    routed = (
        (torch.randn(1, tokens, LATENT, generator=gen) * 0.7)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    shared_act = (
        (torch.randn(tokens, i_local, generator=gen) * 0.6)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
    y = torch.empty((tokens, LATENT), dtype=torch.bfloat16, device=device)
    kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    assert torch.equal(y, expected["y"])
    _assert_close("out", out, expected["out"], ATOL, RTOL)


def _tail_case(device, tp, rank, tokens, num_partials, seed):
    w = shard(_weights(device), tp, rank)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(seed)
    routed = (
        (torch.randn(num_partials, tokens, LATENT, generator=gen) * 0.7)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    shared_act = (
        (torch.randn(tokens, i_local, generator=gen) * 0.6)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    out = torch.full(
        (tokens, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
    )
    y = torch.full((tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device)
    return w, routed, shared_act, out, y


@pytest.mark.parametrize("tokens", (300, 511))
def test_tail_tp8_between_the_single_wave_rows(tokens):
    """TP8 tails for 256 < T < 512 run the norm launch with the late trigger before the GEMM."""
    tp, rank = 8, 0
    device = _require_program("tail", tp, tokens)
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, 1, 900 + tokens
    )
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == "prefill" and runner.launch_count == 2
    assert not runner.plan["early_trigger"] and not runner.plan["fused_norm"]
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    _assert_close("y", y, expected["y"], ATOL, RTOL)
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    graph = _graph_replay(runner)
    y.fill_(float("nan"))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)


@pytest.mark.parametrize("tokens", (3, 8, 16, 64))
@pytest.mark.parametrize("num_partials", (2, 3))
def test_tail_decode_sums_routed_partials(tokens, num_partials):
    """The fused decode tail reduces P >= 2 routed partials on device before the norm."""
    tp, rank = 8, 1
    device = _require_program("tail", tp, tokens)
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, num_partials, 1000 + tokens * 10 + num_partials
    )
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == "decode" and runner.launch_count == 1
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    assert torch.equal(y, expected["y"]), (
        f"y differs from the reference in {int((y != expected['y']).sum())} elements"
    )
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    runner()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)


def test_rejects_invalid_operands():
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device=device)
    gate = torch.zeros(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)
    down = torch.zeros(LATENT, HIDDEN, dtype=torch.bfloat16, device=device)
    sgu = torch.zeros(2 * 768, HIDDEN, dtype=torch.bfloat16, device=device)
    logits = torch.zeros(4, NUM_EXPERTS, dtype=torch.float32, device=device)
    latent = torch.zeros(4, LATENT, dtype=torch.bfloat16, device=device)
    act = torch.zeros(4, 768, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_front(
            x.float(), gate, down, sgu, logits, latent, act
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_front(
            x,
            gate,
            down,
            torch.zeros(2 * 512, HIDDEN, dtype=torch.bfloat16, device=device),
            logits,
            latent,
            act,
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_front(
            x, gate, down, sgu, logits.to(torch.bfloat16), latent, act
        )
    routed = torch.zeros(1, 4, LATENT, dtype=torch.bfloat16, device=device)
    nw = torch.zeros(LATENT, dtype=torch.bfloat16, device=device)
    up = torch.zeros(HIDDEN, LATENT, dtype=torch.bfloat16, device=device)
    sd = torch.zeros(HIDDEN, 768, dtype=torch.bfloat16, device=device)
    out = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device=device)
    y = torch.zeros(4, LATENT, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_tail(
            routed, nw, up, act, sd, out, tp=4, rank=0, y_workspace=y
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_tail(
            routed, nw, up, act, sd, out, tp=8, rank=8, y_workspace=y
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_tail(
            routed[0], nw, up, act, sd, out, tp=8, rank=0, y_workspace=y
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_tail(
            routed, nw, up, act, sd, out, tp=8, rank=0, y_workspace=y[:, : LATENT // 2]
        )
    with pytest.raises(ValueError):
        kimi_k3_latent_moe_front(
            x, gate, down, sgu, logits, latent, act, backend="cutlass"
        )
