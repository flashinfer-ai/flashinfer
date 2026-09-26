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
from flashinfer.experimental.kimi_k3_latent_moe.cake_backend import (
    DECODE_MAX_T,
    HIDDEN,
    LATENT,
    NUM_EXPERTS,
    RMS_EPS,
    ROW_TOKENS,
    SHARED_INTERMEDIATE,
    SM_COUNT,
    SUPPORTED_COMPUTE_CAPABILITIES,
    SUPPORTED_TP,
    decode_front_plan,
    decode_kernel_key,
    decode_symbol,
    decode_tail_plan,
    i_local_for_tp,
    prefill_tail_plan,
    required_kernel_keys,
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
SMOKE_TOKENS = (1, 8, 16, 128, 256, 4096)


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
    for plan in (tp1, tp8):
        assert (
            plan["kdepth"] == 1
            and 1 <= plan["stages"] <= cb.MAX_STAGES
            and not plan["fused"]
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
    # The resident B operand with staged rows serves the smallest TP8 tails; T16 falls back to the global-y protocol.
    assert decode_tail_plan(1, i_local_for_tp(8), 8)["smem_b1"]
    assert decode_tail_plan(1, i_local_for_tp(8), 8)["rows_smem"]
    assert decode_tail_plan(8, i_local_for_tp(8), 8)["smem_b1"]
    assert not decode_tail_plan(16, i_local_for_tp(8), 8)["smem_b1"]
    assert not decode_tail_plan(32, i_local_for_tp(1), 1)["smem_b1"]
    # A second routed partial disables the staged rows (P == 1 only).
    assert not decode_tail_plan(1, i_local_for_tp(8), 8, num_partials=2)["rows_smem"]
    with pytest.raises(ValueError):
        cb.n_pad_for(DECODE_MAX_T + 1)


def test_decode_symbol_encodes_the_plan():
    plan = decode_tail_plan(1, i_local_for_tp(8), 8)
    symbol = decode_symbol(plan)
    assert symbol.startswith("kimi_k3_latent_moe_decode_g112_n8_r")
    assert "_t0_56_0_k7_12_o7168_c2_f" in symbol and symbol.endswith("_sb_rs")
    assert decode_kernel_key(plan) == "decode:" + symbol
    front = decode_symbol(decode_front_plan(1, i_local_for_tp(1)))
    assert "_t7_28_96_k112_0_o3584_c1" in front and "_f" not in front


def test_split_plan_rules():
    # No remainder wave, short K loops (TP8: 19 iterations) or a small-reuse region keep whole tiles.
    whole = split_plan(148, 152, SM_COUNT, 1)
    assert whole == dict(
        num_items=148, full_items=148, sk_ipc=152, sk_max_seg=1, sk_total=0, sk_tiles=0
    )
    assert split_plan(56, 19, SM_COUNT, 1)["sk_tiles"] == 0
    assert split_plan(60, 152, SM_COUNT, 4)["sk_tiles"] == 0
    # TP1 T=2048 (224 pair tiles on 74 resident clusters): 2 x 74 whole + one stream-K wave (module docstring).
    sk = split_plan(224, 152, SM_COUNT, 8)
    assert sk["full_items"] == 148 and sk["sk_tiles"] == 76 and sk["sk_ipc"] == 157
    assert (
        sk["num_items"] == 148 + -(-76 * 152 // 157)
        and 1 < sk["sk_max_seg"] <= cb.MAX_SEG
    )


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
    assert cb.norm_early_trigger(300, 64, SM_COUNT) and not cb.norm_early_trigger(
        112, 64, SM_COUNT
    )
    assert cb.front_grid(cb.m_tiles_for(256), i_local_for_tp(1)) == 1 * 66 * 2
    assert cb.front_grid(cb.m_tiles_for(300), i_local_for_tp(8)) == 2 * 24 * 2


def test_route_keys_cover_the_row_set():
    keys = required_kernel_keys()
    assert len(keys) == len(set(keys))
    assert {k.split(":")[0] for k in keys} == {
        "decode",
        "front",
        "tail_norm",
        "tail_gemm",
    }
    assert "front:i6144" in keys and "front:i768" in keys
    assert "tail_gemm:tp1" in keys and "tail_gemm:tp8" in keys
    assert route_kernel_keys("front", 1, 128)[0].startswith("decode:")
    assert route_kernel_keys("front", 1, 256) == ("front:i6144",)
    assert route_kernel_keys("tail", 8, 256) == ("tail_norm:e1", "tail_gemm:tp8")
    assert len(route_kernel_keys("tail", 1, 16384)) == 2
    for stage in ("front", "tail"):
        for tp in SUPPORTED_TP:
            for tokens in ROW_TOKENS:
                for key in route_kernel_keys(stage, tp, tokens):
                    assert key in keys
    with pytest.raises(ValueError):
        route_kernel_keys("front", 4, 8)
    with pytest.raises(ValueError):
        route_kernel_keys("block", 1, 8)


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
    if not cb.generated_program_available(device, stage, tp, tokens):
        pytest.skip(
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
    assert runner.launch_count == (1 if tokens <= DECODE_MAX_T else 2)
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
